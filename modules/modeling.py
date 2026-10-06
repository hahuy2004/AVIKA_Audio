from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import json
import os

from modules.until_module import PreTrainedModel, AllGather, CrossEn, HardNegativeNCE
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from modules.co_attention_transformer_module import Co_attention_block

import math
from torch import Tensor
from functools import partial
import numpy as np
from typing import Tuple, Union, Optional

try:
    from mamba_ssm import Mamba
    from einops import rearrange
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    MUSE_AVAILABLE = True
except ImportError:
    MUSE_AVAILABLE = False

logger = logging.getLogger(__name__)
allgather = AllGather.apply

if MUSE_AVAILABLE:
    class Mamba_Out(nn.Module):
        def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba_type="none",
            if_devide_out=False,
            init_layer_scale=None,
        ):
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.d_conv = d_conv
            self.expand = expand
            self.d_inner = int(self.expand * self.d_model)
            self.use_fast_path = use_fast_path
            self.layer_idx = layer_idx
            self.bimamba_type = bimamba_type
            self.if_devide_out = if_devide_out

            self.init_layer_scale = init_layer_scale
            if init_layer_scale is not None:
                self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.activation = "silu"
            self.act = nn.SiLU()
            self.conv1d_b = nn.Conv1d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    groups=self.d_inner,
                    padding=d_conv - 1,
                    **factory_kwargs,
                )
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        def forward(self, hidden_states, inference_params=None):
            """
            hidden_states: (B, L, D)
            Returns: same shape as hidden_states
            """
            batch, seqlen, dim = hidden_states.shape
            # We do matmul and transpose BLH -> HBL at the same time
            xz = rearrange(
                self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen,
            )
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
            
            if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
                x, z = xz.chunk(2, dim=1)
                out = self.conv1d(x)
                x, z = xz.flip([-1]).chunk(2, dim=1)
                out_b = self.conv1d_b(x)
                if not self.if_devide_out:
                    out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                
            if self.init_layer_scale is not None:
                out = out * self.gamma    
            return out

    class MultiheadAttention_flash(nn.MultiheadAttention):
        def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                    need_weights: bool = True, attn_mask: Optional[Tensor] = None):

            return flash_attn_func(
                    q=query, k=key, v=value, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), alibi_slopes=None, deterministic=False)

    class LayerNorm_conv(nn.LayerNorm):
        """Subclass torch's LayerNorm to handle fp16."""
        def __init__(self, normalized_shape):
            super().__init__(normalized_shape=normalized_shape)

        def forward(self, x: torch.Tensor):
            x = x.permute(0,2,3,1)
            orig_type = x.dtype
            ret = super().forward(x.type(torch.float32))# add ssf
            return ret.type(orig_type).permute(0,3,1,2)

    class Mamba_head(nn.Module):
        def __init__(self, embed_dim, layer_num=0.1):
            super().__init__()
            self.embed_dim = embed_dim
            self.mamba = Mamba(self.embed_dim, d_conv=4, bimamba_type='v2', use_fast_path=True, expand=1)
            # self.mamba_out = Mamba_Out(self.embed_dim, d_conv=1, bimamba_type='v2', use_fast_path=True, expand=1)
            # self.transformer = nn.MultiheadAttention(self.embed_dim, self.embed_dim // 64)
            # self.flash_attn = MultiheadAttention_flash(self.embed_dim, self.embed_dim // 64)
            self.layer_norm1 = nn.LayerNorm(self.embed_dim)
            # self.layer_norm1 = RMSNorm(hidden_size=self.embed_dim)

            self.proj_drop = nn.Dropout(layer_num)
            self.temporal_fc = nn.Linear(self.embed_dim, self.embed_dim)
            nn.init.constant_(self.temporal_fc.weight, 0.)
            nn.init.constant_(self.temporal_fc.bias, 0.)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            causal_attention_mask=None,
        ):
            residual = hidden_states
            hidden_states = self.layer_norm1(hidden_states)
            # hidden_states = self.transformer((hidden_states, None))[0] [L,B,D]
            hidden_states = self.mamba(hidden_states)
            # hidden_states = self.mamba_out(hidden_states)
            # hidden_states = self.flash_attn(hidden_states, hidden_states, hidden_states, need_weights=False, attn_mask=None)
            res_temporal = self.proj_drop(hidden_states.contiguous())
            
            res_temporal = self.temporal_fc(res_temporal)
            hidden_states = residual + res_temporal
            outputs = hidden_states

            return outputs
else:
    Mamba_Out = None
    MultiheadAttention_flash = None
    LayerNorm_conv = None
    Mamba_head = None

class CLIPKG4VidPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIPKG4VidPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        # Note: For video
        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        # -------------------------------------------------
        
        # ---------- New: Create for narration ------------
        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if not contain_cross:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_caption_position = False
            for key in state_dict.keys():
                if key.find("caption_position_embeddings") > -1:
                    contain_caption_position = True
                    break
            if not contain_caption_position:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["caption_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerCaption.")] = val.clone()
                            continue
        # -------------------------------------------------

        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIPKG4Vid(CLIPKG4VidPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIPKG4Vid, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        # ---------- New: Co-Attention Transformer for tightTransf ------------
        # Based on Cap4Video
        # Mục đích: Tạo một mô-đun transformer để thực hiện co-attention giữa video và text, giúp tăng cường khả năng tương tác giữa hai loại dữ liệu này. 
        # Điều này đặc biệt hữu ích trong trường hợp tightTransf, nơi mà sự tương tác chặt chẽ giữa video và text là cần thiết để đạt được hiệu suất tốt nhất.
        if getattr(self.task_config, "co_attention_block", False):
            self.co_connetion_transformer_model_block = nn.Sequential(*[Co_attention_block(hidden_size=embed_dim, num_attention_heads=transformer_heads, dropout_rate=0.1) for i in range(1)])
        else:
            self.co_connetion_transformer_model_block = None
        # ---------------------------------------------------------------------

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
            # -------------- New: Create for narration -------------
            self.caption_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
            # ------------------------------------------------------ 
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                    heads=transformer_heads, )
            # -------------- New: Create for narration -------------
            self.transformerCaption = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                    heads=transformer_heads)
            # ------------------------------------------------------
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                        batch_first=True, bidirectional=False, num_layers=1)
            # -------------- New: Create for narration -------------
            self.lstm_caption = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                        batch_first=True, bidirectional=False, num_layers=1)
            # ------------------------------------------------------

        if self.sim_header == "MUSE":
            if not MUSE_AVAILABLE:
                raise ImportError("Run sim_header='MUSE' but not install required dependencies."
                                  "Requirement install: pip install einops mamba-ssm[causal-conv1d] flash-attn --no-build-isolation")
            scale_factors = [0.5, 1.0, 2.0] # feature scales used
            dim = transformer_width
            mamba_stage_list = []
            for idx, scale in enumerate(scale_factors):
                out_dim = dim
                out_channels = dim
                if scale == 4.0:
                    layers = [
                        nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                        LayerNorm_conv(dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                    ]
                    out_dim = dim // 4
                elif scale == 2.0:
                    layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                    out_dim = dim // 2
                elif scale == 1.0:
                    layers = []
                elif scale == 0.5:
                    layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
                elif scale == 0.25:
                    layers = [nn.MaxPool2d(kernel_size=4, stride=4)]
                else:
                    raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

                layers.extend(
                    [
                        nn.Conv2d(
                            out_dim,
                            out_channels,
                            kernel_size=1,
                        ),
                        LayerNorm_conv(out_channels),
                        nn.GELU(),
                        nn.Conv2d(
                            out_channels,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                        ),
                        LayerNorm_conv(out_channels)
                    ]
                )
                mamba_stage_list.append(nn.Sequential(*layers))
            # Use ModuleList so DDP/device placement handles all parameters correctly
            self.mamba_stages = nn.ModuleList(mamba_stage_list)
            depth = 4
            dpr = np.linspace(0, 0.1, depth)
            self.MS_mamba = nn.ModuleList([Mamba_head(transformer_width, dpr[i]) for i in range(depth)])
            # -------------- New: Mamba stack for narration sequence (MUSE) -------------
            depth_nar = 4
            dpr_nar = np.linspace(0, 0.1, depth_nar)
            self.MS_mamba_nar = nn.ModuleList([Mamba_head(transformer_width, dpr_nar[i]) for i in range(depth_nar)])
            # --------------------------------------------------------------------------

        # ---------- New: Weighted Token-Video Interaction for tightTransf -------------
        # Based on Cap4Video (https://github.com/whwu95/Cap4Video)   
        self.interaction = 'wti'
        if self.interaction == 'wti':
            # For raw text data
            self.text_weight_fc = nn.Sequential(
                nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                nn.Linear(transformer_width, 1))
            # For raw video data
            self.video_weight_fc = nn.Sequential(
                nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                nn.Linear(transformer_width, 1))
            # For caption (narration) data
            self.caption_weight_fc = nn.Sequential(
                nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                nn.Linear(transformer_width, 1))
        # ---------------------------------------------------------------------------------

        self.loss_fct = CrossEn()
        self.apply(self.init_weights)

    # ------ Function for [Phase 1]: Entry Point & Feature Extraction ----------
    # Note in image: Frame encoder, Text Encoder, Narration Encoder
    # Output: [B, 1, D] - CLS token embedding
    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden
    
    # Original version: get_sequence_output of CLIP4Clip (without words_hidden - return_hidden=False)
    # Output: sequence_output, words_output
    # - sequence_output: CLS token embedding [B, 1, D]
    # - word_output: All word embeddings [B, N_words, D]
    def get_sequence_words_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        # Encoder text like CLIP4Clip
        # return_hidden=True to get both sequence_hidden (CLS token) and words_hidden (all token embeddings)
        sequence_hidden, words_hidden = self.clip.encode_text(input_ids, return_hidden=True)
        sequence_hidden = sequence_hidden.float().view(bs_pair, -1, sequence_hidden.size(-1))
        words_hidden = words_hidden.float()

        return sequence_hidden, words_hidden

    # Output: [B, N_frames, D] - Frame-level features
    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        if self.sim_header == "MUSE":
            # MUSE needs CLS tokens for Co-Attention and all patch tokens for multi-scale Mamba.
            # encode_image(return_hidden=True) returns (cls [B*T, D], hidden [B*T, L, D])
            visual_cls, visual_hidden = self.clip.encode_image(video, return_hidden=True, video_frame=video_frame)
            visual_cls = visual_cls.float().view(bs_pair, -1, visual_cls.size(-1))           # [B, T, D]
            visual_hidden = visual_hidden.float().view(bs_pair, -1, visual_hidden.size(-1))  # [B, T*L, D]
            return visual_cls, visual_hidden  # tuple: (CLS, all-patch-tokens)
        else:
            visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
            visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
            return visual_hidden

    # ---------- New: Get narration output -----------
    # Output: [B, N_narration_words, D] - Narration word-level features
    def get_narration_output(self, narration, attention_mask, batch_mask, shaped=False, video_frame=-1):
        if shaped is False:
            narration = narration.view(-1, narration.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            batch_mask = batch_mask.view(-1, batch_mask.shape[-1])

        bs_pair = batch_mask.size(0)
        frame_num = batch_mask.size(1)
        narration_hidden = self.clip.encode_text(narration).float()
        narration_hidden = narration_hidden.view(bs_pair, frame_num, narration_hidden.size(-1))

        return narration_hidden
    # ------------------------------------------------------

    # Original version: get_sequence_visual_output of CLIP4Clip (without narration)
    # Output: sequence_output, narration_output, visual_output
    # - sequence_output: CLS token embedding [B, 1, D]
    # - narration_output: All narration word embeddings [B, N_narration_words, D]
    # - visual_output: All frame embeddings [B, N_frames, D]
    def get_sequence_narration_visual_output(self, input_ids, token_type_ids, attention_mask, narration, 
                                   nar_attention_mask, nar_batch_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            # --------- New: Reshape for narration input -----------
            narration = narration.view(-1, narration.shape[-1])
            nar_attention_mask = nar_attention_mask.view(-1, nar_attention_mask.shape[-1])
            nar_batch_mask = nar_batch_mask.view(-1, nar_batch_mask.shape[-1])
            # ------------------------------------------------------
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts


        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        # --------- New: Get narration output ------------------
        narration_output = self.get_narration_output(narration, nar_attention_mask, nar_batch_mask, shaped=True)
        # ------------------------------------------------------
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)
        return sequence_output, narration_output, visual_output

    # Original version: get_sequence_visual_output of CLIP4Clip (without narration)
    # Output: sequence_output, words_output, narration_output, visual_output
    # - sequence_output: CLS token embedding [B, 1, D]
    # - words_output: All word embeddings [B, N_words, D]
    # - narration_output: All narration word embeddings [B, N_narration_words, D]
    # - visual_output: All frame embeddings [B, N_frames, D]
    def get_sequence_words_narration_visual_output(self, input_ids, token_type_ids, attention_mask, narration, 
                                    nar_attention_mask, nar_batch_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            # ---------- New: Reshape for narration input -----------
            narration = narration.view(-1, narration.shape[-1])
            nar_attention_mask = nar_attention_mask.view(-1, nar_attention_mask.shape[-1])
            nar_batch_mask = nar_batch_mask.view(-1, nar_batch_mask.shape[-1])
            # ------------------------------------------------------
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        # sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        # ---------- New: Get both sequence_output and words_output -----------
        sequence_output, words_output = self.get_sequence_words_output(input_ids, token_type_ids, attention_mask, shaped=True)
        # ------------------------------------------------------
        # ---------- New: Get narration output -----------------
        narration_output = self.get_narration_output(narration, nar_attention_mask, nar_batch_mask, shaped=True)
        # -------------------------------------------------------
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)
        # return sequence_output, visual_output
        return sequence_output, words_output, narration_output, visual_output
    
    # ------ End phase 1 ----------------
    
    # --------- Function for [Phase 2]: Multi-granularity Matching & Loss Computation (Co-Attention + Weighted Token-Video Interaction + Multi-granularity Similarity + Loss Computation) ----------
    # --- Function for sub-phase [2.2]
    # Note in image: Module Temporal Block (agg_video_feat, agg_narration_feat)
    
    # --- Based on Cap4Video
    # --- Agg video features with different sim_header types (meanP, seqLSTM, seqTransf, MUSE)
    def agg_video_feat(self, visual_output, video_mask, sim_header="meanP", visual_hidden=None):
        visual_output = visual_output.contiguous()
        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif "seqTransf" in sim_header:
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original
        elif sim_header == "MUSE":
            # visual_output: [B, T, D]   - CLS tokens already enhanced by Co-Attention
            # visual_hidden: [B, T*L, D] - raw patch tokens from the CLIP encoder (L=50)
            assert visual_hidden is not None, "MUSE requires visual_hidden (patch tokens) in agg_video_feat"
            B, T, C = visual_output.shape
            _, TL, _ = visual_hidden.shape
            L = TL // T
            H = W = int(math.sqrt(L - 1))

            # Use Co-Attention-enhanced CLS as the frame-level base (not the raw CLS from patches)
            visual_output_original = visual_output  # [B, T, D]
            # Extract spatial patch tokens (skip index 0 = CLS) and reshape for 2D conv
            visual_mamba = visual_hidden.view(B, T, L, C)[:, :, 1:, :].reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # [B*T, C, H, W]

            agg_mode = "scale_wise"
            assert agg_mode in ["scale_wise", "spatial_wise", "frame_wise"]
            visual_mamba_ms = []
            for stage in self.mamba_stages:
                visual_mamba_ms.append(stage(visual_mamba).view(B, T, C, -1).permute(0, 1, 3, 2))
            if agg_mode == "scale_wise":
                visual_mamba_ms = [vi.view(B, -1, C) for vi in visual_mamba_ms]
                visual_mamba_st = torch.cat(visual_mamba_ms, dim=1)
                visual_mamba_output = torch.cat((visual_output_original, visual_mamba_st), dim=1)
                # visual_mamba_output = visual_output_original
            elif agg_mode == "spatial_wise":
                visual_mamba_ms = [vi.mean(dim=1).squeeze() for vi in visual_mamba_ms]
                visual_mamba_st = torch.cat(visual_mamba_ms, dim=1)
                visual_mamba_output = torch.cat((visual_output_original, visual_mamba_st), dim=1)
            elif agg_mode == "frame_wise":
                visual_mamba_output = []
                for t in range(T):
                    visual_mamba_output.append(visual_output_original[:, t, :].unsqueeze(1))
                    for vi in visual_mamba_ms:
                        visual_mamba_output.append(vi[:, t, :])
                visual_mamba_output = torch.cat(visual_mamba_output, dim=1)

            for layer in range(len(self.MS_mamba)):
                visual_mamba_output = self.MS_mamba[layer](visual_mamba_output)
            # visual_mamba_output = self.transformerClip(visual_mamba_output, None)
            if agg_mode == "frame_wise":
                visual_output = visual_mamba_output[:, ::visual_mamba_output.shape[1] // T, :].contiguous()
            else:
                visual_output = visual_mamba_output[:, :T, :].contiguous()
        
        return visual_output
    
    # --- Based on Cap4Video
    def agg_narration_feat(self, narration_output, narrations_batch_mask, sim_header="meanP"):
        narration_output = narration_output.contiguous()
        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            narration_output_original = narration_output
            narration_output = pack_padded_sequence(narration_output, torch.sum(narrations_batch_mask, dim=-1).cpu(),
                                                batch_first=True, enforce_sorted=False)
            narration_output, _ = self.lstm_caption(narration_output)
            if self.training: self.lstm_caption.flatten_parameters()
            narration_output, _ = pad_packed_sequence(narration_output, batch_first=True)
            narration_output = torch.cat((narration_output, narration_output_original[:, narration_output.size(1):, ...].contiguous()), dim=1)
            narration_output = narration_output + narration_output_original
        elif "seqTransf" in sim_header:
            # Sequential type: Transformer Encoder
            narration_output_original = narration_output
            seq_length = narration_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=narration_output.device)
            position_ids = position_ids.unsqueeze(0).expand(narration_output.size(0), -1)
            caption_position_embeddings = self.caption_position_embeddings(position_ids)
            narration_output = narration_output + caption_position_embeddings

            extended_narrations_batch_mask = (1.0 - narrations_batch_mask.unsqueeze(1)) * -1000000.0
            extended_narrations_batch_mask = extended_narrations_batch_mask.expand(-1, narrations_batch_mask.size(1), -1)
            narration_output = narration_output.permute(1, 0, 2)  # NLD -> LND
            narration_output = self.transformerCaption(narration_output, extended_narrations_batch_mask)
            narration_output = narration_output.permute(1, 0, 2)  # LND -> NLD
            narration_output = narration_output + narration_output_original
        elif sim_header == "MUSE":
            # Apply Mamba temporal modeling across the narration sequence [B, N_nar, D]
            for layer in range(len(self.MS_mamba_nar)):
                narration_output = self.MS_mamba_nar[layer](narration_output)
            
        return narration_output
    
    # --- Function for sub-phase [2.4] - [Section 3.4] - Query-Aware Adaptive Filtering
    # Note in image: Module Query-Aware Adaptive Filtering

    # Based on Cap4Video (https://github.com/whwu95/Cap4Video)
    def wti_interaction(self, word_output, word_mask, visual_output, temperature=1):
        B_t, N_t, _ = word_output.shape
        B_v, N_v, _ = visual_output.shape

        text_weight = self.text_weight_fc(word_output).squeeze(2)  # B_t x N_t x D -> B_t x N_t
        text_weight.masked_fill_(torch.tensor((1 - word_mask), dtype=torch.bool), float("-inf"))
        text_weight = F.softmax(text_weight/temperature, dim=-1)  # B_t x N_t
        text_weight_expanded = text_weight.unsqueeze(1).expand(B_t, B_v, N_t) #B_t X B_v X N_t

        return text_weight_expanded
    
    # Note in image: Sub-module softmax in Module Query-Aware Adaptive Filtering
    def get_softmax_weights(self, sequence_output, narration_output, existing_mask, temperature):
        # Normalize sequence and narration outputs
        narration_output_norm = F.normalize(narration_output, p=2, dim=-1)
        sequence_output_norm = F.normalize(sequence_output, p=2, dim=-1)

        # Check batch sizes of sequence output and narration output
        seq_batch_size, nar_batch_size = sequence_output_norm.size(0), narration_output_norm.size(0)

        sequence_output_norm = sequence_output_norm.transpose(0, 1)
        if seq_batch_size == nar_batch_size:
            # Expand sequence output from [b, 1, d] to [b, b, d]
            sequence_output_expanded = sequence_output_norm.repeat(sequence_output_norm.size(1), 1, 1)
        else:
            # Expand sequence output from [seq_b, 1, d] to [nar_b, seq_b, d]
            sequence_output_expanded = sequence_output_norm.repeat(narration_output_norm.size(0), 1, 1)

        # Transpose narration output from [b, f, d] to [b, d, f]
        narration_output_transposed = narration_output_norm.transpose(1, 2)

        # Calculate cosine similarity
        cosine_sim = torch.matmul(sequence_output_expanded, narration_output_transposed)  
        cosine_sim = cosine_sim.transpose(0, 1)
        # Resulting shape: [sequence_batch_size, narration_batch_size, f]

        # Modify masked_cosine_sim to set positions with 0 to a very small value
        masked_cosine_sim = cosine_sim.clone()
        masked_cosine_sim[existing_mask == 0] = float('-inf')
        
        # Apply softmax using temperature parameter
        weights = F.softmax(masked_cosine_sim / temperature, dim=-1) 

        return weights
    
    # Note in image: Sub-module Filtering & softmax in Module Query-Aware Adaptive Filtering
    # Chỉ giữ lại top-p% frames/narrations có probability cao nhất
    # Loại bỏ noise, tập trung vào relevant content
    def apply_nucleus_filtering(self, weights, p):
        # Shape of softmax_weights: [sequence batchsize, video_batchsize, number of frames]
        
        # Sort the probability distribution in descending order
        sorted_weights, sorted_indices = torch.sort(weights, descending=True, dim=-1)
        
        # Find indices where cumulative probability exceeds p
        cumulative_probs = torch.cumsum(sorted_weights, dim=-1)
        cutoff_indices = cumulative_probs > p
        # Keep the first index to ensure at least one option remains
        cutoff_indices[..., 0] = False
        
        # Set probabilities of indices exceeding p to 0
        sorted_weights[cutoff_indices] = 0
        
        # Restore the probability distribution to the original order
        _, original_indices = torch.sort(sorted_indices, descending=False, dim=-1)
        
        filtered_weights = torch.gather(sorted_weights, -1, original_indices)
        
        # Re-normalize the probability distribution
        filtered_weights_sum = filtered_weights.sum(dim=-1, keepdim=True)
        normalized_filtered_weights = filtered_weights / filtered_weights_sum

        return normalized_filtered_weights

    # Note in image: Adjust weights and mask after filtering and softmax
    # If a weight is 0, set the corresponding mask to 0 to exclude it from further computations
    def adjust_weights_and_mask(self, weights, existing_mask):
        weights_masked = weights * existing_mask
        updated_existing_mask = existing_mask.clone()
        updated_existing_mask[weights == 0] = 0
        return weights_masked, updated_existing_mask
    
    # Note in image: Implement Module Query-Aware Adaptive Filtering to get weights and adjusted masks for words, video frames, and narration tokens 
    def get_weights_and_mask(self, sequence_output, word_output, visual_output, narration_output, word_mask, video_mask, narration_mask):
        temperature = self.task_config.temperature
        nucleus_P = self.task_config.nucleus_P

        # Expand the masking vectors to match the sequence dimensions
        word_mask_expanded = word_mask.unsqueeze(1).repeat(1, visual_output.size(0), 1)
        video_mask_expanded = video_mask.unsqueeze(0).repeat(sequence_output.size(0), 1, 1)
        narration_mask_expanded = narration_mask.unsqueeze(0).repeat(sequence_output.size(0), 1, 1)

        word_weights = self.wti_interaction(word_output, word_mask, visual_output, temperature)
        # Calculate softmax weights between sequences
        seq_nar_weights = self.get_softmax_weights(sequence_output, narration_output, narration_mask_expanded, temperature)
        seq_vis_weights = self.get_softmax_weights(sequence_output, visual_output, video_mask_expanded, temperature)
        seq_nar_weights = self.apply_nucleus_filtering(seq_nar_weights, nucleus_P)
        seq_vis_weights = self.apply_nucleus_filtering(seq_vis_weights, nucleus_P)

        word_weights, adjusted_word_mask = self.adjust_weights_and_mask(word_weights, word_mask_expanded)
        vis_weights, adjusted_vis_mask = self.adjust_weights_and_mask(seq_vis_weights, video_mask_expanded)
        nar_weights, adjusted_nar_mask = self.adjust_weights_and_mask(seq_nar_weights, narration_mask_expanded)

        return word_weights, vis_weights, nar_weights, adjusted_word_mask, adjusted_vis_mask, adjusted_nar_mask
    
    # --- Function for sub-phase [2.5]
    # Note in image: Sub-module Multi-granularity Matching (Multi-granularity Similarity Computation)

    # Same CLIP4Clip
    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask
    
    def _weighted_pooling_for_similarity_visual(self, visual_output, video_mask, visual_weights):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_weights_un = visual_weights.to(dtype=torch.float).unsqueeze(-1)
        visual_output_weighted = visual_output * visual_weights_un
        visual_output_masked = visual_output_weighted * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un*visual_weights_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output_masked, dim=1) / video_mask_un_sum
        return video_out

    # [Section 3.5] - Coarse-grained matching / global matching
    # --- Based on Cap4Video: 
    # Xem ý nghĩa tổng thể của cả câu tìm kiếm có khớp với nội dung chính của toàn bộ video hay không
    def _loose_coarse_similarity(self, sequence_output, visual_output, word_mask, video_mask, visual_weights):
        device = sequence_output.device

        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        sequence_batch_size, _, _ = sequence_output.shape
        visual_batch_size, frame_count, embedding_dim = visual_output.shape
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        
        visual_output_pooled = torch.zeros(sequence_batch_size, visual_batch_size, embedding_dim).to(device)
        for i in range(sequence_batch_size):
            pooled_visual_output = self._weighted_pooling_for_similarity_visual(visual_output, video_mask[i], visual_weights[i])
            visual_output_pooled[i] = pooled_visual_output
        
        # [seq_batch, vis_batch, embed]
        visual_output_pooled = visual_output_pooled / visual_output_pooled.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.matmul(sequence_output, visual_output_pooled.transpose(1,2))

        # [seq_batch, 1, vis_batch] -> [seq_batch, vis_batch]
        retrieve_logits = retrieve_logits.squeeze(1)

        return retrieve_logits
    
    # [Section 3.5] - Fine-grained matching / local matching
    # --- Based on Cap4Video (https://github.com/whwu95/Cap4Video)
    # Tính độ tương đồng giữa từng từ đơn lẻ (word embeddings) trong câu với từng khung hình (frame embeddings) của video.
    def _loose_fine_similarity(self, word_output, visual_output, word_mask, video_mask, word_weights, visual_weights):
        word_output, visual_output = word_output.contiguous(), visual_output.contiguous()
        word_output = word_output / word_output.norm(dim=-1, keepdim=True)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [word_output, visual_output])
        retrieve_logits = torch.einsum('abtv,abt->abtv', [retrieve_logits, word_mask])
        retrieve_logits = torch.einsum('abtv,abv->abtv', [retrieve_logits, video_mask])

        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('abt,abt->ab', [t2v_logits, word_weights])

        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('abv,abv->ab', [v2t_logits, visual_weights])
        retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        return retrieve_logits

    # --- CORE MATCHING PIPELINE: Co-Attention + Weighted Token-Video Interaction + Multi-granularity Similarity + Loss Computation
    def get_similarity_logits(self, sequence_output, word_output, visual_output, narration_output, 
                                attention_mask, video_mask, narration_mask, shaped=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            narration_mask = narration_mask.view(-1, narration_mask.shape[-1])

        # co-attention method based on Cap4Video (Wu et al., 2023b).
        cross_video_mask = video_mask.reshape(video_mask.shape[0],1,1,video_mask.shape[-1])
        cross_narration_mask = torch.ones((narration_mask.shape[0],narration_mask.shape[1]),device=narration_output.device)
        cross_narration_mask = cross_narration_mask.reshape(cross_narration_mask.shape[0],1,1,cross_narration_mask.shape[-1])

        # ----- Unpack MUSE visual tuple (CLS for Co-Attention, patches for Mamba) -----
        visual_hidden_patches = None
        if self.sim_header == "MUSE":
            visual_output, visual_hidden_patches = visual_output  # [B,T,D], [B,T*L,D]

        # ----- Phase [2.1]: Co-Attention Fusion (visual CLS ↔ narration) -----
        # Note in image: Module Frame-Level Co-attention Block
        if getattr(self.task_config, "co_attention_block", False) and self.co_connetion_transformer_model_block is not None:
            for co_layer in self.co_connetion_transformer_model_block:
                visual_output, narration_output, co_attention_probs = co_layer(visual_output, cross_video_mask, narration_output, cross_narration_mask)

        # ----- Phase [2.2]: Temporal Aggregation (seqLSTM/seqTransformer/MUSE) -----
        # Note in image: Module Temporal Block (For visual and narration features)
        visual_output = self.agg_video_feat(visual_output, video_mask, self.sim_header, visual_hidden=visual_hidden_patches)
        narration_output = self.agg_narration_feat(narration_output, narration_mask, self.sim_header)

        # ----- Phase [2.3] Distributed Training (if training) -----
        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            narration_output = allgather(narration_output, self.task_config)
            narration_mask = allgather(narration_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            word_output = allgather(word_output, self.task_config)
            attention_mask = allgather(attention_mask, self.task_config)
            torch.distributed.barrier()

        # ----- Phase [2.4] Adaptive Weighting & Filtering (WTI + Nucleus Filtering) -----
        # Note in image: Module Query-Aware Adaptive Filtering 
        # to get weights and adjusted masks for words, video frames, and narration tokens
        word_weights, visual_weights, narration_weights, word_mask, video_mask, narration_mask = self.get_weights_and_mask(sequence_output, 
                                    word_output, visual_output, narration_output, attention_mask, video_mask, narration_mask)
        
        # ----- Phase [2.5] Multi-granularity Similarity Computation (Coarse-grained Matching + Fine-grained Matching) -----
        # Note in image: Sub-module Multi-granularity Matching (Multi-granularity Similarity Computation)
        # Compute both coarse-grained similarity (between sequence CLS and pooled visual/narration features) 
        # and fine-grained similarity (between word embeddings and frame/narration token embeddings), 
        # incorporating the adaptive weights and masks.
        retrieve_logits_T2V_coarse = self._loose_coarse_similarity(sequence_output, visual_output, word_mask, video_mask, visual_weights)
        retrieve_logits_T2N_coarse = self._loose_coarse_similarity(sequence_output, narration_output, word_mask, narration_mask, narration_weights)
        retrieve_logits_T2V_fine = self._loose_fine_similarity(word_output, visual_output, word_mask, video_mask, word_weights, visual_weights)
        retrieve_logits_T2N_fine = self._loose_fine_similarity(word_output, narration_output, word_mask, narration_mask, word_weights, narration_weights)

        return retrieve_logits_T2V_coarse, retrieve_logits_T2N_coarse, retrieve_logits_T2V_fine, retrieve_logits_T2N_fine
    # -------------------------------------------------------------

    # --- Function for sub-phase [3]: NCE Loss Computation (L_NCE)
    # Note in image: Sub-module NCE Loss (L_NCE)
    def cal_nce_loss(self, sim_matrix):
        logit_scale = self.clip.logit_scale.exp()
        sim_matrix = logit_scale * sim_matrix
        sim_loss1 = self.loss_fct(sim_matrix)
        sim_loss2 = self.loss_fct(sim_matrix.T)
        sim_loss = (sim_loss1 + sim_loss2) / 2
        return sim_loss 
    
    # --- Function for sub-phase [4]: Cross-View Hard Negative Loss (L_CVH) - [Section 3.6.1] -------------
    # Note in image: Sub-module Cross-View Hard Negative Loss (L_CVH)
    # Calculate the hard negative hinge loss based on the similarity matrix and hard negative indices
    # [Section 3.6.1] - Cross-View Hard Negative Loss (L_CVH)
    def get_std_hard_negative_indices(self, sim_matrix):
        threshold_factor = self.task_config.hard_negative_selection_factor
        positive_similarities = sim_matrix.diag()
        means = torch.mean(sim_matrix, dim=1)
        stds = torch.std(sim_matrix, dim=1)
        thresholds = positive_similarities - threshold_factor * stds

        sim_matrix_copy = sim_matrix.clone()
        sim_matrix_copy.fill_diagonal_(-float('inf'))
        candidates = sim_matrix_copy > thresholds.unsqueeze(1)
        hard_negative_indices = [candidates[i].nonzero(as_tuple=True)[0] for i in range(sim_matrix_copy.size(0))]
        
        return hard_negative_indices, positive_similarities-thresholds
    
    def cal_hinge_loss(self, sim_matrix, hard_negative_indices, margin):
        pos_scores = sim_matrix.diagonal()

        hinge_losses = []
        for i, hn_indices in enumerate(hard_negative_indices):
            indices = hn_indices[hn_indices != i]
            neg_scores = sim_matrix[i, indices]
            hinge_loss = torch.nn.functional.relu(neg_scores - pos_scores[i] + margin[i]).sum()
            hinge_losses.append(hinge_loss)

        final_hinge_loss = torch.stack(hinge_losses).mean()
        
        return final_hinge_loss

    def get_hn_hinge_loss(self, tv_sim_matrix, tn_sim_matrix):
        margin_factor = self.task_config.hard_negative_loss_factor
        device = tv_sim_matrix.device

        # Get hard negative indices and thresholds
        hard_negative_indices_tv, threshold_tv = self.get_std_hard_negative_indices(tv_sim_matrix)
        hard_negative_indices_tn, threshold_tn = self.get_std_hard_negative_indices(tn_sim_matrix)

        # Transpose matrices and get hard negative indices for transposed matrices
        hard_negative_indices_tv_T, threshold_tv_T = self.get_std_hard_negative_indices(tv_sim_matrix.T)
        hard_negative_indices_tn_T, threshold_tn_T = self.get_std_hard_negative_indices(tn_sim_matrix.T)

        hard_negative_indices_tv = [hn_tv.to(device) for hn_tv in hard_negative_indices_tv]
        hard_negative_indices_tn = [hn_tc.to(device) for hn_tc in hard_negative_indices_tn]
        hard_negative_indices_tv_T = [hn_tv.to(device) for hn_tv in hard_negative_indices_tv_T]
        hard_negative_indices_tn_T = [hn_tc.to(device) for hn_tc in hard_negative_indices_tn_T]

        # Combine hard negative indices
        combined_hard_negative_indices = [torch.unique(torch.cat((hn_tv, hn_tc))) for hn_tv, hn_tc in zip(hard_negative_indices_tv, hard_negative_indices_tn)]
        combined_hard_negative_indices_T = [torch.unique(torch.cat((hn_tv, hn_tc))) for hn_tv, hn_tc in zip(hard_negative_indices_tv_T, hard_negative_indices_tn_T)]
        
        # Calculate hard negative hinge losses based on threshold margin
        tv_hard_neg_loss = self.cal_hinge_loss(tv_sim_matrix, combined_hard_negative_indices, margin_factor*threshold_tv)
        tn_hard_neg_loss = self.cal_hinge_loss(tn_sim_matrix, combined_hard_negative_indices, margin_factor*threshold_tn)
        tv_hard_neg_loss_T = self.cal_hinge_loss(tv_sim_matrix.T, combined_hard_negative_indices_T, margin_factor*threshold_tv_T)
        tn_hard_neg_loss_T = self.cal_hinge_loss(tn_sim_matrix.T, combined_hard_negative_indices_T, margin_factor*threshold_tn_T)

        tv_loss = (tv_hard_neg_loss + tv_hard_neg_loss_T)/2
        tn_loss = (tn_hard_neg_loss + tn_hard_neg_loss_T)/2

        return tv_loss, tn_loss
    # --------------------------------------------------------------

    # Forward function
    # Using for training and evaluation (retrieval) for both video-to-text and text-to-video
        # Input: (text_query, video_frames, narration_captions)
        # ↓
        # 1. Extract features via CLIP encoders
        # ↓
        # 2. Co-Attention (visual ↔ narration)
        # ↓
        # 3. Temporal aggregation (LSTM/Transformer)
        # ↓
        # 4. Compute adaptive weights (WTI + Nucleus Filtering)
        # ↓
        # 5. Multi-granularity matching (coarse + fine)
        # ↓
        # 6. Compute similarity matrices
        # ↓
        # 7. Calculate losses (NCE + Hard Negative)
        # loss = tv_nce_loss + tn_nce_loss + α * (tv_hard_neg_loss + tn_hard_neg_loss)
        # ↓
        # Output: Loss (training) / Similarity scores (inference)
    def forward(self, input_ids, token_type_ids, attention_mask, video, 
                video_mask, narration, narration_word_mask, narration_mask):
        
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # ---------- New: Reshape for narration input -----------
        narration = narration.view(-1, narration.shape[-1])
        narration_word_mask = narration_word_mask.view(-1, narration_word_mask.shape[-1])
        narration_mask = narration_mask.view(-1, narration_mask.shape[-1])
        # ------------------------------------------------------

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        # Phase 1: Entry Point & Feature Extraction
        # ├── [Input Reshaping]
        # │   ├── input_ids, attention_mask (Text query)
        # │   ├── video, video_mask (Video frames)
        # │   └── narration, narration_word_mask, narration_mask (Narrations)
        # │
        # └── get_sequence_words_narration_visual_output()
        #     ├── get_sequence_words_output()
        #     │   └── self.clip.encode_text(input_ids, return_hidden=True)  [CLIP Text Encoder]
        #     │       └── Returns: sequence_output [B,1,D], words_output [B,N_words,D]
        #     │
        #     ├── get_narration_output()
        #     │   └── self.clip.encode_text(narration)  [CLIP Text Encoder]
        #     │       └── Returns: narration_output [B,N_narrations,D]
        #     │
        #     └── get_visual_output()
        #         └── self.clip.encode_image(video, video_frame)  [CLIP Visual Encoder]
        #             └── Returns: visual_output [B,N_frames,D]
        
        # sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
        #                                                                  video, video_mask, shaped=True, video_frame=video_frame)
        sequence_output, word_output, narration_output, visual_output = self.get_sequence_words_narration_visual_output(input_ids, token_type_ids, attention_mask, 
                                                        narration, narration_word_mask, narration_mask, video, video_mask, shaped=True, video_frame=video_frame)

        if self.training:
            # Phase 2: Multi-granularity Matching & Loss Computation
            # ├── get_similarity_logits()  <- CORE MATCHING PIPELINE
            # │   │
            # │   ├── [2.1] Co-Attention Fusion
            # │   │   └── for co_layer in self.co_connetion_transformer_model_block:
            # │   │       └── co_layer(visual_output, narration_output)  [Co_attention_block]
            # │   │           └── Returns: enhanced visual_output, enhanced narration_output
            # │   │
            # │   ├── [2.2] Temporal Aggregation
            # │   │   ├── agg_video_feat(visual_output, video_mask, sim_header)
            # │   │   │   ├── if sim_header == "seqTransf":
            # │   │   │   │   ├── + self.frame_position_embeddings(position_ids)
            # │   │   │   │   └── self.transformerClip(visual_output)  [TransformerClip]
            # │   │   │   ├── if sim_header == "seqLSTM":
            # │   │   │   │   └── self.lstm_visual(visual_output)
            # │   │   │   └── Returns: aggregated visual_output
            # │   │   │
            # │   │   └── agg_narration_feat(narration_output, narration_mask, sim_header)
            # │   │       ├── if sim_header == "seqTransf":
            # │   │       │   ├── + self.caption_position_embeddings(position_ids)
            # │   │       │   └── self.transformerCaption(narration_output)  [TransformerClip]
            # │   │       ├── if sim_header == "seqLSTM":
            # │   │       │   └── self.lstm_caption(narration_output)
            # │   │       └── Returns: aggregated narration_output
            # │   │
            # │   ├── [2.3] Distributed Training (if training)
            # │   │   └── allgather() ×7 (visual, video_mask, narration, narration_mask, sequence, word, attention_mask)
            # │   │
            # │   ├── [2.4] Adaptive Weighting & Filtering
            # │   │   └── get_weights_and_mask()
            # │   │       ├── wti_interaction(word_output, word_mask, visual_output)
            # │   │       │   └── self.text_weight_fc(word_output)  [MLP]
            # │   │       │       └── Returns: word_weights [B_t,B_v,N_t]
            # │   │       │
            # │   │       ├── get_softmax_weights(sequence_output, narration_output) ×2
            # │   │       │   ├── F.normalize() - L2 normalization
            # │   │       │   ├── torch.matmul() - Cosine similarity
            # │   │       │   └── F.softmax(cosine_sim/temperature)
            # │   │       │       └── Returns: seq_nar_weights, seq_vis_weights
            # │   │       │
            # │   │       ├── apply_nucleus_filtering(seq_nar_weights, nucleus_P) ×2  [Section 3.4]
            # │   │       │   ├── torch.sort() - Sort descending
            # │   │       │   ├── torch.cumsum() - Cumulative probability
            # │   │       │   ├── Filter out low-prob frames (p > nucleus_P)
            # │   │       │   └── Re-normalize
            # │   │       │       └── Returns: filtered weights
            # │   │       │
            # │   │       └── adjust_weights_and_mask() ×3
            # │   │           └── Returns: word_weights, visual_weights, narration_weights + masks
            # │   │
            # │   └── [2.5] Coarse-grained Matching & Fine-grained Matching [Section 3.5]
            # |       |
            # |       (1) Coarse-grained Matching
            # │       ├── _loose_coarse_similarity(sequence, visual, word_mask, video_mask, visual_weights)
            # │       │   ├── for i in range(sequence_batch_size):
            # │       │   │   └── _weighted_pooling_for_similarity_visual()
            # │       │   │       └── Weighted pooling: sum(visual×weights×mask)/sum(mask×weights)
            # │       │   └── torch.matmul(sequence, visual_pooled.T)
            # │       │       └── Returns: retrieve_logits_T2V_coarse [B_text, B_video]
            # │       │
            # │       └── _loose_coarse_similarity(sequence, narration, ...)
            # │       |   └── Returns: retrieve_logits_T2N_coarse [B_text, B_narration]
            # │       |
            # │       (2) Fine-grained Matching
            # │       ├── _loose_fine_similarity(word_output, visual_output, masks, weights)
            # │       │   ├── torch.einsum('atd,bvd->abtv') - All word-frame pairs
            # │       │   ├── retrieve_logits.max(dim=-1) - T2V: max over frames
            # │       │   ├── torch.einsum('abt,abt->ab') - Weight by word importance
            # │       │   ├── retrieve_logits.max(dim=-2) - V2T: max over words
            # │       │   ├── torch.einsum('abv,abv->ab') - Weight by frame importance
            # │       │   └── (t2v_logits + v2t_logits)/2
            # │       │       └── Returns: retrieve_logits_T2V_fine [B_text, B_video]
            # │       │
            # │       └── _loose_fine_similarity(word_output, narration_output, ...)
            # │           └── Returns: retrieve_logits_T2N_fine [B_text, B_narration]
            # │
            # ├── [Combine Coarse & Fine]
            # │   ├── tv_sim_matrix = (tv_sim_matrix_coarse + tv_sim_matrix_fine)/2
            # │   └── tn_sim_matrix = (tn_sim_matrix_coarse + tn_sim_matrix_fine)/2
            # │
            # ├── [3] NCE Loss Computation
            # │   ├── cal_nce_loss(tv_sim_matrix)
            # │   │   ├── logit_scale = self.clip.logit_scale.exp()
            # │   │   ├── sim_loss1 = self.loss_fct(sim_matrix)  [CrossEn]
            # │   │   ├── sim_loss2 = self.loss_fct(sim_matrix.T)
            # │   │   └── Returns: (sim_loss1 + sim_loss2)/2
            # │   │       └── tv_nce_loss
            # │   │
            # │   └── cal_nce_loss(tn_sim_matrix)
            # │       └── tn_nce_loss
            # │
            # └── [4] Hard Negative Loss Computation  [Section 3.6.1]
            #     └── get_hn_hinge_loss(tv_sim_matrix, tn_sim_matrix)
            #         │
            #         ├── [4.1] Identify Hard Negatives
            #         │   ├── get_std_hard_negative_indices(tv_sim_matrix)
            #         │   │   ├── positive_sim = sim_matrix.diag()
            #         │   │   ├── stds = torch.std(sim_matrix, dim=1)
            #         │   │   ├── thresholds = positive_sim - threshold_factor×stds
            #         │   │   ├── candidates = sim_matrix > thresholds
            #         │   │   └── Returns: hard_negative_indices_tv, threshold_tv
            #         │   │
            #         │   ├── get_std_hard_negative_indices(tn_sim_matrix)
            #         │   ├── get_std_hard_negative_indices(tv_sim_matrix.T)
            #         │   └── get_std_hard_negative_indices(tn_sim_matrix.T)
            #         │
            #         ├── [4.2] Combine Cross-View Hard Negatives
            #         │   ├── combined_hn = torch.unique(torch.cat(hn_tv, hn_tn))
            #         │   └── combined_hn_T = torch.unique(torch.cat(hn_tv_T, hn_tn_T))
            #         │
            #         ├── [4.3] Compute Hinge Losses
            #         │   ├── cal_hinge_loss(tv_sim_matrix, combined_hn, margin)
            #         │   │   ├── pos_scores = sim_matrix.diagonal()
            #         │   │   ├── for i, hn_indices in enumerate(hard_negative_indices):
            #         │   │   │   ├── neg_scores = sim_matrix[i, hn_indices]
            #         │   │   │   └── hinge_loss = relu(neg_scores - pos_scores[i] + margin[i])
            #         │   │   └── Returns: mean(hinge_losses)
            #         │   │
            #         │   ├── cal_hinge_loss(tn_sim_matrix, ...)
            #         │   ├── cal_hinge_loss(tv_sim_matrix.T, ...)
            #         │   └── cal_hinge_loss(tn_sim_matrix.T, ...)
            #         │
            #         └── Returns: tv_hard_neg_loss, tn_hard_neg_loss
            loss = 0.
            alpha = self.task_config.hard_negative_weighting

            tv_sim_matrix_coarse, tn_sim_matrix_coarse, tv_sim_matrix_fine, tn_sim_matrix_fine = self.get_similarity_logits(
                sequence_output, word_output, visual_output, narration_output, attention_mask, video_mask, narration_mask, shaped=True)

            tv_sim_matrix = (tv_sim_matrix_coarse + tv_sim_matrix_fine)/2
            tn_sim_matrix = (tn_sim_matrix_coarse + tn_sim_matrix_fine)/2

            tv_nce_loss = self.cal_nce_loss(tv_sim_matrix)
            tn_nce_loss = self.cal_nce_loss(tn_sim_matrix)

            tv_hard_neg_loss, tn_hard_neg_loss = self.get_hn_hinge_loss(tv_sim_matrix, tn_sim_matrix)

            # Phase 3: Final Loss Aggregation
            # └── loss = tv_nce_loss + tn_nce_loss + α×(tv_hard_neg_loss + tn_hard_neg_loss)
            # │
            # └── return loss
            loss = tv_nce_loss + tn_nce_loss + alpha * (tv_hard_neg_loss + tn_hard_neg_loss)
            return loss
        else:
            return None