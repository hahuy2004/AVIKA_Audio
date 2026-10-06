from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
import json
import copy
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIPKG4Vid
from modules.optimization import BertAdam

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT
from preprocess.query_generator.aggregator import Aggregator
import datetime

# torch.distributed.init_process_group(backend="nccl")
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=540000))

global logger

def get_args(description='CLIPKG4Vid on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--load_path', type=str, default=None, help='')

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data json file path')
    # --- New: add --narration_path for frame-level caption data
    parser.add_argument('--narration_path', type=str, default='data/caption.pickle', help='caption json file path')
    # ------------------------------------------------------
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    # --- NEW APPROACH - Statistical-based Hard Negative Selection ---
    # parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    # parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--hard_negative_selection_factor', type=float, default=0.7, help='scale factor for hard negative selection')
    parser.add_argument('--hard_negative_loss_factor', type=float, default=1.8, help='scale factor for hard negative loss')
    # alpha in formula loss = tv_nce_loss + tn_nce_loss + α * (tv_hard_neg_loss + tn_hard_neg_loss)
    parser.add_argument('--hard_negative_weighting', type=float, default=1, help='Weight the hard negative loss')
    # --- NEW - Query-Aware Adaptive Filtering
    # add --nucleus_P and --temperature for nucleus sampling in hard negative selection
    parser.add_argument('--nucleus_P', type=float, default=0.4, help='Cumulative value for nucleus filtering')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature value for softmax')
    # -------------------------------------------------------
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')
    parser.add_argument('--max_steps', type=int, default=-1, help='Maximum number of steps to run for training/evaluation (-1 means no limit)')
    parser.add_argument('--video_data_type', type=str, default='frames', choices=['video', 'frames'],
                        help='Type of video data to use: "video" for raw video files (.mp4), "frames" for extracted frames')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf", "MUSE"],
                        help="choice a similarity header.")
    # --- New: add co-attention block between visual and narration features
    parser.add_argument('--co_attention_block', action='store_true',
                        help='Enable co-attention block between visual and narration features.')
    # ------------------------------------------

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    # ==========================================
    # ENRICHED EVALUATION ARGUMENTS (ICLR 2025)
    # ==========================================
    parser.add_argument('--aug_json_path', type=str, default=None, 
                        help='Đường dẫn chính xác đến file JSON chứa FQS queries. Nếu None, chạy baseline.')
    parser.add_argument('--aggregation_strategy', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Aggregation strategy: 1=Weighted RRF, 2=Average Similarity, 3=True Majority Voting, 4=Max Similarity')
    parser.add_argument('--fqs_k', type=int, default=2,
                        help='Số lượng augmented queries mỗi video (k). Dùng để khai báo kích thước tensor (tổng k+1).')
    parser.add_argument('--save_jsons', action='store_true',
                        help='Bật cờ này để xuất kết quả dự đoán ra file JSON')
    parser.add_argument('--save_jsons_path', type=str, default='eval_results.json',
                        help='Đường dẫn/tên file JSON lưu kết quả (tùy chọn)')
    
    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIPKG4Vid.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]


    weight_decay = 0.2
    if args.sim_header == "MUSE":
        decay_noclip_param_tp_seq = [(n, p) for n, p in decay_param_tp if "clip." not in n and 'mamba' in n] 
        decay_noclip_param_tp_noseq = [(n, p) for n, p in decay_param_tp if "clip." not in n and 'mamba' not in n]
        no_decay_noclip_param_tp_seq = [(n, p) for n, p in no_decay_param_tp if "clip." not in n and 'mamba' in n] 
        no_decay_noclip_param_tp_noseq = [(n, p) for n, p in no_decay_param_tp if "clip." not in n and 'mamba' not in n]

        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in decay_noclip_param_tp_seq], 'weight_decay': weight_decay, 'lr': args.lr * 10},
            {'params': [p for n, p in decay_noclip_param_tp_noseq], 'weight_decay': weight_decay},
            {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in no_decay_noclip_param_tp_seq], 'weight_decay': 0.0, 'lr': args.lr * 10},
            {'params': [p for n, p in no_decay_noclip_param_tp_noseq], 'weight_decay': 0.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
            {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
        ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def save_best_model(args, model, type_name="best_result"):
    # Save only model weights (no optimizer state) to a single overwrite file
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(args.output_dir, "{}.bin".format(type_name))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Best model weights saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIPKG4Vid.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        logger.info("Model file %s doesn't exist", model_file)
        model = None
    return model

def should_save_best_score(r1, r5, best_r1, best_r5):
    # Save when R1 improves, or when R1 ties and R5 is not worse.
    return (r1 > best_r1) or (np.isclose(r1, best_r1) and r5 >= best_r5)

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0,
                test_dataloader=None, best_score=0.00001, best_score_r5=0.00001):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        # --- New: add narration to the batch ---
        # input_ids, input_mask, segment_ids, video, video_mask = batch
        # loss = model(input_ids, segment_ids, input_mask, video, video_mask)
        input_ids, input_mask, segment_ids, video, video_mask, narration, narration_word_mask, narration_mask = batch
        loss = model(input_ids, segment_ids, input_mask, video, video_mask, narration, narration_word_mask, narration_mask)
        # ---------------------------------------

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

            # Step-based evaluation
            should_step_eval = False
            if args.datatype == "msrvtt":
                should_step_eval = ((global_step % 150 == 0) or global_step == 1)
            elif args.datatype == "didemo":
                should_step_eval = ((global_step % 50 == 0) or global_step == 1)
                
            if should_step_eval:
                if test_dataloader is not None:
                    if n_gpu > 1:
                        torch.distributed.barrier()
                    if local_rank == 0:
                        logger.info("[Step Eval][%s] Evaluating at global_step %d...", args.datatype, global_step)
                        R1, R5, _ = eval_epoch(args, model, test_dataloader, device, n_gpu)
                        logger.info("[Step Eval] R1: %.4f, R5: %.4f | Best so far -> R1: %.4f, R5: %.4f",
                                    R1, R5, best_score, best_score_r5)
                        if should_save_best_score(R1, R5, best_score, best_score_r5):
                            best_score = R1
                            best_score_r5 = R5
                            save_best_model(args, model)
                            logger.info("[Step Eval] New best! R1: %.4f, R5: %.4f, weights saved to best_result.bin",
                                        R1, R5)
                    if n_gpu > 1:
                        torch.distributed.barrier()
                    model.train()

        # Check if max_steps is reached
        if args.max_steps > 0 and step + 1 >= args.max_steps:
            if local_rank == 0:
                logger.info("Reached max_steps (%d), stopping training for this epoch.", args.max_steps)
            break

    # Calculate average loss based on actual number of steps executed
    actual_steps = min(step + 1, len(train_dataloader)) if args.max_steps <= 0 else min(step + 1, args.max_steps)
    # total_loss = total_loss / len(train_dataloader)
    total_loss = total_loss / actual_steps
    return total_loss, global_step, best_score, best_score_r5

# New: Using 4 separate similarity matrices to calculate the metrics, 
# which is more efficient than calculating the metrics for each pair of sentences and videos.
# Despite 1 sim_matrix like CLIP4Clip
def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_list_n, batch_sequence_output_list, batch_word_output_list, 
                       batch_visual_output_list, batch_narration_output_list):
    sim_matrix_T2V_coarse = []
    sim_matrix_T2N_coarse = []
    sim_matrix_T2V_fine = []
    sim_matrix_T2N_fine = []

    total_text_batches = len(batch_list_t)
    
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        word_output = batch_word_output_list[idx1]
        each_row_T2V_coarse = []
        each_row_T2N_coarse = []
        each_row_T2V_fine = []
        each_row_T2N_fine = []
        for idx2, (b2, b3) in enumerate(zip(batch_list_v, batch_list_n)):
            video_mask, *_tmp = b2
            narration_mask, *_tmp = b3
            visual_output = batch_visual_output_list[idx2]
            narration_output = batch_narration_output_list[idx2]
            b1b2_logits_T2V_coarse, b1b3_logits_T2N_coarse, b1b2_logits_T2V_fine, b1b3_logits_T2N_fine = model.get_similarity_logits(sequence_output,
                                                            word_output, visual_output, narration_output, input_mask, video_mask, narration_mask)
            
            b1b2_logits_T2V_coarse = b1b2_logits_T2V_coarse.cpu().detach().numpy()
            each_row_T2V_coarse.append(b1b2_logits_T2V_coarse)
            b1b2_logits_T2V_fine = b1b2_logits_T2V_fine.cpu().detach().numpy()
            each_row_T2V_fine.append(b1b2_logits_T2V_fine)
            b1b3_logits_T2N_coarse = b1b3_logits_T2N_coarse.cpu().detach().numpy()
            each_row_T2N_coarse.append(b1b3_logits_T2N_coarse)
            b1b3_logits_T2N_fine = b1b3_logits_T2N_fine.cpu().detach().numpy()
            each_row_T2N_fine.append(b1b3_logits_T2N_fine)

        each_row_T2V_coarse = np.concatenate(tuple(each_row_T2V_coarse), axis=-1)
        sim_matrix_T2V_coarse.append(each_row_T2V_coarse)
        each_row_T2V_fine = np.concatenate(tuple(each_row_T2V_fine), axis=-1)
        sim_matrix_T2V_fine.append(each_row_T2V_fine)
        each_row_T2N_coarse = np.concatenate(tuple(each_row_T2N_coarse), axis=-1)
        sim_matrix_T2N_coarse.append(each_row_T2N_coarse)
        each_row_T2N_fine = np.concatenate(tuple(each_row_T2N_fine), axis=-1)
        sim_matrix_T2N_fine.append(each_row_T2N_fine)
        
        # Log progress every 10 text batches or at the end
        if (idx1 + 1) % 10 == 0 or (idx1 + 1) == total_text_batches:
            logger.info("Computing similarity: {}/{} text batches ({:.1f}%)".format(
                idx1 + 1, total_text_batches, 100.0 * (idx1 + 1) / total_text_batches))
        
    return sim_matrix_T2V_coarse, sim_matrix_T2V_fine, sim_matrix_T2N_coarse, sim_matrix_T2N_fine


def eval_epoch(args, model, test_dataloader, device, n_gpu):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        # # =====================================================================
        # # BẮT ĐẦU ĐO GFLOPs: STANDARD EVAL (Dùng Dummy Tensors - An toàn tuyệt đối)
        # # =====================================================================
        # if args.do_eval:
        #     try:
        #         from thop import profile
        #         logger.info("--- ĐANG TÍNH TOÁN GFLOPs (STANDARD EVAL) ---")

        #         # KHÔNG dùng dataloader nữa. Tự tạo Dummy Tensors (Batch Size = 1)
        #         # Kích thước chuẩn: max_frames, 3 kênh màu, ảnh 224x224
        #         b_size = 1
        #         vid = torch.randn(b_size, 1, 1, args.max_frames, 3, 224, 224).to(device)
        #         vid_mask = torch.ones(b_size, 1, args.max_frames).to(device)

        #         # Kích thước chuẩn Text: max_words
        #         in_ids = torch.randint(0, 30000, (b_size, 1, args.max_words)).to(device)
        #         in_mask = torch.ones(b_size, 1, args.max_words).to(device)
        #         seg_ids = torch.zeros(b_size, 1, args.max_words, dtype=torch.long).to(device)

        #         # Kích thước chuẩn Narration (Nếu có dùng, giả sử giống text)
        #         nar = torch.randint(0, 30000, (b_size, 1, args.max_frames, args.max_words)).to(device)
        #         nar_w_mask = torch.ones(b_size, 1, args.max_frames, args.max_words).to(device)
        #         nar_mask = torch.ones(b_size, 1, args.max_frames).to(device)

        #         class VisWrap(torch.nn.Module):
        #             def __init__(self, m):
        #                 super().__init__()
        #                 self.m = m

        #             def forward(self, v, vm):
        #                 return self.m.get_visual_output(v, vm)

        #         class TextWrap(torch.nn.Module):
        #             def __init__(self, m):
        #                 super().__init__()
        #                 self.m = m

        #             def forward(self, i, s, m):
        #                 return self.m.get_sequence_words_output(i, s, m)

        #         class NarWrap(torch.nn.Module):
        #             def __init__(self, m):
        #                 super().__init__()
        #                 self.m = m

        #             def forward(self, n, nw, nm):
        #                 return self.m.get_narration_output(n, nw, nm)

        #         profile_model = copy.deepcopy(model.module if hasattr(model, 'module') else model).to(device)
        #         profile_model.eval()

        #         flops_v, _ = profile(VisWrap(profile_model), inputs=(vid, vid_mask), verbose=False)
        #         flops_t, _ = profile(TextWrap(profile_model), inputs=(in_ids, seg_ids, in_mask), verbose=False)
        #         flops_n, _ = profile(NarWrap(profile_model), inputs=(nar, nar_w_mask, nar_mask), verbose=False)

        #         del profile_model

        #         total_flops = flops_v + flops_t + flops_n
        #         logger.info("Visual FLOPs: %.4f GFLOPs", flops_v / 1e9)
        #         logger.info("Text FLOPs: %.4f GFLOPs", flops_t / 1e9)
        #         logger.info("Narration FLOPs: %.4f GFLOPs", flops_n / 1e9)
        #         logger.info("Total FLOPs: %.4f GFLOPs / 1 Query", total_flops / 1e9)
        #         logger.info("-----------------------------------------------------")
        #     except Exception as e:
        #         logger.warning("Unable to calculate GFLOPs: %s", e)
        # # =====================================================================

        batch_list_t = []
        batch_list_v = []
        batch_list_n = []
        batch_sequence_output_list, batch_word_output_list, batch_visual_output_list, batch_narration_output_list = [], [], [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        logger.info("[start] extract features")
        total_batches = len(test_dataloader)
        logger.info("Total batches: {}".format(total_batches))
        
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask, narration, narration_word_mask, narration_mask = batch

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output, word_output = model.get_sequence_words_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_word_output_list.append(word_output)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    narration, narration_mask = narration[filter_inds, ...], narration_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    narration_output = model.get_narration_output(narration, narration_word_mask, narration_mask)
                    
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                    
                    batch_narration_output_list.append(narration_output)
                    batch_list_n.append((narration_mask,))
                    
                total_video_num += b
            else:
                sequence_output, word_output, narration_output, visual_output = model.get_sequence_words_narration_visual_output(input_ids, segment_ids, input_mask, 
                                                        narration, narration_word_mask, narration_mask, video, video_mask)

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                batch_word_output_list.append(word_output)

                batch_narration_output_list.append(narration_output)
                batch_list_n.append((narration_mask,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))
            
            # print("{}/{}\r".format(bid, len(test_dataloader)), end="")
            # Log progress every 10 batches or at the end
            if (bid + 1) % 10 == 0 or (bid + 1) == total_batches:
                logger.info("Extracting features: {}/{} batches ({:.1f}%)".format(
                    bid + 1, total_batches, 100.0 * (bid + 1) / total_batches))
 
        logger.info("[finish] extract features")
        logger.info("Cached {} text batches, {} video batches".format(
            len(batch_list_t), len(batch_list_v)))
        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        logger.info("[start] calculate the similarity")
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_list_n_splits = []
            batch_t_output_splits = []
            batch_w_output_splits = []
            batch_v_output_splits = []
            batch_n_output_splits = []
            batch_len = len(batch_list_t)
            split_len = (batch_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)
                    batch_list_n_splits.append(batch_list_n)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_w_output_splits.append(batch_word_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                    batch_n_output_splits.append(batch_narration_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_n]
                    batch_list_n_splits.append(devc_batch_list)

                    devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_word_output_list[s_:e_]]
                    batch_w_output_splits.append(devc_batch_list)
                    # visual_output can be either Tensor (non-MUSE) or Tuple[Tensor, Tensor] (MUSE)
                    if args.sim_header == "MUSE":
                        # Tuple[Tensor, Tensor] (MUSE)
                        devc_batch_list = [tuple(x.to(devc) for x in b) for b in batch_visual_output_list]
                    else:
                        # Tensor (non-MUSE)
                        devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_narration_output_list]
                    batch_n_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id], batch_list_n_splits[dev_id], 
                                    batch_t_output_splits[dev_id], batch_w_output_splits[dev_id], batch_v_output_splits[dev_id], batch_n_output_splits[dev_id]) for dev_id in device_ids]

            logger.info("Running similarity computation on {} GPUs...".format(n_gpu))
            parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)

            logger.info("Aggregating results from {} GPUs...".format(n_gpu))
            TV_sim_matrix_coarse, TV_sim_matrix_fine, TN_sim_matrix_coarse, TN_sim_matrix_fine = [],[],[],[]
            for idx in range(len(parallel_outputs)):
                TV_sim_matrix_coarse += parallel_outputs[idx][0]
                TV_sim_matrix_fine += parallel_outputs[idx][1]
                TN_sim_matrix_coarse += parallel_outputs[idx][2]
                TN_sim_matrix_fine += parallel_outputs[idx][3]
            TV_sim_matrix_coarse = np.concatenate(tuple(TV_sim_matrix_coarse), axis=0)
            TV_sim_matrix_fine = np.concatenate(tuple(TV_sim_matrix_fine), axis=0)
            TN_sim_matrix_coarse = np.concatenate(tuple(TN_sim_matrix_coarse), axis=0)
            TN_sim_matrix_fine = np.concatenate(tuple(TN_sim_matrix_fine), axis=0)

            TV_sim_matrix = (TV_sim_matrix_coarse+TV_sim_matrix_fine)/2
            TN_sim_matrix = (TN_sim_matrix_coarse+TN_sim_matrix_fine)/2

        else:
            TV_sim_matrix_coarse, TV_sim_matrix_fine, TN_sim_matrix_coarse, TN_sim_matrix_fine = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_list_n, 
                                                                batch_sequence_output_list, batch_word_output_list, batch_visual_output_list, batch_narration_output_list)
            TV_sim_matrix_coarse = np.concatenate(tuple(TV_sim_matrix_coarse), axis=0)
            TN_sim_matrix_coarse = np.concatenate(tuple(TN_sim_matrix_coarse), axis=0)
            TV_sim_matrix_fine = np.concatenate(tuple(TV_sim_matrix_fine), axis=0)
            TN_sim_matrix_fine = np.concatenate(tuple(TN_sim_matrix_fine), axis=0)
            
            TV_sim_matrix = (TV_sim_matrix_coarse+TV_sim_matrix_fine)/2
            TN_sim_matrix = (TN_sim_matrix_coarse+TN_sim_matrix_fine)/2
    
    logger.info("[finish] calculate the similarity")
    logger.info("Similarity matrix shape: ({}, {})".format(TV_sim_matrix.shape[0], TV_sim_matrix.shape[1]))
    
    R1, R5, raw_2d_sim_matrix = get_score(TV_sim_matrix, TN_sim_matrix, multi_sentence_, cut_off_points_)

    return R1, R5, raw_2d_sim_matrix

def eval_epoch_for_fqs(args, model, test_dataloader, device, n_gpu):
    """
    Enriched Evaluation (FQS): evaluates using original + augmented queries.

    Flow:
      1. For each batch, text tensors (batch, k+1, max_words) are flattened to
         (batch*(k+1), max_words) inside the model via .view(-1, last_dim).
      2. Video / narration features are extracted normally (batch, ...).
      3. The raw similarity matrix has shape (N*(k+1), M).
      4. Reshape to (N, k+1, M) → transpose to (k+1, N, M).
      5. Aggregator reduces to (N, M) via WeightedRRF / Average / MajorityVoting.
      6. Pass aggregated matrix to get_score() for metric computation.
    """
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("[FQS] Eval under the multi-sentence per video clip setting.")
        logger.warning("[FQS] sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    k_plus_1 = 1 + args.fqs_k  # total query variants per sample (original + augmented)

    model.eval()
    with torch.no_grad():
        # # =====================================================================
        # # BẮT ĐẦU ĐO GFLOPs: ENRICHED EVAL (Dùng Dummy Tensors - An toàn tuyệt đối)
        # # =====================================================================
        # if args.do_eval:
        #     try:
        #         from thop import profile
        #         logger.info("--- Calculating GFLOPs (ENRICHED EVAL - 1 Ori + %d Aug) ---", args.fqs_k)

        #         # Video Encoder vẫn chỉ chạy 1 lần (Batch Size = 1)
        #         b_vid = 1
        #         vid = torch.randn(b_vid, 1, 1, args.max_frames, 3, 224, 224).to(device)
        #         vid_mask = torch.ones(b_vid, 1, args.max_frames).to(device)

        #         # Narration Encoder cũng chạy 1 lần
        #         nar = torch.randint(0, 30000, (b_vid, 1, args.max_frames, args.max_words)).to(device)
        #         nar_w_mask = torch.ones(b_vid, 1, args.max_frames, args.max_words).to(device)
        #         nar_mask = torch.ones(b_vid, 1, args.max_frames).to(device)

        #         # NHƯNG Text Encoder phải chạy k_plus_1 lần (1 câu gốc + k câu aug)
        #         b_txt = k_plus_1
        #         in_ids = torch.randint(0, 30000, (b_vid, b_txt, args.max_words)).to(device)
        #         in_mask = torch.ones(b_vid, b_txt, args.max_words).to(device)
        #         seg_ids = torch.zeros(b_vid, b_txt, args.max_words, dtype=torch.long).to(device)

        #         class VisWrap(torch.nn.Module):
        #             def __init__(self, m):
        #                 super().__init__()
        #                 self.m = m

        #             def forward(self, v, vm):
        #                 return self.m.get_visual_output(v, vm)

        #         class TextWrap(torch.nn.Module):
        #             def __init__(self, m):
        #                 super().__init__()
        #                 self.m = m

        #             def forward(self, i, s, m):
        #                 return self.m.get_sequence_words_output(i, s, m)

        #         class NarWrap(torch.nn.Module):
        #             def __init__(self, m):
        #                 super().__init__()
        #                 self.m = m

        #             def forward(self, n, nw, nm):
        #                 return self.m.get_narration_output(n, nw, nm)

        #         profile_model = copy.deepcopy(model.module if hasattr(model, 'module') else model).to(device)
        #         profile_model.eval()

        #         flops_v, _ = profile(VisWrap(profile_model), inputs=(vid, vid_mask), verbose=False)
        #         flops_t, _ = profile(TextWrap(profile_model), inputs=(in_ids, seg_ids, in_mask), verbose=False)
        #         flops_n, _ = profile(NarWrap(profile_model), inputs=(nar, nar_w_mask, nar_mask), verbose=False)

        #         del profile_model

        #         total_flops = flops_v + flops_t + flops_n
        #         logger.info("Visual FLOPs: %.4f GFLOPs", flops_v / 1e9)
        #         logger.info("Text FLOPs: %.4f GFLOPs (%d queries)", flops_t / 1e9, k_plus_1)
        #         logger.info("Narration FLOPs: %.4f GFLOPs", flops_n / 1e9)
        #         logger.info("Total FLOPs: %.4f GFLOPs / 1 Enriched Query", total_flops / 1e9)
        #         logger.info("-----------------------------------------------------")
        #     except Exception as e:
        #         logger.warning("Unable to calculate GFLOPs: %s", e)
        # # =====================================================================

        batch_list_t = []
        batch_list_v = []
        batch_list_n = []
        batch_sequence_output_list, batch_word_output_list = [], []
        batch_visual_output_list, batch_narration_output_list = [], []
        total_video_num = 0

        logger.info("[FQS] [start] extract features")
        total_batches = len(test_dataloader)
        logger.info("[FQS] Total batches: {}, k+1={}".format(total_batches, k_plus_1))

        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask, narration, narration_word_mask, narration_mask = batch

            # Keep all text tensors aligned in 2D to avoid shape mismatch later in similarity computation.
            if len(input_ids.shape) == 3:
                _b_text, _k_dim, seq_len = input_ids.shape
                input_ids = input_ids.contiguous().view(-1, seq_len)
                input_mask = input_mask.contiguous().view(-1, seq_len)
                segment_ids = segment_ids.contiguous().view(-1, seq_len)

            # --- Text encoding ---
            # input_ids shape: (batch, k+1, max_words) when aug is active.
            # model.get_sequence_words_output internally calls
            # input_ids.view(-1, input_ids.shape[-1]), so it correctly flattens
            # (batch, k+1, max_words) -> (batch*(k+1), max_words) before encoding.
            # Resulting sequence_output: (batch*(k+1), 1, D)
            #           word_output:     (batch*(k+1), max_words, D)
            if multi_sentence_:
                b, *_t = video.shape
                sequence_output, word_output = model.get_sequence_words_output(
                    input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_word_output_list.append(word_output)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    narration, narration_mask = narration[filter_inds, ...], narration_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    narration_output = model.get_narration_output(narration, narration_word_mask, narration_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                    batch_narration_output_list.append(narration_output)
                    batch_list_n.append((narration_mask,))

                total_video_num += b
            else:
                sequence_output, word_output = model.get_sequence_words_output(
                    input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_word_output_list.append(word_output)
                batch_list_t.append((input_mask, segment_ids,))

                visual_output = model.get_visual_output(video, video_mask)
                narration_output = model.get_narration_output(narration, narration_word_mask, narration_mask)
                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))
                batch_narration_output_list.append(narration_output)
                batch_list_n.append((narration_mask,))

            if (bid + 1) % 10 == 0 or (bid + 1) == total_batches:
                logger.info("[FQS] Extracting features: {}/{} batches ({:.1f}%)".format(
                    bid + 1, total_batches, 100.0 * (bid + 1) / total_batches))

        logger.info("[FQS] [finish] extract features")
        logger.info("[FQS] Cached {} text batches, {} video batches".format(
            len(batch_list_t), len(batch_list_v)))

        # ------------------------------------------------------------------
        # 2. Calculate the similarity (reuse _run_on_single_gpu exactly)
        # ------------------------------------------------------------------
        logger.info("[FQS] [start] calculate the similarity")
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_list_n_splits = []
            batch_t_output_splits = []
            batch_w_output_splits = []
            batch_v_output_splits = []
            batch_n_output_splits = []
            batch_len = len(batch_list_t)
            split_len = (batch_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)
                    batch_list_n_splits.append(batch_list_n)
                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_w_output_splits.append(batch_word_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                    batch_n_output_splits.append(batch_narration_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_n]
                    batch_list_n_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_word_output_list[s_:e_]]
                    batch_w_output_splits.append(devc_batch_list)
                    if args.sim_header == "MUSE":
                        devc_batch_list = [tuple(x.to(devc) for x in b) for b in batch_visual_output_list]
                    else:
                        devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_narration_output_list]
                    batch_n_output_splits.append(devc_batch_list)

            parameters_tuple_list = [
                (batch_list_t_splits[dev_id], batch_list_v_splits[dev_id], batch_list_n_splits[dev_id],
                 batch_t_output_splits[dev_id], batch_w_output_splits[dev_id],
                 batch_v_output_splits[dev_id], batch_n_output_splits[dev_id])
                for dev_id in device_ids]

            logger.info("[FQS] Running similarity computation on {} GPUs...".format(n_gpu))
            parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)

            TV_sim_matrix_coarse, TV_sim_matrix_fine = [], []
            TN_sim_matrix_coarse, TN_sim_matrix_fine = [], []
            for idx in range(len(parallel_outputs)):
                TV_sim_matrix_coarse += parallel_outputs[idx][0]
                TV_sim_matrix_fine   += parallel_outputs[idx][1]
                TN_sim_matrix_coarse += parallel_outputs[idx][2]
                TN_sim_matrix_fine   += parallel_outputs[idx][3]
            TV_sim_matrix_coarse = np.concatenate(tuple(TV_sim_matrix_coarse), axis=0)
            TV_sim_matrix_fine   = np.concatenate(tuple(TV_sim_matrix_fine),   axis=0)
            TN_sim_matrix_coarse = np.concatenate(tuple(TN_sim_matrix_coarse), axis=0)
            TN_sim_matrix_fine   = np.concatenate(tuple(TN_sim_matrix_fine),   axis=0)

            TV_sim_matrix = (TV_sim_matrix_coarse + TV_sim_matrix_fine) / 2
            TN_sim_matrix = (TN_sim_matrix_coarse + TN_sim_matrix_fine) / 2
        else:
            TV_sim_matrix_coarse, TV_sim_matrix_fine, TN_sim_matrix_coarse, TN_sim_matrix_fine = \
                _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_list_n,
                                   batch_sequence_output_list, batch_word_output_list,
                                   batch_visual_output_list, batch_narration_output_list)
            TV_sim_matrix_coarse = np.concatenate(tuple(TV_sim_matrix_coarse), axis=0)
            TN_sim_matrix_coarse = np.concatenate(tuple(TN_sim_matrix_coarse), axis=0)
            TV_sim_matrix_fine   = np.concatenate(tuple(TV_sim_matrix_fine),   axis=0)
            TN_sim_matrix_fine   = np.concatenate(tuple(TN_sim_matrix_fine),   axis=0)

            TV_sim_matrix = (TV_sim_matrix_coarse + TV_sim_matrix_fine) / 2
            TN_sim_matrix = (TN_sim_matrix_coarse + TN_sim_matrix_fine) / 2

        logger.info("[FQS] [finish] calculate the similarity")
        logger.info("[FQS] Raw similarity matrix shape: ({}, {})".format(
            TV_sim_matrix.shape[0], TV_sim_matrix.shape[1]))

        # ------------------------------------------------------------------
        # 3. Reshape & Aggregate: (N*k_plus_1, M) → (k+1, N, M) → (N, M)
        # ------------------------------------------------------------------
        raw_2d_sim_matrix = TV_sim_matrix + TN_sim_matrix
        total_text_queries, n_videos = TV_sim_matrix.shape
        n_unique = total_text_queries // k_plus_1

        # (N*(k+1), M) → (N, k+1, M) → (k+1, N, M)
        TV_stacked = TV_sim_matrix.reshape(n_unique, k_plus_1, n_videos).transpose(1, 0, 2)
        TN_stacked = TN_sim_matrix.reshape(n_unique, k_plus_1, n_videos).transpose(1, 0, 2)

        logger.info("[FQS] Aggregating {} query variants with strategy: {} "
                    "(1=Weighted RRF, 2=Average, 3=Majority Voting, 4=Max Similarity)".format(
                        k_plus_1, args.aggregation_strategy))
        aggregator = Aggregator(strategy=args.aggregation_strategy)
        TV_sim_agg = aggregator.aggregate(TV_stacked)  # (N, M)
        TN_sim_agg = aggregator.aggregate(TN_stacked)  # (N, M)

        logger.info("[FQS] Aggregated similarity matrix shape: ({}, {})".format(
            TV_sim_agg.shape[0], TV_sim_agg.shape[1]))

    # Skip z-score normalization when rank/vote-based aggregators are used,
    # because their score scales are intentionally not raw logits.
    skip_normalization = True if args.aggregation_strategy in [1, 3] else False
    R1, R5, _ = get_score(TV_sim_agg, TN_sim_agg, multi_sentence_, cut_off_points_, skip_norm=skip_normalization)
    agg_2d_sim_matrix = TV_sim_agg + TN_sim_agg
    return R1, R5, raw_2d_sim_matrix, agg_2d_sim_matrix

def get_score(TV_sim_matrix, TN_sim_matrix, multi_sentence_, cut_off_points_, skip_norm=False):

    if not skip_norm:
        TV_mean = np.mean(TV_sim_matrix)
        TV_std = np.std(TV_sim_matrix)
        TV_sim_matrix = (TV_sim_matrix - TV_mean) / TV_std

        TC_mean = np.mean(TN_sim_matrix)
        TC_std = np.std(TN_sim_matrix)
        TN_sim_matrix = (TN_sim_matrix - TC_mean) / TC_std

    raw_2d_sim_matrix = TV_sim_matrix + TN_sim_matrix
    T2V_sim_matrix = V2T_sim_matrix = raw_2d_sim_matrix

    logger.info("[start] compute_metrics")
    
    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(T2V_sim_matrix.shape[0], T2V_sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        T2V_sim_matrix_new = []
        V2T_sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            T2V_sim_matrix_new.append(np.concatenate((T2V_sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, T2V_sim_matrix.shape[1]), -np.inf)), axis=0))
            V2T_sim_matrix_new.append(np.concatenate((V2T_sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, V2T_sim_matrix.shape[1]), -np.inf)), axis=0))
        T2V_sim_matrix = np.stack(tuple(T2V_sim_matrix_new), axis=0)
        V2T_sim_matrix = np.stack(tuple(V2T_sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(T2V_sim_matrix.shape[0], T2V_sim_matrix.shape[1], T2V_sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(T2V_sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(V2T_sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(T2V_sim_matrix.shape[0], T2V_sim_matrix.shape[1]))
        tv_metrics = compute_metrics(T2V_sim_matrix)
        vt_metrics = compute_metrics(V2T_sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(T2V_sim_matrix), len(T2V_sim_matrix[0])))
    
    logger.info("[finish] compute_metrics")
    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    R5 = tv_metrics['R5']

    return R1, R5, raw_2d_sim_matrix

def save_baseline_retrieval(sim_matrix, dataset, save_path="baseline_preds.json", top_k=5):
    """Save baseline retrieval results using original queries only."""
    logger.info("Saving Baseline retrieval JSON to %s...", save_path)

    if sim_matrix is None:
        logger.warning("sim_matrix is None, skip saving baseline JSON.")
        return

    vid_ids = list(getattr(dataset, 'video_ids', []))
    if not vid_ids:
        vid_ids = ["video{}".format(i) for i in range(sim_matrix.shape[1])]

    texts = list(getattr(dataset, 'sentences', []))
    if not texts:
        texts = ["Query_{}".format(i) for i in range(sim_matrix.shape[0])]

    sim_tensor = torch.tensor(sim_matrix)
    k = min(top_k, sim_tensor.shape[1])
    _, topk_indices = torch.topk(sim_tensor, k=k, dim=-1)

    results = {}
    max_rows = min(sim_matrix.shape[0], len(vid_ids))
    for i in range(max_rows):
        gt_vid = vid_ids[i]
        query = texts[i] if i < len(texts) else "Query_{}".format(i)
        pred_vids = [vid_ids[idx.item()] if idx.item() < len(vid_ids) else "video{}".format(idx.item())
                     for idx in topk_indices[i]]

        results[gt_vid] = {
            "cap_1": {
                "original": query,
                "original_answer": pred_vids
            }
        }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info("Finished saving Baseline retrieval JSON.")

def save_augmented_retrieval(raw_sim_matrix, dataset, aug_json_path, save_path="augmented_preds.json", top_k=5, agg_sim_matrix=None):
    """Save retrieval results for original + augmented queries."""
    logger.info("Saving Augmented retrieval JSON to %s...", save_path)

    if raw_sim_matrix is None:
        logger.warning("raw_sim_matrix is None, skip saving augmented JSON.")
        return

    vid_ids = list(getattr(dataset, 'video_ids', []))
    if not vid_ids:
        vid_ids = ["video{}".format(i) for i in range(raw_sim_matrix.shape[1])]

    all_texts = list(getattr(dataset, 'sentences', []))

    with open(aug_json_path, 'r', encoding='utf-8') as f:
        aug_data = json.load(f)

    sim_tensor = torch.tensor(raw_sim_matrix)
    k = min(top_k, sim_tensor.shape[1])
    _, topk_indices = torch.topk(sim_tensor, k=k, dim=-1)

    agg_topk_indices = None
    if agg_sim_matrix is not None:
        agg_tensor = torch.tensor(agg_sim_matrix)
        agg_k = min(top_k, agg_tensor.shape[1])
        _, agg_topk_indices = torch.topk(agg_tensor, k=agg_k, dim=-1)

    if hasattr(dataset, 'fqs_k'):
        block_size = int(dataset.fqs_k) + 1
    else:
        block_size = max(1, raw_sim_matrix.shape[0] // max(1, len(vid_ids)))

    results = {}
    for vid_idx, gt_vid in enumerate(vid_ids):
        base_row = vid_idx * block_size
        if base_row >= raw_sim_matrix.shape[0]:
            break

        cap_key = "cap_1"
        aug_list = []
        if gt_vid in aug_data and isinstance(aug_data[gt_vid], dict):
            cap_key = list(aug_data[gt_vid].keys())[0] if len(aug_data[gt_vid]) > 0 else "cap_1"
            cap_payload = aug_data[gt_vid].get(cap_key, {})
            if isinstance(cap_payload, dict):
                aug_list = cap_payload.get("augment", [])
            elif isinstance(cap_payload, list):
                aug_list = cap_payload

        orig_query = all_texts[base_row] if base_row < len(all_texts) else "Query_{}".format(base_row)
        orig_vids = [vid_ids[idx.item()] if idx.item() < len(vid_ids) else "video{}".format(idx.item())
                     for idx in topk_indices[base_row]]
        agg_vids = []
        if agg_topk_indices is not None and vid_idx < agg_topk_indices.shape[0]:
            agg_vids = [vid_ids[idx.item()] if idx.item() < len(vid_ids) else "video{}".format(idx.item())
                        for idx in agg_topk_indices[vid_idx]]

        augment_results = []
        for i in range(block_size - 1):
            row_idx = base_row + i + 1
            if row_idx >= raw_sim_matrix.shape[0]:
                break

            default_aug = aug_list[i] if i < len(aug_list) else ""
            aug_query = all_texts[row_idx] if row_idx < len(all_texts) else default_aug
            aug_vids = [vid_ids[idx.item()] if idx.item() < len(vid_ids) else "video{}".format(idx.item())
                        for idx in topk_indices[row_idx]]

            augment_results.append({
                "query": aug_query,
                "answer": aug_vids
            })

        results[gt_vid] = {
            cap_key: {
                "original": orig_query,
                "original_answer": orig_vids,
                "aggregator_answer": agg_vids,
                "augment": augment_results
            }
        }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info("Finished saving Augmented retrieval JSON.")

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert  args.task_type == "retrieval"
    if args.do_eval or args.load_path is not None:
        model = load_model(-1, args, n_gpu, device, model_file=args.load_path)
    else:
        model = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    if args.local_rank == 0:
        all_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("All parameters: %d", all_params)
        logger.info("Trainable parameters: %d", trainable_params)
        logger.info("Percentage of training parameters: %.4f", trainable_params / all_params)

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    ## ####################################
    # train and eval
    ## ####################################````````````
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_score_r5 = 0.00001
        best_output_model_file = "None"
        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
            resumed_loss = checkpoint['loss']
        
        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step, best_score, best_score_r5 = train_epoch(
                epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                scheduler, global_step, local_rank=args.local_rank,
                test_dataloader=test_dataloader, best_score=best_score, best_score_r5=best_score_r5)

            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                ## Run on val dataset, this process is *TIME-consuming*.
                # logger.info("Eval on val dataset")
                # R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)

                R1, R5, _ = eval_epoch(args, model, test_dataloader, device, n_gpu)
                if should_save_best_score(R1, R5, best_score, best_score_r5):
                    best_score = R1
                    best_score_r5 = R5
                    best_output_model_file = output_model_file
                    save_best_model(args, model)
                    logger.info("New best score at end of epoch: R1=%.4f, R5=%.4f, weights saved to best_result.bin",
                                best_score, best_score_r5)
                logger.info("The best model is: {}, best R1: {:.4f}, best R5: {:.4f}".format(
                    best_output_model_file, best_score, best_score_r5))

        ## Uncomment if want to test on the best checkpoint
        # if args.local_rank == 0:
        #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
        #     eval_epoch(args, model, test_dataloader, device, n_gpu)

    elif args.do_eval:
        if args.local_rank == 0:
            # Điều hướng tự động dựa trên tham số aug_json_path
            save_path = args.save_jsons_path
            if not os.path.isabs(save_path):
                save_path = os.path.join(args.output_dir, save_path)

            if args.aug_json_path is not None:
                logger.info("Starting Enriched Evaluation (FQS)...")
                R1, R5, raw_sim_matrix, agg_sim_matrix = eval_epoch_for_fqs(args, model, test_dataloader, device, n_gpu)
                if args.save_jsons:
                    save_augmented_retrieval(
                        raw_sim_matrix=raw_sim_matrix,
                        dataset=test_dataloader.dataset,
                        aug_json_path=args.aug_json_path,
                        save_path=save_path,
                        top_k=5,
                        agg_sim_matrix=agg_sim_matrix,
                    )
            else:
                logger.info("Starting Baseline Evaluation...")
                R1, R5, sim_matrix = eval_epoch(args, model, test_dataloader, device, n_gpu)
                if args.save_jsons:
                    save_baseline_retrieval(
                        sim_matrix=sim_matrix,
                        dataset=test_dataloader.dataset,
                        save_path=save_path,
                        top_k=5,
                    )

if __name__ == "__main__":
    main()
