from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
import torch
from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.rawframes_util import RawFrameExtractor

class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            csv_path,
            narration_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            video_data_type='frames',
            aug_json_path=None,
            fqs_k=2,
    ):
        self.data = pd.read_csv(csv_path)
        # ----------- New: Load narration data -----------
        self.narration = json.load(open(narration_path, 'r'))
        # -------------------------------------------
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        # ----------- New: video_data_type -----------
        self.video_data_type = video_data_type
        assert self.video_data_type in ['video', 'frames']
        # -------------------------------------------

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        # ----------- New: rawFrameExtractor -----------
        self.rawFrameExtractor = RawFrameExtractor(size=image_resolution)
        # ----------------------------------------------
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        
        # ----------- New: narration dict -----------
        self.narration_dict = {}
        for item in self.narration:
            video_file = item['video_file']
            # narration_msvd.json:
            # {
            #     "video_file": "...",
            #     "caption_1": "..."
            #     "caption_2": "..."
            #     "caption_3": "..."
            # }
            # => len(item) = 4
            narration = [item[f'caption_{i}'] for i in range(1, len(item))]
            self.narration_dict[video_file] = narration
        # -------------------------------------------

        # ----------- New: aug_data & fqs_k -----------
        self.fqs_k = fqs_k
        self.aug_data = None
        if aug_json_path is not None and os.path.exists(aug_json_path):
            print(f"DataLoader loading augmented queries from {aug_json_path}...")
            with open(aug_json_path, 'r') as f:
                self.aug_data = json.load(f)

        # Keep explicit mapping between retrieval matrix rows/cols and raw inputs.
        # - video_ids maps to columns
        # - sentences maps to rows in the order fed to the model
        self.video_ids = self.data['video_id'].astype(str).tolist()
        self.sentences = []
        raw_sentences = self.data['sentence'].astype(str).tolist()
        if self.aug_data is not None:
            for video_id, sentence in zip(self.video_ids, raw_sentences):
                self.sentences.append(sentence)
                aug_sentences = []
                if video_id in self.aug_data:
                    video_aug = self.aug_data[video_id]
                    if isinstance(video_aug, dict):
                        for _cap_key, cap_data in video_aug.items():
                            if isinstance(cap_data, dict) and cap_data.get("original", "") == sentence:
                                aug_sentences = cap_data.get("augment", [])
                                break
                        if not aug_sentences and len(video_aug) > 0:
                            first_cap = list(video_aug.values())[0]
                            if isinstance(first_cap, dict):
                                aug_sentences = first_cap.get("augment", [])
                    elif isinstance(video_aug, list):
                        aug_sentences = video_aug

                for i in range(self.fqs_k):
                    aug_query = aug_sentences[i] if i < len(aug_sentences) else sentence
                    self.sentences.append(aug_query)
        else:
            self.sentences = raw_sentences
        # -------------------------------------------

    # ------ Same CLIP4Clip --------
    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    # ------ New: get text for original + augmented queries --------
    def _get_text_with_aug(self, video_id, sentence):
        """Tokenize the original sentence plus self.fqs_k augmented queries.
        Returns shape (1 + fqs_k, max_words) for text/mask/segment.
        """
        choice_video_ids = [video_id]
        total_queries = 1 + self.fqs_k

        pairs_text = np.zeros((total_queries, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((total_queries, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((total_queries, self.max_words), dtype=np.long)

        def _tokenize_sentence(sent):
            words = self.tokenizer.tokenize(sent)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            return input_ids, input_mask, segment_ids

        # Row 0: original query
        input_ids, input_mask, segment_ids = _tokenize_sentence(sentence)
        pairs_text[0] = np.array(input_ids)
        pairs_mask[0] = np.array(input_mask)
        pairs_segment[0] = np.array(segment_ids)

        # Rows 1..fqs_k: augmented queries
        aug_sentences = []
        if self.aug_data is not None and video_id in self.aug_data:
            video_aug = self.aug_data[video_id]
            # Find the cap whose "original" matches this sentence
            for cap_key, cap_data in video_aug.items():
                if cap_data.get("original", "") == sentence:
                    aug_sentences = cap_data.get("augment", [])
                    break
            # Fallback: use the first cap's augmented queries
            if not aug_sentences and video_aug:
                first_cap = list(video_aug.values())[0]
                aug_sentences = first_cap.get("augment", [])

        for i in range(self.fqs_k):
            aug_sent = aug_sentences[i] if i < len(aug_sentences) else sentence
            input_ids, input_mask, segment_ids = _tokenize_sentence(aug_sent)
            pairs_text[1 + i] = np.array(input_ids)
            pairs_mask[1 + i] = np.array(input_mask)
            pairs_segment[1 + i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids
    # ----------------------------------------------

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask
    # ----------------------------------------------

    # ------ New: get rawframes instead of rawvideo --------
    def _get_rawframes(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                        self.rawFrameExtractor.size, self.rawFrameExtractor.size), dtype=float)
        
        for i, video_id in enumerate(choice_video_ids):
            frames_path = os.path.join(self.features_path, "{}".format(video_id))

            if not os.path.isdir(frames_path):
                print("Frames path: {} does not exist. Video id: {}".format(frames_path, video_id))
                continue

            raw_frames_data = self.rawFrameExtractor.get_frames_data(frames_path)['frames']

            if len(raw_frames_data.shape) > 3:
                raw_frames_data_clip = self.rawVideoExtractor.process_raw_data(raw_frames_data)
                
                if self.max_frames < raw_frames_data_clip.shape[0]:
                    if self.slice_framepos == 0:
                        frame_slice = raw_frames_data_clip[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        frame_slice = raw_frames_data_clip[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_frames_data_clip.shape[0] - 1, num=self.max_frames, dtype=int)
                        frame_slice = raw_frames_data_clip[sample_indx, ...]
                else:
                    frame_slice = raw_frames_data_clip

                slice_len = frame_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = frame_slice
            else:
                print("Frames path: {} error. Video id: {}".format(frames_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask
    # ----------------------------------------------
    
    # ------ New: get narration data --------
    def _get_narration(self, choice_video_ids):

        narration = np.zeros((len(choice_video_ids), self.max_frames, self.max_words), dtype=np.long)
        caption_word_masks = np.zeros((len(choice_video_ids), self.max_frames, self.max_words), dtype=np.long)
    
        for video_idx, video_id in enumerate(choice_video_ids):
            video_narration = self.narration_dict.get(video_id, [])
            
            for caption_idx, caption in enumerate(video_narration):
                words = self.tokenizer.tokenize(caption)
                words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
                total_length_with_CLS = self.max_words - 1
                if len(words) > total_length_with_CLS:
                    words = words[:total_length_with_CLS]
                words += [self.SPECIAL_TOKEN["SEP_TOKEN"]]
                
                input_ids = self.tokenizer.convert_tokens_to_ids(words)
                input_mask = [1] * len(input_ids)
                while len(input_ids) < self.max_words:
                    input_ids.append(0)
                    input_mask.append(0)
                    
                assert len(input_ids) == self.max_words
                assert len(input_mask) == self.max_words
                
                narration[video_idx][caption_idx] = np.array(input_ids)
                caption_word_masks[video_idx][caption_idx] = np.array(input_mask)

        return narration, caption_word_masks
    # ----------------------------------------------

    # ------ Same CLIP4Clip, but now can choose between rawvideo and rawframes --------
    # ------ And supports augmented eval when aug_data is loaded ------------------
    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        sentence = self.data['sentence'].values[idx]

        if self.aug_data is not None:
            # Augmented eval mode: original query + fqs_k augmented queries
            pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text_with_aug(video_id, sentence)
        else:
            # Baseline mode: original query only
            pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)

        narration, caption_word_mask = self._get_narration(choice_video_ids)
        # Choose between raw video or raw frames based on video_data_type
        if self.video_data_type == 'video':
            video, video_mask = self._get_rawvideo(choice_video_ids)
        else:  # 'frames'
            video, video_mask = self._get_rawframes(choice_video_ids)
        narration_mask = video_mask

        # return pairs_text, pairs_mask, pairs_segment, video, video_mask
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, narration, caption_word_mask, narration_mask

class MSRVTT_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            narration_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            video_data_type='frames',
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        ## ----------- New: Load narration data -----------
        self.narration = json.load(open(narration_path, 'r'))
        ## --------------------------------------------
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        # ----------- New: video_data_type -----------
        self.video_data_type = video_data_type
        assert self.video_data_type in ['video', 'frames']
        # -------------------------------------------

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        if self.unfold_sentences:
            train_video_ids = list(self.csv['video_id'].values)
            self.sentences_dict = {}
            for itm in self.data['sentences']:
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)
        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])
                num_sentences += 1
                s_video_id_set.add(itm['video_id'])

            # Use to find the clips in the same video
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)
            for itm in self.data['videos']:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        # ----------- New: rawFrameExtractor -----------
        self.rawFrameExtractor = RawFrameExtractor(size=image_resolution)
        # ----------------------------------------------
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        
        # ----------- New: narration dict -----------
        self.narration_dict = {}
        for item in self.narration:
            video_file = item['video_file']
            narration = [item[f'caption_{i}'] for i in range(1, len(item))]
            self.narration_dict[video_file] = narration
        # -------------------------------------------

    # ------ Same CLIP4Clip --------
    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask
    # ----------------------------------------------

    # ----- New: get rawframes instead of rawvideo --------
    def _get_rawframes(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                        self.rawFrameExtractor.size, self.rawFrameExtractor.size), dtype=float)
        
        for i, video_id in enumerate(choice_video_ids):
            frames_path = os.path.join(self.features_path, "{}".format(video_id))
            if not os.path.isdir(frames_path):
                print("Frames path: {} does not exist. Video id: {}".format(frames_path, video_id))
                continue

            raw_frames_data = self.rawFrameExtractor.get_frames_data(frames_path)['frames']

            if len(raw_frames_data.shape) > 3:
                raw_frames_data_clip = self.rawVideoExtractor.process_raw_data(raw_frames_data)
                
                if self.max_frames < raw_frames_data_clip.shape[0]:
                    if self.slice_framepos == 0:
                        frame_slice = raw_frames_data_clip[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        frame_slice = raw_frames_data_clip[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_frames_data_clip.shape[0] - 1, num=self.max_frames, dtype=int)
                        frame_slice = raw_frames_data_clip[sample_indx, ...]
                else:
                    frame_slice = raw_frames_data_clip

                slice_len = frame_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = frame_slice
            else:
                print("Frames path: {} error. Video id: {}".format(frames_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask
    # ----------------------------------------------
    
    # ----- New: get narration data --------
    def _get_narration(self, choice_video_ids):

        narration = np.zeros((len(choice_video_ids), self.max_frames, self.max_words), dtype=np.long)
        caption_word_masks = np.zeros((len(choice_video_ids), self.max_frames, self.max_words), dtype=np.long)
    
        for video_idx, video_id in enumerate(choice_video_ids):
            video_narration = self.narration_dict.get(video_id, [])
            
            for caption_idx, caption in enumerate(video_narration):
                words = self.tokenizer.tokenize(caption)
                words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
                total_length_with_CLS = self.max_words - 1
                if len(words) > total_length_with_CLS:
                    words = words[:total_length_with_CLS]
                words += [self.SPECIAL_TOKEN["SEP_TOKEN"]]
                
                input_ids = self.tokenizer.convert_tokens_to_ids(words)
                input_mask = [1] * len(input_ids)
                while len(input_ids) < self.max_words:
                    input_ids.append(0)
                    input_mask.append(0)
                    
                assert len(input_ids) == self.max_words
                assert len(input_mask) == self.max_words
                
                narration[video_idx][caption_idx] = np.array(input_ids)
                caption_word_masks[video_idx][caption_idx] = np.array(input_mask)

        return narration, caption_word_masks
    # ----------------------------------------------

    # ------ Same CLIP4Clip, but now can choose between rawvideo and rawframes --------
    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        # ------------------- New: get narration data -----------
        narration, caption_word_mask = self._get_narration(choice_video_ids)
        # -----------------------------------------------------------
        # Choose between raw video or raw frames based on video_data_type
        if self.video_data_type == 'video':
            video, video_mask = self._get_rawvideo(choice_video_ids)
        else:  # 'frames'
            video, video_mask = self._get_rawframes(choice_video_ids)
        narration_mask = video_mask
        
        # return pairs_text, pairs_mask, pairs_segment, video, video_mask
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, narration, caption_word_mask, narration_mask


