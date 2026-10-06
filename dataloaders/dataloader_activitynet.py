from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import json
import math
from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.rawframes_util import RawFrameExtractor

class ActivityNet_DataLoader(Dataset):
    def __init__(
            self,
            subset,
            data_path,
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
        self.data_path = data_path
        self.narration = json.load(open(narration_path, 'r'))
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
        # video_data_type: 'video' or 'frames'
        self.video_data_type = video_data_type
        assert self.video_data_type in ['video', 'frames']

        # ----------- New: aug_data & fqs_k -----------
        self.fqs_k = fqs_k
        self.aug_data = None
        if aug_json_path is not None and os.path.exists(aug_json_path):
            print(f"DataLoader loading augmented queries from {aug_json_path}...")
            with open(aug_json_path, 'r') as f:
                self.aug_data = json.load(f)
        # -------------------------------------------

        self.subset = subset
        assert self.subset in ["train", "val"]

        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_ids.json")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_ids.json")

        video_json_path_dict = {}
        video_json_path_dict["train"] = os.path.join(self.data_path, "train.json")
        video_json_path_dict["val"] = os.path.join(self.data_path, "val_1.json")

        pseudo_video_id_list, video_id_list = self._get_video_id_single(video_id_path_dict[self.subset])
        pseudo_caption_dict = self._get_captions_single(video_json_path_dict[self.subset])

        print("video id list: {}".format(len(video_id_list)))
        print("pseudo caption dict: {}".format(len(pseudo_caption_dict.keys())))

        video_dict = {}
        if self.video_data_type == 'frames':
            for video_id_ in video_id_list:
                frames_dir = os.path.join(self.features_path, video_id_)
                if os.path.isdir(frames_dir):
                    video_dict[video_id_] = frames_dir
        else:  # 'video'
            for root, dub_dir, video_files in os.walk(self.features_path):
                for video_file in video_files:
                    video_id_ = ".".join(video_file.split(".")[:-1])
                    if video_id_ not in video_id_list:
                        continue
                    file_path_ = os.path.join(root, video_file)
                    video_dict[video_id_] = file_path_
        self.video_dict = video_dict
        print("video dict: {}".format(len(video_dict)))

        self.pseudo_video_id_list = pseudo_video_id_list
        self.video_id_list = video_id_list
        self.pseudo_caption_dict = pseudo_caption_dict

        # Get iterator video ids
        self.video_id2idx_dict = {pseudo_video_id: id for id, pseudo_video_id in enumerate(self.pseudo_video_id_list)}
        # Get all captions
        self.iter2video_pairs_dict = {}
        for pseudo_video_id, video_id in zip(self.pseudo_video_id_list, self.video_id_list):
            if pseudo_video_id not in self.pseudo_caption_dict or video_id not in self.video_dict:
                continue
            caption = self.pseudo_caption_dict[pseudo_video_id]
            n_caption = len(caption['start'])
            for sub_id in range(n_caption):
                self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (pseudo_video_id, sub_id)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.rawFrameExtractor = RawFrameExtractor(size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        
        # Build narration dict
        self.narration_dict = {}
        for item in self.narration:
            video_file = item['video_file']
            narration = [item[f'caption_{i}'] for i in range(1, len(item))]
            self.narration_dict[video_file] = narration

    def __len__(self):
        return len(self.iter2video_pairs_dict)

    def _get_video_id_from_pseduo(self, pseudo_video_id):
        video_id = pseudo_video_id[2:]
        return video_id

    def _get_video_id_single(self, path):
        pseudo_video_id_list = []
        video_id_list = []
        print('Loading json: {}'.format(path))
        with open(path, 'r') as f:
            json_data = json.load(f)

        for pseudo_video_id in json_data:
            if pseudo_video_id in pseudo_video_id_list:
                print("reduplicate.")
            else:
                video_id = self._get_video_id_from_pseduo(pseudo_video_id)
                pseudo_video_id_list.append(pseudo_video_id)
                video_id_list.append(video_id)
        return pseudo_video_id_list, video_id_list

    def _get_captions_single(self, path):
        pseudo_caption_dict = {}
        with open(path, 'r') as f:
            json_data = json.load(f)

        for pseudo_video_id, v_ in json_data.items():
            pseudo_caption_dict[pseudo_video_id] = {}
            duration = v_["duration"]
            pseudo_caption_dict[pseudo_video_id]["start"] = np.array([0], dtype=object)
            pseudo_caption_dict[pseudo_video_id]["end"] = np.array([int(math.ceil(float(duration)))], dtype=object)
            pseudo_caption_dict[pseudo_video_id]["text"] = np.array([" ".join(v_["sentences"])], dtype=object)
        return pseudo_caption_dict

    def _get_text(self, pseudo_video_id, sub_id):
        caption = self.pseudo_caption_dict[pseudo_video_id]
        k = 1
        r_ind = [sub_id]

        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i in range(k):
            ind = r_ind[i]
            words = self.tokenizer.tokenize(caption['text'][ind])

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

        return pairs_text, pairs_mask, pairs_segment

    # ------ New: get text for original + augmented queries --------
    def _get_text_with_aug(self, pseudo_video_id, sub_id):
        """Tokenize the original sentence plus self.fqs_k augmented queries.
        Returns shape (1 + fqs_k, max_words) for text/mask/segment.
        """
        caption = self.pseudo_caption_dict[pseudo_video_id]
        sentence = caption['text'][sub_id]
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
            words += [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            
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
            
            return input_ids, input_mask, segment_ids

        # Row 0: original query
        input_ids, input_mask, segment_ids = _tokenize_sentence(sentence)
        pairs_text[0] = np.array(input_ids)
        pairs_mask[0] = np.array(input_mask)
        pairs_segment[0] = np.array(segment_ids)

        # Rows 1..fqs_k: augmented queries
        aug_sentences = []
        if self.aug_data is not None and pseudo_video_id in self.aug_data:
            aug_payload = self.aug_data[pseudo_video_id]
            if isinstance(aug_payload, dict):
                for _cap_key, cap_data in aug_payload.items():
                    if isinstance(cap_data, dict) and cap_data.get("original", "") == sentence:
                        aug_sentences = cap_data.get("augment", [])
                        break

                if not aug_sentences and len(aug_payload) > 0:
                    first_val = next(iter(aug_payload.values()), [])
                    if isinstance(first_val, dict):
                        aug_sentences = first_val.get("augment", [])
                    elif isinstance(first_val, list):
                        aug_sentences = first_val
            elif isinstance(aug_payload, list):
                aug_sentences = aug_payload

            aug_sentences = aug_sentences[:self.fqs_k]

        for i in range(self.fqs_k):
            if i < len(aug_sentences):
                input_ids, input_mask, segment_ids = _tokenize_sentence(aug_sentences[i])
            else:
                input_ids, input_mask, segment_ids = _tokenize_sentence("")
            
            pairs_text[i + 1] = np.array(input_ids)
            pairs_mask[i + 1] = np.array(input_mask)
            pairs_segment[i + 1] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment
    # ----------------------------------------------

    def _get_rawvideo(self, choice_video_ids, s, e):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)
        try:
            for i, video_id in enumerate(choice_video_ids):
                video_path = self.video_dict[video_id]
                start_time = int(s[i])
                end_time = int(e[i])
                start_time = start_time if start_time >= 0. else 0.
                end_time = end_time if end_time >= 0. else 0.
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = end_time + 1

                # Should be optimized by gathering all asking of this video
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time)
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
                    print("video path: {} error. video id: {}, start: {}, end: {}".format(video_path, video_id, start_time, end_time))
        except Exception as excep:
            print("video path: {} error. video id: {}, start: {}, end: {}, Error: {}".format(video_path, choice_video_ids, s, e, excep))
            raise excep

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def _get_rawframes(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                        self.rawFrameExtractor.size, self.rawFrameExtractor.size), dtype=np.float)
        
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

    def _get_narration(self, choice_video_ids):
        narration = np.zeros((len(choice_video_ids), self.max_frames, self.max_words), dtype=np.long)
        caption_word_masks = np.zeros((len(choice_video_ids), self.max_frames, self.max_words), dtype=np.long)
    
        for video_idx, video_id in enumerate(choice_video_ids):
            video_narration = self.narration_dict.get(video_id, [])
            
            for caption_idx, caption in enumerate(video_narration):
                if caption_idx >= self.max_frames:
                    break
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

    def __getitem__(self, feature_idx):
        pseudo_video_id, sub_id = self.iter2video_pairs_dict[feature_idx]

        # Get text with or without augmentation
        if self.aug_data is not None:
            pairs_text, pairs_mask, pairs_segment = self._get_text_with_aug(pseudo_video_id, sub_id)
        else:
            pairs_text, pairs_mask, pairs_segment = self._get_text(pseudo_video_id, sub_id)
        
        video_id = self.video_id_list[self.video_id2idx_dict[pseudo_video_id]]
        choice_video_ids = [video_id]
        narration, captions_word_mask = self._get_narration(choice_video_ids)
        
        # Choose between raw video or raw frames based on video_data_type
        if self.video_data_type == 'video':
            s = self.pseudo_caption_dict[pseudo_video_id]['start']
            e = self.pseudo_caption_dict[pseudo_video_id]['end']
            video, video_mask = self._get_rawvideo(choice_video_ids, s, e)
        else:  # 'frames'
            video, video_mask = self._get_rawframes(choice_video_ids)
        
        narration_mask = video_mask
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, narration, captions_word_mask, narration_mask
