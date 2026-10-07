from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from dataloaders.rawframes_util import RawFrameExtractor
from dataloaders.rawvideo_util import RawVideoExtractor


class Charades_DataLoader(Dataset):
    """Charades loader with AVIKA narration/frames and AVIGATE audio."""

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
            audio_path=None,
    ):
        self.data = pd.read_csv(csv_path)
        self.narration = json.load(open(narration_path, 'r'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        self.video_data_type = video_data_type
        assert self.video_data_type in ['video', 'frames']
        self.audios_path = audio_path

        self.fqs_k = fqs_k
        self.aug_data = None
        if aug_json_path is not None and os.path.exists(aug_json_path):
            print(f"DataLoader loading augmented queries from {aug_json_path}...")
            with open(aug_json_path, 'r') as f:
                self.aug_data = json.load(f)

        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution)
        self.rawFrameExtractor = RawFrameExtractor(size=image_resolution)
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

        self.narration_dict = {}
        for item in self.narration:
            video_file = item['video_file']
            caption_keys = sorted(
                (key for key in item if key.startswith('caption_')),
                key=lambda key: int(key.split('_')[-1]))
            self.narration_dict[video_file] = [item[key] for key in caption_keys]

        self.video_ids = self.data['id'].astype(str).tolist()
        raw_sentences = self.data['script'].astype(str).tolist()
        self.sentences = []
        if self.aug_data is not None:
            for video_id, sentence in zip(self.video_ids, raw_sentences):
                self.sentences.append(sentence)
                aug_sentences = self._find_augmented_sentences(video_id, sentence)
                for i in range(self.fqs_k):
                    self.sentences.append(
                        aug_sentences[i] if i < len(aug_sentences) else sentence)
        else:
            self.sentences = raw_sentences

    def __len__(self):
        return len(self.data)

    def _tokenize_sentence(self, sentence):
        words = self.tokenizer.tokenize(sentence)
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

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        pairs_text = np.zeros((1, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((1, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((1, self.max_words), dtype=np.long)

        input_ids, input_mask, segment_ids = self._tokenize_sentence(sentence)
        pairs_text[0] = np.array(input_ids)
        pairs_mask[0] = np.array(input_mask)
        pairs_segment[0] = np.array(segment_ids)
        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _find_augmented_sentences(self, video_id, sentence):
        aug_sentences = []
        if self.aug_data is None or video_id not in self.aug_data:
            return aug_sentences

        aug_payload = self.aug_data[video_id]
        if isinstance(aug_payload, dict):
            for cap_data in aug_payload.values():
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
        return aug_sentences[:self.fqs_k]

    def _get_text_with_aug(self, video_id, sentence):
        total_queries = 1 + self.fqs_k
        pairs_text = np.zeros((total_queries, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((total_queries, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((total_queries, self.max_words), dtype=np.long)

        sentences = [sentence] + self._find_augmented_sentences(video_id, sentence)
        while len(sentences) < total_queries:
            sentences.append("")

        for i, current_sentence in enumerate(sentences[:total_queries]):
            input_ids, input_mask, segment_ids = self._tokenize_sentence(current_sentence)
            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
        return pairs_text, pairs_mask, pairs_segment, [video_id]

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        for i, video_id in enumerate(choice_video_ids):
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)['video']
            if len(raw_video_data.shape) > 3:
                video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data)
                if self.max_frames < video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(
                            0, video_slice.shape[0] - 1,
                            num=self.max_frames, dtype=int)
                        video_slice = video_slice[sample_indx, ...]
                video_slice = self.rawVideoExtractor.process_frame_order(
                    video_slice, frame_order=self.frame_order)
                slice_len = video_slice.shape[0]
                max_video_length[i] = slice_len
                if slice_len > 0:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length
        return video, video_mask

    def _get_rawframes(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawFrameExtractor.size, self.rawFrameExtractor.size), dtype=float)

        for i, video_id in enumerate(choice_video_ids):
            frames_path = os.path.join(self.features_path, "{}".format(video_id))
            if not os.path.isdir(frames_path):
                print("Frames path: {} does not exist. Video id: {}".format(
                    frames_path, video_id))
                continue

            raw_frames_data = self.rawFrameExtractor.get_frames_data(frames_path)['frames']
            if len(raw_frames_data.shape) > 3:
                frame_slice = self.rawVideoExtractor.process_raw_data(raw_frames_data)
                if self.max_frames < frame_slice.shape[0]:
                    if self.slice_framepos == 0:
                        frame_slice = frame_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        frame_slice = frame_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(
                            0, frame_slice.shape[0] - 1,
                            num=self.max_frames, dtype=int)
                        frame_slice = frame_slice[sample_indx, ...]
                slice_len = frame_slice.shape[0]
                max_video_length[i] = slice_len
                if slice_len > 0:
                    video[i][:slice_len, ...] = frame_slice
            else:
                print("Frames path: {} error. Video id: {}".format(
                    frames_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length
        return video, video_mask

    def _get_narration(self, choice_video_ids):
        narration = np.zeros(
            (len(choice_video_ids), self.max_frames, self.max_words), dtype=np.long)
        caption_word_masks = np.zeros(
            (len(choice_video_ids), self.max_frames, self.max_words), dtype=np.long)

        for video_idx, video_id in enumerate(choice_video_ids):
            video_narration = self.narration_dict.get(video_id, [])
            for caption_idx, caption in enumerate(video_narration):
                if caption_idx >= self.max_frames:
                    break
                input_ids, input_mask, _ = self._tokenize_sentence(caption)
                narration[video_idx][caption_idx] = np.array(input_ids)
                caption_word_masks[video_idx][caption_idx] = np.array(input_mask)
        return narration, caption_word_masks

    def _get_rawaudio(self, choice_audio_ids, sample_rate=16000):
        target_length = 1024
        norm_mean = -5.118
        norm_std = 3.2527153
        fbanks = torch.zeros((len(choice_audio_ids), target_length, 128))
        for i, audio_id in enumerate(choice_audio_ids):
            audio_path = os.path.join(self.audios_path, "{}.wav".format(audio_id))

            if os.path.exists(audio_path) == False:
                pass
            else:
                waveform, sr = torchaudio.load(audio_path)
                if sample_rate != sr:
                    Resample = torchaudio.transforms.Resample(sr, sample_rate)
                    waveform = Resample(waveform)
                waveform -= waveform.mean()
                f_shift = waveform.shape[1] * 1000 / (sample_rate * target_length)
                fbank = torchaudio.compliance.kaldi.fbank(
                    waveform, htk_compat=True, sample_frequency=sample_rate,
                    use_energy=False, window_type='hanning', num_mel_bins=128,
                    dither=0.0, frame_shift=f_shift)

                n_frames = fbank.shape[0]
                p = target_length - n_frames
                if p > 0:
                    m = torch.nn.ZeroPad2d((0, 0, 0, p))
                    fbank = m(fbank)
                elif p < 0:
                    fbank = fbank[0:target_length, :]

                fbank = (fbank - norm_mean) / (norm_std * 2)
                fbanks[i] = fbank
        return fbanks

    def __getitem__(self, idx):
        video_id = self.data['id'].values[idx]
        sentence = self.data['script'].values[idx]

        if self.aug_data is not None:
            pairs_text, pairs_mask, pairs_segment, choice_video_ids = \
                self._get_text_with_aug(video_id, sentence)
        else:
            pairs_text, pairs_mask, pairs_segment, choice_video_ids = \
                self._get_text(video_id, sentence)

        narration, caption_word_mask = self._get_narration(choice_video_ids)
        if self.video_data_type == 'video':
            video, video_mask = self._get_rawvideo(choice_video_ids)
        else:
            video, video_mask = self._get_rawframes(choice_video_ids)
        fbank = self._get_rawaudio(choice_video_ids)
        narration_mask = video_mask

        return (pairs_text, pairs_mask, pairs_segment, video, video_mask,
                narration, caption_word_mask, narration_mask, fbank)
