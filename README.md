# CLIPKG4Vid: Boosting Text-Video Retrieval via Comprehensive Utilization of Frame-Level Captions

The official implementation of CLIPKG4Vid — a framework that enhances text-video retrieval by leveraging frame-level captions (narration) to improve semantic understanding and retrieval accuracy. CLIPKG4Vid employs cross-modal interactions, query-aware filtering, dual-modal matching, and hard-negative loss, achieving good results on MSR-VTT, MSVD, DiDeMo.

## Requirements
This project requires two environments: one for `CLIPKG4Vid`, which serves as the main framework, 
and one for `LLaVa`, which is used for preprocessing to generate narrations.

### Setting up the Main CLIPKG4Vid Environment
```sh
# CLIP4Clip
conda install --yes -c pytorch pytorch=1.13.1 torchvision cudatoolkit=11.6
pip install opencv-python==4.9.0.80 numpy==1.23.0 ftfy regex tqdm boto3 requests pandas
# Mamba
pip install -e causal_conv1d>=1.1.0
pip install -e mamba-1p1p1
```
Install Causal_conv1d and Mamba_ssm following [Vim](https://github.com/doodleima/vision_mamba).


### Setting up the LLaVa Environment

For setting up the llava environment, please refer to the official GitHub repository: [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main).

## Data Preparation

### For MSRVTT

Raw videos can be download from [link](https://cove.thecvf.com/datasets/839)
The splits can be found in the job [collaborative-experts](https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/msrvtt)

### For MSVD

Raw videos can be download from [link](https://www.cs.utexas.edu/~ml/clamp/videoDescription/)
The splits can be found in the job [collaborative-experts](https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/msvd)

<!-- ### For VATEX

Raw Videos and split can be download from [Vatex](https://eric-xw.github.io/vatex-website/download.html)  -->

### For DiDeMo

Raw videos can be download from [LisaAnne/LocalizingMoments.](https://github.com/LisaAnne/LocalizingMoments). The splits can be found in the job [collaborative-experts](https://github.com/albanie/collaborative-experts/blob/master/misc/datasets/didemo/README.md).


## Data Preprocessing

For convenient reproduction of our research, we provide both data preprocessing scripts and pre-generated narration files.

### Compress Video for Speed-up (optional)
```sh
python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]
```
This script will compress the video to *3fps* with width *224* (or height *224*). Modify the variables for your customization.

### Extract Video Frames

Before generating captions for each frame, you need to perform preprocessing on the raw video to extract the frames.

```sh
python preprocess/video_frame_extractor.py --raw_video_path [your_raw_video_folder_path] --extracted_frame_path [your_output_frame_path]
```

### Generate Narration from Frames

Based on the extracted video frames, use LLaVa to generate captions for each frame.

```sh
python preprocess/narration/narration_generator.py --video_frames_path [your_frame_path] --video_id_list_path [your_video_id.json]
```

## How to Run 

（1）Ensure that the data preparation and preprocessing are completed

You can check the narration for the MSR-VTT, MSVD, and VATEX datasets in the `narration_data` directory. Please note that the Didemo dataset is not included due to file size limitations

（2）About the pretrained CLIP checkpoints 

Download CLIP (ViT-B/32) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```
or, download CLIP (ViT-B/16) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```
