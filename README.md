# COVER

Official Code for [CVPR Workshop2024] Paper *"COVER: A Comprehensive Video Quality Evaluator"*. 
Official Code, Demo, Weights for the [Comprehensive Video Quality Evaluator (COVER)].

# Todo:: update date, hugging face model below
- xx xxx, 2024: We upload weights of [COVER](https://github.com/vztu/COVER/release/Model/COVER.pth) and [COVER++](TobeContinue) to Hugging Face models.
- xx xxx, 2024: We upload Code of [COVER](https://github.com/vztu/COVER)
- 12 Apr, 2024: COVER has been accepted by CVPR Workshop2024.


# Todo:: update [visitors](link) below
![visitors](https://visitor-badge.laobi.icu/badge?page_id=teowu/TobeContinue) [![](https://img.shields.io/github/stars/vztu/COVER)](https://github.com/vztu/COVER)
[![State-of-the-Art](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/QualityAssessment/COVER)
<a href="https://colab.research.google.com/github/taskswithcode/COVER/blob/master/TWCCOVER.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 


# Todo:: update predicted score for YT-UGC challenge dataset specified by AIS
**COVER** Pseudo-labelled Quality scores of [YT-UGC](https://www.deepmind.com/open-source/kinetics): [CSV](https://github.com/QualityAssessment/COVER/raw/master/cover_predictions/kinetics_400_1.csv)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangling-aesthetic-and-technical-effects/video-quality-assessment-on-youtube-ugc)](https://paperswithcode.com/sota/video-quality-assessment-on-youtube-ugc?p=disentangling-aesthetic-and-technical-effects)


## Introduction
#  Todo:: Add Introduction here

### the proposed COVER

*This inspires us to*

![Fig](figs/approach.png)

## Install

The repository can be installed via the following commands:
```shell
git clone https://github.com/vztu/COVER 
cd COVER 
pip install -e . 
mkdir pretrained_weights 
cd pretrained_weights 
wget https://github.com/vztu/COVER/release/Model/COVER.pth
cd ..
```


## Evaluation: Judge the Quality of Any Video

### Try on Demos
You can run a single command to judge the quality of the demo videos in comparison with videos in VQA datasets. 

```shell
    python evaluate_one_video.py -v ./demo/video_1.mp4
```

or 

```shell
    python evaluate_one_video.py -v ./demo/video_2.mp4
```

Or choose any video you like to predict its quality:


```shell
    python evaluate_one_video.py -v $YOUR_SPECIFIED_VIDEO_PATH$
```

### Outputs

#### ITU-Standarized Overall Video Quality Score

The script can directly score the video's overall quality (considering all perspectives).

```shell
    python evaluate_one_video.py -v $YOUR_SPECIFIED_VIDEO_PATH$
```

The final output score is averaged among all perspectives.


## Evaluate on a Exsiting Video Dataset


```shell
    python evaluate_one_dataset.py -in $YOUR_SPECIFIED_DIR$ -out $OUTPUT_CSV_PATH$
```

## Evaluate on a Set of Unlabelled Videos


```shell
    python evaluate_a_set_of_videos.py -in $YOUR_SPECIFIED_DIR$ -out $OUTPUT_CSV_PATH$
```

The results are stored as `.csv` files in cover_predictions in your `OUTPUT_CSV_PATH`.

Please feel free to use COVER to pseudo-label your non-quality video datasets.


## Data Preparation

We have already converted the labels for most popular datasets you will need for Blind Video Quality Assessment,
and the download links for the **videos** are as follows:

:book: LSVQ: [Github](https://github.com/baidut/PatchVQ)

:book: KoNViD-1k: [Official Site](http://database.mmsp-kn.de/konvid-1k-database.html)

:book: LIVE-VQC: [Official Site](http://live.ece.utexas.edu/research/LIVEVQC)

:book: YouTube-UGC: [Official Site](https://media.withyoutube.com)

*(Please contact the original authors if the download links were unavailable.)*

After downloading, kindly put them under the `../datasets` or anywhere but remember to change the `data_prefix` respectively in the [config file](cover.yml).

# Training: Adapt COVER to your video quality dataset!

Now you can employ ***head-only/end-to-end transfer*** of COVER to get dataset-specific VQA prediction heads. 

We still recommend **head-only** transfer. As we have evaluated in the paper, this method has very similar performance with *end-to-end transfer* (usually 1%~2% difference), but will require **much less** GPU memory, as follows:

```shell
    python transfer_learning.py -t $YOUR_SPECIFIED_DATASET_NAME$
```

For existing public datasets, type the following commands for respective ones:

- `python transfer_learning.py -t val-kv1k` for KoNViD-1k.
- `python transfer_learning.py -t val-ytugc` for YouTube-UGC.
- `python transfer_learning.py -t val-cvd2014` for CVD2014.
- `python transfer_learning.py -t val-livevqc` for LIVE-VQC.


As the backbone will not be updated here, the checkpoint saving process will only save the regression heads with only `398KB` file size (compared with `200+MB` size of the full model). To use it, simply replace the head weights with the official weights [COVER.pth](https://github.com/vztu/COVER/release/Model/COVER.pth).

We also support ***end-to-end*** fine-tune right now (by modifying the `num_epochs: 0` to `num_epochs: 15` in `./cover.yml`). It will require more memory cost and more storage cost for the weights (with full parameters) saved, but will result in optimal accuracy.

Fine-tuning curves by authors can be found here: [Official Curves](https://wandb.ai/timothyhwu/COVER) for reference.


## Visualization

### WandB Training and Evaluation Curves

You can be monitoring your results on WandB!

## Acknowledgement

Thanks for every participant of the subjective studies!

## Citation

Should you find our work interesting and would like to cite it, please feel free to add these in your references! 


# Todo, add bibtex of cover below
```bibtex
%cover

```
