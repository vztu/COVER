# 🏆 [CVPRW 2024] [COVER](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/He_COVER_A_Comprehensive_Video_Quality_Evaluator_CVPRW_2024_paper.pdf): A Comprehensive Video Quality Evaluator. 

🏆 🥇 **Winner solution for [Video Quality Assessment Challenge](https://codalab.lisn.upsaclay.fr/competitions/17340) at the 1st [AIS 2024](https://ai4streaming-workshop.github.io/) workshop @ CVPR 2024** 

Official Code for [CVPR Workshop 2024] Paper *"COVER: A Comprehensive Video Quality Evaluator"*. 
Official Code, Demo, Weights for the [Comprehensive Video Quality Evaluator (COVER)](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/He_COVER_A_Comprehensive_Video_Quality_Evaluator_CVPRW_2024_paper.pdf).

- 29 May, 2024: We create a space for [COVER](https://huggingface.co/spaces/Sorakado/COVER) on Hugging Face.
- 09 May, 2024: We upload Code of [COVER](https://github.com/vztu/COVER).
- 12 Apr, 2024: COVER has been accepted by CVPR Workshop2024.

![visitors](https://visitor-badge.laobi.icu/badge?page_id=vztu/COVER) [![](https://img.shields.io/github/stars/vztu/COVER)](https://github.com/vztu/COVER)
[![State-of-the-Art](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/vztu/COVER)
<a href="https://huggingface.co/spaces/Sorakado/COVER"><img src="./figs/deploy-on-spaces-sm-dark.svg" alt="hugging face log"></a> 

## Introduction
- Existing UGC VQA models strive to quantify quality degradation mainly from technical aspect, with a few considering aesthetic or semantic aspects, but no model has addressed all three aspects simultaneously.
- The demand for high-resolution and high-frame-rate videos on social media platforms presents new challenges for VQA tasks, as they must ensure effectiveness while also meeting real-time requirements.

## the proposed COVER

*This inspires us to develop comprehensive and efficient model for UGC VQA task*

![Fig](./figs/approach.jpg)

### COVER

Results comparison:
|  Dataset: YT-UGC    | SROCC | KROCC | PLCC | RMSE | Run Time  |
| ----  |    ----   |   ----  |      ----     |   ----  | ---- |
| [**COVER**](https://github.com/vztu/COVER/release/Model/COVER.pth) | 0.9143 | 0.7413 | 0.9122 | 0.2519 | 79.37ms |
| TVQE (Wang *et al*, CVPRWS 2024) | 0.9150 | 0.7410 | 0.9182 | ------- | 705.30ms |
| Q-Align (Zhang *et al, CVPRWS 2024) | 0.9080 | 0.7340 | 0.9120 | ------- | 1707.06ms |
| SimpleVQA+ (Sun *et al, CVPRWS 2024) | 0.9060 | 0.7280 | 0.9110 | ------- | 245.51ms |

The run time is measured on an NVIDIA A100 GPU. A clip
of 30 frames of 4K resolution 3840×2160 is used as input.

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

The script can directly score the video's overall quality (considering all perspectives).

```shell
    python evaluate_one_video.py -v $YOUR_SPECIFIED_VIDEO_PATH$
```

The final output score is the sum of all perspectives.


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

```shell
    python transfer_learning.py -t $YOUR_SPECIFIED_DATASET_NAME$
```

For existing public datasets, type the following commands for respective ones:

- `python transfer_learning.py -t val-kv1k` for KoNViD-1k.
- `python transfer_learning.py -t val-ytugc` for YouTube-UGC.
- `python transfer_learning.py -t val-cvd2014` for CVD2014.
- `python transfer_learning.py -t val-livevqc` for LIVE-VQC.

As the backbone will not be updated here, the checkpoint saving process will only save the regression heads. To use it, simply replace the head weights with the official weights [COVER.pth](https://github.com/vztu/COVER/release/Model/COVER.pth).

We also support ***end-to-end*** fine-tune right now (by modifying the `num_epochs: 0` to `num_epochs: 15` in `./cover.yml`). It will require more memory cost and more storage cost for the weights (with full parameters) saved, but will result in optimal accuracy.

## Visualization

### WandB Training and Evaluation Curves

You can be monitoring your results on WandB!

## Acknowledgement

Thanks for every participant of the subjective studies!

## Citation

Should you find our work interesting and would like to cite it, please feel free to add these in your references! 

```bibtex
%AIS 2024 VQA challenge
@article{conde2024ais,
  title={AIS 2024 challenge on video quality assessment of user-generated content: Methods and results},
  author={Conde, Marcos V and Zadtootaghaj, Saman and Barman, Nabajeet and Timofte, Radu and He, Chenlong and Zheng, Qi and Zhu, Ruoxi and Tu, Zhengzhong and Wang, Haiqiang and Chen, Xiangguang and others},
  journal={arXiv preprint arXiv:2404.16205},
  year={2024}
}

%cover
@article{cover2024cpvrws,
  title={COVER: A comprehensive video quality evaluator},
  author={Chenlong, He and Qi, Zheng and Ruoxi, Zhu and Xiaoyang, Zeng and
Yibo, Fan and Zhengzhong, Tu},
  journal={In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2024}
}
```
