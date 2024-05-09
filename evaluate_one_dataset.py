import argparse
import os
import csv

import torch

import pandas as pd
import numpy as np
import pickle as pkl
import decord
import yaml

from scipy import stats
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from cover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from cover.models import COVER

# use case
# python evaluate_on_ytugc.py -o cover.yml -d cuda:3 --output result.csv -uh 0

def save_to_csv(video_name, pre_smos, pre_tmos, pre_amos, pre_overall, filename):
    combined_data = list(zip(video_name, pre_smos, pre_tmos, pre_amos, pre_overall))

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Video', 'semantic score', 'technical score', 'aesthetic score', 'overall/final score'])
        writer.writerows(combined_data)

mean_cover, std_cover = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)

mean_clip, std_clip = (
    torch.FloatTensor([122.77, 116.75, 104.09]),
    torch.FloatTensor([68.50, 66.63, 70.32])
)

def fuse_results(results: list):
    x = (results[0] + results[1] + results[2])
    return {
        "semantic" : results[0],
        "technical": results[1],
        "aesthetic": results[2],
        "overall"  : x,
    }


def gaussian_rescale(pr):
    # The results should follow N(0,1)
    pr = (pr - np.mean(pr)) / np.std(pr)
    return pr

 
def uniform_rescale(pr):
    # The result scores should follow U(0,1)
    return np.arange(len(pr))[np.argsort(pr).argsort()] / len(pr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt"   , type=str, default="./cover.yml", help="the option file")
    parser.add_argument('-d', "--device", type=str, default="cuda:0"     , help='CUDA device id')
    parser.add_argument("-t", "--target_set", type=str, default="val-ytugc", help="target_set")
    parser.add_argument(      "--output", type=str, default="ytugc.csv" , help='output file to store predict mos value')
    args = parser.parse_args()
    return args


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


if __name__ == '__main__':
    args = parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    
    ### Load COVER
    evaluator = COVER(**opt["model"]["args"]).to(args.device)
    state_dict = torch.load(opt["test_load_path"], map_location=args.device)

    # set strict=False here to avoid error of missing
    # weight of prompt_learner in clip-iqa+, cross-gate
    evaluator.load_state_dict(state_dict['state_dict'], strict=False)

    dopt = opt["data"][args.target_set]["args"]
    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        temporal_samplers[stype] = UnifiedFrameSampler(
            sopt["clip_len"] // sopt["t_frag"],
            sopt["t_frag"],
            sopt["frame_interval"],
            sopt["num_clips"],
        )

    if args.target_set == 'val-livevqc':
        videos_dir = './datasets/LIVE_VQC/Video/'
        datainfo = './datasets/LIVE_VQC/metainfo/LIVE_VQC_metadata.csv'
        df = pd.read_csv(datainfo)
        files = df['File'].tolist()
        mos = df['MOS'].tolist()
    elif args.target_set == 'val-kv1k':
        videos_dir = './datasets/KoNViD/KoNViD_1k_videos/'
        datainfo = './datasets/KoNViD/metainfo/KoNVid_metadata.csv'
        df = pd.read_csv(datainfo)
        files = df['Filename'].tolist()
        files = [str(file) + '.mp4' for file in files]
        mos = df['MOS'].tolist()
    elif args.target_set == 'val-ytugc':
        videos_dir = './datasets/YouTubeUGC/'
        datainfo = './datasets/YouTubeUGC/../meta_info/Youtube-UGC_metadata.csv'
        df = pd.read_csv(datainfo)
        files = df['filename'].tolist()
        mos = df['MOSFull'].tolist()
        files = [str(file) + '_crf_10_ss_00_t_20.0.mp4' for file in files]
    else:
        print("unsupported video dataset for evaluation")
        assert(0)

    print(len(files))

    pure_name_list = []
    pre_overall = np.zeros(len(mos))
    pre_smos = np.zeros(len(mos))
    pre_tmos = np.zeros(len(mos))
    pre_amos = np.zeros(len(mos))
    gt_mos = np.array(mos)
    count = 0

    for vi in range(len(mos)):
        video = files[vi]
        pure_name = os.path.splitext(video)[0]
        video_path = os.path.join(videos_dir, video)

        views, _ = spatial_temporal_view_decomposition(
            video_path, dopt["sample_types"], temporal_samplers
        )
        
        for k, v in views.items():
            num_clips = dopt["sample_types"][k].get("num_clips", 1)
            if k == 'technical' or k == 'aesthetic':
                views[k] = (
                    ((v.permute(1, 2, 3, 0) - mean_cover) / std_cover)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                    .transpose(0, 1)
                    .to(args.device)
                )
            elif k == 'semantic':
                views[k] = (
                    ((v.permute(1, 2, 3, 0) - mean_clip) / std_clip)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                    .transpose(0, 1)
                    .to(args.device)
                )

        results = [r.mean().item() for r in evaluator(views)]

    
        pre_overall[count] = fuse_results(results)['overall']
        pre_smos[count] = results[0]
        pre_tmos[count] = results[1]
        pre_amos[count] = results[2]
        pure_name_list.append(pure_name)
        print("Process ", video, ", predicted quality score is ", pre_overall[count])
        count += 1


    SROCC = stats.spearmanr(pre_overall, gt_mos)[0]
    KROCC = stats.stats.kendalltau(pre_overall, gt_mos)[0]

    # logistic regression btw y_pred & y
    beta_init = [np.max(gt_mos), np.min(gt_mos), np.mean(pre_overall), 0.5]
    popt, _ = curve_fit(logistic_func, pre_overall, gt_mos, p0=beta_init, maxfev=int(1e8))
    pre_overall_logistic = logistic_func(pre_overall, *popt)

    PLCC = stats.pearsonr(gt_mos, pre_overall_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(gt_mos, pre_overall_logistic))

    print("Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
          .format(SROCC, KROCC, PLCC, RMSE))

    save_to_csv(pure_name_list, pre_smos, pre_tmos, pre_amos, pre_overall, args.output)