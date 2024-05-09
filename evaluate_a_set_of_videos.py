import torch

import argparse
import os
import pickle as pkl

import decord
import numpy as np
import yaml
from tqdm import tqdm

from cover.datasets import (
    UnifiedFrameSampler,
    ViewDecompositionDataset,
    spatial_temporal_view_decomposition,
)
from cover.models import COVER

mean, std = (
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt"   , type=str, default="./cover.yml", help="the option file")
    parser.add_argument('-d', "--device", type=str, default="cuda"       , help='CUDA device id')
    parser.add_argument("-i", "--input_video_dir", type=str, default="./demo", help="the input video dir")
    parser.add_argument(      "--output", type=str, default="./demo.csv" , help='output file to store predict mos value')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    ### Load COVER
    evaluator = COVER(**opt["model"]["args"]).to(args.device)
    state_dict = torch.load(opt["test_load_path"], map_location=args.device)
    
    # set strict=False here to avoid error of missing
    # weight of prompt_learner in clip-iqa+, cross-gate
    evaluator.load_state_dict(state_dict['state_dict'], strict=False)


    video_paths = []
    all_results = {}

    with open(args.output, "w") as w:
        w.write(f"path, semantic score, technical score, aesthetic score, overall/final score\n")

    dopt = opt["data"]["val-l1080p"]["args"]

    dopt["anno_file"] = None
    dopt["data_prefix"] = args.input_video_dir

    dataset = ViewDecompositionDataset(dopt)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
    )

    sample_types = ["semantic", "technical", "aesthetic"]

    for i, data in enumerate(tqdm(dataloader, desc="Testing")):
        if len(data.keys()) == 1:
            ##  failed data
            continue

        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(args.device)
                b, c, t, h, w = video[key].shape
                video[key] = (
                    video[key]
                    .reshape(
                        b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                    )
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(
                        b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                    )
                )
    
        with torch.no_grad():
            results = evaluator(video, reduce_scores=False)
            results = [np.mean(l.cpu().numpy()) for l in results]

        rescaled_results = fuse_results(results)
        # all_results[data["name"][0]] = rescaled_results

        # with open(
        #    f"cover_predictions/val-custom_{args.input_video_dir.split('/')[-1]}.pkl", "wb"
        # ) as wf:
        # pkl.dump(all_results, wf)
        
        with open(args.output, "a") as w:
            w.write(
                f'{data["name"][0].split("/")[-1]},{rescaled_results["semantic"]:4f},{rescaled_results["technical"]:4f},{rescaled_results["aesthetic"]:4f},{rescaled_results["overall"]:4f}\n'
            )
