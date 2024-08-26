#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


def minmax_standardize(x, dim=(-1,-2,-3)):
    return (x - x.amin(dim=dim)) / (x.amax(dim=dim) - x.amin(dim=dim))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        renders = []
        for rep in range(32):
            torch.manual_seed(hash(str(args)+str(iteration)+str(rep)+str(name)))
            rendering = render(view, gaussians, pipeline, background)["render"]
            rendering = torch.clamp(rendering, 0.0, 1.0)
            renders.append(rendering)
            # torchvision.utils.save_image(rendering, os.path.join(render_path, 'idx{0:05d}_rep{1:05d}'.format(idx, rep) + ".png"))
        renders = torch.stack(renders)
        torch.save(renders, os.path.join(render_path, 'idx{0:05d}'.format(idx) + ".pt"))
        torchvision.utils.save_image(minmax_standardize(renders.std(dim=0)), os.path.join(render_path, 'idx{0:05d}_stddev'.format(idx) + ".png"))
        torchvision.utils.save_image(renders.mean(dim=0).clip(0.,1.), os.path.join(render_path, 'idx{0:05d}_mean'.format(idx) + ".png"))
        torchvision.utils.save_image(renders.median(dim=0).values.clip(0.,1.), os.path.join(render_path, 'idx{0:05d}_median'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)