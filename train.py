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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from sparsification_curves import SparsificationCurves

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        state = torch.load(checkpoint)
        gaussians.load_state_dict(state)
        gaussians.training_setup(opt)
        first_iter = state['iteration']

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", disable=True)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0 and iteration < args.switch_to_beyesian:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        gt_image = viewpoint_cam.original_image.cuda()
        yhat = []
        for rep in range(8 if gaussians.am_i_bayesian else 1):
            torch.manual_seed(hash(str(args)+str(iteration)+str(rep)))
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            image = torch.clamp(image, 0.0, 1.0)
            yhat.append(image)

        if gaussians.am_i_bayesian:
            loss = 0.0
            image_stack = torch.stack(yhat, dim=-1)
            mean_image = image_stack.mean(dim=-1)
            # var_image = image_stack.var(dim=-1)
            # nll = torch.nn.functional.gaussian_nll_loss(mean_image, gt_image, var_image)
            # loss += (1.0 - opt.lambda_dssim) * nll + opt.lambda_dssim * (1.0 - ssim(mean_image, gt_image))
            Ll1 = l1_loss(mean_image, gt_image)
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(mean_image, gt_image))
            if args.KL_weight > 0:
                loss += args.KL_weight*gaussians.compute_posterior_prior_KL()
            if args.AUSE_weight > 0:
                sc = SparsificationCurves(
                    image_stack,
                    gt_image,
                )
                # curve = sc._get_sparsification_curve(sc.uncertainties['us'], sc.errors['us'])
                # loss += sc._compute_area_under_sparsification_curve(curve)
                loss += args.AUSE_weight * sc.compute_AUSE()
        else:
            mean_image = yhat[0]
            Ll1 = l1_loss(mean_image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(mean_image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                checkpoint_path = os.path.join(scene.model_path, "checkpoint_{}.pth".format(iteration))
                state = gaussians.state_dict()
                state['iteration'] = iteration
                torch.save(state, checkpoint_path)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()    

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration >= args.switch_to_beyesian:
                # Now the number of gaussians is finalized
                # We can start to compute the uncertainty
                gaussians.transition_to_bayesian(args)          

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    assert tb_writer is not None
    tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})
        for config in validation_configs:
            images_test = []
            gts_test = []
            for viewpoint in config['cameras']:
                repeat = 10 if scene.gaussians.am_i_bayesian else 1
                images = []
                for rep in range(repeat):
                    torch.manual_seed(hash(str(args)+str(iteration)+str(rep)+str(config)))
                    images.append(torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0))
                images_test.append(torch.stack(images))
                gts_test.append(torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0))
            images_test = torch.stack(images_test).permute(0,2,3,4,1) # [cameras, 3, H, W, repeat]
            gts_test = torch.stack(gts_test).squeeze() # [cameras, 3, H, W]
            assert images_test.ndim == 5 and gts_test.ndim == 4
            assert images_test.shape[:4] == gts_test.shape
            assert images_test.shape[-1] == repeat
            if not scene.gaussians.am_i_bayesian:
                images_test = images_test.mean(dim=-1)
                psnr_test = (images_test - gts_test).square().mean(dim=(-1,-2,-3)).log10().mul(-10).mean()
                ausc_test = torch.nan
                ause_test = torch.nan
            else:
                sc = SparsificationCurves(
                    images_test,
                    gts_test,
                    # RMSE
                    diff_to_error_fn=lambda x: x.square().mean(dim=-3),
                    summarizer_fn=lambda x: x.mean(dim=-1).sqrt().mean(),
                    # # PSNR
                    # diff_to_error_fn=lambda x: x.square().mean(dim=-3),
                    # summarizer_fn=lambda x: x.mean(dim=-1).log10().mul(-10).mean(),
                    # # MAE
                    # diff_to_error_fn=lambda x: x.abs().mean(dim=-3),
                    # summarizer_fn=lambda x: x.mean(),
                )
                _, all_results = sc.get_all(basic_only=True)
                psnr_test = all_results['psnr']['us']
                ausc_test = all_results['ausc']['us']
                ause_test = sc.compute_AUSE()

            print("\n[ITER {}] Evaluating {}: PSNR {} AUSC {} AUSE {}".format(iteration, config['name'], psnr_test, ausc_test, ause_test))
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ausc', ausc_test, iteration)
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ause', ause_test, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # Bayesian regime parameters
    parser.add_argument("--switch_to_beyesian", type=int, default=15_100)
    parser.add_argument("--prior_xyz_variance", type=float)
    parser.add_argument("--posterior_xyz_lr", type=float)
    parser.add_argument("--prior_features_variance", type=float)
    parser.add_argument("--posterior_features_lr", type=float)
    parser.add_argument("--prior_opacity_variance", type=float)
    parser.add_argument("--posterior_opacity_lr", type=float)
    parser.add_argument("--KL_weight", type=float)
    parser.add_argument("--AUSE_weight", type=float)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.switch_to_beyesian-1)
    args.save_iterations.append(args.switch_to_beyesian+1)
    args.save_iterations.append(args.iterations)
    # Sanity checks for bayesian regime
    assert args.densify_until_iter < args.switch_to_beyesian, "densify_until_iter must be smaller than switch_to_beyesian"
    # assert args.compute_cov3D_python == False
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
