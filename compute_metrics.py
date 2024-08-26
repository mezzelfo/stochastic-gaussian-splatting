import torch
from sparsification_curves import SparsificationCurves, get_GS
import argparse

@torch.no_grad()
def compute_metrics(experiment_path, basic_only, device):
    pred, gt = get_GS(experiment_path, device=device)
    sc = SparsificationCurves(
        predictions=pred,
        gts=gt,
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
    _, all_results = sc.get_all(basic_only=basic_only)
    all_results.pop('names')
    if basic_only:
        all_results.pop('ssim')
        all_results.pop('lpips')
    mean = {k:v['us'].mean().item() for k,v in all_results.items()}
    return mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--basic_only', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    df = compute_metrics(args.experiment_path, args.basic_only, args.device)
    print(df)
