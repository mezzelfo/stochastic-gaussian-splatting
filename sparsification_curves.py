import torch
import os
import glob
import re
import numpy as np
from torchvision.io import read_image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

### Metrics Objects
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
lpips = LearnedPerceptualImagePatchSimilarity()

class SparsificationCurves:
    def __init__(
            self,
            predictions,
            gts,
            diff_to_error_fn=lambda x: x.square().mean(dim=-3),
            summarizer_fn=lambda x: x.mean(dim=-1).sqrt().mean(),
            P=100,
        ) -> None:
        '''
        predictions:        dict of tensors of shape (..., C, H, W, S) or a single tensor of shape (..., C, H, W, S)
                            where C is the number of channels, H and W are the height and width of the image, and S is the number of samples
        gts:                tensor of shape (..., C, H, W)
        diff_to_error_fn:   function that takes a tensor of shape (..., C, H, W) and returns a tensor of shape (..., H, W)
                            default: lambda x: x.square().mean(dim=-3) (mean squared error per pixel averaged over channels dim=-3)
                            needed for computing the error aggregating the channel dimension
        summarizer_fn:      function that takes a tensor of shape (..., M) and returns a scalar
                            where M is the number of "survived" pixels after sparsification
                            default: lambda x: x.mean(dim=-1).log10().mul(-10).mean() (PSNR)
                            needed for computing the error aggregating the spatial dimension

        # RMSE
        diff_to_error_fn=lambda x: x.square().mean(dim=-3),
        summarizer_fn=lambda x: x.mean(dim=-1).sqrt().mean(),
        # PSNR
        diff_to_error_fn=lambda x: x.square().mean(dim=-3),
        summarizer_fn=lambda x: x.mean(dim=-1).log10().mul(-10).mean(),
        # MAE
        diff_to_error_fn=lambda x: x.abs().mean(dim=-3),
        summarizer_fn=lambda x: x.mean(),

        '''
        if type(predictions) != dict:
            predictions = {'us': predictions}

        self.predictions = predictions  # dict of tensor of shape (..., C, H, W, S)
        self.gts = gts.unsqueeze(-1)    # a single tensor of shape (..., C, H, W, 1)
        self.diff_to_error_fn = diff_to_error_fn
        self.summarizer_fn = summarizer_fn

        self.errors = {}
        self.uncertainties = {}

        for k in predictions.keys():
            # check that self.predictions is broadcastable to self.gts
            assert torch.broadcast_shapes(self.predictions[k].shape, self.gts.shape)
            error = self._compute_error(self.predictions[k], self.gts)      # (..., H, W)
            uncertainty = self._compute_uncertainty(self.predictions[k])    # (..., H, W)

            error = error.flatten(start_dim=-2)                             # (..., H*W)
            uncertainty = uncertainty.flatten(start_dim=-2)                 # (..., H*W)

            self.errors[k] = error
            self.uncertainties[k] = uncertainty

        self.HW = self.gts.shape[-3] * self.gts.shape[-2]                 # H * W        
        self.percentage_ax = torch.arange(0, self.HW, self.HW/P, device=self.gts.device).round().int()

        if ssim.device != self.gts.device:
            ssim.to(self.gts.device)
        if lpips.device != self.gts.device:
            lpips.to(self.gts.device)

    def _compute_error(self, predictions, gts):
        # predictions:  (..., C, H, W, S)
        # gts:          (..., C, H, W, 1)
        # output:       (..., H, W)
        mean_pred = predictions.mean(dim=-1, keepdim=True)
        diff = mean_pred - gts
        assert diff.shape[-1] == 1
        diff = diff.squeeze(-1)
        output = self.diff_to_error_fn(diff)
        return output
    
    def _compute_uncertainty(self, predictions):
        # predictions:  (..., C, H, W, S)
        # output:       (..., H, W)
        assert predictions.shape[-1] > 1
        output = predictions.std(dim=-1).mean(dim=-3)
        return output
    
    def _get_sparsification_curve(self, statistic, error):
        res = []
        order = torch.argsort(statistic, descending=True, dim=-1)
        reordered_error = torch.gather(error, -1, order)
        for percentage in self.percentage_ax:
            summary = self.summarizer_fn(reordered_error[...,percentage:])
            res.append(summary)

        res = torch.stack(res)
        return res
    
    def _compute_area_under_sparsification_curve(self, sparsification_curve):
        # sparsification_curve: (..., P)
        # output:               (...)
        return torch.trapezoid(
            y=sparsification_curve,
            x=self.percentage_ax/self.HW,
            dim=-1,
        )
    
    def _compute_psnr(self, predictions, gts):
        # predictions:  (..., C, H, W, S)
        # gts:          (..., C, H, W, 1)
        # output:       (...)
        mean_pred = predictions.mean(dim=-1, keepdim=True)
        diff = mean_pred - gts
        assert diff.shape[-1] == 1
        diff = diff.squeeze(-1)
        output = diff.square().mean(dim=(-3,-2,-1)).log10().mul(-10).mean()
        return output
    
    def _compute_ssim(self, predictions, gts):
        # predictions:  (..., C, H, W, S)
        # gts:          (..., C, H, W, 1)
        # output:       (...)
        mean_pred = predictions.mean(dim=-1, keepdim=True)
        output = ssim(mean_pred.squeeze(-1), gts.squeeze(-1))
        return output.mean()
    
    def _compute_lpips(self, predictions, gts):
        # predictions:  (..., C, H, W, S)
        # gts:          (..., C, H, W, 1)
        # output:       (...)
        mean_pred = predictions.mean(dim=-1, keepdim=True)
        output = lpips(mean_pred.squeeze(-1), gts.squeeze(-1))
        return output.mean()
    
    def compute_AUSE(self, k='us'):
        sc = self._get_sparsification_curve(self.uncertainties[k], self.errors[k])
        oracle = self._get_sparsification_curve(self.errors[k], self.errors[k])
        ause = self._compute_area_under_sparsification_curve(oracle-sc).abs()
        return ause


    def get_all(self, basic_only=False):
        all_results = {
            'names': [],
            'sc': {},
            'sc_oracle': {},
            'sc_random': {},
            'psnr': {},
            'ausc': {},
            'ause': {},
            'ssim': {},
            'lpips': {},
        }
        rand = torch.rand(self.gts.shape[:-4]+(self.HW,), device=self.gts.device)
        for k in self.predictions.keys():
            all_results['names'].append(k)
            all_results['sc'][k] = self._get_sparsification_curve(self.uncertainties[k], self.errors[k])
            all_results['sc_oracle'][k] = self._get_sparsification_curve(self.errors[k], self.errors[k])
            all_results['sc_random'][k] = self._get_sparsification_curve(rand, self.errors[k])
            all_results['ausc'][k] = self._compute_area_under_sparsification_curve(all_results['sc'][k])
            all_results['ause'][k] = self._compute_area_under_sparsification_curve(all_results['sc_oracle'][k]-all_results['sc'][k]).abs()
            all_results['psnr'][k] = self._compute_psnr(self.predictions[k], self.gts)
            if basic_only:
                continue
            all_results['ssim'][k] = self._compute_ssim(self.predictions[k], self.gts)
            all_results['lpips'][k] = self._compute_lpips(self.predictions[k], self.gts)
        return self.percentage_ax/self.HW, all_results

def get_CFNeRF(root, device='cpu'):
    rgbs_path = glob.glob(os.path.join(root, f'rgbs_*.npy'))
    rgbs_path += glob.glob(os.path.join(root, f'data*.npz'))
    print(root, len(rgbs_path))
    def extract_iter(path:str):
        if path.endswith('.npz'):
            # example *data_123.npy.npz -> 123
            return int(re.search(r'data_(\d+).npy.npz', path).group(1))
        else:
            # example rgbs_123.npy -> 123
            return int(re.search(r'rgbs_(\d+).npy', path).group(1))
    rgbs_path = sorted(rgbs_path, key=extract_iter)
    rgbs = []
    for rgb_path in tqdm(rgbs_path):
        if rgb_path.endswith('.npz'):
            yolo = np.load(rgb_path)['rgbs']
        else:
            yolo = np.load(rgb_path)
        rgbs.append(torch.from_numpy(yolo).permute(2,0,1,3).float())
    rgbs = torch.stack(rgbs).to(device)
    return rgbs

def get_GS(root, device='cpu'):
    rgbs_path = sorted(glob.glob(os.path.join(root, f'renders/idx*.pt')))
    gts_path = sorted(glob.glob(os.path.join(root, f'gt/*.png')))
    print(root, len(rgbs_path))
    rgbs = []
    gts = []
    for rgb_path, gt_path in tqdm(zip(rgbs_path, gts_path)):
        rgbs.append(torch.load(rgb_path, 'cpu').permute(1,2,3,0).float())
        gts.append(read_image(gt_path).float() / 255.)
    rgbs = torch.stack(rgbs).to(device)
    gts = torch.stack(gts).to(device)
    return rgbs, gts
