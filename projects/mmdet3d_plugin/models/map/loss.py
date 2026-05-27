import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmdet.models.builder import LOSSES
from mmdet.models.losses import l1_loss, smooth_l1_loss
from mmdet.models import weighted_loss
from torch.distributions.laplace import Laplace

# UniUncer: Negative log-likelihood loss under Laplace distribution for map line regression.
# Uses predicted point-wise means and stds to compute NLL against line targets.
@LOSSES.register_module()
class LinesNLLLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0,num_sample=20, map_velo=False):
        super(LinesNLLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.num_sample = num_sample
        self.map_velo = map_velo

    def forward(self,
                line,
                stds,
                line_target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        n = line_target.shape[0]
        if n > 0:
            num_pts = n*self.num_sample

            pts_pred= line.reshape(n, self.num_sample, 2)

            # print(pts_pred.shape)
            xy_means = pts_pred[...,:2]

            pts_betas = stds.reshape(n, self.num_sample, 2)
            xy_stds = pts_betas[...,:2]

            target = line_target.reshape(n, self.num_sample, 2)
            xy_target = target[...,:2]

            xy_means = xy_means.reshape(num_pts,2)
            xy_stds = xy_stds.reshape(num_pts,2)
            xy_target = xy_target.reshape(num_pts,2)

            m_xy = Laplace(xy_means, xy_stds)
            log_prob = m_xy.log_prob(xy_target)
            nll_loss_xy = -log_prob
            # prob = torch.exp(log_prob)
            # print("Max prob:", prob.max().item())

            if self.reduction == 'mean':
                nll_loss = nll_loss_xy.mean() * self.loss_weight
                nll_loss = torch.clamp(nll_loss, min=1e-3, max=10)   # 这里控制一下最大 最小值
            else:
                nll_loss = nll_loss_xy.sum() * self.loss_weight
                nll_loss = torch.clamp(nll_loss, min=1e-3, max=10)
        else:
            # keep the training happy for find 0 element 
            num_pts = self.num_sample

            xy_means = line[...,:40]
            xy_stds = stds[...,:40]
            xy_target = line_target[...,:40]

            m_xy = Laplace(xy_means, xy_stds)
            nll_loss_xy = -m_xy.log_prob(xy_target)
            nll_loss = nll_loss_xy.mean() * self.loss_weight
            nll_loss = torch.clamp(nll_loss, min=1e-3, max=10)
            # nll_loss = (nll_loss_xy.mean() + nll_loss_v.mean()) * self.loss_weight

        return nll_loss

        
@LOSSES.register_module()
class LinesL1Loss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
        """
            L1 loss. The same as the smooth L1 loss
            Args:
                reduction (str, optional): The method to reduce the loss.
                    Options are "none", "mean" and "sum".
                loss_weight (float, optional): The weight of loss.
        """

        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
                shape: [bs, ...]
            target (torch.Tensor): The learning target of the prediction.
                shape: [bs, ...]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None. 
                it's useful when the predictions are not all valid.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.beta > 0:
            loss = smooth_l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
        
        else:
            loss = l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        
        num_points = pred.shape[-1] // 2
        loss = loss / num_points

        return loss*self.loss_weight


@LOSSES.register_module()
# UniUncer: Now jointly computes L1 loss and Laplace NLL loss using predicted point-wise stds.
class SparseLineLoss(nn.Module):
    def __init__(
        self,
        loss_line_l1,
        loss_line_nll,
        num_sample=20,
        roi_size=(30, 60),
    ):
        super().__init__()

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.loss_line_l1 = build(loss_line_l1, LOSSES)
        self.loss_line_nll = build(loss_line_nll, LOSSES)
        self.num_sample = num_sample
        self.roi_size = roi_size


    def forward(
        self,
        line,
        stds,
        line_target,
        weight=None,
        avg_factor=None,
        prefix="",
        suffix="",
        **kwargs,
    ):
        output = {}
        # print("when computing loss_reg, line shape, line_target shape:", line.shape, line_target.shape)
        n = line.shape[0]
        if n > 0:
            line_reshape = line.reshape(n, self.num_sample, 2) # (n, self.num_sample, 4) # 这里的line shape是bsx40，无法reshape成nxnum_samplex4，将4修改为2
            line_l1 = line_reshape[...,:2]
            line_l1 = line_l1.reshape(n, self.num_sample*2)
        
            line_target_reshape = line_target.reshape(n, self.num_sample, 2)  # (n, self.num_sample, 4)
            line_l1_target = line_target_reshape[...,:2]

            line_l1_target = line_l1_target.reshape(n, self.num_sample*2)

            line_l1 = self.normalize_line(line_l1)
            line_target_l1 = self.normalize_line(line_l1_target)

            line_loss_l1 = self.loss_line_l1(
                line_l1, line_target_l1, weight=weight[:,:40], avg_factor=avg_factor
            )

            output[f"{prefix}loss_line_l1{suffix}"] = line_loss_l1

            # print("line shape, stds shape, line_target shape", line.shape, stds.shape, line_target.shape)  都是bsx40
            # 使用归一化nll
            # norm_laplace_line, norm_laplace_stds = self.normalize_laplace_line(line, stds)
            # line_loss_nll = self.loss_line_nll(norm_laplace_line, norm_laplace_stds, self.normalize_line(line_target))

            # 不使用归一化nll
            line_loss_nll = self.loss_line_nll(line, stds, line_target)

            output[f"{prefix}loss_line_nll{suffix}"] = line_loss_nll

        else:
            line_l1 = line[...,:40]
            # line_velo = line[...,40:]
            line_target_l1 = line_target[...,:40]
            line_loss_l1 = self.loss_line_l1(
                line_l1, line_target_l1, weight=weight[:,:40], avg_factor=avg_factor
            )
            output[f"{prefix}loss_line_l1{suffix}"] = line_loss_l1
            
            line_loss_nll = self.loss_line_nll(line, stds, line_target)
            # norm_laplace_line, norm_laplace_stds = self.normalize_laplace_line(line, stds)
            # line_loss_nll = self.loss_line_nll(norm_laplace_line, norm_laplace_stds, self.normalize_line(line_target))
            output[f"{prefix}loss_line_nll{suffix}"] = line_loss_nll

        return output

    def normalize_line(self, line):
        if line.shape[0] == 0:
            return line

        line = line.view(line.shape[:-1] + (self.num_sample, -1))
        
        origin = -line.new_tensor([self.roi_size[0]/2, self.roi_size[1]/2])
        line = line - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps
        line = line / norm
        line = line.flatten(-2, -1)

        return line
    
    def normalize_laplace_line(self, line, stds):
        if line.shape[0] == 0:
            return line

        line = line.view(line.shape[:-1] + (self.num_sample, -1))
        
        origin = -line.new_tensor([self.roi_size[0]/2, self.roi_size[1]/2])
        line = line - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps
        line = line / norm
        line = line.flatten(-2, -1)

        stds = stds.view(stds.shape[:-1] + (self.num_sample, -1))
        stds = stds / norm
        stds = torch.clamp(stds, min=1e-6, max=1e-1)
        stds = stds.flatten(-2, -1)
        
        return line, stds