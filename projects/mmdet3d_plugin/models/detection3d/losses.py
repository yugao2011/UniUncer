import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmdet.models.builder import LOSSES

from projects.mmdet3d_plugin.core.box3d import *

from torch.distributions.laplace import Laplace
from torch.distributions.von_mises import VonMises
import math


# UniUncer: Negative log-likelihood loss under Laplace distribution for box regression.
# Uses predicted means and stds to compute NLL against box targets.
@LOSSES.register_module()
class BoxNLLLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(
            self,
            reduction='mean',
            loss_weight=1.0,

    ):
        super(BoxNLLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = 1e-6

    def forward(self,
                box_means,
                box_stds,
                box_target,
                weight=None,
                avg_factor=None,
                prefix="",
                suffix="",
                cls_target=None,
        ):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """
        # X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ
        n = box_means.shape[0]
        if n > 0:
            xyz_means = box_means[...,:3]
            # print("xyz mean shape",xyz_means.shape)
            xyz_stds = box_stds[...,:3]
            xyz_target = box_target[...,:3]
            # wlh_means = box_means[...,3:6]
            # wlh_stds = box_stds[...,3:6]
            # wlh_target = box_target[...,3:6]
            
            # sin_yaw_means = box_means[...,6]
            # cos_yaw_means = box_means[...,7]

            # sin_yaw_stds = box_stds[...,6]
            # cos_yaw_stds = box_stds[...,7]

            # yaw_means = torch.atan2(box_means[..., SIN_YAW], box_means[..., COS_YAW])
            # yaw_stds = torch.atan2(box_stds[..., SIN_YAW], box_stds[..., COS_YAW])
            # yaw_means = box_means[...,6:8]
            # yaw_stds = box_stds[...,6:8]


            # yaw_target = box_target[...,6:8]
            # yaw_target = torch.atan2(box_target[..., SIN_YAW], box_target[..., COS_YAW])
            # v_means = box_means[...,8:]
            # v_stds = box_stds[...,8:]
            # v_target = box_target[...,8:]

            m_xyz = Laplace(xyz_means, xyz_stds)
            # m_yaw = VonMises(yaw_means,yaw_stds)
            # m_wlh = Laplace(wlh_means, wlh_stds)
            # m_v = Laplace(v_means, v_stds)
            
            nll_loss_xyz = -m_xyz.log_prob(xyz_target)
            # nll_loss_yaw = -m_yaw.log_prob(yaw_target)
            # nll_loss_wlh = -m_wlh.log_prob(wlh_target)
            # nll_loss_v = -m_v.log_prob(v_target)
            


            if self.reduction == 'mean':
                # nll_loss = (nll_loss_xyz.mean() + nll_loss_wlh.mean() + nll_loss_yaw.mean() + nll_loss_v.mean())  * self.loss_weight
                nll_loss = nll_loss_xyz.mean() * self.loss_weight
                # nll_loss = (nll_loss_xyz.mean() + nll_loss_yaw.mean() + nll_loss_wlh.mean())  * self.loss_weight
                nll_loss = torch.clamp(nll_loss, min=0, max=10)
            else:
                # nll_loss = (nll_loss_xyz.sum() + nll_loss_wlh.sum() + nll_loss_yaw.sum() + nll_loss_v.sum()) * self.loss_weight
                nll_loss = nll_loss_xyz.sum() * self.loss_weight
                # nll_loss = (nll_loss_xyz.sum() + nll_loss_yaw.sum() + nll_loss_wlh.sum())  * self.loss_weight
                nll_loss = torch.clamp(nll_loss, min=0, max=10)
            
        else:
            nll_loss = torch.tensor(0., device=box_means.device, requires_grad=True)
            nll_loss = torch.clamp(nll_loss, min=0, max=10)

        return nll_loss


@LOSSES.register_module()
# UniUncer: Now jointly computes L1 loss (on a subset of dims) and Laplace NLL loss
# using predicted box means and stds from laplace_branch.
class SparseBox3DLoss(nn.Module):
    def __init__(
        self,
        loss_box_l1,
        loss_box_nll,
        loss_centerness=None,
        loss_yawness=None,
        cls_allow_reverse=None,
    ):
        super().__init__()

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.loss_box_l1 = build(loss_box_l1, LOSSES)
        self.loss_box_nll = build(loss_box_nll, LOSSES)
        self.loss_cns = build(loss_centerness, LOSSES)
        self.loss_yns = build(loss_yawness, LOSSES)
        self.cls_allow_reverse = cls_allow_reverse

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        prefix="",
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        # seperate stds from quality
        if quality is not None:
            stds = quality[..., 0:10]
            quality = quality[..., 10:]
        # Some categories do not distinguish between positive and negative
        # directions. For example, barrier in nuScenes dataset.
        if self.cls_allow_reverse is not None and cls_target is not None:
            if_reverse = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                < 0
            )
            if_reverse = (
                torch.isin(
                    cls_target, cls_target.new_tensor(self.cls_allow_reverse)
                )
                & if_reverse
            )
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                if_reverse[..., None],
                -box_target[..., [SIN_YAW, COS_YAW]],
                box_target[..., [SIN_YAW, COS_YAW]],
            )

        output = {}
        l1_index = [3,4,5,6,7,8,9]
        box_l1 = box[...,l1_index]
        box_target_l1 = box_target[...,l1_index]
        weight_l1 = weight[...,l1_index]
        box_loss_l1 = self.loss_box_l1(
            box_l1, box_target_l1, weight=weight_l1, avg_factor=avg_factor
        )
        output[f"{prefix}loss_box_l1{suffix}"] = box_loss_l1

        box_loss_nll = self.loss_box_nll(
            box, stds, box_target, weight=weight, avg_factor=avg_factor
        )
        output[f"{prefix}loss_box_nll{suffix}"] = box_loss_nll

        if quality is not None:
            cns = quality[..., CNS]
            
            cns_target = torch.norm(
                box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], p=2, dim=-1
            )
            cns_target = torch.exp(-cns_target)
            cns_loss = self.loss_cns(cns, cns_target, avg_factor=avg_factor)
            output[f"{prefix}loss_cns{suffix}"] = cns_loss
            
            yns = quality[..., YNS].sigmoid()
            yns_target = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                > 0
            )
            yns_target = yns_target.float()
            yns_loss = self.loss_yns(yns, yns_target, avg_factor=avg_factor)
            output[f"{prefix}loss_yns{suffix}"] = yns_loss
        return output
