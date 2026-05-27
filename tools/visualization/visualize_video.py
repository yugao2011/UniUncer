import os
import glob
import argparse
from tqdm import tqdm

import cv2
import numpy as np
import torch
from PIL import Image

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from shapely.geometry import Polygon
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners

from tools.visualization.video_bev_render import (
    BEVRenderCustom,
    BEVRenderCustom_Origin,
)
from tools.visualization.custom_cam_render import CustomCamRender


plot_choices = dict(
    draw_pred=True,  # True: draw gt and pred; False: only draw gt
    det=True,
    cam_det=False, # draw detections on the image
    track=False,  # True: draw history tracked boxes
    # motion = True,
    motion=False,
    motion_top_k=1,
    map=True,
    planning=True,
    planning_top_k=1,
    with_uncer_value=False,
    with_frame_index=False,
    with_metrics=False,
    with_pred_cmd_legend=False,
    cam_planning_gt=False,
    dpi=0,
)

START = 4811
END = 4843
INTERVAL = 1


def check_collision(ego_box, boxes):
    """
    ego_box: tensor with shape [7], [x, y, z, w, l, h, yaw]
    boxes: tensor with shape [N, 7]
    """

    if boxes.shape[0] == 0:
        return False, -1

    # follow uniad, add a 0.5m offset
    ego_box[0] += 0.5 * torch.cos(ego_box[6])
    ego_box[1] += 0.5 * torch.sin(ego_box[6])
    ego_corners_box = box3d_to_corners(ego_box.unsqueeze(0))[0, [0, 3, 7, 4], :2]

    corners_box = box3d_to_corners(boxes)[:, [0, 3, 7, 4], :2]
    ego_poly = Polygon([(point[0], point[1]) for point in ego_corners_box])
    for i in range(len(corners_box)):
        box_poly = Polygon([(point[0], point[1]) for point in corners_box[i]])
        collision = ego_poly.intersects(box_poly)
        if collision:
            return True, i

    return False, -1


def get_yaw(traj):
    start = traj[0]
    end = traj[-1]
    dist = torch.linalg.norm(end - start, dim=-1)
    if dist < 0.5:
        return traj.new_ones(traj.shape[0]) * np.pi / 2

    zeros = traj.new_zeros((1, 2))
    traj_cat = torch.cat([zeros, traj], dim=0)
    yaw = traj.new_zeros(traj.shape[0] + 1)
    yaw[..., 1:-1] = torch.atan2(
        traj_cat[..., 2:, 1] - traj_cat[..., :-2, 1],
        traj_cat[..., 2:, 0] - traj_cat[..., :-2, 0],
    )
    yaw[..., -1] = torch.atan2(
        traj_cat[..., -1, 1] - traj_cat[..., -2, 1],
        traj_cat[..., -1, 0] - traj_cat[..., -2, 0],
    )
    return yaw[1:]


class PlanningMetric:
    def __init__(
        self,
        n_future=6,
        compute_on_step: bool = False,
    ):
        self.W = 1.85
        self.H = 4.084

        self.n_future = n_future
        self.reset()

    def reset(self):
        self.obj_col = torch.zeros(self.n_future)
        self.obj_box_col = torch.zeros(self.n_future)
        self.L2 = torch.zeros(self.n_future)
        self.total = torch.tensor(0)

    def evaluate_single_coll(self, traj, fut_boxes):
        n_future = traj.shape[0]
        yaw = get_yaw(traj)
        ego_box = traj.new_zeros((n_future, 7))
        ego_box[:, :2] = traj
        ego_box[:, 3:6] = ego_box.new_tensor([self.H, self.W, 1.56])
        ego_box[:, 6] = yaw
        collision = torch.zeros(n_future, dtype=torch.bool)
        collision_index = torch.zeros(n_future)

        for t in range(n_future):
            ego_box_t = ego_box[t].clone()
            boxes = fut_boxes[t].clone()
            collision[t], collision_index[t] = check_collision(ego_box_t, boxes)
        return collision, collision_index

    def evaluate_coll(self, trajs, gt_trajs, fut_boxes):
        B, n_future, _ = trajs.shape
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=trajs.device)
        obj_box_coll_sum = torch.zeros(n_future, device=trajs.device)

        assert B == 1, "only supprt bs=1"
        for i in range(B):
            gt_box_coll, gt_collision_index = self.evaluate_single_coll(
                gt_trajs[i], fut_boxes
            )
            box_coll, collision_index = self.evaluate_single_coll(trajs[i], fut_boxes)
            box_coll = torch.logical_and(
                box_coll, torch.logical_not(gt_box_coll)
            )  # 排除gt中已经发生碰撞的部分，只保留预测轨迹新增加的碰撞

            obj_coll_sum += gt_box_coll.long()
            obj_box_coll_sum += box_coll.long()

        return obj_coll_sum, obj_box_coll_sum, collision_index

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        """
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        """
        return torch.sqrt(
            (((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)
        )

    def update(self, trajs, gt_trajs, gt_trajs_mask, fut_boxes):
        assert trajs.shape == gt_trajs.shape
        trajs[..., 0] = -trajs[..., 0]
        gt_trajs[..., 0] = -gt_trajs[..., 0]

        L2 = self.compute_L2(
            trajs, gt_trajs, gt_trajs_mask
        )  # 这3个元素的shape都是6x2，显示的是未来6个时间步的轨迹坐标
        # fut_boxes是一个长度为6的list，显示的是未来6个时间步的boxes，在每个时间步，与任何一个boxes发生碰撞都视为发生碰撞
        obj_coll_sum, obj_box_coll_sum, collision_index = self.evaluate_coll(
            trajs[:, :, :2], gt_trajs[:, :, :2], fut_boxes
        )
        return L2.mean(), obj_coll_sum, obj_box_coll_sum, collision_index


class Visualizer:
    def __init__(
        self,
        args,
        plot_choices,
    ):
        self.out_dir = args.out_dir
        self.combine_dir = os.path.join(self.out_dir, "combine")
        os.makedirs(self.combine_dir, exist_ok=True)

        cfg = Config.fromfile(args.config)
        self.dataset = build_dataset(cfg.data.val)
        self.results = mmcv.load(args.result_path)  # load保存的结果
        self.sparsedrive_results = mmcv.load(
            "work_dirs/sparsedrive_small_stage2/results.pkl"
        )  # load sparsedrive的results
        self.bev_render = BEVRenderCustom(plot_choices, self.out_dir)
        self.bev_render_origin = BEVRenderCustom_Origin(plot_choices, self.out_dir)
        self.select_better = args.select_better
        self.cam_render = CustomCamRender(plot_choices, self.out_dir)

    def add_vis(self, index):
        data = self.dataset.get_data_info(index)
        result = self.results[index]["img_bbox"]
        sparsedrive_result = self.sparsedrive_results[index]["img_bbox"]

        planning_metrics = PlanningMetric()
        sdc_planning = (
            torch.from_numpy(data["gt_ego_fut_trajs"])
            .unsqueeze(0)
            .cumsum(dim=-2)
            .unsqueeze(1)
        )
        sdc_planning_mask = (
            torch.from_numpy(data["gt_ego_fut_masks"])
            .unsqueeze(-1)
            .repeat(1, 1, 2)
            .unsqueeze(1)
        )
        fut_boxes = [torch.from_numpy(arr) for arr in data["fut_boxes"]]
        if sdc_planning_mask.all():  ## for incomplete gt, we do not count this sample
            pred_sdc_traj = result["final_planning"].unsqueeze(0)
            sparsedrive_pred_sdc_traj = sparsedrive_result["final_planning"].unsqueeze(
                0
            )
            L2, obj_coll_sum, obj_box_coll_sum, collision_index = (
                planning_metrics.update(
                    pred_sdc_traj[:, :6, :2],
                    sdc_planning[0, :, :6, :2],
                    sdc_planning_mask[0, :, :6, :2],
                    fut_boxes,
                )
            )
            (
                sparsedrive_L2,
                sparsedrive_obj_coll_sum,
                sparsedrive_obj_box_coll_sum,
                sparsedrive_collision_index,
            ) = planning_metrics.update(
                sparsedrive_pred_sdc_traj[:, :6, :2],
                sdc_planning[0, :, :6, :2],
                sdc_planning_mask[0, :, :6, :2],
                fut_boxes,
            )
            if self.select_better and (
                L2 > sparsedrive_L2
                or obj_box_coll_sum.mean() > sparsedrive_obj_box_coll_sum.mean()
            ):
                return
            # spd
            sparsedrive_bev_pred_path = self.bev_render_origin.render(
                data, sparsedrive_result, index, sparsedrive_L2, colormap="winter"
            )
            # ours
            bev_pred_path, _, _, _, _ = self.bev_render.render(
                data, result, index, L2, colormap="autumn"
            )
            # cam_pred_path = self.cam_render.render(data, result, index)
            cam_pred_path = self.cam_render.render(
                data, result, sparsedrive_result, index
            )
            self.combine_custom(
                bev_pred_path, cam_pred_path, sparsedrive_bev_pred_path, index
            )

    def combine(self, bev_gt_path, bev_pred_path, cam_pred_path, index):
        bev_gt = cv2.imread(bev_gt_path)
        bev_image = cv2.imread(bev_pred_path)
        cam_image = cv2.imread(cam_pred_path)
        merge_image = cv2.hconcat([cam_image, bev_image, bev_gt])
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + ".jpg")
        cv2.imwrite(save_path, merge_image)

    def combine_ours(
        self,
        bev_gt_path,
        bev_pred_path,
        cam_pred_path,
        sparsedrive_bev_pred_path,
        index,
    ):
        bev_gt = cv2.imread(bev_gt_path)
        bev_image = cv2.imread(bev_pred_path)
        cam_image = cv2.imread(cam_pred_path)
        sparsedrive_bev_image = cv2.imread(sparsedrive_bev_pred_path)
        merge_image = cv2.hconcat([cam_image, bev_image, sparsedrive_bev_image, bev_gt])
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + ".jpg")
        cv2.imwrite(save_path, merge_image)

    def combine_custom(
        self, bev_pred_path, cam_pred_path, sparsedrive_bev_pred_path, index
    ):
        bev_image = cv2.imread(bev_pred_path)
        cam_image = cv2.imread(cam_pred_path)
        sparsedrive_bev_image = cv2.imread(sparsedrive_bev_pred_path)

        target_h, _ = bev_image.shape[:2]
        h, w = cam_image.shape[:2]
        if h != target_h:
            scale = target_h / h
            target_w = int(w * scale)
            cam_image = cv2.resize(
                cam_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        merge_image = cv2.hconcat([cam_image, sparsedrive_bev_image, bev_image])
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + ".jpg")
        cv2.imwrite(save_path, merge_image)

    def image2video(self, fps=12, downsample=4, start=0, end=0):
        imgs_path = glob.glob(os.path.join(self.combine_dir, "*.jpg"))
        imgs_path = sorted(imgs_path)
        img_array = []
        for img_path in tqdm(imgs_path):
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(
                img,
                (width // downsample, height // downsample),
                interpolation=cv2.INTER_AREA,
            )
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out_path = os.path.join(self.out_dir, "%d_%d_video.mp4" % (start, end))
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize groundtruth and results")
    parser.add_argument("config", help="config file path")
    parser.add_argument(
        "--result-path",
        default=None,
        help="prediction result to visualize"
        "If submission file is not provided, only gt will be visualized",
    )
    parser.add_argument(
        "--out-dir",
        default="vis_ours_collision",
        help="directory where visualize results will be saved",
    )
    parser.add_argument(
        "--select_better",
        action="store_true",
        help="whether to vis better scenes",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    visualizer = Visualizer(args, plot_choices)

    for idx in tqdm(range(START, END + 1, INTERVAL)):  # 0-81，索引间隔是1
        if idx > len(visualizer.results):
            break
        visualizer.add_vis(idx) 

    visualizer.image2video(fps=2, downsample=2, start=START, end=END)


if __name__ == "__main__":

    main()
