import os
import numpy as np
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners

CMD_LIST = ["Turn Right", "Turn Left", "Go Straight"]
COLOR_VECTORS = ["cornflowerblue", "royalblue", "slategrey"]
SCORE_THRESH = 0.3
MAP_SCORE_THRESH = 0.3
MAP_SCORE_THRESH_OURS = 0.3
color_mapping = (
    np.asarray(
        [
            [0, 0, 0],
            [255, 179, 0],
            [128, 62, 117],
            [255, 104, 0],
            [166, 189, 215],
            [193, 0, 32],
            [206, 162, 98],
            [129, 112, 102],
            [0, 125, 52],
            [246, 118, 142],
            [0, 83, 138],
            [255, 122, 92],
            [83, 55, 122],
            [255, 142, 0],
            [179, 40, 81],
            [244, 200, 0],
            [127, 24, 13],
            [147, 170, 0],
            [89, 51, 21],
            [241, 58, 19],
            [35, 44, 22],
            [112, 224, 255],
            [70, 184, 160],
            [153, 0, 255],
            [71, 255, 0],
            [255, 0, 163],
            [255, 204, 0],
            [0, 255, 235],
            [255, 0, 235],
            [255, 0, 122],
            [255, 245, 0],
            [10, 190, 212],
            [214, 255, 0],
            [0, 204, 255],
            [20, 0, 255],
            [255, 255, 0],
            [0, 153, 255],
            [0, 255, 204],
            [41, 255, 0],
            [173, 0, 255],
            [0, 245, 255],
            [71, 0, 255],
            [0, 255, 184],
            [0, 92, 255],
            [184, 255, 0],
            [255, 214, 0],
            [25, 194, 194],
            [92, 0, 255],
            [220, 220, 220],
            [255, 9, 92],
            [112, 9, 255],
            [8, 255, 214],
            [255, 184, 6],
            [10, 255, 71],
            [255, 41, 10],
            [7, 255, 255],
            [224, 255, 8],
            [102, 8, 255],
            [255, 61, 6],
            [255, 194, 7],
            [0, 255, 20],
            [255, 8, 41],
            [255, 5, 153],
            [6, 51, 255],
            [235, 12, 255],
            [160, 150, 20],
            [0, 163, 255],
            [140, 140, 140],
            [250, 10, 15],
            [20, 255, 0],
        ]
    )
    / 255
)


class BEVRenderCustom:
    def __init__(
        self,
        plot_choices,
        out_dir,
        xlim=40,
        ylim=40,
    ):
        self.plot_choices = plot_choices
        self.xlim = xlim
        self.ylim = ylim
        self.gt_dir = os.path.join(out_dir, "bev_gt")
        self.pred_dir = os.path.join(out_dir, "bev_pred")
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)
        self.amp = 2

    def reset_canvas(self, render_gt=False):
        plt.close()
        if render_gt:
            figsize = (20, 20)
            self.xlim = 40
        else:
            figsize = (15, 20)
            self.xlim = 30
        self.fig, self.axes = plt.subplots(1, 1, figsize=figsize)
        self.axes.set_xlim(-self.xlim, self.xlim)
        self.axes.set_ylim(-self.ylim, self.ylim)
        self.axes.axis("off")

    def render(self, data, result, index, error, colormap):

        self.reset_canvas(render_gt=True)
        # self.draw_detection_gt(data)
        obj_beta_x, obj_beta_y = self.draw_detection_pred(result)
        self.draw_track_pred(result)
        self.draw_motion_pred(result, top_k=self.plot_choices["motion_top_k"])
        map_beta_x, map_beta_y = self.draw_map_pred(result)
        # draw preds with gts
        self.draw_planning_gt(data)
        self.draw_planning_pred(
            data, result, top_k=self.plot_choices["planning_top_k"], colormap=colormap
        )
        # # draw preds with gts
        # self.draw_planning_gt(data)
        self._render_sdc_car()
        # self._render_command(data)
        self._render_legend()
        if self.plot_choices["with_frame_index"]:
            self.axes.text(-38, -25, "index: {}".format(index), fontsize=40)
        if self.plot_choices["with_metrics"]:
            self.axes.text(-38, -30, "L2: {}".format(error), fontsize=40)
        save_path_pred = os.path.join(self.pred_dir, str(index).zfill(4) + ".jpg")
        self.save_fig(save_path_pred)

        return (
            save_path_pred,
            obj_beta_x,
            obj_beta_y,
            map_beta_x,
            map_beta_y,
        )

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        if self.plot_choices["dpi"] > 0:
            plt.savefig(filename, dpi=self.plot_choices["dpi"], bbox_inches="tight")
        else:
            plt.savefig(filename)

    def plot_points_with_laplace_variances(
        self, x, y, beta_x, beta_y, color, ax, std, mode
    ):
        # 通过这个函数，实现对分布的可视化
        if mode == "detection":
            # 对于object只分布化中心点，并乘以一定的放大倍数所以这里分开处理
            var_x = 2 * beta_x * self.amp**2
            var_y = 2 * beta_y * self.amp**2
            if isinstance(x, (list, tuple, np.ndarray)):
                for j in range(len(x)):
                    if std:
                        width = np.sqrt(var_x[j]) * 2
                        height = np.sqrt(var_y[j]) * 2
                    else:
                        width, height = 0, 0
                    ellipse = Ellipse(
                        (x[j], y[j]),
                        width=width,
                        height=height,
                        fc=color,
                        lw=0.5,
                        alpha=0.3,
                    )
                    ax.add_patch(ellipse)
            else:
                if std:
                    width = np.sqrt(var_x) * 2
                    height = np.sqrt(var_y) * 2
                else:
                    width, height = 0, 0
                ellipse = Ellipse(
                    (x, y), width=width, height=height, fc=color, lw=0.5, alpha=0.3
                )
                ax.add_patch(ellipse)

        elif mode == "map":
            ax.plot(
                x,
                y,
                color=color,
                linewidth=3,
                marker="o",
                linestyle="-",
                markersize=7,
                alpha=0.8,
                zorder=-1,
            )
            # ax.scatter(x, y, color=color, s=2, alpha=0.8, zorder=-1)
            var_x = 2 * beta_x**2
            var_y = 2 * beta_y**2
            for j in range(len(x)):
                if std:
                    width = np.sqrt(var_x[j]) * 2
                    height = np.sqrt(var_y[j]) * 2
                else:
                    width, height = 0, 0
                ellipse = Ellipse(
                    (x[j], y[j]),
                    width=width,
                    height=height,
                    fc=color,
                    lw=0.5,
                    alpha=0.3,
                )
                ax.add_patch(ellipse)

    def draw_detection_gt(self, data):
        if not self.plot_choices["det"]:
            return

        for i in range(data["gt_labels_3d"].shape[0]):
            label = data["gt_labels_3d"][i]
            if label == -1:
                continue
            color = color_mapping[i % len(color_mapping)]

            # draw corners
            corners = box3d_to_corners(data["gt_bboxes_3d"])[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle="-")

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle="-")

    def draw_detection_pred(self, result):
        if not (
            self.plot_choices["draw_pred"]
            and self.plot_choices["det"]
            and "boxes_3d" in result
        ):
            return None, None

        bboxes = result["boxes_3d"]  # shape: 300 x 10
        bboxes_quality = result["quality_3d"]  # shape: 300 x 13

        for i in range(
            result["labels_3d"].shape[0]
        ):  # shape[0]也是300，这里是逐个计算的
            score = result["scores_3d"][i]
            if score < SCORE_THRESH:
                continue
            color = color_mapping[result["instance_ids"][i] % len(color_mapping)]

            # draw corners
            corners = box3d_to_corners(bboxes)[
                i, [0, 3, 7, 4, 0]
            ]  # 这里为每个boxes_3d计算8个角点的坐标，并取出其中5个作为corner数据
            corners_quality = bboxes_quality[i]
            # print('detection corner shape:', corners.shape) # shape: (5, 3)
            x = corners[:, 0]  # shape: (5,)
            y = corners[:, 1]  # shape: (5,)

            self.axes.plot(x, y, color=color, linewidth=3, linestyle="-")

            # draw line to indicate forward direction
            forward_center = np.mean(
                corners[2:4], axis=0
            )  # 选取索引2和3的角点（通常是 3D 边界框前方的两个角）并计算均值作为前边中心点
            center = np.mean(
                corners[0:4], axis=0
            )  # 选取索引 0、1、2、3 的角点（通常是整个 3D 边界框底部的四个角），并计算中心点
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            beta_x = corners_quality[0]
            beta_y = corners_quality[1]

            self.axes.plot(x, y, color=color, linewidth=3, linestyle="-")
            self.plot_points_with_laplace_variances(
                center[0],
                center[1],
                beta_x,
                beta_y,
                color,
                self.axes,
                True,
                "detection",
            )
        obj_beta_x, obj_beta_y = (
            torch.mean(bboxes_quality[:, 0]).item(),
            torch.mean(bboxes_quality[:, 1]).item(),
        )
        if self.plot_choices["with_uncer_value"]:
            self.axes.text(-38, -15, "obj_x: {}".format(obj_beta_x), fontsize=40)
            self.axes.text(-38, -20, "obj_y: {}".format(obj_beta_y), fontsize=40)
        return obj_beta_x, obj_beta_y

    def draw_track_pred(self, result):
        if not (
            self.plot_choices["draw_pred"]
            and self.plot_choices["track"]
            and "anchor_queue" in result
        ):
            return
        temp_bboxes = result["anchor_queue"]  # shape: 300 x 2 x 10
        period = result["period"]  # 300
        bboxes = result["boxes_3d"]
        bboxes_quality = result["quality_3d"]
        for i in range(result["labels_3d"].shape[0]):
            score = result["scores_3d"][i]
            if score < SCORE_THRESH:
                continue
            color = color_mapping[result["instance_ids"][i] % len(color_mapping)]
            center = bboxes[i, :3]
            centers = [center]
            centers_quality = [bboxes_quality[i]]
            for j in range(period[i]):
                # draw corners
                corners = box3d_to_corners(temp_bboxes[:, -1 - j])[
                    i, [0, 3, 7, 4, 0]
                ]  # 获取倒数j帧的预测框并转化为8个角点
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle="-")

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle="-")
                centers.append(center)
                centers_quality.append(
                    bboxes_quality[i]
                )  # 这里可以将当前boxes的uncertainty用做历史track的吗？或许应该找到result["anchor_queue"]对应的uncertainty数值

            centers = np.stack(centers)
            centers_quality = np.stack(centers_quality)
            xs = centers[:, 0]
            ys = centers[:, 1]
            xs_beta = centers_quality[:, 0]
            ys_beta = centers_quality[:, 1]
            self.axes.plot(xs, ys, color=color, linewidth=2, linestyle="-")
            self.plot_points_with_laplace_variances(
                xs, ys, xs_beta, ys_beta, color, self.axes, True, "detection"
            )

    def draw_motion_gt(self, data):
        if not self.plot_choices["motion"]:
            return

        for i in range(data["gt_labels_3d"].shape[0]):
            label = data["gt_labels_3d"][i]
            if label == -1:
                continue
            color = color_mapping[i % len(color_mapping)]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            center = data["gt_bboxes_3d"][i, :2]
            masks = data["gt_agent_fut_masks"][i].astype(bool)
            if masks[0] == 0:
                continue
            trajs = data["gt_agent_fut_trajs"][i][masks]
            trajs = trajs.cumsum(axis=0) + center
            trajs = np.concatenate([center.reshape(1, 2), trajs], axis=0)

            self._render_traj(
                trajs, traj_score=1.0, colormap="winter", dot_size=dot_size
            )

    def draw_motion_pred(self, result, top_k=3):
        if not (
            self.plot_choices["draw_pred"]
            and self.plot_choices["motion"]
            and "trajs_3d" in result
        ):
            return

        bboxes = result["boxes_3d"]
        labels = result["labels_3d"]
        for i in range(result["labels_3d"].shape[0]):
            score = result["scores_3d"][i]
            if score < SCORE_THRESH:
                continue
            label = labels[i]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            traj_score = result["trajs_score"][i].numpy()
            traj = result["trajs_3d"][i].numpy()
            num_modes = len(traj_score)
            center = bboxes[i, :2][None, None].repeat(num_modes, 1, 1).numpy()
            traj = np.concatenate([center, traj], axis=1)

            sorted_ind = np.argsort(traj_score)[::-1]
            sorted_traj = traj[sorted_ind, :, :2]
            sorted_score = traj_score[sorted_ind]
            norm_score = np.exp(sorted_score[0])

            for j in range(top_k - 1, -1, -1):
                viz_traj = sorted_traj[j]
                traj_score = np.exp(sorted_score[j]) / norm_score
                self._render_traj(
                    viz_traj,
                    traj_score=traj_score,
                    colormap="winter",
                    dot_size=dot_size,
                )

    def draw_map_gt(self, data):
        if not self.plot_choices["map"]:
            return
        vectors = data["map_infos"]
        for label, vector_list in vectors.items():
            color = COLOR_VECTORS[label]
            for vector in vector_list:
                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                self.axes.plot(
                    x,
                    y,
                    color=color,
                    linewidth=3,
                    marker="o",
                    linestyle="-",
                    markersize=7,
                )

    def draw_map_pred(self, result):
        if not (
            self.plot_choices["draw_pred"]
            and self.plot_choices["map"]
            and "vectors" in result
        ):
            return

        for i in range(result["scores"].shape[0]):
            score = result["scores"][i]
            if score < MAP_SCORE_THRESH_OURS:
                continue
            color = COLOR_VECTORS[result["labels"][i]]
            pts = result["vectors"][i]
            pts_quality = result["vectors_quality"][i]
            x = pts[:, 0]  # shape (20,)
            y = pts[:, 1]  # shape (20,)
            beta_x = pts_quality[:, 0]
            beta_y = pts_quality[:, 1]

            # plt.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7) # 原本的可视化函数
            # 新的分布可视化函数，需要包含进quality变量
            self.plot_points_with_laplace_variances(
                x, y, beta_x, beta_y, color, self.axes, True, "map"
            )

        map_beta_x = np.mean([np.mean(vec[:, 0]) for vec in result["vectors_quality"]])
        map_beta_y = np.mean([np.mean(vec[:, 1]) for vec in result["vectors_quality"]])
        if self.plot_choices["with_uncer_value"]:
            self.axes.text(-38, -5, "map_x: {}".format(map_beta_x), fontsize=40)
            self.axes.text(-38, -10, "map_y: {}".format(map_beta_y), fontsize=40)
        return map_beta_x, map_beta_y

    def draw_planning_gt(self, data):
        if not self.plot_choices["planning"]:
            return

        # draw planning gt
        masks = data["gt_ego_fut_masks"].astype(bool)
        if masks[0] != 0:
            plan_traj = data["gt_ego_fut_trajs"][masks]  # 表示未来真实的轨迹
            cmd = data["gt_ego_fut_cmd"]  # 表示未来的控制命令
            plan_traj[abs(plan_traj) < 0.01] = 0.0
            plan_traj = plan_traj.cumsum(axis=0)
            plan_traj = np.concatenate(
                (np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0
            )
            self._render_traj(
                plan_traj, traj_score=1.0, colormap="Purples", dot_size=30
            )
            # self._render_traj_gt(
            #     plan_traj, color="purple", dot_size=5
            # )

    def draw_planning_pred(self, data, result, top_k=3, colormap="autumn"):
        if not (
            self.plot_choices["draw_pred"]
            and self.plot_choices["planning"]
            and "planning" in result
        ):
            return

        if self.plot_choices["track"] and "ego_anchor_queue" in result:
            ego_temp_bboxes = result["ego_anchor_queue"]
            ego_period = result["ego_period"]
            for j in range(ego_period[0]):
                # draw corners
                corners = box3d_to_corners(ego_temp_bboxes[:, -1 - j])[
                    0, [0, 3, 7, 4, 0]
                ]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color="mediumseagreen", linewidth=2, linestyle="-")
                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color="mediumseagreen", linewidth=2, linestyle="-")
        # import ipdb; ipdb.set_trace()
        plan_trajs = result["planning"].cpu().numpy()
        # print('plan trajs:', plan_trajs.shape)  (3, 6, 6, 2)
        num_cmd = len(CMD_LIST)
        num_mode = plan_trajs.shape[1]
        plan_trajs = np.concatenate(
            (np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2
        )
        plan_score = result["planning_score"].cpu().numpy()
        # print('plan trajs:', plan_trajs.shape)  (3, 6, 7, 2)

        cmd = data["gt_ego_fut_cmd"].argmax()
        plan_trajs = plan_trajs[cmd]
        plan_score = plan_score[cmd]
        # print('plan trajs:', plan_trajs.shape)   (6, 7, 2)

        sorted_ind = np.argsort(plan_score)[::-1]  # 根据plan score来获取预测轨迹的索引
        sorted_traj = plan_trajs[
            sorted_ind, :, :2
        ]  # 根据plan score来获取按分数排序的预测轨迹
        # print('sorted traj:', sorted_traj.shape)  (6, 7, 2)
        sorted_score = plan_score[sorted_ind]
        norm_score = np.exp(sorted_score[0])

        for j in range(top_k - 1, -1, -1):
            viz_traj = sorted_traj[j]
            traj_score = np.exp(sorted_score[j]) / norm_score
            self._render_traj(
                viz_traj, traj_score=traj_score, colormap=colormap, dot_size=30
            )

    def _render_traj(
        self,
        future_traj,
        traj_score=1,
        colormap="winter",
        points_per_step=20,
        dot_size=25,
    ):
        total_steps = (len(future_traj) - 1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](np.linspace(0, 1, total_steps))[
            :, :3
        ]
        dot_colors = dot_colors * traj_score + (1 - traj_score) * np.ones_like(
            dot_colors
        )
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps - 1):
            unit_vec = (
                future_traj[i // points_per_step + 1]
                - future_traj[i // points_per_step]
            )
            total_xy[i] = (
                i / points_per_step - i // points_per_step
            ) * unit_vec + future_traj[i // points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size)

    def _render_sdc_car(self):
        sdc_car_png = cv2.imread("resources/sdc_car.png")
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        im = self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))
        im.set_zorder(2)

    def _render_legend(self):
        legend = cv2.imread("tools/visualization/ours.png")
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        self.axes.imshow(legend, extent=(15, 40, -40, -30))

    def _render_command(self, data):
        cmd = data["gt_ego_fut_cmd"].argmax()
        self.axes.text(20, -26, CMD_LIST[cmd], fontsize=50)


class BEVRenderCustom_Origin:
    def __init__(
        self,
        plot_choices,
        out_dir,
        xlim=40,
        ylim=40,
    ):
        self.plot_choices = plot_choices
        self.xlim = xlim
        self.ylim = ylim
        self.gt_dir = os.path.join(out_dir, "sparsedrive_bev_gt")
        self.pred_dir = os.path.join(out_dir, "sparsedrive_bev_pred")
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)

    def reset_canvas(self, render_gt=False):
        plt.close()
        if render_gt:
            figsize = (20, 20)
            self.xlim = 40
        else:
            figsize = (15, 20)
            self.xlim = 30
        self.fig, self.axes = plt.subplots(1, 1, figsize=figsize)
        self.axes.set_xlim(-self.xlim, self.xlim)
        self.axes.set_ylim(-self.ylim, self.ylim)
        self.axes.axis("off")

    def render(self, data, result, index, error, colormap):
        # self.reset_canvas(render_gt=True)
        # self.draw_detection_gt(data)
        # self.draw_motion_gt(data)
        # self.draw_map_gt(data)
        # self.draw_planning_gt(data)
        # self._render_sdc_car()
        # self._render_command(data)
        # self._render_legend()
        # save_path_gt = os.path.join(self.gt_dir, str(index).zfill(4) + ".jpg")
        # self.save_fig(save_path_gt)

        self.reset_canvas(render_gt=True)
        self.draw_detection_pred(result)
        self.draw_track_pred(result)
        self.draw_motion_pred(result, top_k=self.plot_choices["motion_top_k"])
        self.draw_map_pred(result)
        self.draw_planning_gt(data)
        self.draw_planning_pred(
            data, result, top_k=self.plot_choices["planning_top_k"], colormap=colormap
        )
        # self.draw_planning_gt(data)
        self._render_sdc_car()
        if self.plot_choices["with_pred_cmd_legend"]:
            self._render_command(data)
        self._render_legend()
        if self.plot_choices["with_frame_index"]:
            self.axes.text(-38, -25, "index: {}".format(index), fontsize=40)
        if self.plot_choices["with_metrics"]:
            self.axes.text(-38, -30, "L2: {}".format(error), fontsize=40)
        save_path_pred = os.path.join(self.pred_dir, str(index).zfill(4) + ".jpg")
        self.save_fig(save_path_pred)

        return save_path_pred

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        if self.plot_choices["dpi"] > 0:
            plt.savefig(filename, dpi=self.plot_choices["dpi"], bbox_inches="tight")
        else:
            plt.savefig(filename)

    def draw_detection_gt(self, data):
        if not self.plot_choices["det"]:
            return

        for i in range(data["gt_labels_3d"].shape[0]):
            label = data["gt_labels_3d"][i]
            if label == -1:
                continue
            color = color_mapping[i % len(color_mapping)]

            # draw corners
            corners = box3d_to_corners(data["gt_bboxes_3d"])[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle="-")

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle="-")

    def draw_detection_pred(self, result):
        if not (
            self.plot_choices["draw_pred"]
            and self.plot_choices["det"]
            and "boxes_3d" in result
        ):
            return

        bboxes = result["boxes_3d"]
        for i in range(result["labels_3d"].shape[0]):
            score = result["scores_3d"][i]
            if score < SCORE_THRESH:
                continue
            color = color_mapping[result["instance_ids"][i] % len(color_mapping)]

            # draw corners
            corners = box3d_to_corners(bboxes)[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle="-")

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle="-")

    def draw_track_pred(self, result):
        if not (
            self.plot_choices["draw_pred"]
            and self.plot_choices["track"]
            and "anchor_queue" in result
        ):
            return

        temp_bboxes = result["anchor_queue"]
        period = result["period"]
        bboxes = result["boxes_3d"]
        for i in range(result["labels_3d"].shape[0]):
            score = result["scores_3d"][i]
            if score < SCORE_THRESH:
                continue
            color = color_mapping[result["instance_ids"][i] % len(color_mapping)]
            center = bboxes[i, :3]
            centers = [center]
            for j in range(period[i]):
                # draw corners
                corners = box3d_to_corners(temp_bboxes[:, -1 - j])[i, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle="-")

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle="-")
                centers.append(center)

            centers = np.stack(centers)
            xs = centers[:, 0]
            ys = centers[:, 1]
            self.axes.plot(xs, ys, color=color, linewidth=2, linestyle="-")

    def draw_motion_gt(self, data):
        if not self.plot_choices["motion"]:
            return

        for i in range(data["gt_labels_3d"].shape[0]):
            label = data["gt_labels_3d"][i]
            if label == -1:
                continue
            color = color_mapping[i % len(color_mapping)]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            center = data["gt_bboxes_3d"][i, :2]
            masks = data["gt_agent_fut_masks"][i].astype(bool)
            if masks[0] == 0:
                continue
            trajs = data["gt_agent_fut_trajs"][i][masks]
            trajs = trajs.cumsum(axis=0) + center
            trajs = np.concatenate([center.reshape(1, 2), trajs], axis=0)

            self._render_traj(
                trajs, traj_score=1.0, colormap="winter", dot_size=dot_size
            )

    def draw_motion_pred(self, result, top_k=3):
        if not (
            self.plot_choices["draw_pred"]
            and self.plot_choices["motion"]
            and "trajs_3d" in result
        ):
            return

        bboxes = result["boxes_3d"]
        labels = result["labels_3d"]
        for i in range(result["labels_3d"].shape[0]):
            score = result["scores_3d"][i]
            if score < SCORE_THRESH:
                continue
            label = labels[i]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            traj_score = result["trajs_score"][i].numpy()
            traj = result["trajs_3d"][i].numpy()
            num_modes = len(traj_score)
            center = bboxes[i, :2][None, None].repeat(num_modes, 1, 1).numpy()
            traj = np.concatenate([center, traj], axis=1)

            sorted_ind = np.argsort(traj_score)[::-1]
            sorted_traj = traj[sorted_ind, :, :2]
            sorted_score = traj_score[sorted_ind]
            norm_score = np.exp(sorted_score[0])

            for j in range(top_k - 1, -1, -1):
                viz_traj = sorted_traj[j]
                traj_score = np.exp(sorted_score[j]) / norm_score
                self._render_traj(
                    viz_traj,
                    traj_score=traj_score,
                    colormap="winter",
                    dot_size=dot_size,
                )

    def draw_map_gt(self, data):
        if not self.plot_choices["map"]:
            return
        vectors = data["map_infos"]
        for label, vector_list in vectors.items():
            color = COLOR_VECTORS[label]
            for vector in vector_list:
                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                self.axes.plot(
                    x,
                    y,
                    color=color,
                    linewidth=3,
                    marker="o",
                    linestyle="-",
                    markersize=7,
                )

    def draw_map_pred(self, result):
        if not (
            self.plot_choices["draw_pred"]
            and self.plot_choices["map"]
            and "vectors" in result
        ):
            return

        for i in range(result["scores"].shape[0]):
            score = result["scores"][i]
            if score < MAP_SCORE_THRESH:
                continue
            color = COLOR_VECTORS[result["labels"][i]]
            pts = result["vectors"][i]
            x = pts[:, 0]
            y = pts[:, 1]
            plt.plot(
                x, y, color=color, linewidth=3, marker="o", linestyle="-", markersize=7
            )

    def draw_planning_gt(self, data):
        if not self.plot_choices["planning"]:
            return

        # draw planning gt
        masks = data["gt_ego_fut_masks"].astype(bool)
        if masks[0] != 0:
            plan_traj = data["gt_ego_fut_trajs"][masks]
            cmd = data["gt_ego_fut_cmd"]
            plan_traj[abs(plan_traj) < 0.01] = 0.0
            plan_traj = plan_traj.cumsum(axis=0)
            plan_traj = np.concatenate(
                (np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0
            )
            self._render_traj(
                plan_traj, traj_score=1.0, colormap="Purples", dot_size=30
            )

    def draw_planning_pred(self, data, result, top_k=3, colormap="winter"):
        if not (
            self.plot_choices["draw_pred"]
            and self.plot_choices["planning"]
            and "planning" in result
        ):
            return

        if self.plot_choices["track"] and "ego_anchor_queue" in result:
            ego_temp_bboxes = result["ego_anchor_queue"]
            ego_period = result["ego_period"]
            for j in range(ego_period[0]):
                # draw corners
                corners = box3d_to_corners(ego_temp_bboxes[:, -1 - j])[
                    0, [0, 3, 7, 4, 0]
                ]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color="mediumseagreen", linewidth=2, linestyle="-")

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color="mediumseagreen", linewidth=2, linestyle="-")
        # import ipdb; ipdb.set_trace()
        plan_trajs = result["planning"].cpu().numpy()
        num_cmd = len(CMD_LIST)
        num_mode = plan_trajs.shape[1]
        plan_trajs = np.concatenate(
            (np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2
        )
        plan_score = result["planning_score"].cpu().numpy()

        cmd = data["gt_ego_fut_cmd"].argmax()
        plan_trajs = plan_trajs[cmd]
        plan_score = plan_score[cmd]

        sorted_ind = np.argsort(plan_score)[::-1]
        sorted_traj = plan_trajs[sorted_ind, :, :2]
        sorted_score = plan_score[sorted_ind]
        norm_score = np.exp(sorted_score[0])

        for j in range(top_k - 1, -1, -1):
            viz_traj = sorted_traj[j]
            traj_score = np.exp(sorted_score[j]) / norm_score
            self._render_traj(
                viz_traj, traj_score=traj_score, colormap=colormap, dot_size=30
            )

    def _render_traj(
        self,
        future_traj,
        traj_score=1,
        colormap="winter",
        points_per_step=20,
        dot_size=25,
    ):
        total_steps = (len(future_traj) - 1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](np.linspace(0, 1, total_steps))[
            :, :3
        ]
        dot_colors = dot_colors * traj_score + (1 - traj_score) * np.ones_like(
            dot_colors
        )
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps - 1):
            unit_vec = (
                future_traj[i // points_per_step + 1]
                - future_traj[i // points_per_step]
            )
            total_xy[i] = (
                i / points_per_step - i // points_per_step
            ) * unit_vec + future_traj[i // points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size)

    def _render_sdc_car(self):
        sdc_car_png = cv2.imread("resources/sdc_car.png")
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        im = self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))
        im.set_zorder(2)

    def _render_legend(self):
        legend = cv2.imread("tools/visualization/spd.png")
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        self.axes.imshow(legend, extent=(15, 40, -40, -30))

    def _render_command(self, data):
        cmd = data["gt_ego_fut_cmd"].argmax()
        self.axes.text(20, -26, CMD_LIST[cmd], fontsize=50)
        # self.axes.text(-38, -38, CMD_LIST[cmd], fontsize=60)
