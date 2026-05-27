import torch

def count_params_in_pth(pth_path):
    state_dict = torch.load(pth_path, map_location="cpu")

    # 如果是直接保存的 state_dict (dict)，遍历所有参数
    if isinstance(state_dict, dict) and all(isinstance(v, torch.Tensor) for v in state_dict.values()):
        total_params = sum(v.numel() for v in state_dict.values())
    # 如果是保存的包含 'state_dict' 的字典 (常见于Lightning/一些框架)
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        total_params = sum(v.numel() for v in state_dict["state_dict"].values())
    else:
        raise ValueError("不识别的 pth 文件结构，请检查 torch.save 时的方式")

    return total_params / 1e6  # 单位 M

# ==== 示例 ====
pth_file = "/home/jwg2sgh/Code/SparseDrive/work_dirs/sparsedrive_stage2.pth"
# pth_file = "/home/jwg2sgh/Code/UniUncer/work_dirs/sparsedrive_small_stage2_uncer_map_uncer_det_clip/latest.pth"
params_M = count_params_in_pth(pth_file)
print(f"参数量: {params_M:.2f} M")