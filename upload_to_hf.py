#!/usr/bin/env python
"""Upload UniUncer checkpoints to Hugging Face Hub."""

from huggingface_hub import HfApi, create_repo, upload_file

# ==================== 修改这里 ====================
REPO_ID = "EricGaoSH/UniUncer"
# =================================================

CKPT_FILES = [
    ("ckpt/uncer_stage1_iter_11720_1e-4.pth", "uncer_stage1_iter_11720_1e-4.pth"),
    ("ckpt/uniuncer_final_iter_11720.pth", "uniuncer_final_iter_11720.pth"),
]

api = HfApi()

# 创建仓库（如果已存在则跳过）
print(f"Creating repo: {REPO_ID} ...")
create_repo(REPO_ID, repo_type="model", exist_ok=True)

# 上传文件
for local_path, repo_filename in CKPT_FILES:
    print(f"Uploading {local_path} -> {repo_filename} ...")
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_filename,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"  Done: https://huggingface.co/{REPO_ID}/blob/main/{repo_filename}")

print("\nAll checkpoints uploaded!")
print(f"Model page: https://huggingface.co/{REPO_ID}")
