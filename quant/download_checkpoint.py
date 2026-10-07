#!/usr/bin/env python3
# download_checkpoint.py - 通过 hf-mirror 下载 checkpoint（huggingface.co 不稳定时用）
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
import time
from huggingface_hub import snapshot_download

repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B-FP8"
local = sys.argv[2] if len(sys.argv) > 2 else "/home/hngs/models/Qwen3-30B-A3B-FP8"

t0 = time.time()
path = snapshot_download(
    repo,
    local_dir=local,
    max_workers=4,
)
print(f"DONE: {path} in {time.time()-t0:.0f}s", flush=True)
