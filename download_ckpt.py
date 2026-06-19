import argparse
from huggingface_hub import snapshot_download

MODELS = {
    "3B": ("ckpt/VINCIE-3B", "ByteDance-Seed/VINCIE-3B"),
    "7B": ("ckpt/VINCIE-7B", "ByteDance-Seed/VINCIE-7B"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS.keys()), default="3B")
args = parser.parse_args()

save_dir, repo_id = MODELS[args.model]
cache_dir = save_dir + "/cache"

snapshot_download(
    cache_dir=cache_dir,
    local_dir=save_dir,
    repo_id=repo_id,
    local_dir_use_symlinks=False,
    resume_download=True,
)
