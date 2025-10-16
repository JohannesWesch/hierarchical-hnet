import argparse
import os
import shutil
import sys


# python scripts/download_checkpoint.py cartesia-ai/hnet_2stage_XL hnet_2stage_XL.pt
def main() -> None:
    parser = argparse.ArgumentParser(description="Download a specific file from a Hugging Face repo to /export/data2/jwesch/checkpoints")
    parser.add_argument("repo_id", type=str, help="HF repo id, e.g. cartesia-ai/hnet_2stage_XL")
    parser.add_argument("filename", type=str, help="Filename inside the repo, e.g. pytorch_model.pt")
    parser.add_argument("--outdir", type=str, default="/export/data2/jwesch/checkpoints", help="Destination directory")
    parser.add_argument("--cache-dir", type=str, default="/export/data2/jwesch", help="HF cache directory")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    dest_path = os.path.join(args.outdir, os.path.basename(args.filename))
    if os.path.exists(dest_path):
        print(f"Destination already exists: {dest_path}")
        return

    snapshot_dir = snapshot_download(repo_id=args.repo_id, cache_dir=args.cache_dir, allow_patterns=[args.filename])
    src = os.path.join(snapshot_dir, args.filename)
    if not os.path.exists(src):
        print(f"File not found in snapshot: {src}")
        sys.exit(1)

    resolved_src = os.path.realpath(src)
    if not os.path.exists(resolved_src):
        print(f"Resolved file not found: {resolved_src}")
        sys.exit(1)

    shutil.copyfile(resolved_src, dest_path)
    print(f"Saved: {dest_path}")


if __name__ == "__main__":
    main()
