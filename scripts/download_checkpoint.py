import argparse
import os
import shutil
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from project root .env
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Falling back to system environment variables only.")


# python scripts/download_checkpoint.py cartesia-ai/hnet_2stage_XL hnet_2stage_XL.pt
def main() -> None:
    parser = argparse.ArgumentParser(description="Download a specific file from a Hugging Face repo")
    parser.add_argument("repo_id", type=str, help="HF repo id, e.g. cartesia-ai/hnet_2stage_XL")
    parser.add_argument("filename", type=str, help="Filename inside the repo, e.g. pytorch_model.pt")
    parser.add_argument(
        "--outdir", 
        type=str, 
        help="Destination directory"
    )
    parser.add_argument(
        "--cache-dir", 
        type=str,  
        help="HF cache directory"
    )
    args = parser.parse_args()
    
    # Set defaults from environment variables if not provided via CLI
    if args.outdir is None:
        args.outdir = os.getenv("HF_CHECKPOINT_DIR")
        if args.outdir is None:
            print("Error: --outdir not provided and HF_CHECKPOINT_DIR not set in .env")
            sys.exit(1)
    
    if args.cache_dir is None:
        args.cache_dir = os.getenv("HF_CACHE_DIR")
        if args.cache_dir is None:
            print("Error: --cache-dir not provided and HF_CACHE_DIR not set in .env")
            sys.exit(1)

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
