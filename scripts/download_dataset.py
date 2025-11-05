#!/usr/bin/env python3
"""
Script to manually download the HuggingFace dataset for offline use.

This downloads the fineweb-edu dataset and saves it locally so the training
script can use it without needing to connect to huggingface.co during training.
"""

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm


def download_dataset(save_dir: str, num_samples: int = None, checkpoint_interval: int = 1000000):
    """
    Download the fineweb-edu dataset and save it locally with resume support.

    Args:
        save_dir: Directory where the dataset will be saved
        num_samples: Optional number of samples to download (None = all)
        checkpoint_interval: Save checkpoint every N samples (default: 1000)
    """

    print(f"Downloading fineweb-edu dataset to: {save_dir}")
    print("=" * 80)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_file = os.path.join(save_dir, ".download_checkpoint.txt")
    temp_dir = os.path.join(save_dir, ".temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    # Check for existing checkpoint
    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_idx = int(f.read().strip())
        print(f"✓ Found checkpoint: resuming from sample {start_idx:,}")

    # Load the dataset with streaming to avoid loading everything into memory at once
    print("\nConnecting to HuggingFace Hub and loading dataset...")
    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True
        )
        print("✓ Successfully connected to dataset")
    except Exception as e:
        print(f"✗ Failed to connect to HuggingFace Hub: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connectivity")
        print("  2. Verify you can access huggingface.co")
        print("  3. Try increasing timeout: export HF_DATASETS_DOWNLOAD_TIMEOUT=300")
        return False

    # Convert streaming dataset to regular dataset and save with checkpointing
    print("\nDownloading and saving dataset with checkpoints...")
    print(f"(Saving checkpoints every {checkpoint_interval:,} samples)")

    import json

    from datasets import Dataset

    _ = num_samples if num_samples else float("inf")
    dataset_list = []

    try:
        for i, example in enumerate(
            tqdm(dataset, total=num_samples, desc="Downloading", initial=start_idx)
        ):
            # Skip already downloaded samples
            if i < start_idx:
                continue

            dataset_list.append(example)

            # Save checkpoint periodically
            if len(dataset_list) > 0 and len(dataset_list) % checkpoint_interval == 0:
                chunk_file = os.path.join(temp_dir, f"chunk_{i}.json")
                with open(chunk_file, "w") as f:
                    json.dump(dataset_list, f)
                dataset_list = []  # Clear memory

                # Update checkpoint
                with open(checkpoint_file, "w") as f:
                    f.write(str(i + 1))

            # Check if we've reached the limit
            if num_samples and i + 1 >= num_samples:
                break

        # Save any remaining samples
        if dataset_list:
            chunk_file = os.path.join(temp_dir, "chunk_final.json")
            with open(chunk_file, "w") as f:
                json.dump(dataset_list, f)

        # Now merge all chunks
        print("\nMerging downloaded chunks...")
        all_data = []
        chunk_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".json")])

        for chunk_file in tqdm(chunk_files, desc="Merging chunks"):
            with open(os.path.join(temp_dir, chunk_file), "r") as f:
                chunk_data = json.load(f)
                all_data.extend(chunk_data)

        # Convert to dataset
        if not all_data:
            print("✗ No data downloaded")
            return False

        final_dataset = Dataset.from_dict(
            {key: [example[key] for example in all_data] for key in all_data[0].keys()}
        )

        # Save the final dataset
        print(f"\nSaving final dataset to {save_dir}...")
        final_dataset.save_to_disk(save_dir)

        # Cleanup temporary files
        import shutil

        shutil.rmtree(temp_dir)
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        print("\n" + "=" * 80)
        print("✓ Dataset successfully downloaded and saved!")
        print(f"  Location: {save_dir}")
        print(f"  Number of examples: {len(final_dataset):,}")

        # Calculate disk space
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(save_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)

        print(f"  Disk space used: ~{total_size / (1024**3):.2f} GB")
        print("\nYou can now use this dataset offline in your training script.")

        return True

    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        print(f"Progress saved at sample {start_idx + len(dataset_list):,}")
        print("Run the script again to resume from this point.")
        return False
    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        print("Progress saved. Run again to resume.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace dataset for offline use")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/fineweb-edu-sample-100BT",
        help="Directory where the dataset will be saved (default: ./data/fineweb-edu-sample-100BT)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to download (default: None = download all samples, or specify a number like 100000 for partial download)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Download timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000000,
        help="Save checkpoint every N samples for resume support (default: 1000)",
    )

    args = parser.parse_args()

    # Set timeout
    import datasets

    datasets.config.HF_DATASETS_DOWNLOAD_TIMEOUT = args.timeout
    print(f"HuggingFace download timeout set to: {args.timeout} seconds")
    print(f"Checkpoint interval: {args.checkpoint_interval:,} samples")

    # Show what we're about to do
    if args.num_samples is None:
        print(
            "\nWARNING: No --num_samples specified, will download FULL dataset (~95GB, 9.7M samples)"
        )
        print("This may take several hours. Press Ctrl+C within 5 seconds to cancel...")
        import time

        time.sleep(5)
        print("Proceeding with full download...\n")
    else:
        print(f"\nWill download {args.num_samples:,} samples\n")

    # Download dataset
    success = download_dataset(args.save_dir, args.num_samples, args.checkpoint_interval)

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
