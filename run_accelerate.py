#!/usr/bin/env python
import os
import subprocess
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Launch StreamingRNNT training with Accelerate")

    # General options
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers per GPU")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["no", "fp16", "bf16", "fp16-mixed", "bf16-mixed"],
                        help="Mixed precision mode")

    # Optional overrides
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--train_manifest", type=str, help="Training manifest path")
    parser.add_argument("--val_manifest", type=str, help="Validation manifest path")
    parser.add_argument("--no_augment", action="store_true", help="Disable augmentation")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Build command
    cmd = ["accelerate", "launch", "train_accelerate.py"]
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--max_epochs", str(args.max_epochs)])
    cmd.extend(["--num_workers", str(args.num_workers)])
    cmd.extend(["--output_dir", args.output_dir])
    cmd.extend(["--precision", args.precision])
    cmd.extend(["--lr", str(args.lr)])

    if args.resume_from_checkpoint:
        cmd.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])

    if args.train_manifest:
        cmd.extend(["--train_manifest", args.train_manifest])

    if args.val_manifest:
        cmd.extend(["--val_manifest", args.val_manifest])

    if args.no_augment:
        cmd.append("--no_augment")

    # Log the command
    print("Running command:", " ".join(cmd))

    # Execute the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main()