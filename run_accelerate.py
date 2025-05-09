#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
from datetime import datetime

def parse_args():
    """Parse command line arguments for the run script"""
    parser = argparse.ArgumentParser(description="Run StreamingRNNT training with Accelerate")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per GPU")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Maximum number of steps (0 for no limit, useful for debugging)")

    # Model configuration
    parser.add_argument("--att_context_size", type=int, default=[-1, 6], nargs="+",
                        help="Attention context size (e.g. -1 6 for left=full, right=6)")

    # Data paths
    parser.add_argument("--train_manifest", type=str, required=True,
                        help="Path to training manifest file")
    parser.add_argument("--val_manifest", type=str, required=True,
                        help="Path to validation manifest file")
    parser.add_argument("--tokenizer_model_path", type=str, required=True,
                        help="Path to tokenizer model file")
    parser.add_argument("--bg_noise_path", type=str, default=None,
                        help="Path to background noise for augmentation")

    # Output and checkpointing
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for logs and checkpoints (default: runs/run_DATETIME)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name to append to output directory")

    # Accelerate configuration
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["no", "fp16", "bf16", "fp16-mixed", "bf16-mixed"],
                        help="Mixed precision mode")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--val_check_interval", type=int, default=2,
                        help="Run validation every N epochs")

    return parser.parse_args()

def main():
    """Main function to prepare and launch training"""
    args = parse_args()

    # Create a timestamped output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = args.experiment_name if args.experiment_name else "run"
        args.output_dir = os.path.join("runs", f"{exp_name}_{timestamp}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert attention context size from list to string if provided
    if isinstance(args.att_context_size, list):
        att_context_size_str = str(args.att_context_size).replace(" ", "")
    else:
        att_context_size_str = str(args.att_context_size)

    # Build the accelerate launch command
    cmd = [
        "accelerate", "launch",
        "train_accelerate.py",
        f"--batch_size={args.batch_size}",
        f"--max_epochs={args.max_epochs}",
        f"--lr={args.lr}",
        f"--train_manifest={args.train_manifest}",
        f"--val_manifest={args.val_manifest}",
        f"--tokenizer_model_path={args.tokenizer_model_path}",
        f"--output_dir={args.output_dir}",
        f"--precision={args.precision}",
        f"--num_workers={args.num_workers}",
        f"--val_check_interval={args.val_check_interval}",
    ]

    # Add conditional arguments
    if args.bg_noise_path:
        cmd.append(f"--bg_noise_path={args.bg_noise_path}")

    if args.resume_from_checkpoint:
        cmd.append(f"--resume_from_checkpoint={args.resume_from_checkpoint}")

    if args.no_augment:
        cmd.append("--no_augment")

    if args.max_steps > 0:
        cmd.append(f"--max_steps={args.max_steps}")

    # Print the command
    print("Running command:", " ".join(cmd))

    # Execute the command
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main())