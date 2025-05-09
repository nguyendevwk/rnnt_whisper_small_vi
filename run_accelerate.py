#!/usr/bin/env python
import os
import subprocess
import argparse
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Launch StreamingRNNT training with Accelerate")

    # Main options
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16", "fp16-mixed", "bf16-mixed"],
                        help="Mixed precision mode")

    # Model options
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")

    # Data options - CHÚ Ý: Bắt buộc phải có cả TRAIN_MANIFEST và VAL_MANIFEST
    parser.add_argument("--train_manifest", type=str, default="/kaggle/working/data/train/train_data.jsonl",
                        help="Training manifest path - QUAN TRỌNG: Phải chính xác")
    parser.add_argument("--val_manifest", type=str, default="/kaggle/working/data/test/test_data.jsonl",
                        help="Validation manifest path - QUAN TRỌNG: Phải chính xác")
    parser.add_argument("--base_path", type=str, default="/kaggle/working/",
                        help="Base path for resolving audio file paths")
    parser.add_argument("--tokenizer_model_path", type=str, default="./weights/tokenizer_spe_bpe_v1024_pad/tokenizer.model",
                        help="Path to tokenizer model")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")

    # Launch options
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--debug_launcher", action="store_true", help="Run with debug info")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Kiểm tra cả hai đường dẫn manifest
    path_warnings = []

    # Kiểm tra train_manifest
    if not os.path.exists(args.train_manifest):
        path_warnings.append(f"LỖI NGHIÊM TRỌNG: Train manifest không tồn tại: {args.train_manifest}")
        path_warnings.append("Đảm bảo rằng đường dẫn tới file train_data.jsonl là chính xác")
        path_warnings.append("Mặc định: /kaggle/working/data/train/train_data.jsonl")

    # Kiểm tra val_manifest
    if not os.path.exists(args.val_manifest):
        path_warnings.append(f"LỖI NGHIÊM TRỌNG: Validation manifest không tồn tại: {args.val_manifest}")
        path_warnings.append("Đảm bảo rằng đường dẫn tới file test_data.jsonl là chính xác")
        path_warnings.append("Mặc định: /kaggle/working/data/test/test_data.jsonl")

    # Kiểm tra tokenizer
    if not os.path.exists(args.tokenizer_model_path):
        path_warnings.append(f"LỖI: Tokenizer model không tồn tại: {args.tokenizer_model_path}")

    # Hiển thị cảnh báo và yêu cầu xác nhận nếu có lỗi
    if path_warnings:
        print("\n" + "!" * 80)
        print("CẢNH BÁO ĐƯỜNG DẪN:")
        print("!" * 80)
        for warning in path_warnings:
            print(warning)

        print("\nKiểm tra hiện trạng thư mục:")
        print(f"Thư mục hiện tại: {os.getcwd()}")

        # Kiểm tra cấu trúc thư mục chứa manifest
        train_dir = os.path.dirname(args.train_manifest)
        val_dir = os.path.dirname(args.val_manifest)

        if os.path.exists(train_dir):
            print(f"\nNội dung thư mục {train_dir}:")
            files = os.listdir(train_dir)
            for f in files:
                if f.endswith('.jsonl'):
                    print(f" - {f} (jsonl)")
                else:
                    print(f" - {f}")

        if os.path.exists(val_dir) and val_dir != train_dir:
            print(f"\nNội dung thư mục {val_dir}:")
            files = os.listdir(val_dir)
            for f in files:
                if f.endswith('.jsonl'):
                    print(f" - {f} (jsonl)")
                else:
                    print(f" - {f}")

        print("\n" + "!" * 80)
        print("BẮT BUỘC PHẢI CÓ CẢ HAI FILE MANIFEST TRAIN VÀ VAL")
        print("!" * 80)

        print("\nTiếp tục? (y/n)")
        if input().lower() != 'y':
            print("Huỷ training")
            return

    # Build accelerate launch command
    cmd = ["accelerate", "launch"]

    # Add debug flag if requested
    if args.debug_launcher:
        cmd.append("--debug")

    # Force CPU if requested
    if args.cpu:
        cmd.extend(["--cpu"])

    # Add training script and arguments
    cmd.append("train_accelerate.py")
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--max_epochs", str(args.max_epochs)])
    cmd.extend(["--num_workers", str(args.num_workers)])
    cmd.extend(["--output_dir", args.output_dir])
    cmd.extend(["--precision", args.precision])
    cmd.extend(["--lr", str(args.lr)])
    cmd.extend(["--grad_clip", str(args.grad_clip)])
    cmd.extend(["--train_manifest", args.train_manifest])
    cmd.extend(["--val_manifest", args.val_manifest])  # Bổ sung val_manifest vào lệnh
    cmd.extend(["--base_path", args.base_path])
    cmd.extend(["--tokenizer_model_path", args.tokenizer_model_path])

    if args.resume_from_checkpoint:
        cmd.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])

    if args.augment:
        cmd.append("--augment")

    # Save command for reference
    command_path = os.path.join(args.output_dir, "last_command.txt")
    cmd_str = " ".join(cmd)
    with open(command_path, "w") as f:
        f.write(cmd_str)

    print("=" * 80)
    print("Khởi chạy training với Accelerate")
    print("=" * 80)
    print(f"Lệnh: {cmd_str}")
    print(f"Cấu hình đã lưu vào: {config_path}")
    print(f"Lệnh đã lưu vào: {command_path}")
    print("=" * 80)

    # Execute the command
    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"Lỗi khi thực thi lệnh: {str(e)}")
        error_log = os.path.join(args.output_dir, "error_log.txt")
        with open(error_log, "w") as f:
            f.write(f"Lỗi: {str(e)}\n")
            f.write(f"Lệnh: {cmd_str}\n")
        print(f"Chi tiết lỗi được lưu tại: {error_log}")

if __name__ == "__main__":
    main()