from models.encoder import AudioEncoder
from models.decoder import Decoder
from models.jointer import Jointer
from torch.utils.data import DataLoader
from train_accelerate import StreamingRNNT
from utils.dataset import AudioDataset, collate_fn
from utils.scheduler import WarmupLR
from constants import *
import torch
import pytorch_lightning as pl
import os

# Xác định số lượng GPU có sẵn
num_gpus = torch.cuda.device_count()
# Đề xuất số workers dựa trên CPU có sẵn
import multiprocessing
suggested_workers = min(multiprocessing.cpu_count(), NUM_WORKERS)

# Điều chỉnh batch size nếu sử dụng nhiều GPU
effective_batch_size = BATCH_SIZE
per_device_batch_size = BATCH_SIZE
if num_gpus > 1:
    # Nếu sử dụng nhiều GPU, chia batch size cho số GPU
    per_device_batch_size = BATCH_SIZE // num_gpus
    print(f"Training with {num_gpus} GPUs, per-GPU batch size: {per_device_batch_size}")
    if per_device_batch_size < 1:
        per_device_batch_size = 1
        effective_batch_size = num_gpus
        print(f"Adjusted per-GPU batch size to 1, effective batch size: {effective_batch_size}")

# Hiển thị thông tin về cấu hình
print(f"Using {suggested_workers} CPU workers for data loading")
print(f"Total batch size: {effective_batch_size}")

# Khởi tạo datasets
train_dataset = AudioDataset(
    manifest_files=TRAIN_MANIFEST,
    bg_noise_path=BG_NOISE_PATH,
    shuffle=True,
    augment=True,
    tokenizer_model_path=TOKENIZER_MODEL_PATH
)

val_dataset = AudioDataset(
    manifest_files=VAL_MANIFEST,
    shuffle=False,
    tokenizer_model_path=TOKENIZER_MODEL_PATH
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=per_device_batch_size,
    shuffle=True,
    num_workers=suggested_workers,
    persistent_workers=True if suggested_workers > 0 else False,
    collate_fn=collate_fn,
    pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=per_device_batch_size,
    shuffle=False,
    num_workers=suggested_workers,
    persistent_workers=True if suggested_workers > 0 else False,
    collate_fn=collate_fn,
    pin_memory=True
)

model = StreamingRNNT(
    att_context_size=ATTENTION_CONTEXT_SIZE,
    vocab_size=VOCAB_SIZE,
    tokenizer_model_path=TOKENIZER_MODEL_PATH
)

# Callbacks
callbacks = [
    pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=LOG_DIR,
        filename='rnnt-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        mode='min'
    ),
    # LR monitor để theo dõi learning rate
    pl.callbacks.LearningRateMonitor(logging_interval='step'),
    # Early stopping để dừng training khi model không cải thiện
    pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )
]

# Xác định strategy dựa trên số GPU
if num_gpus > 1:
    strategy = "ddp"  # Distributed Data Parallel
else:
    strategy = "auto"

# Xác định precision dựa trên hardware
# Một số GPU không hỗ trợ bfloat16
precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "32-true"

# Khởi tạo trainer với cấu hình phù hợp
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    devices=num_gpus if num_gpus > 0 else None,  # None sẽ sử dụng tất cả GPU có sẵn
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    precision=precision,
    strategy=strategy,
    callbacks=callbacks,
    logger=pl.loggers.TensorBoardLogger(LOG_DIR),
    num_sanity_val_steps=2,  # Kiểm tra validation trước khi training
    check_val_every_n_epoch=1,  # Kiểm tra validation sau mỗi epoch
    gradient_clip_val=1.0,  # Thêm gradient clipping để tránh exploding gradients
    accumulate_grad_batches=max(1, BATCH_SIZE // (per_device_batch_size * max(1, num_gpus))),  # Gradient accumulation nếu cần
    log_every_n_steps=50,  # Log metrics sau mỗi 50 steps
)

# Cho phép resume training nếu có checkpoint
ckpt_path = None
if os.path.exists(f"{LOG_DIR}/rnnt-latest.ckpt"):
    ckpt_path = f"{LOG_DIR}/rnnt-latest.ckpt"
    print(f"Resuming from checkpoint: {ckpt_path}")

# Bắt đầu training
trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)