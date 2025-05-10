import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# from warp_rnnt import rnnt_loss
import warprnnt_numba

import sentencepiece as spm
from jiwer import wer

from loguru import logger

from models.encoder import AudioEncoder
from models.decoder import Decoder
from models.jointer import Jointer

from constants import RNNT_BLANK, PAD, VOCAB_SIZE, TOKENIZER_MODEL_PATH, MAX_SYMBOLS
from constants import ATTENTION_CONTEXT_SIZE
from constants import N_STATE, N_LAYER, N_HEAD, N_MELS
from constants import BATCH_SIZE, NUM_WORKERS
from constants import MAX_EPOCHS, TOTAL_STEPS, WARMUP_STEPS, LR, MIN_LR
from constants import PRETRAINED_ENCODER_WEIGHT, TRAIN_MANIFEST, VAL_MANIFEST, LOG_DIR, BG_NOISE_PATH

from utils.dataset import AudioDataset, collate_fn
from utils.scheduler import WarmupLR

class StreamingRNNT(pl.LightningModule):
    def __init__(self, att_context_size, vocab_size, tokenizer_model_path):
        super().__init__()

        # Lưu hyperparameters để có thể lưu và restore đúng
        self.save_hyperparameters()

        # Xác định device phù hợp
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained encoder weights
        encoder_state_dict = torch.load(
            PRETRAINED_ENCODER_WEIGHT,
            map_location=device,
            weights_only=True
        )
        # Create new keys 'conv3.weight', 'conv3.bias' that copy from 'conv2.weight', 'conv2.bias' so that we don't have to initialize conv3 weights
        encoder_state_dict['model_state_dict']['conv3.weight'] = encoder_state_dict['model_state_dict']['conv2.weight']
        encoder_state_dict['model_state_dict']['conv3.bias'] = encoder_state_dict['model_state_dict']['conv2.bias']

        self.encoder = AudioEncoder(
            n_mels=N_MELS,
            n_state=N_STATE,
            n_head=N_HEAD,
            n_layer=N_LAYER,
            att_context_size=att_context_size
        )
        self.encoder.load_state_dict(encoder_state_dict['model_state_dict'], strict=False)

        self.decoder = Decoder(vocab_size=vocab_size + 1)
        self.joint = Jointer(vocab_size=vocab_size + 1)

        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_path)

        # self.loss = torchaudio.transforms.RNNTLoss(reduction="mean") # RNNTLoss has bug with logits number of elements > 2**31
        self.loss = warprnnt_numba.RNNTLossNumba(
            blank=RNNT_BLANK, reduction="mean",
        )

        # Optimizer sẽ được cấu hình trong configure_optimizers()
        self.learning_rate = LR

        # Lưu trữ metrics để tính trung bình
        self.validation_step_outputs = []

        # Enable auto optimization
        self.automatic_optimization = True


    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L416
    def greedy_decoding(self, x, x_len, max_symbols=None):
        enc_out, _ = self.encoder(x, x_len)
        all_sentences = []
        # greedy decoding, handle each sequence independently for easier implementation
        for batch_idx in range(enc_out.shape[0]):
            hypothesis = [[None, None]]  # [label, state]
            seq_enc_out = enc_out[batch_idx, :, :].unsqueeze(0) # [1, T, D]
            seq_ids = []

            for time_idx in range(seq_enc_out.shape[1]):
                curent_seq_enc_out = seq_enc_out[:, time_idx, :].unsqueeze(1) # 1, 1, D

                not_blank = True
                symbols_added = 0

                while not_blank and (max_symbols is None or symbols_added < max_symbols):
                    # In the first timestep, we initialize the network with RNNT Blank
                    # In later timesteps, we provide previous predicted label as input.
                    if hypothesis[-1][0] is None:
                        last_token = torch.tensor([[RNNT_BLANK]], dtype=torch.long, device=seq_enc_out.device)
                        last_seq_h_n = None
                    else:
                        last_token = hypothesis[-1][0]
                        last_seq_h_n = hypothesis[-1][1]

                    if last_seq_h_n is None:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token)
                    else:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token, last_seq_h_n)
                    logits = self.joint(curent_seq_enc_out, current_seq_dec_out)[0, 0, 0, :]  # (B, T=1, U=1, V + 1)

                    del current_seq_dec_out

                    _, token_id = logits.max(0)
                    token_id = token_id.detach().item()  # K is the label at timestep t_s in inner loop, s >= 0.

                    del logits

                    if token_id == RNNT_BLANK:
                        not_blank = False
                    else:
                        symbols_added += 1
                        hypothesis.append([
                            torch.tensor([[token_id]], dtype=torch.long, device=curent_seq_enc_out.device),
                            current_seq_h_n
                        ])
                        seq_ids.append(token_id)
            all_sentences.append(self.tokenizer.decode(seq_ids))
        return all_sentences

    def process_batch(self, batch):
        return batch

    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len = self.process_batch(batch)

        global_rank = self.global_rank if hasattr(self, 'global_rank') else 0
        global_step = self.global_step if hasattr(self, 'global_step') else 0

        # Chỉ đánh giá WER trên rank 0 để tiết kiệm tài nguyên
        if global_rank == 0 and batch_idx != 0 and batch_idx % 2000 == 0:
            # Chuyển sang eval mode để tránh dropout, etc.
            self.encoder.eval()
            self.decoder.eval()
            self.joint.eval()

            with torch.no_grad():
                all_pred = self.greedy_decoding(x, x_len, max_symbols=3)
                all_true = []
                for i, y_i in enumerate(y):
                    y_i = y_i.cpu().numpy().astype(int).tolist()
                    y_i = y_i[:y_len[i]]
                    all_true.append(self.tokenizer.decode_ids(y_i))

                for pred, true in zip(all_pred, all_true):
                    logger.debug(f"Pred: {pred}")
                    logger.debug(f"True: {true}")

                all_wer = wer(all_true, all_pred)
                self.log("train_wer", all_wer, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)

            # Trở lại train mode
            self.encoder.train()
            self.decoder.train()
            self.joint.train()

        # Forward pass để tính loss
        enc_out, x_len = self.encoder(x, x_len) # (B, T, Enc_dim)

        # Add a blank token to the beginning of the target sequence
        # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L185
        # Blank is also the start of sequence token and the sentence will start with blank; https://github.com/pytorch/audio/issues/3750
        y_start = torch.cat([torch.full((y.shape[0], 1), RNNT_BLANK, dtype=torch.int).to(y.device), y], dim=1).to(y.device)
        dec_out, _ = self.decoder(y_start) # (B, U, Dec_dim)
        logits = self.joint(enc_out, dec_out)

        input_lengths = x_len.int()
        target_lengths = y_len.int()
        targets = y.int()

        loss = self.loss(logits.to(torch.float32), targets, input_lengths, target_lengths)

        # Log mỗi 100 steps và khi trên rank 0
        if batch_idx % 100 == 0:
            # Log training loss với format chỉ hai chữ số cuối cùng
            self.log("train_loss", loss.detach().item(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            # Log current learning rate
            self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, x_len, y, y_len = self.process_batch(batch)

        # Chỉ tính WER trên một phần nhỏ của validation set để tiết kiệm thời gian
        if batch_idx < 20:  # Giới hạn số lượng batches để đánh giá WER
            all_pred = self.greedy_decoding(x, x_len, max_symbols=3)
            all_true = []
            for i, y_i in enumerate(y):
                y_i = y_i.cpu().numpy().astype(int).tolist()
                y_i = y_i[:y_len[i]]
                all_true.append(self.tokenizer.decode_ids(y_i))

            all_wer = wer(all_true, all_pred)

            # In một số mẫu để debugging
            if batch_idx % 10 == 0 and self.global_rank == 0:
                for pred, true in zip(all_pred[:2], all_true[:2]):  # Chỉ in 2 mẫu đầu tiên
                    logger.debug(f"Pred: {pred}")
                    logger.debug(f"True: {true}")

            # Thêm WER vào outputs để tính trung bình
            self.validation_step_outputs.append({'wer': all_wer, 'count': len(all_true)})

        # ------------------CALCULATE LOSS------------------
        enc_out, x_len = self.encoder(x, x_len) # (B, T, Enc_dim)

        # Add a blank token to the beginning of the target sequence
        y_start = torch.cat([torch.full((y.shape[0], 1), RNNT_BLANK, dtype=torch.int).to(y.device), y], dim=1).to(y.device)
        dec_out, _ = self.decoder(y_start) # (B, U, Dec_dim)
        logits = self.joint(enc_out, dec_out)

        input_lengths = x_len.int()
        target_lengths = y_len.int()
        targets = y.int()

        loss = self.loss(logits.to(torch.float32), targets, input_lengths, target_lengths)
        # ---------------------------------------------------

        # Log loss sau mỗi batch
        self.log("val_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        # Tính toán WER trung bình
        if len(self.validation_step_outputs) > 0:
            total_wer = 0
            total_count = 0

            for output in self.validation_step_outputs:
                total_wer += output['wer'] * output['count']
                total_count += output['count']

            avg_wer = total_wer / total_count if total_count > 0 else 0

            # Log WER trung bình
            self.log("val_wer", avg_wer, prog_bar=True, sync_dist=True)

        # Reset validation outputs
        self.validation_step_outputs = []

    def on_train_epoch_end(self):
        # Lưu checkpoint sau mỗi epoch
        if self.trainer.is_global_zero:  # Chỉ lưu từ rank 0
            self.trainer.save_checkpoint(f"{LOG_DIR}/rnnt-latest.ckpt", weights_only=True)
        return super().on_train_epoch_end()

    def configure_optimizers(self):
        # Khởi tạo optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=1e-2  # Thêm weight decay để giảm overfitting
        )

        # Khởi tạo scheduler
        scheduler = WarmupLR(self.optimizer, WARMUP_STEPS, TOTAL_STEPS, MIN_LR)

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }