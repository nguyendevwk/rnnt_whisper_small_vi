import torch
import torch.nn as nn
import warprnnt_numba
import sentencepiece as spm
from loguru import logger

from models.encoder import AudioEncoder
from models.decoder import Decoder
from models.jointer import Jointer
from constants import RNNT_BLANK, PRETRAINED_ENCODER_WEIGHT
from constants import N_MELS, N_STATE, N_HEAD, N_LAYER

class StreamingRNNT(nn.Module):
    def __init__(self, att_context_size, vocab_size, tokenizer_model_path):
        super().__init__()

        encoder_state_dict = torch.load(PRETRAINED_ENCODER_WEIGHT, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=True)
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

    def forward(self, x, x_len, y, y_len):
        """
        Forward pass that computes loss. Replaces the Lightning training_step and validation_step.
        """
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

        # Return the loss as is - the training script will handle the logging
        return loss