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

        logger.info(f"Initializing StreamingRNNT with att_context_size={att_context_size}, vocab_size={vocab_size}")

        # Load pretrained encoder weights
        try:
            encoder_state_dict = torch.load(PRETRAINED_ENCODER_WEIGHT, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=True)
            # Create new keys for conv3 from conv2
            encoder_state_dict['model_state_dict']['conv3.weight'] = encoder_state_dict['model_state_dict']['conv2.weight']
            encoder_state_dict['model_state_dict']['conv3.bias'] = encoder_state_dict['model_state_dict']['conv2.bias']
            logger.info(f"Loaded encoder weights from {PRETRAINED_ENCODER_WEIGHT}")
        except Exception as e:
            logger.error(f"Error loading encoder weights: {str(e)}")
            raise

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

        # Load tokenizer
        try:
            self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
            logger.info(f"Loaded tokenizer from {tokenizer_model_path}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise

        # RNNT loss function
        self.loss = warprnnt_numba.RNNTLossNumba(
            blank=RNNT_BLANK, reduction="mean",
        )

    def greedy_decoding(self, x, x_len, max_symbols=None):
        """
        Greedy decoding for generating predictions
        """
        enc_out, _ = self.encoder(x, x_len)
        all_sentences = []

        # Process each sequence in the batch independently
        for batch_idx in range(enc_out.shape[0]):
            hypothesis = [[None, None]]  # [label, state]
            seq_enc_out = enc_out[batch_idx, :, :].unsqueeze(0)  # [1, T, D]
            seq_ids = []

            for time_idx in range(seq_enc_out.shape[1]):
                curent_seq_enc_out = seq_enc_out[:, time_idx, :].unsqueeze(1)  # [1, 1, D]

                not_blank = True
                symbols_added = 0

                while not_blank and (max_symbols is None or symbols_added < max_symbols):
                    # In the first timestep, initialize with RNNT Blank
                    # In later timesteps, provide previous predicted label as input
                    if hypothesis[-1][0] is None:
                        last_token = torch.tensor([[RNNT_BLANK]], dtype=torch.long, device=seq_enc_out.device)
                        last_seq_h_n = None
                    else:
                        last_token = hypothesis[-1][0]
                        last_seq_h_n = hypothesis[-1][1]

                    # Get decoder output
                    if last_seq_h_n is None:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token)
                    else:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token, last_seq_h_n)

                    # Joint network
                    logits = self.joint(curent_seq_enc_out, current_seq_dec_out)[0, 0, 0, :]  # (V + 1)

                    del current_seq_dec_out

                    # Get predicted token
                    _, token_id = logits.max(0)
                    token_id = token_id.detach().item()

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

            # Convert token IDs to text
            all_sentences.append(self.tokenizer.decode(seq_ids))

        return all_sentences

    def process_batch(self, batch):
        """
        Process batch for model input
        """
        return batch

    def forward(self, x, x_len, y, y_len):
        """
        Forward pass that computes loss
        """
        # Encoder
        enc_out, x_len = self.encoder(x, x_len)  # (B, T, Enc_dim)

        # Add blank token to the beginning of target sequence
        y_start = torch.cat([torch.full((y.shape[0], 1), RNNT_BLANK, dtype=torch.int).to(y.device), y], dim=1).to(y.device)

        # Decoder
        dec_out, _ = self.decoder(y_start)  # (B, U, Dec_dim)

        # Joint network
        logits = self.joint(enc_out, dec_out)

        # Prepare inputs for loss function
        input_lengths = x_len.int()
        target_lengths = y_len.int()
        targets = y.int()

        # Compute loss
        loss = self.loss(logits.to(torch.float32), targets, input_lengths, target_lengths)

        return loss