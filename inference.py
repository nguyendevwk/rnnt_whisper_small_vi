import torch
import torchaudio
import sentencepiece as spm
from loguru import logger
import librosa
import numpy as np
import torch.nn.functional as F

# Import các thành phần cần thiết từ project
from models.encoder import AudioEncoder
from models.decoder import Decoder
from models.jointer import Jointer
from constants import RNNT_BLANK, VOCAB_SIZE, TOKENIZER_MODEL_PATH, MAX_SYMBOLS
from constants import ATTENTION_CONTEXT_SIZE, N_STATE, N_LAYER, N_HEAD, N_MELS
from constants import SAMPLE_RATE, N_FFT, HOP_LENGTH

def log_mel_spectrogram(audio, n_mels, padding=0, streaming=False, device=None):
    """Tính toán log mel spectrogram từ audio waveform."""
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))

    window = torch.hann_window(N_FFT).to(audio.device)
    if not streaming:
        stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    else:
        stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, center=False, return_complex=True)

    magnitudes = stft[..., :-1].abs() ** 2

    # Sử dụng librosa để tạo mel filters
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)
    mel_basis = torch.from_numpy(mel_basis).to(audio.device)

    mel_spec = torch.matmul(mel_basis, magnitudes)
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec

def main(checkpoint_path, wav_file_path):
    """Load checkpoint và thực hiện nhận dạng trên file WAV."""
    # Cấu hình device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Sử dụng device: {device}")

    # Khởi tạo model
    logger.info("Đang khởi tạo model...")
    encoder = AudioEncoder(
        n_mels=N_MELS,
        n_state=N_STATE,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        att_context_size=ATTENTION_CONTEXT_SIZE
    )

    decoder = Decoder(vocab_size=VOCAB_SIZE + 1)
    joint = Jointer(vocab_size=VOCAB_SIZE + 1)

    # Load checkpoint
    logger.info(f"Đang load checkpoint từ {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Tách state_dict thành các phần riêng biệt và bỏ qua các tensor alibi
    encoder_weight = {}
    decoder_weight = {}
    joint_weight = {}

    for k, v in checkpoint['state_dict'].items():
        if 'alibi' in k:
            continue
        if 'encoder' in k:
            encoder_weight[k.replace('encoder.', '')] = v
        elif 'decoder' in k:
            decoder_weight[k.replace('decoder.', '')] = v
        elif 'joint' in k:
            joint_weight[k.replace('joint.', '')] = v

    # Load state dict sau khi đã xử lý
    encoder.load_state_dict(encoder_weight, strict=False)
    decoder.load_state_dict(decoder_weight, strict=False)
    joint.load_state_dict(joint_weight, strict=False)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    joint = joint.to(device)

    encoder.eval()
    decoder.eval()
    joint.eval()

    # Khởi tạo tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)

    # Xử lý file WAV
    logger.info(f"Đang xử lý file WAV {wav_file_path}...")
    audio, _ = librosa.load(wav_file_path, sr=SAMPLE_RATE)
    audio = torch.from_numpy(audio).to(device)

    # Offline transcription
    logger.info("Đang thực hiện inference...")
    with torch.no_grad():
        mels = log_mel_spectrogram(audio=audio, n_mels=N_MELS, device=device)
        x = mels.reshape(1, *mels.shape)
        x_len = torch.tensor([x.shape[2]]).to(device)

        enc_out, _ = encoder(x, x_len)

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

                while not_blank and (MAX_SYMBOLS is None or symbols_added < MAX_SYMBOLS):
                    # In the first timestep, we initialize the network with RNNT Blank
                    # In later timesteps, we provide previous predicted label as input.
                    if hypothesis[-1][0] is None:
                        last_token = torch.tensor([[RNNT_BLANK]], dtype=torch.long, device=seq_enc_out.device)
                        last_seq_h_n = None
                    else:
                        last_token = hypothesis[-1][0]
                        last_seq_h_n = hypothesis[-1][1]

                    if last_seq_h_n is None:
                        current_seq_dec_out, current_seq_h_n = decoder(last_token)
                    else:
                        current_seq_dec_out, current_seq_h_n = decoder(last_token, last_seq_h_n)
                    logits = joint(curent_seq_enc_out, current_seq_dec_out)[0, 0, 0, :]  # (B, T=1, U=1, V + 1)

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
            all_sentences.append(tokenizer.decode(seq_ids))

    # In kết quả
    transcription = all_sentences[0]
    logger.info(f"Kết quả nhận dạng: {transcription}")
    return transcription

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR Inference trên file WAV")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến file checkpoint")
    parser.add_argument("--wav", type=str, required=True, help="Đường dẫn đến file WAV")

    args = parser.parse_args()

    result = main(args.checkpoint, args.wav)
    print(f"Kết quả nhận dạng: {result}")