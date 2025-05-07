from datasets import load_dataset
import torchaudio
import json
from tqdm import tqdm
import os
import torch

# Tải dữ liệu từ HuggingFace
ds = load_dataset("Cets/audio-logs-vie", split="train")

# Cấu hình
output_dir = "datatest/audio_clips"
os.makedirs(output_dir, exist_ok=True)
output_jsonl = "datatest/audio_logs_vie.jsonl"
MAX_DURATION = 15.0

valid_count = 0
with open(output_jsonl, "w", encoding="utf-8") as f:
    for idx, item in enumerate(tqdm(ds)):
        try:
            text = item.get("text", "").strip()
            audio_info = item["audio"]
            audio_array = audio_info.get("array", None)
            sr = audio_info.get("sampling_rate", None)

            if not text or audio_array is None or sr is None:
                continue

            waveform = torch.tensor(audio_array).unsqueeze(0)  # (1, N)
            duration = waveform.shape[1] / sr
            if duration > MAX_DURATION:
                continue

            # Ghi file âm thanh vào working directory
            save_path = os.path.join(output_dir, f"clip_{idx}.wav")
            torchaudio.save(save_path, waveform, sr)

            json_record = {
                "audio_filepath": save_path,
                "duration": round(duration, 3),
                "offset": 0,
                "text": text
            }
            f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
            valid_count += 1

        except Exception as e:
            continue

print(f"Tổng số mẫu hợp lệ đã ghi: {valid_count}")

from datasets import load_dataset
from tqdm import tqdm
import os
import shutil
import torchaudio.transforms as T
# Tên datasets từ Hugging Face

from datasets import load_dataset
from tqdm import tqdm
import os
import torchaudio
import torch

# Danh sách các dataset noise
# noise_datasets = {
#     "fsdnoisy18k": "sps44/fsdnoisy18k-test",
#     "musan": "noisy-alpaca-test/MUSAN-noise-audio-only"
# }
noise_datasets = {
    "fsdnoisy18k": "sps44/fsdnoisy18k-test",
}

save_root = "datatest/noise"
os.makedirs(save_root, exist_ok=True)

for name, hf_path in noise_datasets.items():
    print(f"Downloading {name} from {hf_path}...")
    dataset = load_dataset(hf_path, split="train")

    save_dir = os.path.join(save_root, name)
    os.makedirs(save_dir, exist_ok=True)

    for idx, sample in enumerate(tqdm(dataset, desc=f"Saving {name}")):
        audio_info = sample["audio"]
        waveform = torch.tensor(audio_info["array"]).unsqueeze(0)  # [1, T]
        sr = audio_info["sampling_rate"]

        filename = f"{idx:05d}.wav"
        out_path = os.path.join(save_dir, filename)

        # Chuẩn hóa về 16kHz nếu cần
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        torchaudio.save(out_path, waveform, sr)
