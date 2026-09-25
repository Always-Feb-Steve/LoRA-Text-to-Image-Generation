# LoRA Fine-tuning for Stable Diffusion

Fine-tune Stable Diffusion 1.5 with LoRA (Low-Rank Adaptation) on custom image datasets. The pipeline covers everything from data scraping to training and inference.

## Project Structure

```
lora-finetune/
├── data/
│   ├── scratch_data.py            # Step 1: Scrape images from Bing
│   ├── convert_data.py            # Step 2a: Build captions.json from image folders
│   └── convert_data_from_jsonl.py # Step 2b: Convert from existing .jsonl metadata
├── experiments/
│   └── vae_demo.py                # VAE encode/decode sanity check
├── train_LoRA.py                  # Step 3: Train LoRA on SD1.5
├── eval_LoRA.py                   # Step 4: Evaluate a saved checkpoint
├── text_to_image.py               # Baseline inference (SD1.5 / SDXL / SD3)
├── requirements.txt
└── .gitignore
```

## Pipeline Overview

```
Bing Images          Folder of images
     │                      │
scratch_data.py      convert_data.py / convert_data_from_jsonl.py
     │                      │
     └──────────────────────┘
                   │
            captions.json
                   │
            train_LoRA.py          ← LoRA fine-tuning on SD1.5
                   │
         checkpoints/epoch_N/
                   │
             eval_LoRA.py          ← Generate test images from checkpoint
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Scrape training images

```bash
python data/scratch_data.py --tag Kung_Fu_Panda --count 20
```

Images are saved to `lora_train/<tag>/` (override with `--save_dir`).

### 3. Prepare captions

**Option A** — auto-generate captions from folder names (every sub-folder of `lora_train/` becomes a tag):
```bash
python data/convert_data.py                      # all tags
python data/convert_data.py --tags Kung_Fu_Panda # only some tags
```

**Option B** — convert from an existing `metadata.jsonl` (Hugging Face `imagefolder` format, `{"file_name": ..., "text": ...}`):
```bash
python data/convert_data_from_jsonl.py --infolder lora_train/animal
```

Both write `lora_train/captions.json` with entries like:
```json
[
  {"caption": "a photo of 'Kung_Fu_Panda'", "file_path": "lora_train/Kung_Fu_Panda/Kung_Fu_Panda_1.jpg"}
]
```

### 4. (Optional) Verify VAE

Sanity-check that the SD1.5 VAE can reconstruct your images cleanly:
```bash
python experiments/vae_demo.py input.png reconstruction.png
```

### 5. Train LoRA

```bash
python train_LoRA.py
```

Every option in `create_default_config()` can be overridden from the command line, e.g. a quick test run:
```bash
python train_LoRA.py --num_epochs 1 --batch_size 1 --image_size 256
```

| Parameter | Default | Description |
|---|---|---|
| `--captions_file` | `lora_train/captions.json` | Training data |
| `--output_dir` | `./models/lora_trained` | Where checkpoints and the final adapter go |
| `--learning_rate` | `5e-6` | AdamW learning rate |
| `--batch_size` | `2` | Batch size per step |
| `--gradient_accumulation_steps` | `2` | Effective batch = 4 |
| `--num_epochs` | `100` | Training epochs |
| `--image_size` | `512` | Training resolution |
| `--lora_rank` | `4` | LoRA rank (r) |
| `--lora_alpha` | `4` | LoRA scaling factor |
| `--save_steps` | `20` | Save checkpoint every N steps |

Checkpoints are saved to `models/lora_trained/checkpoints/checkpoint_epoch_N/` (plus `checkpoint_<step>/` every `save_steps`), and the final adapter to `models/lora_trained/`.

### 6. Evaluate a checkpoint

```bash
python eval_LoRA.py --prompt "a photo of 'Kung_Fu_Panda'"   # final adapter
python eval_LoRA.py --lora_path models/lora_trained/checkpoints/checkpoint_epoch_40
```

Generated images are saved to `test_generations/`.

### 7. Baseline inference (no LoRA)

```bash
python text_to_image.py --model sd15 --prompt "an astronaut riding a horse"
python text_to_image.py --model sdxl --no_refiner
```

Supports SD1.5, SDXL (with optional refiner), and SD3 (gated on Hugging Face — accept the license and run `huggingface-cli login` first). Images are saved to `generated_images/`.

## How LoRA Works

LoRA freezes the original UNet weights and injects small trainable low-rank matrices into the attention layers. For a weight matrix **W** (M×N), instead of updating all M×N parameters, LoRA learns two smaller matrices **A** (M×r) and **B** (r×N) where r ≪ min(M,N).

Target modules: `to_k`, `to_q`, `to_v`, `to_out`, `proj_in`, `proj_out`, `ff.net.*`

The text encoder and VAE remain fully frozen throughout training.

## Requirements

- Python 3.9+
- An NVIDIA GPU (CUDA) or an Apple Silicon Mac (MPS) — the scripts pick the device automatically; CPU works but is very slow
- ~6GB VRAM for SD1.5 LoRA training at batch size 2, image size 512

## Notes

- The Bing scraper in `scratch_data.py` is for educational use. Please respect Bing's Terms of Service and `robots.txt`.
- Model weights (`.safetensors`, `.bin`, `.pt`) are excluded from git via `.gitignore`. The base model ([stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), ~4 GB) is downloaded automatically by `diffusers` on first run; pass `--model_name` to use a local copy.
