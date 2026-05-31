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
python data/scratch_data.py
```

Edit `tag` and `save_dir` in the script to set your subject and output folder.

### 3. Prepare captions

**Option A** — auto-generate captions from folder name:
```bash
python data/convert_data.py
```

**Option B** — convert from existing `.jsonl` metadata:
```bash
python data/convert_data_from_jsonl.py
```

Both produce a `captions.json` with entries like:
```json
[
  {"caption": "a photo of 'Kung_Fu_Panda'", "file_path": "/path/to/image.jpg"}
]
```

### 4. (Optional) Verify VAE

Sanity-check that the SD1.5 VAE can reconstruct your images cleanly:
```bash
python experiments/vae_demo.py
```

Edit `inpath` and `outpath` in the script.

### 5. Train LoRA

```bash
python train_LoRA.py
```

Key config options inside `create_default_config()`:

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `5e-6` | AdamW learning rate |
| `batch_size` | `2` | Batch size per step |
| `gradient_accumulation_steps` | `2` | Effective batch = 4 |
| `num_epochs` | `100` | Training epochs |
| `lora_rank` | `4` | LoRA rank (r) |
| `lora_alpha` | `4` | LoRA scaling factor |
| `save_steps` | `20` | Save checkpoint every N steps |

Checkpoints are saved to `models/lora_trained_animal/checkpoints/checkpoint_epoch_N/`.

### 6. Evaluate a checkpoint

Edit `epoch` and `lora_path` in `eval_LoRA.py`, then:
```bash
python eval_LoRA.py
```

Generated images are saved to `test_generations/`.

### 7. Baseline inference (no LoRA)

```bash
python text_to_image.py
```

Supports SD1.5, SDXL (with optional refiner), and SD3.

## How LoRA Works

LoRA freezes the original UNet weights and injects small trainable low-rank matrices into the attention layers. For a weight matrix **W** (M×N), instead of updating all M×N parameters, LoRA learns two smaller matrices **A** (M×r) and **B** (r×N) where r ≪ min(M,N).

Target modules: `to_k`, `to_q`, `to_v`, `to_out`, `proj_in`, `proj_out`, `ff.net.*`

The text encoder and VAE remain fully frozen throughout training.

## Requirements

- Python 3.9+
- CUDA GPU recommended (training on CPU is very slow)
- ~6GB VRAM for SD1.5 LoRA training at batch size 2, image size 512

## Notes

- The Bing scraper in `scratch_data.py` is for educational use. Please respect Bing's Terms of Service and `robots.txt`.
- Model weights (`.safetensors`, `.bin`, `.pt`) are excluded from git via `.gitignore`. Download the base model from [Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5) or let `diffusers` fetch it automatically on first run.
