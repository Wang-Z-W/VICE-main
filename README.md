# VICE

Our paper **Diagnosing and Mitigating Evidence Misgrounding in Multimodal Knowledge Editing** has been accepted to **Findings of EMNLP 2026**.

## Setup

- Use Python 3.10 and install the dependencies with `pip install -r requirements.txt`.
- Prepare LLaVA-v1.5-7B or MiniGPT-4-vicuna-7B and configure their paths in `utils/GLOBAL.py`. MiniGPT-4 requires its model configuration and associated weights. VICE also uses CLIP-ViT-B/32 and `all-MiniLM-L6-v2`.
- Please follow [LiveEdit](https://github.com/qizhou000/LiveEdit) to prepare **MMEdit (EVQA/EIC)** and **VLKEB** datasets.
- **WaterBird-Edit:** Annotations are included in `data/WaterBird-v2.0/`; see its [data README](data/WaterBird-v2.0/README.md) to prepare the referenced images.

## Getting Started

Generate image summaries and train the scope head:

```bash
# EVQA, EIC, and VLKEB; use DATA_NAME=waterv2 for WaterBird-Edit
DATA_NAME=all bash scripts/generate_summaries.sh

# Edit DATA_NAMES in the script to select the datasets to train
bash scripts/train_scope_head.sh
```

Summary preprocessing defaults to eight GPUs. Adjust `NUM_GPUS` in `preprocess_image_summaries.py` to match your hardware.

Run a reference experiment:

```bash
SCOPE_CKPT="/path/to/your/evqa_scope_head.pt" \
  bash scripts/main/vice_vl/run_vice_llava_evqa.sh
```

Scope-head paths in the scripts are placeholders. Replace them with your own **checkpoint files**, or set `SCOPE_CKPT` as above. Relative paths are resolved from the repository root. Trained scope-head weights are saved under `checkpoints/`.

The scripts provide example settings. Adjust GPU IDs, editing scale, similarity thresholds, and other parameters for your experiments. VICE evaluation launchers run in the background; check `logs/` for progress and `eval_results/` for outputs.

## WaterBird-Edit

We release 1,136 editing annotations (636 train / 500 test). Edit and image-rephrase images come directly from CUB-200-2011; image-locality queries use MMEdit images. See the [data README](data/WaterBird-v2.0/README.md) for image preparation.

## Acknowledgements

This repository draws on [LiveEdit](https://github.com/qizhou000/LiveEdit) and [ComprehendEdit](https://github.com/yaohui120/ComprehendEdit). We thank the authors for sharing their code.
