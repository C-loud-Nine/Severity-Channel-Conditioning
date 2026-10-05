# Severity-Channel Conditioning

Code for **Severity-Channel Conditioning: An Injection-Level Study for Consumer-Device Deblurring**
(A. A. Shafi, K. S. Alam, S. R. Suma, S. I. Hossain; Khulna University of Engineering & Technology).

We study where a per-sample blur-severity descriptor should enter a converged latent-diffusion deblurring
model (DiffIR) fine-tuned on the iPhoneBlur consumer-device benchmark. Under a shared update budget, we compare
input-level conditioning (a severity channel appended to the image), global and spatial feature modulation,
a tier-contrastive output objective, and an unconditioned spectral-loss control. The descriptor is partly
derived from the paired sharp target and is supplied at inference, so the main results are an oracle diagnostic.

## Results

iPhoneBlur test set (1,686 images from 15 held-out videos), PSNR in dB.

| Configuration | Easy | Medium | Hard | Overall | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| Schedule-matched baseline | 35.14 | 31.62 | 27.99 | 31.23 | 0.948 | 0.049 |
| FiLM, global | 34.90 | 31.28 | 27.66 | 30.91 | 0.945 | 0.053 |
| FiLM, spatial (SFT) | 35.12 | 31.59 | 27.98 | 31.20 | 0.947 | 0.050 |
| Contrastive | 35.04 | 31.49 | 27.87 | 31.11 | 0.946 | 0.051 |
| Spectral-loss control | 35.10 | 31.59 | 28.02 | 31.21 | 0.944 | 0.052 |
| **Severity channel (ours)** | **35.20** | **31.70** | **28.08** | **31.30** | **0.949** | **0.048** |

All reported numbers are in `results/`.

## Repository

```
scc/            severity map, models (SCC, FiLM, SFT, Restormer), losses, data, metrics, tiled inference
scripts/        precompute_severity.py, train.py, evaluate.py, stats.py, make_split_manifest.py
configs/        one YAML per configuration, including the descriptor controls and Restormer runs
data/           split_manifest.csv (36/15 source-video split), test_videos.txt, normalization.json
results/        reported results as CSV
docs/           REPRODUCIBILITY.md (training budget, signal and evaluation settings)
```

## Setup

```bash
git clone https://github.com/C-loud-Nine/Severity-Channel-Conditioning.git
cd Severity-Channel-Conditioning
git clone https://github.com/Zj-BinXia/DiffIR.git
git clone https://github.com/swz30/Restormer.git          # only for the Restormer runs
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/DiffIR:$PWD/Restormer
```

Download iPhoneBlur ([arXiv:2605.05990](https://arxiv.org/abs/2605.05990)) with `complete_metadata.csv`, and the
GoPro-pretrained DiffIR S2 (and Restormer) weights from their official releases into `checkpoints/`.
Set `root` and `metadata` in the configs.

## Usage

```bash
# severity maps
python scripts/precompute_severity.py --root DATA --metadata complete_metadata.csv

# baseline adaptation, then each configuration (12 epochs from the adapted baseline)
python scripts/train.py --config configs/dab_parent.yaml
python scripts/train.py --config configs/scc.yaml

# evaluation (final-epoch EMA weights); --mode zero removes the descriptor at inference
python scripts/evaluate.py --config configs/scc.yaml --ckpt runs/scc/final.pth --out scc.csv

# significance tests
python scripts/stats.py --base dab_matched.csv --method scc.csv
python scripts/stats.py --per-video results/per_video.csv
```

## Data and ethics

iPhoneBlur videos were recorded by the benchmark authors in public outdoor places in compliance with local
regulations. No footage focuses on individuals and no personally identifiable information is present. Use for
surveillance, biometric identification, or other privacy-violating purposes is not permitted.

## Citation

```bibtex
@misc{shafi2026scc,
  author       = {Shafi, Abdullah Al and Alam, Kazi Saeed and Suma, Sumaiya Rahim and Hossain, Sk Imran},
  title        = {Severity-Channel Conditioning: An Injection-Level Study for Consumer-Device Deblurring},
  year         = {2026},
  howpublished = {\url{https://github.com/C-loud-Nine/Severity-Channel-Conditioning}}
}
```

## License

Released under the [MIT License](LICENSE). DiffIR and Restormer are subject to their own licenses.
