# Reproducibility record

## Training stages
* **DAB adaptation:** 12 epochs from GoPro-pretrained DiffIR S2, no descriptor (`configs/dab_parent.yaml`).
* **Branch stage:** 12 further epochs from the DAB for every DiffIR arm, including the schedule-matched baseline.
* **Restormer:** 20 epochs from GoPro-pretrained weights for both the baseline and the SCC run.
* Every reported result uses the **final-epoch EMA checkpoint**. No test image or video informs checkpoint choice or training length.

## Budget (every DiffIR branch)
| Setting | Value |
|---|---|
| Physical batch / accumulation | 2 / 2 (effective 4); contrastive: 4 / 1 |
| Micro-batches per epoch | 2,857 (trailing incomplete batch dropped) |
| Optimizer updates | 1,428 per epoch, 17,136 per arm |
| Loss scaling | divided by the accumulation factor, AMP GradScaler |
| Optimizer | AdamW, betas (0.9, 0.99), no weight decay |
| Schedule | cosine from 2e-5 (min 1e-6), period 34,284 steps, stepped once per update; final rate about 1.05e-5 |
| EMA | 0.999, updated per optimizer step |
| Patch / augmentation | 256x256 random crop, horizontal flip |
| Seeds | 42, 123, 456 |

## Signal specification
* Severity map: 3x3 Sobel on grayscale, 15x15 Gaussian (sigma 2.6), normalized by its maximum with eps 1e-6;
  gamma = 1.5 - m~, m~ = isp_diff min-max normalized over all released metadata, range [-3830.9, -33.6]
  (transductive, consistent with the oracle-diagnostic scope); clipped to [0, 1]; stored at 256x256, bilinearly upsampled.
* isp_diff = high-frequency energy of the blurred frame minus that of the sharp frame.
* FiLM inputs: motion, isp_diff, blur_window, blur_duration_ms, noise_estimate; z-scored with training-split statistics
  (`data/normalization.json`); no missing values.

## Evaluation specification
* Full resolution, reflect-padded to a multiple of 8, 768 px tiles with 48 px overlap and averaging.
* PSNR and SSIM on 8-bit RGB (data range 255, scikit-image, SSIM over colour channels); LPIPS with AlexNet.
* Diffusion latent fixed (`torch.manual_seed(0)`) across compared conditions.

## Parameter counts (DiffIR)
SCC +9,216 (16 unshuffled severity channels x 64 x 3 x 3); FiLM +1,680,864; SFT +4,053,536.
