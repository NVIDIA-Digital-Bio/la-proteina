# Stochastic Heun-100 vs SDE Euler Baseline

**Date:** 2026-03-14
**Config:** `inference_ucond_notri_stochastic_heun` (inherits `inference_ucond_notri`, sampling_mode: sc)

## Method

Stochastic Heun applies the trapezoidal rule to the SDE drift while keeping the
same Brownian increment for both the trial and corrected steps:

```
dW = sqrt(dt) * randn()           # sample once
x_trial = x + f(x,t)*dt + s(t)*dW  # Euler-Maruyama trial
f2 = drift(x_trial, t+dt)          # 2nd function eval
x_new = x + 0.5*(f1+f2)*dt + s(t)*dW  # trapezoidal correction on drift
```

Strong convergence order 1.0 (vs 0.5 for Euler-Maruyama). Works because the
diffusion coefficient depends only on t (additive noise), so no Ito correction
is needed.

## Results Summary

Outliers with scRMSD > 20A are excluded from mean/median computations.
Designability threshold: scRMSD < 2.0A.

### Designability (ProteinMPNN + ESMFold scRMSD)

| Run | NFEs | Samples | Outliers | Designable | Rate | Mean scRMSD | Median scRMSD |
|-----|------|---------|----------|------------|------|-------------|---------------|
| SDE Euler-400 (1) | 400 | 1890 | 374 | 1052/1516 | 69.4% | 4.679 A | 0.836 A |
| **Stoch-Heun-100** | **200** | **150** | **2** | **126/148** | **85.1%** | **1.449 A** | **0.695 A** |

### Designability by Length

| Length | Euler-400 Rate | Euler-400 Med | Stoch-Heun Rate | Stoch-Heun Med |
|--------|---------------|---------------|-----------------|----------------|
| L=100 | 60.5% (362/598) | 0.939 A | 100.0% (49/49) | 0.498 A |
| L=200 | 71.2% (353/496) | 0.810 A | 82.0% (41/50) | 0.759 A |
| L=300 | 79.9% (337/422) | 0.818 A | 73.5% (36/49) | 1.255 A |

### Backbone Perplexity

| Length | SDE Euler (2) | Stoch-Heun-100 | Delta |
|--------|---------------|----------------|-------|
| L=100 | 3.183 +/- 0.345 | 3.201 +/- 0.373 | +0.018 |
| L=200 | 3.087 +/- 0.352 | 3.226 +/- 0.461 | +0.139 |
| L=300 | 3.013 +/- 0.353 | 2.915 +/- 0.391 | -0.098 |
| **Overall** | **3.095** | **3.114** | **+0.019** |

### Timing and Memory (A10G, 10 samples per batch, BF16+SDPA)

| Length | SDE Euler-200 (2) | Stoch-Heun-100 | Speedup |
|--------|-------------------|----------------|---------|
| L=100 | 13.9s / 2.50 GB | 8.9s / 2.73 GB | 1.56x |
| L=200 | 40.5s / 4.96 GB | 28.3s / 4.89 GB | 1.43x |
| L=300 | 76.9s / 9.07 GB | 58.5s / 8.48 GB | 1.31x |
| **Total (150 samples)** | **657s / 9.07 GB** | **478s / 8.48 GB** | **1.37x** |

Both Euler-200 and Stoch-Heun-100 use 200 NFEs. The speedup comes from fewer
Python loop iterations (100 vs 200) while the NN eval cost per NFE is the same.

Estimated SDE Euler-400 (400 NFEs) total: ~1314s. Stoch-Heun-100 speedup vs
Euler-400: **~2.75x**.

## Notes

1. **SDE Euler-400 designability** is from a Feb 16 run (commit 3c5e5a9, same
   model weights, 630 samples/length). The larger sample and older code version
   may not be perfectly matched to the Stoch-Heun-100 run. A matched 50-sample
   Euler-400 SDE eval would give a cleaner comparison.

2. **Perplexity and timing baseline** comes from the Mar 12 benchmark_baseline
   run, which used nsteps=200 (not 400). The perplexity comparison is therefore
   Euler-200 vs Stoch-Heun-100 (both 200 NFEs); Euler-400 perplexity would be
   slightly higher (more steps typically improves quality).

3. **Memory usage** is comparable between methods; the stochastic Heun holds one
   extra noise tensor per modality, which is negligible.

4. The high outlier rate in the Euler-400 baseline (374/1890 = 19.8% with scRMSD
   > 20A) versus Stoch-Heun-100 (2/150 = 1.3%) is a notable difference. This
   could be due to the larger sample size or a genuine improvement from the
   higher-order solver.

## Reproduction

```bash
# Full three-way benchmark (Euler-400 vs Heun-100 vs Stoch-Heun-100)
bash scripts/benchmark_solvers.sh 50 "100 200 300"

# Or run Stochastic Heun only:
python proteinfoundation/generate.py \
    --config_name inference_ucond_notri_stochastic_heun \
    --nsamples 50 --lengths 100 200 300 \
    --output_dir ./inference/stochastic_heun

python proteinfoundation/eval_trajectory.py \
    --input_dir ./inference/stochastic_heun
```
