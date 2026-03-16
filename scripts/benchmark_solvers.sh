#!/bin/bash
# Benchmark solver comparison: Euler-400 vs Heun-100 (ODE) vs Stochastic-Heun-100 (SDE)
#
# Generates samples with each solver, runs both fast eval (structural metrics,
# seconds) and full designability eval (ProteinMPNN + ESMFold scRMSD, hours),
# then prints a side-by-side summary.
#
# Requires: GPU with CUDA, checkpoints in ./checkpoints_laproteina/
#
# Usage:
#   bash scripts/benchmark_solvers.sh                      # default: 10 samples, L=100,200,300
#   bash scripts/benchmark_solvers.sh 50 "100 200 300"     # custom sample count and lengths
#
# Output goes to ./inference/benchmark/<solver>/ for each solver.

set -euo pipefail

NSAMPLES="${1:-10}"
LENGTHS="${2:-100 200 300}"

BENCHMARK_DIR="./inference/benchmark"
CONFIG="inference_ucond_notri"

SOLVERS=("euler" "heun" "stochastic_heun")
NSTEPS=(400 100 100)
LABELS=("Euler-400 (SDE baseline)" "Heun-100 (ODE, 200 NFEs)" "Stoch-Heun-100 (SDE, 200 NFEs)")

echo "=============================================="
echo "  Solver Benchmark"
echo "=============================================="
echo "  Samples per length: $NSAMPLES"
echo "  Lengths: $LENGTHS"
echo "  Solvers: ${SOLVERS[*]}"
echo "  Output:  $BENCHMARK_DIR"
echo "=============================================="
echo ""

# ── 1. Generate samples with each solver ─────────────────────────────────────

for i in "${!SOLVERS[@]}"; do
    solver="${SOLVERS[$i]}"
    nsteps="${NSTEPS[$i]}"
    label="${LABELS[$i]}"
    outdir="${BENCHMARK_DIR}/${solver}"

    echo "$(date '+%H:%M:%S') Generating: $label (nsteps=$nsteps)..."
    python proteinfoundation/generate.py \
        --config_name "$CONFIG" \
        --solver "$solver" \
        --nsteps "$nsteps" \
        --nsamples "$NSAMPLES" \
        --lengths $LENGTHS \
        --output_dir "$outdir"

    echo "  -> Saved to $outdir"
    echo ""
done

# ── 2. Fast eval (structural metrics; seconds) ──────────────────────────────

echo "=============================================="
echo "  Fast Eval (structural metrics)"
echo "=============================================="
echo ""

for i in "${!SOLVERS[@]}"; do
    solver="${SOLVERS[$i]}"
    label="${LABELS[$i]}"
    outdir="${BENCHMARK_DIR}/${solver}"

    echo "$(date '+%H:%M:%S') Fast eval: $label ..."
    python proteinfoundation/eval_trajectory_fast.py \
        --input_dir "$outdir"

    echo "  -> ${outdir}/trajectory_fast_eval.csv"
    echo ""
done

# ── 3. Full designability eval (ProteinMPNN + ESMFold; hours) ────────────────

echo "=============================================="
echo "  Full Eval (ProteinMPNN + ESMFold scRMSD)"
echo "=============================================="
echo ""

for i in "${!SOLVERS[@]}"; do
    solver="${SOLVERS[$i]}"
    label="${LABELS[$i]}"
    outdir="${BENCHMARK_DIR}/${solver}"

    echo "$(date '+%H:%M:%S') Full eval: $label ..."
    python proteinfoundation/eval_trajectory.py \
        --input_dir "$outdir"

    echo "  -> ${outdir}/trajectory_eval.csv"
    echo ""
done

# ── 4. Summary ───────────────────────────────────────────────────────────────

echo "=============================================="
echo "  Benchmark Complete"
echo "=============================================="
echo ""

python3 -c "
import pandas as pd
import os

solvers = ['euler', 'heun', 'stochastic_heun']
labels = ['Euler-400 (SDE)', 'Heun-100 (ODE)', 'Stoch-Heun-100 (SDE)']
base = '${BENCHMARK_DIR}'

# ── Designability (full eval) ──
rows = []
for solver, label in zip(solvers, labels):
    csv_path = os.path.join(base, solver, 'trajectory_eval.csv')
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    if 'step' in df.columns:
        df = df.loc[df.groupby('sample')['step'].idxmax()]
    n = len(df)
    scrmsd_col = [c for c in df.columns if 'scrmsd' in c.lower() or 'sc_rmsd' in c.lower()]
    if scrmsd_col:
        scrmsd = df[scrmsd_col[0]]
        valid = scrmsd[scrmsd <= 20.0]
        designable = (valid < 2.0).sum()
        rows.append({
            'Solver': label,
            'Samples': n,
            'Outliers': n - len(valid),
            'Designable': f'{designable}/{len(valid)}',
            'Rate': f'{100*designable/len(valid):.1f}%' if len(valid) > 0 else 'N/A',
            'Median scRMSD': f'{valid.median():.3f}A',
        })

if rows:
    print('Designability (scRMSD < 2.0A, outliers > 20A excluded):')
    print(pd.DataFrame(rows).to_string(index=False))
    print()

# ── Structural metrics (fast eval) ──
rows2 = []
for solver, label in zip(solvers, labels):
    csv_path = os.path.join(base, solver, 'trajectory_fast_eval.csv')
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    if 'step' in df.columns:
        df = df.loc[df.groupby('sample')['step'].idxmax()]
    info = {'Solver': label, 'N': len(df)}
    for col in ['radius_of_gyration', 'helix_fraction', 'sheet_fraction']:
        if col in df.columns:
            info[col] = f'{df[col].mean():.3f}'
    rows2.append(info)

if rows2:
    print('Structural metrics (fast eval):')
    print(pd.DataFrame(rows2).to_string(index=False))
" 2>/dev/null || echo "(install pandas for automatic summary)"

echo ""
echo "=============================================="
