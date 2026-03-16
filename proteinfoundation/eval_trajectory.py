"""Evaluate trajectory snapshots from protein generation.

Loads ESMFold once and evaluates all trajectory steps + final outputs.
Produces a single CSV with a 'step' column and prints a summary table.

Usage:
    python proteinfoundation/eval_trajectory.py \
        --config_name inference_ucond_notri \
        --input_dir ./inference/inference_ucond_notri
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict


def _numsort_key(path: str):
    """Sort key that orders numeric segments numerically, not alphabetically."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path)]

root = os.path.abspath(".")
sys.path.insert(0, root)
# isort: split

import pandas as pd
import torch
from loguru import logger

from proteinfoundation.metrics.designability import (
    extract_seq_from_pdb,
    pdb_name_from_path,
    scRMSD,
)
from proteinfoundation.metrics.folding_models import load_esmfold


def discover_steps(input_dir: str) -> list[dict]:
    """Discover trajectory step PDBs and the final output PDBs.

    Supports per-sample trajectory directories (new layout):
        job_0_n_100_trajectory_0/job_0_step_0002_n_100_sample_0.pdb
    and legacy per-step directories:
        trajectory_step_0002/step_0002_sample_0.pdb

    Returns a list of dicts: [{"step": int, "pdb_paths": [str, ...], "label": str}, ...]
    Sorted by step number. Final outputs get label "final".
    """
    # Collect trajectory PDBs grouped by step number.
    step_pdbs: dict[int, list[str]] = defaultdict(list)

    # New layout: per-sample dirs named *_trajectory_*
    for d in sorted(glob.glob(os.path.join(input_dir, "*_trajectory_*")), key=_numsort_key):
        if not os.path.isdir(d):
            continue
        for pdb in sorted(glob.glob(os.path.join(d, "*.pdb")), key=_numsort_key):
            match = re.search(r"_step_(\d+)_", os.path.basename(pdb))
            if match:
                step_pdbs[int(match.group(1))].append(pdb)

    # Legacy layout: per-step dirs named trajectory_step_NNNN
    for d in sorted(glob.glob(os.path.join(input_dir, "trajectory_step_*")), key=_numsort_key):
        if not os.path.isdir(d):
            continue
        match = re.search(r"trajectory_step_(\d+)$", d)
        if not match:
            continue
        step = int(match.group(1))
        for pdb in sorted(glob.glob(os.path.join(d, "*.pdb")), key=_numsort_key):
            step_pdbs[step].append(pdb)

    entries = []
    for step in sorted(step_pdbs):
        entries.append({
            "step": step,
            "pdb_paths": sorted(step_pdbs[step], key=_numsort_key),
            "label": f"step_{step:04d}",
        })

    # Find final output PDBs (in job_*_id_* subdirectories, excluding trajectory dirs)
    final_pdbs = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        if "_trajectory_" in dirpath or "trajectory_step_" in dirpath:
            continue
        if "eval_tmp" in dirpath:
            continue
        for f in filenames:
            if f.endswith(".pdb"):
                final_pdbs.append(os.path.join(dirpath, f))

    if final_pdbs:
        max_traj_step = max(step_pdbs) if step_pdbs else 0
        final_step = max_traj_step if max_traj_step > 0 else 9999
        entries.append({"step": final_step, "pdb_paths": sorted(final_pdbs, key=_numsort_key), "label": "final"})

    return entries


def evaluate_step(
    step_info: dict,
    preloaded_esmfold: tuple,
    tmp_base: str,
) -> list[dict]:
    """Evaluate all PDBs for a single trajectory step.

    Returns a list of per-sample result dicts.
    """
    step = step_info["step"]
    label = step_info["label"]
    results = []

    for pdb_path in step_info["pdb_paths"]:
        name = pdb_name_from_path(pdb_path)
        seq = extract_seq_from_pdb(pdb_path)
        n = len(seq)

        # Create tmp dir for this sample's eval artifacts
        tmp_dir = os.path.join(tmp_base, f"{label}_{name}")
        os.makedirs(tmp_dir, exist_ok=True)

        row = {
            "step": step,
            "label": label,
            "pdb_path": pdb_path,
            "length": n,
            "sequence": seq,
        }

        try:
            res = scRMSD(
                pdb_file_path=pdb_path,
                ret_min=False,
                tmp_path=tmp_dir,
                use_pdb_seq=False,
                rmsd_modes=["ca"],
                folding_models=["esmfold"],
                preloaded_esmfold=preloaded_esmfold,
                keep_outputs=True,
            )
            # res["ca"]["esmfold"] is a list of RMSD values (one per ProteinMPNN seq)
            rmsds = res["ca"]["esmfold"]
            row["scRMSD_ca_min"] = min(rmsds) if rmsds else float("inf")
            row["scRMSD_ca_mean"] = sum(rmsds) / len(rmsds) if rmsds else float("inf")
            row["scRMSD_ca_all"] = rmsds
        except Exception as e:
            logger.error(f"Failed to evaluate {pdb_path}: {e}")
            row["scRMSD_ca_min"] = float("inf")
            row["scRMSD_ca_mean"] = float("inf")
            row["scRMSD_ca_all"] = []

        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectory snapshots")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Generation output dir containing trajectory_step_*/ subdirs and final PDBs",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path for output CSV (default: <input_dir>/trajectory_eval.csv)",
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_csv = args.output_csv or os.path.join(input_dir, "trajectory_eval.csv")

    # Discover steps
    steps = discover_steps(input_dir)
    if not steps:
        logger.error(f"No PDB files found in {input_dir}")
        sys.exit(1)

    total_pdbs = sum(len(s["pdb_paths"]) for s in steps)
    logger.info(f"Found {len(steps)} steps with {total_pdbs} total PDBs")
    for s in steps:
        logger.info(f"  {s['label']}: {len(s['pdb_paths'])} PDBs")

    # Load ESMFold once
    logger.info("Loading ESMFold model (one-time)...")
    esmfold_model, esmfold_tokenizer = load_esmfold()
    preloaded_esmfold = (esmfold_model, esmfold_tokenizer)
    logger.info("ESMFold loaded.")

    # Load existing results to skip already-evaluated PDBs (resume support)
    existing_pdbs = set()
    prior_results = []
    if os.path.exists(output_csv):
        prior_df = pd.read_csv(output_csv)
        existing_pdbs = set(prior_df["pdb_path"].tolist())
        prior_results = prior_df.to_dict("records")
        logger.info(f"Resuming: found {len(existing_pdbs)} already-evaluated PDBs in {output_csv}")

    # Filter out already-evaluated PDBs from each step
    for step_info in steps:
        before = len(step_info["pdb_paths"])
        step_info["pdb_paths"] = [p for p in step_info["pdb_paths"] if p not in existing_pdbs]
        skipped = before - len(step_info["pdb_paths"])
        if skipped:
            logger.info(f"  {step_info['label']}: skipping {skipped}/{before} already evaluated")

    remaining = sum(len(s["pdb_paths"]) for s in steps)
    if remaining == 0:
        logger.info("All PDBs already evaluated, nothing to do.")
    else:
        logger.info(f"{remaining} PDBs remaining to evaluate")

    # Evaluate each step
    tmp_base = os.path.join(input_dir, "eval_tmp")
    all_results = list(prior_results)

    for step_info in steps:
        if not step_info["pdb_paths"]:
            continue
        logger.info(f"Evaluating {step_info['label']} ({len(step_info['pdb_paths'])} PDBs)...")
        step_results = evaluate_step(step_info, preloaded_esmfold, tmp_base)
        all_results.extend(step_results)

        # Save incrementally after each step so progress survives crashes
        df_inc = pd.DataFrame(all_results)
        df_inc.drop(columns=["scRMSD_ca_all"], errors="ignore").to_csv(output_csv, index=False)

    # Final save
    df = pd.DataFrame(all_results)
    df_csv = df.drop(columns=["scRMSD_ca_all"], errors="ignore")
    df_csv.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")

    # Print summary table
    print("\n" + "=" * 70)
    print("TRAJECTORY EVALUATION SUMMARY")
    print("=" * 70)
    summary = df.groupby(["step", "label"]).agg(
        n_samples=("scRMSD_ca_min", "count"),
        mean_scRMSD=("scRMSD_ca_min", "mean"),
        median_scRMSD=("scRMSD_ca_min", "median"),
        designability_rate=("scRMSD_ca_min", lambda x: (x < 2.0).mean()),
    ).reset_index()

    print(f"\n{'Step':>6}  {'Label':<12}  {'N':>4}  {'Mean scRMSD':>12}  {'Med scRMSD':>11}  {'Design. Rate':>13}")
    print("-" * 70)
    for _, row in summary.iterrows():
        print(
            f"{row['step']:>6}  {row['label']:<12}  {row['n_samples']:>4}  "
            f"{row['mean_scRMSD']:>12.3f}  {row['median_scRMSD']:>11.3f}  "
            f"{row['designability_rate']:>12.1%}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
