"""Fast trajectory evaluation using coordinate-only metrics (no neural networks).

Computes per-step:
  - CA RMSD vs final structure (convergence)
  - CA RMSD vs previous step (rate of change)
  - Radius of gyration
  - Secondary structure content (helix/sheet fraction via CA distances)
  - (optional) ProteinMPNN backbone perplexity (cheap designability proxy)

Runs in seconds, not hours. Use for trajectory analysis; reserve full scRMSD
eval (eval_trajectory.py) for final structures only.

Usage:
    python proteinfoundation/eval_trajectory_fast.py \
        --input_dir ./inference/inference_ucond_notri/2026-02-14_205759

    # With backbone perplexity (adds ~0.5s per PDB on CPU):
    python proteinfoundation/eval_trajectory_fast.py \
        --input_dir ./inference/inference_ucond_notri/2026-02-14_205759 \
        --perplexity
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np


def _numsort_key(path: str):
    """Sort key that orders numeric segments numerically, not alphabetically."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path)]

root = os.path.abspath(".")
sys.path.insert(0, root)
# isort: split

import pandas as pd
from loguru import logger

from proteinfoundation.utils.pdb_utils import load_pdb

# CA is atom index 1 in atom37 representation.
CA_IDX = 1


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def extract_ca_coords(pdb_path: str) -> np.ndarray:
    """Load a PDB and return CA coordinates as (n, 3) array."""
    prot = load_pdb(pdb_path)
    return prot.atom_positions[:, CA_IDX, :]


def kabsch_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute CA RMSD after optimal superposition (Kabsch alignment).

    Both inputs should be (n, 3) arrays with matching residue count.
    """
    assert coords1.shape == coords2.shape
    # Center
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)
    # Kabsch
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    c1_aligned = c1 @ R.T
    return float(np.sqrt(np.mean(np.sum((c1_aligned - c2) ** 2, axis=1))))


def radius_of_gyration(coords: np.ndarray) -> float:
    """Radius of gyration from CA coordinates."""
    center = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))


def secondary_structure_fractions(coords: np.ndarray) -> dict:
    """Estimate helix/sheet content from CA-CA distance patterns.

    Uses the characteristic CA-CA distances:
      - Alpha helix: CA(i) to CA(i+3) ~ 5.1-5.5 A
      - Beta sheet: CA(i) to CA(i+2) ~ 6.5-7.5 A (extended)
    """
    n = len(coords)
    if n < 4:
        return {"helix_frac": 0.0, "sheet_frac": 0.0}

    helix_count = 0
    sheet_count = 0
    for i in range(n - 3):
        d_i3 = np.linalg.norm(coords[i] - coords[i + 3])
        if 4.8 <= d_i3 <= 5.8:
            helix_count += 1
    for i in range(n - 2):
        d_i2 = np.linalg.norm(coords[i] - coords[i + 2])
        if 6.2 <= d_i2 <= 7.8:
            sheet_count += 1

    return {
        "helix_frac": helix_count / max(n - 3, 1),
        "sheet_frac": sheet_count / max(n - 2, 1),
    }


def discover_samples(input_dir: str) -> dict[int, dict]:
    """Discover per-sample trajectory directories and final PDBs.

    Returns {sample_idx: {"trajectory": [(step, path), ...], "final": path_or_None}}
    """
    samples: dict[int, dict] = defaultdict(lambda: {"trajectory": [], "final": None})

    # New layout: per-sample dirs *_trajectory_N
    for d in sorted(glob.glob(os.path.join(input_dir, "*_trajectory_*")), key=_numsort_key):
        if not os.path.isdir(d):
            continue
        dir_match = re.search(r"_trajectory_(\d+)$", d)
        if not dir_match:
            continue
        sample_idx = int(dir_match.group(1))
        for pdb in sorted(glob.glob(os.path.join(d, "*.pdb")), key=_numsort_key):
            step_match = re.search(r"_step_(\d+)_", os.path.basename(pdb))
            if step_match:
                samples[sample_idx]["trajectory"].append(
                    (int(step_match.group(1)), pdb)
                )

    # Legacy layout: per-step dirs trajectory_step_NNNN
    for d in sorted(glob.glob(os.path.join(input_dir, "trajectory_step_*")), key=_numsort_key):
        if not os.path.isdir(d):
            continue
        step_match = re.search(r"trajectory_step_(\d+)$", d)
        if not step_match:
            continue
        step = int(step_match.group(1))
        for pdb in sorted(glob.glob(os.path.join(d, "*.pdb")), key=_numsort_key):
            # Extract sample index from filename
            sample_match = re.search(r"sample_(\d+)", os.path.basename(pdb))
            sample_idx = int(sample_match.group(1)) if sample_match else 0
            samples[sample_idx]["trajectory"].append((step, pdb))

    # Final PDBs (in job_*_n_*_id_* subdirectories)
    final_pdbs = []
    for dirpath, _, filenames in os.walk(input_dir):
        if "_trajectory_" in dirpath or "trajectory_step_" in dirpath:
            continue
        if "eval_tmp" in dirpath:
            continue
        for f in filenames:
            if f.endswith(".pdb"):
                final_pdbs.append(os.path.join(dirpath, f))

    # Match finals to trajectory samples by protein length.
    # Extract length from path (_n_<num>_) or by loading the PDB.
    def _get_length(path: str) -> int:
        m = re.search(r"_n_(\d+)[_./]", path)
        if m:
            return int(m.group(1))
        # Fallback: load PDB and count residues
        return len(extract_ca_coords(path))

    finals_by_length: dict[int, list[str]] = defaultdict(list)
    for pdb in sorted(final_pdbs, key=_numsort_key):
        finals_by_length[_get_length(pdb)].append(pdb)

    assigned_by_length: dict[int, int] = defaultdict(int)
    for sample_idx in sorted(samples):
        traj = samples[sample_idx]["trajectory"]
        if not traj:
            continue
        length = _get_length(traj[0][1])
        idx = assigned_by_length[length]
        if idx < len(finals_by_length.get(length, [])):
            samples[sample_idx]["final"] = finals_by_length[length][idx]
            assigned_by_length[length] += 1

    # If no trajectory samples were matched to finals, assign by order
    if not any(s["final"] for s in samples.values()):
        for i, pdb in enumerate(sorted(final_pdbs, key=_numsort_key)):
            samples[i]["final"] = pdb

    # Sort trajectories by step
    for s in samples.values():
        s["trajectory"].sort(key=lambda x: x[0])

    return dict(samples)


def evaluate_sample(sample_idx: int, sample_info: dict, pmpnn_model=None) -> list[dict]:
    """Evaluate one sample's trajectory + final structure.

    Args:
        sample_idx: Index of the sample.
        sample_info: Dict with "trajectory" and "final" keys.
        pmpnn_model: Pre-loaded ProteinMPNN model for backbone perplexity.
            If None, perplexity is skipped.
    """
    results = []
    trajectory = sample_info["trajectory"]
    final_path = sample_info["final"]

    # Load final CA coords for RMSD-to-final
    final_ca = None
    if final_path:
        final_ca = extract_ca_coords(final_path)

    prev_ca = None
    all_paths = [(step, path) for step, path in trajectory]
    if final_path:
        max_step = max(s for s, _ in trajectory) if trajectory else 0
        all_paths.append((max_step, final_path))

    for step, pdb_path in all_paths:
        ca = extract_ca_coords(pdb_path)
        is_final = pdb_path == final_path
        label = "final" if is_final else f"step_{step:04d}"

        row = {
            "sample": sample_idx,
            "step": step,
            "label": label,
            "pdb_path": pdb_path,
            "length": len(ca),
            "radius_of_gyration": radius_of_gyration(ca),
        }

        # RMSD vs final
        if is_final:
            row["rmsd_to_final"] = 0.0
        elif final_ca is not None:
            try:
                row["rmsd_to_final"] = kabsch_rmsd(ca, final_ca)
            except Exception:
                row["rmsd_to_final"] = float("nan")
        else:
            row["rmsd_to_final"] = float("nan")

        # RMSD vs previous step
        if prev_ca is not None and len(ca) == len(prev_ca):
            row["rmsd_to_prev"] = kabsch_rmsd(ca, prev_ca)
        else:
            row["rmsd_to_prev"] = float("nan")

        # Secondary structure
        ss = secondary_structure_fractions(ca)
        row["helix_frac"] = ss["helix_frac"]
        row["sheet_frac"] = ss["sheet_frac"]

        # AA logit statistics (if saved alongside PDB)
        logits_path = pdb_path.replace(".pdb", "_aa_logits.npy")
        if os.path.exists(logits_path):
            logits = np.load(logits_path)  # [n, 20]
            probs = _softmax(logits)  # [n, 20]
            # Entropy: how uncertain the AA prediction is (max = log(20) ≈ 3.0)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)  # [n]
            row["aa_entropy_mean"] = float(np.mean(entropy))
            # Max probability: how confident the top AA pick is
            row["aa_max_prob_mean"] = float(np.mean(np.max(probs, axis=-1)))
        else:
            row["aa_entropy_mean"] = float("nan")
            row["aa_max_prob_mean"] = float("nan")

        # Backbone perplexity via ProteinMPNN (if model provided)
        if pmpnn_model is not None:
            try:
                from proteinfoundation.metrics.backbone_perplexity import (
                    backbone_perplexity,
                )
                bp = backbone_perplexity(pdb_path, model=pmpnn_model)
                row["bb_perplexity"] = bp["perplexity"]
                row["bb_mean_nll"] = bp["mean_nll"]
            except Exception as e:
                logger.warning(f"Perplexity failed for {pdb_path}: {e}")
                row["bb_perplexity"] = float("nan")
                row["bb_mean_nll"] = float("nan")
        else:
            row["bb_perplexity"] = float("nan")
            row["bb_mean_nll"] = float("nan")

        results.append(row)
        prev_ca = ca

    return results


def main():
    parser = argparse.ArgumentParser(description="Fast trajectory evaluation (no GPU)")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Generation output dir with trajectory and final PDBs",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path for output CSV (default: <input_dir>/trajectory_fast_eval.csv)",
    )
    parser.add_argument(
        "--convergence_threshold",
        type=float,
        default=0.5,
        help="RMSD threshold (Angstroms) for convergence detection (default: 0.5)",
    )
    parser.add_argument(
        "--perplexity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute ProteinMPNN backbone perplexity. Default: auto (on if GPU available, off otherwise). Use --no-perplexity to force off.",
    )
    parser.add_argument(
        "--perplexity_device",
        type=str,
        default=None,
        help="Device for ProteinMPNN backbone perplexity (default: auto-detect)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_csv = args.output_csv or os.path.join(input_dir, "trajectory_fast_eval.csv")

    samples = discover_samples(input_dir)
    if not samples:
        logger.error(f"No PDB files found in {input_dir}")
        sys.exit(1)

    total_pdbs = sum(
        len(s["trajectory"]) + (1 if s["final"] else 0) for s in samples.values()
    )
    logger.info(f"Found {len(samples)} samples with {total_pdbs} total PDBs")

    # Auto-detect: enable perplexity by default if a GPU is available
    import torch
    has_gpu = torch.cuda.is_available()
    run_perplexity = args.perplexity if args.perplexity is not None else has_gpu
    perplexity_device = args.perplexity_device or ("cuda" if has_gpu else "cpu")

    pmpnn_model = None
    if run_perplexity:
        from proteinfoundation.metrics.backbone_perplexity import load_proteinmpnn
        logger.info(f"Loading ProteinMPNN for backbone perplexity (device={perplexity_device})")
        pmpnn_model = load_proteinmpnn(device=perplexity_device)
        logger.info("ProteinMPNN loaded.")

    all_results = []
    for sample_idx in sorted(samples):
        info = samples[sample_idx]
        logger.info(
            f"  Sample {sample_idx}: {len(info['trajectory'])} trajectory steps"
            + (f" + final" if info["final"] else "")
        )
        all_results.extend(evaluate_sample(sample_idx, info, pmpnn_model=pmpnn_model))

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")

    # Print summary table per sample
    has_logits = "aa_entropy_mean" in df.columns and df["aa_entropy_mean"].notna().any()
    has_perplexity = "bb_perplexity" in df.columns and df["bb_perplexity"].notna().any()

    print("\n" + "=" * 130)
    print("FAST TRAJECTORY EVALUATION")
    print("=" * 130)

    for sample_idx in sorted(samples):
        sdf = df[df["sample"] == sample_idx]
        n = sdf["length"].iloc[0]
        print(f"\nSample {sample_idx} (n={n}):")
        header = (
            f"  {'Step':>6}  {'Label':<12}  {'RMSD->final':>12}  "
            f"{'RMSD->prev':>12}  {'Rg':>8}  {'Helix%':>7}  {'Sheet%':>7}"
        )
        if has_logits:
            header += f"  {'AA Entropy':>11}  {'AA MaxProb':>11}"
        if has_perplexity:
            header += f"  {'BB Perpl':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in sdf.iterrows():
            rmsd_final = f"{row['rmsd_to_final']:>12.3f}" if not np.isnan(row["rmsd_to_final"]) else f"{'--':>12}"
            rmsd_prev = f"{row['rmsd_to_prev']:>12.3f}" if not np.isnan(row["rmsd_to_prev"]) else f"{'--':>12}"
            line = (
                f"  {row['step']:>6}  {row['label']:<12}  {rmsd_final}  "
                f"{rmsd_prev}  {row['radius_of_gyration']:>8.2f}  "
                f"{row['helix_frac']:>6.1%}  {row['sheet_frac']:>6.1%}"
            )
            if has_logits:
                aa_ent = f"{row['aa_entropy_mean']:>11.3f}" if not np.isnan(row.get("aa_entropy_mean", float("nan"))) else f"{'--':>11}"
                aa_max = f"{row['aa_max_prob_mean']:>11.3f}" if not np.isnan(row.get("aa_max_prob_mean", float("nan"))) else f"{'--':>11}"
                line += f"  {aa_ent}  {aa_max}"
            if has_perplexity:
                bb_ppl = f"{row['bb_perplexity']:>9.2f}" if not np.isnan(row.get("bb_perplexity", float("nan"))) else f"{'--':>9}"
                line += f"  {bb_ppl}"
            print(line)

    # Convergence analysis
    traj_df = df[df["label"] != "final"].copy()
    thresholds = [1.0, 0.5, 0.1]
    primary_threshold = args.convergence_threshold

    def find_convergence_step(rmsd_series, threshold):
        """Find first step where rmsd_to_prev < threshold and stays below for all subsequent steps."""
        values = rmsd_series.values
        for i in range(len(values)):
            if np.isnan(values[i]):
                continue
            if all(v < threshold for v in values[i:] if not np.isnan(v)):
                return i
        return None

    print("\n" + "-" * 110)
    print("CONVERGENCE SUMMARY")
    print("-" * 110)

    if not traj_df.empty:
        max_step = int(traj_df["step"].max())

        # Per-sample convergence
        for sample_idx in sorted(samples):
            sdf = traj_df[traj_df["sample"] == sample_idx].sort_values("step")
            if sdf.empty:
                continue
            n = sdf["length"].iloc[0]
            first_rmsd = sdf.iloc[0]["rmsd_to_final"]
            last_rmsd = sdf.iloc[-1]["rmsd_to_final"]
            r_first = f"{first_rmsd:>8.3f}" if not np.isnan(first_rmsd) else f"{'--':>8}"
            r_last = f"{last_rmsd:>8.3f}" if not np.isnan(last_rmsd) else f"{'--':>8}"

            conv_parts = []
            for t in thresholds:
                idx = find_convergence_step(sdf["rmsd_to_prev"], t)
                if idx is not None:
                    conv_step = int(sdf.iloc[idx]["step"])
                    pct = (max_step - conv_step) / max_step * 100
                    conv_parts.append(f"<{t}A @ step {conv_step:>4}")
                else:
                    conv_parts.append(f"<{t}A @ {'--':>7}")

            print(
                f"  Sample {sample_idx:>3} (n={n:>3}):  "
                f"RMSD->final {r_first} -> {r_last}    "
                f"{' | '.join(conv_parts)}"
            )

        # Aggregate adaptive stopping potential
        print("\n" + "-" * 110)
        print("ADAPTIVE STOPPING POTENTIAL")
        print("-" * 110)

        for t in thresholds:
            conv_steps = []
            for sample_idx in sorted(samples):
                sdf = traj_df[traj_df["sample"] == sample_idx].sort_values("step")
                if sdf.empty:
                    continue
                idx = find_convergence_step(sdf["rmsd_to_prev"], t)
                if idx is not None:
                    conv_steps.append(int(sdf.iloc[idx]["step"]))

            if conv_steps:
                median_step = int(np.median(conv_steps))
                savings = (max_step - median_step) / max_step * 100
                n_converged = len(conv_steps)
                n_total = len(samples)
                marker = "  <--" if t == primary_threshold else ""
                print(
                    f"  RMSD < {t:.1f} A:  "
                    f"median convergence at step {median_step:>4}/{max_step} "
                    f"({savings:>4.0f}% savings)  "
                    f"[{n_converged}/{n_total} samples converged]{marker}"
                )
            else:
                print(f"  RMSD < {t:.1f} A:  no samples converged")

    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
