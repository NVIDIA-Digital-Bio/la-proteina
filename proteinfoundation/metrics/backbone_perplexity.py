"""ProteinMPNN backbone perplexity: a fast, cheap proxy for designability.

Backbone perplexity measures how well the backbone geometry constrains amino
acid identity. ProteinMPNN's `unconditional_probs` mode computes p(s_i | backbone)
for all 20 standard amino acids at each position in a single forward pass (no
sequence context, just backbone geometry).

Low perplexity (2-5) = backbone strongly constrains AA choice = likely designable.
High perplexity (>10) = backbone is ambiguous or unphysical = likely not designable.

Runs in ~0.1-1s per structure on CPU. Compare to full designability eval
(ProteinMPNN + ESMFold + scRMSD) which takes 5-30 min per structure on GPU.

Usage:
    from proteinfoundation.metrics.backbone_perplexity import (
        load_proteinmpnn, backbone_perplexity,
    )
    model = load_proteinmpnn(device="cpu")
    result = backbone_perplexity("path/to/protein.pdb", model=model)
    print(f"Perplexity: {result['perplexity']:.2f}")
"""

import math
import os
import sys

import numpy as np
import torch

# ProteinMPNN lives at <repo_root>/ProteinMPNN/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PMPNN_DIR = os.path.join(_ROOT, "ProteinMPNN")


def _ensure_pmpnn_importable():
    if _PMPNN_DIR not in sys.path:
        sys.path.insert(0, _PMPNN_DIR)


def load_proteinmpnn(model_name="v_48_020", device="cpu", ca_only=True):
    """Load a ProteinMPNN model for backbone perplexity scoring.

    Args:
        model_name: Weight file stem (default v_48_020, the 0.20A noise model).
        device: torch device string.
        ca_only: If True, use CA-only model (recommended for generated backbones
            that may lack full backbone atoms).

    Returns:
        ProteinMPNN model in eval mode on the specified device.
    """
    _ensure_pmpnn_importable()
    from protein_mpnn_utils import ProteinMPNN as PMPNNModel

    weight_dir = "ca_model_weights" if ca_only else "vanilla_model_weights"
    ckpt_path = os.path.join(_PMPNN_DIR, weight_dir, f"{model_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"ProteinMPNN weights not found at {ckpt_path}. "
            "Run script_utils/download_pmpnn_weights.sh first."
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = PMPNNModel(
        ca_only=ca_only,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,  # no noise at inference
        k_neighbors=checkpoint["num_edges"],
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def featurize_pdb(pdb_path, device="cpu", ca_only=True):
    """Parse a PDB and return tensors needed by ProteinMPNN.unconditional_probs.

    Returns:
        (X, mask, residue_idx, chain_encoding_all) tensors.
    """
    _ensure_pmpnn_importable()
    from protein_mpnn_utils import parse_PDB, tied_featurize

    pdb_dict_list = parse_PDB(pdb_path, ca_only=ca_only)

    # All chains are "designed" (masked); no fixed chains.
    all_chains = [
        item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == "seq_chain"
    ]
    chain_dict = {pdb_dict_list[0]["name"]: (all_chains, [])}

    # tied_featurize returns a 20-element tuple; we need elements 0, 2, 5, 12.
    result = tied_featurize(
        batch=pdb_dict_list,
        device=device,
        chain_dict=chain_dict,
        ca_only=ca_only,
    )
    X = result[0]                   # [1, L, 3] for ca_only
    mask = result[2]                # [1, L]
    chain_encoding_all = result[5]  # [1, L]
    residue_idx = result[12]        # [1, L]

    return X, mask, residue_idx, chain_encoding_all


def backbone_perplexity(pdb_path, model=None, device="cpu", ca_only=True):
    """Compute backbone perplexity for a PDB file.

    Args:
        pdb_path: Path to PDB file.
        model: Pre-loaded ProteinMPNN model (from load_proteinmpnn).
            If None, loads a fresh model (slow; prefer pre-loading).
        device: torch device string.
        ca_only: Use CA-only features.

    Returns:
        dict with:
            mean_nll: Mean negative log-likelihood of the best AA per position.
            perplexity: exp(mean_nll).
            per_residue_nll: List of per-residue NLL values.
            length: Number of residues.
    """
    if model is None:
        model = load_proteinmpnn(device=device, ca_only=ca_only)

    # Infer device from model parameters so inputs land on the same device
    model_device = next(model.parameters()).device
    X, mask, residue_idx, chain_encoding_all = featurize_pdb(
        pdb_path, device=model_device, ca_only=ca_only
    )

    with torch.no_grad():
        log_probs = model.unconditional_probs(X, mask, residue_idx, chain_encoding_all)

    # log_probs: [1, L_padded, 21] (21 = 20 standard AAs + X)
    L = int(mask.sum().item())

    # NLL of the most likely AA at each position (exclude X at index 20)
    log_probs_aa = log_probs[0, :L, :20]  # [L, 20]
    max_log_probs = log_probs_aa.max(dim=-1).values  # [L]
    per_residue_nll = (-max_log_probs).cpu().tolist()

    mean_nll = float(np.mean(per_residue_nll))
    perplexity = float(math.exp(mean_nll))

    return {
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "per_residue_nll": per_residue_nll,
        "length": L,
    }
