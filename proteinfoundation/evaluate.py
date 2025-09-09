import os
import sys
from typing import Dict, List, Tuple

root = os.path.abspath(".")
sys.path.insert(0, root)  # Adds project's root directory
# isort: split


import pandas as pd
import torch
from biotite.structure.io import load_structure
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf

from proteinfoundation.generate import parse_args_and_cfg, setup
from proteinfoundation.metrics.designability import (
    extract_seq_from_pdb,
    rmsd_metric,
    sc_sequence_recovery,
    scRMSD,
)
from proteinfoundation.utils.motif_utils import (
    extract_motif_from_pdb,
    pad_motif_to_full_length,
    pad_motif_to_full_length_unindexed,
)
from proteinfoundation.utils.pdb_utils import load_pdb


def parse_cfg_for_table(cfg: Dict) -> Tuple[List[str], Dict]:
    """
    Flatten config and uses it to initialize results dataframes columns.

    Returns:
        2-tuple, with the columns (list of strings) and the flattened dictionary.
    """
    flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    flat_dict = pd.json_normalize(flat_cfg, sep="_").to_dict(orient="records")[0]
    flat_dict = {k: str(v) for k, v in flat_dict.items()}
    columns = list(flat_dict.keys())
    # if present, remove columns containing generation_dataset or generation_metric
    columns = [col for col in columns if "generation_dataset" not in col and "generation_metric" not in col]
    # do the same for the keys in flat_dict
    flat_dict = {k: v for k, v in flat_dict.items() if "generation_dataset" not in k and "generation_metric" not in k}
    return columns, flat_dict


def split_by_job(cfg: Dict, job_id: int, is_des: bool = True) -> List[str]:
    """
    Split evaluation jobs by job id.
    For designability, select files starting with `job_{job_id}_`, as each eval job will start after the corresponding generation job finishes
    For FID, uniformly assign files to each job. We ususally only use 1 eval job for FID.

    Returns:
        List of paths to where PDBs are stored (each PDB is at a different path).
    """
    if is_des:
        sample_root_paths = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.startswith(f"job_{job_id}_") and file.endswith(".pdb"):
                    sample_root_paths.append(os.path.join(root, file))

        logger.info(
            f"Job id {job_id} for designability or novelty evaluation for {len(sample_root_paths)} files starting with `job_{job_id}_`"
        )
    else:
        raise NotImplementedError("New metrics not implemented.")
    return sample_root_paths


def compute_traditional_metrics(
    cfg_metric: Dict, samples_paths: List[str], job_id: int, ncpus: int
) -> pd.DataFrame:
    """
    Given a path where samples are stored and the job ID, it computes the metrics requested, and returns the values in
    a pandas dataframe.

    Args:
        cfg_metric: Dict with the configuration for the metrics. Configurable options include:
            - designability_modes: List of RMSD modes for designability evaluation 
              (default: ["ca"]. Options: "ca", "bb3o", "all_atom")
            - codesignability_modes: List of RMSD modes for codesignability evaluation
              (default: ["ca", "bb3o", "all_atom"]. Options: "ca", "bb3o", "all_atom")
            - designability_motif_eval: Enable motif-specific designability evaluation
            - codesignability_motif_eval: Enable motif-specific codesignability evaluation
            - compute_motif_rmsd: Direct comparison between generated structure and ground truth motif
            - designability_folding_models: List of folding models for designability (default: ["esmfold"])
            - codesignability_folding_models: List of folding models for codesignability (default: ["esmfold"])  
        samples_paths: List of paths, one for each PDB that should be evaluated.
        job_id: Job id for the evaluation.
        ncpus: Number of CPUs to use.

    Returns:
        Pandas dataframe with values for traditional metrics.
        
    Note:
        Supported RMSD modes:
        - "ca": CA atoms only
        - "bb3o": Backbone atoms (N, CA, C, O)
        - "all_atom": All available atoms
        
        When motif evaluation is enabled, additional metrics are computed that focus
        only on the motif region:
        - Direct motif RMSD: Compares generated structure directly against ground truth motif
        - Designability motif metrics: Uses ProteinMPNN + folding models, evaluated on motif region
        - Codesignability motif metrics: Uses ground truth sequence + folding models, evaluated on motif region
    """
    columns, flat_dict = parse_cfg_for_table(cfg)
    # Add some columns to store per-sample results
    columns += ["id_gen", "pdb_path", "L"]

    # Configure evaluation modes and models
    designability_modes = cfg_metric.get("designability_modes", ["ca"])
    designability_folding_models = cfg_metric.get("designability_folding_models", ["esmfold"])
    designability_motif_eval = cfg_metric.get("designability_motif_eval", False)

    codesignability_modes = cfg_metric.get("codesignability_modes", ["ca", "bb3o", "all_atom"])
    codesignability_folding_models = cfg_metric.get("codesignability_folding_models", ["esmfold"])
    codesignability_motif_eval = cfg_metric.get("codesignability_motif_eval", False)

    # Check if any motif evaluation is needed
    is_motif_task = "motif_task_name" in cfg.generation.dataset
    needs_motif_setup = (
        is_motif_task and 
        (designability_motif_eval or codesignability_motif_eval or cfg_metric.get("compute_motif_rmsd", False))
    )

    metrics = {}
    
    # Standard designability metrics
    if cfg_metric.compute_designability:
        for model in designability_folding_models:
            for mode in designability_modes:
                metrics[f"_res_scRMSD_{mode}_{model}"] = []
                metrics[f"_res_scRMSD_all_{mode}_{model}"] = []

    # Standard codesignability metrics
    if cfg_metric.compute_codesignability:
        for model in codesignability_folding_models:
            for m in codesignability_modes:
                metrics[f"_res_co_scRMSD_{m}_{model}"] = []
                metrics[f"_res_co_scRMSD_all_{m}_{model}"] = []

    if cfg_metric.compute_co_sequence_recovery:
        metrics["_res_co_seq_rec"] = []
        metrics["_res_co_seq_rec_all"] = []

    # Motif-specific metrics
    if needs_motif_setup:
        # Direct motif RMSD metrics (generated structure vs ground truth motif)
        if cfg_metric.get("compute_motif_rmsd", True):
            # Use custom motif_rmsd_modes if specified, otherwise fall back to designability_modes
            motif_rmsd_modes = cfg_metric.get("motif_rmsd_modes", designability_modes)
            for m in motif_rmsd_modes:
                metrics[f"_res_motif_rmsd_{m}"] = []
            metrics[f"_res_motif_seq_rec"] = []

        # Designability motif metrics
        if designability_motif_eval:
            for model in designability_folding_models:
                for m in designability_modes:
                    metrics[f"_res_des_motif_scRMSD_{m}_{model}"] = []
            for model in designability_folding_models:
                metrics[f"_res_des_motif_seq_rec_{model}"] = []

        # Codesignability motif metrics  
        if codesignability_motif_eval:
            for model in codesignability_folding_models:
                for m in codesignability_modes:
                    metrics[f"_res_co_motif_scRMSD_{m}_{model}"] = []
            for model in codesignability_folding_models:
                metrics[f"_res_co_motif_seq_rec_{model}"] = []

        # Setup motif data
        motif_task_name = cfg.generation.dataset.motif_task_name
        motif_cfg = cfg.generation.dataset.motif_dict_cfg[motif_task_name]
        
        # Use 'all_atom' atom selection mode to ensure all atoms are available for any RMSD mode
        # The specific RMSD computation will then use the appropriate atoms based on rmsd_modes
        motif_mask, x_motif, residue_type = extract_motif_from_pdb(
            motif_cfg.contig_string,
            motif_cfg.motif_pdb_path,
            motif_only=motif_cfg.motif_only,
            atom_selection_mode="all_atom",
            coors_to_nm=False,
        )
        motif_csv = f"{motif_task_name}_{job_id}_motif_info.csv"
        motif_csv = os.path.join(root_path, motif_csv)
        motif_info = pd.read_csv(motif_csv)

    results = []

    for i, pdb_path in enumerate(samples_paths):
        seq = extract_seq_from_pdb(pdb_path)
        n = len(seq)

        res_row = list(flat_dict.values()) + [i, pdb_path, n]
        results.append(res_row)

        # create tmp_dir for this sample
        tmp_dir = os.path.splitext(pdb_path)[0]  # removes extension ".pdb"
        assert not os.path.exists(tmp_dir), f"tmp_dir {tmp_dir} already exists"
        os.makedirs(tmp_dir, exist_ok=False)

        # Initialize motif-related variables
        motif_index = None
        motif_residue_indices = None
        
        if needs_motif_setup:
            sample_id = int(os.path.basename(pdb_path).split("_")[3])
            contig_string = motif_info[motif_info["sample_num"] == sample_id][
                "contig"
            ].values[0]

            gen_prot = load_pdb(pdb_path)
            gen_coors = torch.Tensor(gen_prot.atom_positions)
            gen_mask = torch.Tensor(gen_prot.atom_mask).bool()
            gen_aa_type = torch.Tensor(gen_prot.aatype)

            #######################################################
            # Some manual changes needed here for the motif task
            #######################################################

            # This is for indexed models
            motif_mask_full, x_motif_full, residue_type_full = pad_motif_to_full_length(
                motif_mask, x_motif, residue_type, contig_string
            )

            # # This is for unindexed models
            # motif_mask_full, x_motif_full, residue_type_full = pad_motif_to_full_length_unindexed(
            #     motif_mask=motif_mask,
            #     x_motif=x_motif,
            #     residue_type=residue_type,
            #     gen_coors=gen_coors,
            #     gen_mask=gen_mask,
            #     gen_aa_type=gen_aa_type,
            # )

            #######################################################
            #######################################################
            #######################################################

            # Get motif index for ProteinMPNN fixing and sequence recovery
            from openfold.np.residue_constants import restype_num, restype_order

            gen_residue_type = torch.as_tensor(
                [restype_order.get(r, restype_num) for r in seq]
            )
            logger.info(f"Gen residue type: {gen_residue_type.shape}")
            motif_sequence_mask = motif_mask_full.any(dim=1)

            motif_index = []
            motif_residue_indices = []
            for i in motif_sequence_mask.nonzero():
                # Convert to 1-indexed for ProteinMPNN (PDB residue numbering)
                motif_index.append(f"A{i.item() + 1}")
                # Keep 0-indexed for tensor operations
                motif_residue_indices.append(i.item())

            # Direct motif RMSD computation (generated structure vs ground truth motif)
            if cfg_metric.get("compute_motif_rmsd", True):
                for m in motif_rmsd_modes:
                    metrics[f"_res_motif_rmsd_{m}"].append(
                        rmsd_metric(
                            coors_1_atom37=gen_coors,
                            coors_2_atom37=x_motif_full,
                            mask_atom_37=gen_mask * motif_mask_full,
                            mode=m,
                        )
                    )
                
                # Direct motif sequence recovery computation
                is_same_motif_residue = (gen_residue_type == residue_type_full)[motif_sequence_mask]
                metrics["_res_motif_seq_rec"].append(is_same_motif_residue.float().mean().item())

        # Designability evaluation
        if cfg_metric.compute_designability:
            # Use unified scRMSD function that computes both normal and motif RMSD when needed
            res_designability = scRMSD(
                pdb_file_path=pdb_path,
                ret_min=False,
                tmp_path=tmp_dir,
                use_pdb_seq=False,
                rmsd_modes=designability_modes,
                motif_index=motif_index,  # Fix motif positions if in motif task
                motif_residue_indices=motif_residue_indices if (designability_motif_eval and needs_motif_setup) else None,
                folding_models=designability_folding_models,
                keep_outputs=cfg_metric.get("keep_folding_outputs", False),
            )
            
            # Extract normal designability results
            for model in designability_folding_models:
                for mode in designability_modes:
                    if res_designability[mode][model]:
                        metrics[f"_res_scRMSD_{mode}_{model}"].append(
                            min(res_designability[mode][model])
                        )
                        metrics[f"_res_scRMSD_all_{mode}_{model}"].append(
                            res_designability[mode][model]
                        )
                    else:
                        metrics[f"_res_scRMSD_{mode}_{model}"].append(float("inf"))
                        metrics[f"_res_scRMSD_all_{mode}_{model}"].append([float("inf")])

            # Extract motif designability results if they were computed
            if designability_motif_eval and needs_motif_setup:
                for model in designability_folding_models:
                    for m in designability_modes:
                        motif_key = f"{m}_motif"
                        col_name = f"_res_des_motif_scRMSD_{m}_{model}"
                        if motif_key in res_designability and res_designability[motif_key][model]:
                            metrics[col_name].append(min(res_designability[motif_key][model]))
                        else:
                            metrics[col_name].append(float("inf"))

                # Designability-style sequence recovery for each model
                for model in designability_folding_models:
                    col_name = f"_res_des_motif_seq_rec_{model}"
                    is_same_motif_residue = (gen_residue_type == residue_type_full)[
                        motif_sequence_mask
                    ]
                    metrics[col_name].append(is_same_motif_residue.float().mean().item())

        # Codesignability evaluation
        if cfg_metric.compute_codesignability:
            # Use unified scRMSD function that computes both normal and motif RMSD when needed
            res_codesignability = scRMSD(
                pdb_file_path=pdb_path,
                ret_min=False,
                tmp_path=tmp_dir,
                use_pdb_seq=True,
                rmsd_modes=codesignability_modes,
                motif_index=motif_index,  # Fix motif positions if in motif task
                motif_residue_indices=motif_residue_indices if (codesignability_motif_eval and needs_motif_setup) else None,
                folding_models=codesignability_folding_models,
                keep_outputs=cfg_metric.get("keep_folding_outputs", False),
            )
            
            # Extract normal codesignability results
            for model in codesignability_folding_models:
                for m in codesignability_modes:
                    if res_codesignability[m][model]:
                        metrics[f"_res_co_scRMSD_{m}_{model}"].append(
                            min(res_codesignability[m][model])
                        )
                        metrics[f"_res_co_scRMSD_all_{m}_{model}"].append(
                            res_codesignability[m][model]
                        )
                    else:
                        metrics[f"_res_co_scRMSD_{m}_{model}"].append(float("inf"))
                        metrics[f"_res_co_scRMSD_all_{m}_{model}"].append(
                            [float("inf")]
                        )

            # Extract motif codesignability results if they were computed
            if codesignability_motif_eval and needs_motif_setup:
                for model in codesignability_folding_models:
                    for m in codesignability_modes:
                        motif_key = f"{m}_motif"
                        col_name = f"_res_co_motif_scRMSD_{m}_{model}"
                        if motif_key in res_codesignability and res_codesignability[motif_key][model]:
                            metrics[col_name].append(min(res_codesignability[motif_key][model]))
                        else:
                            metrics[col_name].append(float("inf"))

                # Codesignability-style sequence recovery for each model
                for model in codesignability_folding_models:
                    col_name = f"_res_co_motif_seq_rec_{model}"
                    is_same_motif_residue = (gen_residue_type == residue_type_full)[
                        motif_sequence_mask
                    ]
                    metrics[col_name].append(is_same_motif_residue.float().mean().item())

        if cfg_metric.compute_co_sequence_recovery:
            res_seqres = sc_sequence_recovery(
                pdb_file_path=pdb_path,
                ret_max=False,
                tmp_path=tmp_dir,
                motif_index=motif_index,  # Fix motif positions if in motif task
            )
            metrics["_res_co_seq_rec"].append(max(res_seqres))
            metrics["_res_co_seq_rec_all"].append(res_seqres)
    df = pd.DataFrame(results, columns=columns)
    for metric in metrics:
        df[metric] = metrics[metric]

    return df


if __name__ == "__main__":
    load_dotenv()
    torch.set_float32_matmul_precision("high")
    # Parse arguments, load appropriate config, and set up root path
    # (should already exist, since this happens after generation)
    args, cfg, config_name = parse_args_and_cfg()
    run_name = cfg.run_name_
    ncpus = cfg.ncpus_
    root_path = setup(
        cfg, create_root=False, config_name=config_name, job_id=args.job_id
    )

    cfg_metric = cfg.generation.metric
    
    # Code for designability
    if cfg_metric.compute_designability:
        gen_njobs = cfg.get("gen_njobs", 1)
        eval_njobs = cfg.get("eval_njobs", 1)
        assert (
            gen_njobs == eval_njobs
        ), f"The numbers of generation and evaluation jobs for traditaional metrics should be equal."
        samples_paths = split_by_job(cfg, args.job_id, is_des=True)
        df = compute_traditional_metrics(cfg_metric, samples_paths, args.job_id, ncpus)
        if "motif_task_name" in cfg.generation.dataset:
            csv_filename = f"results_{config_name}_{cfg.generation.dataset.motif_task_name}_{args.job_id}.csv"
        else:
            csv_filename = f"results_{config_name}_{args.job_id}.csv"
        csv_path = os.path.join(root_path, "..", csv_filename)

    # Code for FID results
    if cfg_metric.compute_fid:
        raise NotImplementedError("New metrics not implemented.")
    df.to_csv(csv_path, index=False)
