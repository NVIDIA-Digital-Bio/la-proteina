import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np

root = os.path.abspath(".")
sys.path.insert(0, root)  # Adds project's root directory
# isort: split

import argparse
import json

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader

from proteinfoundation.datasets.gen_dataset import GenDataset
from proteinfoundation.proteina import Proteina
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb


def parse_args_and_cfg() -> Tuple[Dict, Dict, str]:
    """
    Parses command line arguments and loads the corresponding config file.

    Returns:
        Command line arguments (dict)
        Config file (dict)
        config_name (string)
    """
    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--config_name",
        type=str,
        default="inference_base",
        help="Name of the config yaml file.",
    )
    parser.add_argument(
        "--config_number", type=int, default=-1, help="Number of the config yaml file."
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="Job id for this config to determine which split to use.",
    )
    parser.add_argument(
        "--config_subdir",
        type=str,
        help="(Optional) Name of directory with config files, if not included uses base inference config.\
            Likely only used when submitting to the cluster with script.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Name of the data path",
    )
    args = parser.parse_args()
    if args.data_path is not None:
        os.environ["DATA_PATH"] = args.data_path
    # Inference config
    # If config_subdir is None then use base inference config
    # Otherwise use config_subdir/some_config
    if args.config_subdir is None:
        config_path = "../configs"
    else:
        config_path = f"../configs/{args.config_subdir}"

    with hydra.initialize(config_path, version_base=hydra.__version__):
        # If number provided use it, otherwise name
        if args.config_number != -1:
            config_name = f"inf_{args.config_number}"
        else:
            config_name = args.config_name
        cfg = hydra.compose(config_name=config_name)
        logger.info(f"Inference config {cfg}")

    return args, cfg, config_name


def setup(
    cfg: Dict, create_root: bool = True, config_name: str = ".", job_id: int = 0
) -> str:
    """
    Checks if metrics being computed are compatible, sets the right seed, and creates the root directory
    where the run will store things.

    Returns:
        Path of the root directory (string)
    """
    logger.info(" ".join(sys.argv))

    assert (
        torch.cuda.is_available()
    ), "CUDA not available"  # Needed for ESMfold and designability
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )  # Send to stdout

    assert (
        not (
            cfg.generation.metric.compute_designability
            or cfg.generation.metric.compute_novelty_pdb
            or cfg.generation.metric.compute_novelty_afdb
        )
        or not cfg.generation.metric.compute_fid
    ), "Designability/Novelty cannot be computed together with FID"

    # Set root path for this inference run
    if "motif_task_name" in cfg.generation.dataset:
        root_path = (
            f"./inference/{config_name}_{cfg.generation.dataset.motif_task_name}"
        )
    else:
        root_path = f"./inference/{config_name}"
    if create_root:
        os.makedirs(root_path, exist_ok=True)
    else:
        if not os.path.exists(root_path):
            raise ValueError("Results path %s does not exist" % root_path)

    # Set seed
    cfg.seed = cfg.seed + job_id  # Different seeds for different splits ids
    logger.info(f"Seeding everything to seed {cfg.seed}")
    L.seed_everything(cfg.seed)

    return root_path


def check_cfg_validity(cfg_data: Dict, cfg_sample_args: Dict) -> None:
    """
    Checks if guidance arguments (CFG and AG) are valid.
    """
    # Logging CFG
    if cfg_sample_args.guidance_w != 1.0:
        logger.info(
            f"Guidance is turned on with guidance weight {cfg_sample_args.guidance_w} and autoguidance ratio {cfg_sample_args.ag_ratio}."
        )
        assert (
            cfg_sample_args.ag_ratio >= 0.0 and cfg_sample_args.ag_ratio <= 1.0
        ), f"Autoguidance ratio should be between 0 and 1, but now is {cfg_sample_args.ag_ratio}."
        assert (cfg_sample_args.ag_ratio == 0.0) or (
            cfg_sample_args.ag_ckpt_path is not None
        ), f"Autoguidance checkpoint path should be provided"
    else:
        logger.info(f"Guidance is turned off.")

    # Logging conditional generation
    if cfg_sample_args.fold_cond:
        logger.info("Conditional generation is turned on.")
        assert (
            cfg_data.empirical_distribution_cfg.len_cath_code_path is not None
        ), "Empirical (len, cath_code) distribution file should be provided when using conditional generation."
    else:
        logger.info("Conditional generation is turned off.")
        assert (
            cfg_data.empirical_distribution_cfg.len_cath_code_path is None
        ), "Empirical (len, cath_code) distribution file shouldn't be provided when using unconditional generation."


def load_ag_ckpt(cfg: Dict) -> Union[None, torch.nn.Module]:
    """
    Loads the neural network for the "bad" checkpoint in autoguidance, if requested.

    Returns:
        A nn module, if autogudance enabled.
    """
    nn_ag = None
    if cfg.ag_ratio > 0 and cfg.guidance_w != 1.0:
        logger.info(
            f"Using autoguidance with guidance weight {cfg.guidance_w} and autoguidance ratio {cfg.ag_ratio} based on the checkpoint {cfg.ag_ckpt_path}"
        )
        ckpt_ag_file = cfg.ag_ckpt_path
        assert os.path.exists(ckpt_ag_file), f"Not a valid checkpoint {ckpt_ag_file}"
        model_ag = Proteina.load_from_checkpoint(ckpt_ag_file, strict=False)

        # OPTIMIZATION: Remove encoder from autoguidance model autoencoder during generation (only decoder needed)
        if model_ag.autoencoder is not None:
            logger.info(
                "Removing autoencoder encoder from autoguidance model during generation to save memory"
            )
            del model_ag.autoencoder.encoder
            model_ag.autoencoder.encoder = None

        nn_ag = model_ag.nn
    return nn_ag


def load_ckpt_n_configure_inference(cfg: Dict) -> Proteina:
    """
    Loads the model, potentially the autoguidance checkpoint as well, if requested.

    Returns:
        Model (Proteina)
    """
    # Load model from checkpoint
    ckpt_path = cfg.ckpt_path
    ckpt_file = os.path.join(ckpt_path, cfg.ckpt_name)
    logger.info(f"Using checkpoint {ckpt_file}")
    assert os.path.exists(ckpt_file), f"Not a valid checkpoint {ckpt_file}"

    model = Proteina.load_from_checkpoint(ckpt_file, strict=False, autoencoder_ckpt_path=cfg.get("autoencoder_ckpt_path", None))

    # Set inference variables and potentially load autoguidance
    nn_ag = load_ag_ckpt(cfg.generation.args)

    model.configure_inference(cfg.generation, nn_ag=nn_ag)

    return model


def split_by_job(cfg: Dict, job_id: int, njobs: int) -> Dict:
    """
    Since generation may be split across multiple jobs, this function determines how many samples are produced per job.
    Then, it sets the right value in the config dict, and returns the updated config.

    Returns:
        Config updated with the correct number of samples to generate.
    """
    nsamples = cfg.dataset.nsamples
    nsamples_per_split = (nsamples - 1) // njobs + 1
    if nsamples_per_split * job_id >= nsamples:
        logger.info(f"Job id {job_id} get 0 samples. Finishing job...")
        exit(0)
    else:
        cfg.dataset.nsamples = min(
            nsamples_per_split, nsamples - nsamples_per_split * job_id
        )
    return cfg


def binder_split_by_job(cfg: Dict, job_id: int, njobs: int) -> Dict:
    """
    Since generation may be split across multiple jobs, this function determines how many samples are produced per job.
    Then, it sets the right value in the config dict, and returns the updated config.

    Returns:
        Config updated with the correct number of samples to generate.
    """
    nsamples = cfg.dataset.nlens_cfg.random_lens[2]
    nsamples_per_split = (nsamples - 1) // njobs + 1
    if nsamples_per_split * job_id >= nsamples:
        logger.info(f"Job id {job_id} get 0 samples. Finishing job...")
        exit(0)
    else:
        cfg.dataset.nlens_cfg.random_lens[2] = min(
            nsamples_per_split, nsamples - nsamples_per_split * job_id
        )
    return cfg


def save_predictions(
    root_path: str,
    predictions: List[List[Tuple[torch.tensor]]],
    job_id: int = 0,
    chain_indexes: np.ndarray = None,
    cath_codes: List[List[List[str]]] = None,
) -> None:
    """
    Saves generated samples.

    Args:
        root_path: root directory where samples will be stored (within subdirectories)/
        predictions: List of lists of tuples. Each tuple represents a sample, has to components
            (coors [n, 37, 3], aatype [n])
        job_id: job number, used to store files.
        chain_indexes: chain indexes for each sample, used to store files.
        cath_codes: conditional sampling...
    """
    predictions = [sample for sublist in predictions for sample in sublist]
    # List[tuple] where each tuple is (coors [n, 37, 3], aatype [n])

    samples_per_length = defaultdict(int)
    for j, pred in enumerate(predictions):
        coors_atom37, residue_type = pred  # [n, 37, 3] and [n]
        n = coors_atom37.shape[-3]
        if chain_indexes:
            chain_index = chain_indexes[j].numpy()
        else:
            chain_index = None

        # Create directory where everything related to this sample will be stored
        suffix = ""
        dir_name = f"job_{job_id}_n_{n}_id_{samples_per_length[n]}{suffix}"
        samples_per_length[n] += 1
        sample_root_path = os.path.join(
            root_path, dir_name
        )
        os.makedirs(sample_root_path, exist_ok=False)

        # Save generated structure as pdb
        fname = dir_name + ".pdb"
        pdb_path = os.path.join(sample_root_path, fname)
        write_prot_to_pdb(
            prot_pos=coors_atom37.float().detach().cpu().numpy(),
            aatype=residue_type.detach().cpu().numpy(),
            file_path=pdb_path,
            chain_index=chain_index,
            overwrite=True,
            no_indexing=True,
        )



def save_motif_predictions(
    root_path: str,
    predictions: List[List[Tuple[torch.tensor]]],
    job_id: int = 0,
    motif_pdb_name: str = None,
) -> None:
    predictions = [sample for sublist in predictions for sample in sublist]
    print([(p[0].shape, p[1].shape) for p in predictions])
    samples_per_length = defaultdict(int)
    for j, pred in enumerate(predictions):
        coors_atom37, residue_type = pred  # [n, 37, 3] and [n]
        n = coors_atom37.shape[-3]
        dir_name = f"job_{job_id}_id_{j}_motif_{motif_pdb_name}"
        samples_per_length[n] += 1
        sample_root_path = os.path.join(root_path, dir_name)
        os.makedirs(sample_root_path, exist_ok=False)
        fname = dir_name + ".pdb"
        pdb_path = os.path.join(sample_root_path, fname)
        write_prot_to_pdb(
            prot_pos=coors_atom37.float().detach().cpu().numpy(),
            aatype=residue_type.detach().cpu().numpy(),
            file_path=pdb_path,
            overwrite=True,
            no_indexing=True,
        )


def main():
    load_dotenv()

    # Parse arguments, load appropriate config, and set up root path
    args, cfg, config_name = parse_args_and_cfg()
    # cfg.run_name_
    motif_cond = cfg.generation.args.get("motif_cond", False)
    target_cond = cfg.generation.args.get("target_cond", False)
    cfg.generation.args.get("multi_cond", False)
    cfg.generation.args.get("fold_cond", False)
    njobs = cfg.get("gen_njobs", 1)
    root_path = setup(
        cfg, create_root=True, config_name=config_name, job_id=args.job_id
    )

    # Exit if results from analysis already exist (assumes samples already there)
    # File to store analysis (next step, this is generate) results
    csv_filename = f"results_{config_name}_{args.job_id}.csv"
    csv_path = os.path.join(root_path, "..", csv_filename)
    # Exit if results from analysis already exist
    if os.path.exists(csv_path):
        logger.info(f"Results already exist at {csv_path}. Exiting generate.py.")
        sys.exit(0)

    cfg_gen = cfg.generation
    check_cfg_validity(cfg_gen.dataset, cfg_gen.args)

    # Load model
    model = load_ckpt_n_configure_inference(cfg)

    # Create generation dataset
    cfg_gen = split_by_job(cfg_gen, args.job_id, njobs)

    # Motif-specific dataset creation
    if motif_cond or ("motif_task_name" in cfg.generation.dataset):
        motif_csv_path = os.path.join(
            root_path,
            f"{cfg_gen.dataset.get('motif_task_name', 'motif')}_{args.job_id}_motif_info.csv",
        )
        """
        Motif Configuration Examples:
        
        The motif dataset supports two modes for specifying which atoms to include:
        
        1. **Atom-level specification** (precise control):
           motif_dict_cfg:
             my_motif:
               motif_pdb_path: "path/to/motif.pdb"
               motif_atom_spec: "A64: [O, CG]; A65: [N, CA]; A66: [CB, CD]"
               # atom_selection_mode is ignored when motif_atom_spec is provided
        
        2. **Residue/range-based specification** (automatic atom selection):
           motif_dict_cfg:
             my_motif:
               motif_pdb_path: "path/to/motif.pdb" 
               contig_string: "A1-7/A28-79"
               atom_selection_mode: "tip_atoms"  # NEW: Choose atom selection mode
               
           Available atom_selection_mode options:
           - "ca_only": Only CA atoms (default, fastest)
           - "all": All available atoms (most complete motif)
           - "backbone": Backbone atoms only (N, CA, C, O)
           - "sidechain": Sidechain atoms only
           - "tip_atoms": Tip atoms of sidechains (e.g., OH for Ser, NH2 for Arg)
           - "random": Random subset of available atoms
           
        If atom_selection_mode is not specified, defaults to "ca_only" for backward compatibility.
        """
        dataset = GenDataset(motif_csv_path=motif_csv_path, **cfg_gen.dataset)
    else:
        dataset = GenDataset(**cfg_gen.dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Sample model
    trainer = L.Trainer(accelerator="gpu", devices=1)
    predictions = trainer.predict(model, dataloader)

    chain_indexes = None

    if motif_cond or ("motif_task_name" in cfg.generation.dataset):
        save_motif_predictions(
            root_path,
            predictions,
            job_id=args.job_id,
            motif_pdb_name=cfg_gen.dataset.get("motif_task_name", None),
        )
        import shutil

        motif_csv = f"./{cfg_gen.dataset.get('motif_task_name', '')}_motif_info.csv"
        if os.path.exists(motif_csv):
            shutil.copy(motif_csv, root_path)
    else:
        save_predictions(
            root_path,
            predictions,
            job_id=args.job_id,
            chain_indexes=chain_indexes,
            cath_codes=dataset.cath_codes,
        )


if __name__ == "__main__":
    main()
