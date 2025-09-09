# La-Proteina Overview

## Description:  
La-Proteina is a state-of-the-art generative model that designs fully atomistic protein structures, generating both the sequence and all atomic coordinates. It is trained using a partially latent flow matching objective, where the protein backbone is modeled explicitly while side-chain details and sequence information are captured in a fixed-size latent space. New proteins are generated iteratively starting from random noise, using stochastic sampling. This framework enables a protein designer to generate novel, fully atomistic protein structures unconditionally or to perform complex conditional tasks like atomistic motif scaffolding, where the model can build a protein around a predefined functional site in both indexed and unindexed setups.

This model is ready for commercial/non-commercial use.


### License/Terms of Use:   
GOVERNING TERMS: Use of the model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

### Deployment Geography:  
Global

### Use Case:  
La-Proteina can be used by protein designers interested in generating novel fully atomistic protein structures and their corresponding sequences.

### Release Date:
**Github** [09/10/2025] via [https://github.com/NVIDIA-Digital-Bio/la-proteina]

**NGC** [09/10/2025] via [https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/collections/laproteina_weights_data]

## Reference(s):
https://arxiv.org/pdf/2507.09466

## Model Architecture:
**Architecture Type:** Autoencoder + Flow model.
**Network Architecture:** Transformer.

La-Proteina uses three neural networks, an encoder, a decoder, and a denoiser, all of which share a core non-equivariant transformer architecture with pair-biased attention mechanisms. For refining the pair representation, optional triangle multiplicative layers can be included within the denoiser network. The architecture operates on a partially latent representation, epxlicitly modeling the protein's three-dimensional alpha-carbon coordinates while capturing the sequence and all other atomistic details in per-residue eight-dimensional latent variables. The denoiser network parametrizes the flow that maps a noise distribution to the joint distribution of alpha-carbon coordinates and latent variables, which are iteratively updated during the generation process. The decoder then generates the final fully atomistic structure from these outputs.
 
**This model was developed based on:**  Proteina (https://github.com/NVIDIA-Digital-Bio/proteina).
**Number of model parameters:** 4.2*10^8
 
## Input:  
**Input Type(s):**

- Text (time step schedules, noise schedules, sampling modes, motif coordinates) <br>

- Number (number of residues, noise scales, time step sizes, seed, noise schedule exponents) <br>

- Binary (use of self-conditioning) <br>

**Input Format(s):** 

- Text: Strings (time step schedules, noise schedules, sampling modes), PDB file (motif coordinates) <br>

- Number: Integers (number of residues, seed), floats (noise scales, time step sizes, noise schedule exponents) <br>

- Binary: Booleans <br>

**Input Parameters:** 

- Text: One-dimensional (1D) or text file (PDB file)

- Number: One-dimensional (1D)

- Binary: One-dimensional (1D)

**Other Properties Related to Input:** All inputs are handled and specified in the config yaml files, see README. 

## Output: <br>
**Output Type(s):** Text (generated atomistic coordinates and sequence) <br>

**Output Format:** Text: PDB file (generated protein with sequence and all atom coordinates) <br>

**Output Parameters**: One-dimensional (1D)

**Other Properties Related to Output:** The model output is stored as a PDB file containing the protein sequence and three-dimensional coordinates for all atoms.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems NVIDIA A100 or equivalent. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:  
**Runtime Engine(s):** Pytorch.

**Supported Hardware Microarchitecture Compatibility:**  <br>
NVIDIA Ampere (tested on A100) <br>

**[Preferred/Supported] Operating System(s):** <br>
Linux <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s): 
We release ten model checkpoints, seven for latent diffusion models and three for the corresponding autoencoders.

Latent diffusion checkpoints (used together with the corresponding autoencoder below):
- LaProteinaDiff v1.1 (unconditional generation up to 500 residues, no triangle layers). Used with LaProteinaAE v1.1.
- LaProteinaDiff v1.2 (unconditional generation up to 500 residues, triangle layers). Used with LaProteinaAE v1.1.
- LaProteinaDiff v1.3 (unconditional generation between 300 and 800 residues, no triangle layers). Used with LaProteinaAE v1.2.
- LaProteinaDiff v1.4 (indexed, all-atom atomistic motif scaffolding). Used with LaProteinaAE v1.3.
- LaProteinaDiff v1.5 (indexed, tip-atom atomistic motif scaffolding). Used with LaProteinaAE v1.3.
- LaProteinaDiff v1.6 (unindexed, all-atom atomistic motif scaffolding). Used with LaProteinaAE v1.3.
- LaProteinaDiff v1.7 (unindexed, tip-atom atomistic motif scaffolding). Used with LaProteinaAE v1.3.

Autoencoder checkpoins:
- LaProteinaAE v1.1 (trained with proteins up to 512 residues).
- LaProteinaAE v1.2 (trained with proteins up to 896 residues).
- LaProteinaAE v1.3 (trained with proteins up to 256 residues).
 

## Training, Testing, and Evaluation Datasets:

For additional information regarding the datasets, please see the paper [here](https://arxiv.org/pdf/2507.09466).

### Dataset Overview: 
**Total Size:** Approximately 46,000,000 data points  
**Total Number of Datasets:** 2 (AFDB and PDB)
**Dataset partition:** Training 99.9%%, Validation 0.1%%  
**Time period for training data collection:** The AFDB was generated using Alpha Fold 2, published in 2021, the PDB consists of experimental data, which started being collected on 1971.
**Time period for testing data collection:** See above.
**Time period for validation data collection:** See above.


## Training Dataset:
**Link:** https://alphafold.ebi.ac.uk/ 

**Data Modality:**
* Text (PDB files) 
 
**Non-Audio, Image, Text Training Data Size:**  
* Between 0.5MB and 2MB per sample (PDB file)

**Data Collection Method by dataset:**  
Synthetic (AlphaFold predictions)

**Labeling Method by dataset:** N/A

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**  The AlphaFold Protein Structure Database (AFDB) contains approximately 214M synthetic three-dimensional protein structures predicted by AlphaFold2, along with their corresponding sequences. We trained La-Proteina models on two different filtered subsets of the AFDB, one comprising 344.508 structures, the other one comprising 46.942.694 structures.

## Testing Dataset:

(1) AFDB
**Link:** https://alphafold.ebi.ac.uk/ 

**Data Modality:**
* Text (PDB files) 
 
**Non-Audio, Image, Text Training Data Size:**  
* Between 0.5MB and 2MB per sample (PDB file)

**Data Collection Method by dataset:**  
Synthetic (AlphaFold predictions)

**Labeling Method by dataset:** N/A

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**  We use a subset of the AFDB subset of 734,658 structures as a reference set in evaluations. 

(2) PDB
**Link:** https://www.rcsb.org/

**Data Modality:**
* Text (PDB files) 
 
**Non-Audio, Image, Text Training Data Size:**  
* Between 0.5MB and 2MB per sample (PDB file)

**Data Collection Method by dataset:**  
Synthetic (AlphaFold predictions)

**Labeling Method by dataset:** N/A

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**  We use the entire PDB as reference set in evaluations. We also extract 19 files, and modify them to create the benchmark for motif scaffolding. We release these modified files with the codebase.

## Evaluation Dataset:

**Link:** https://alphafold.ebi.ac.uk/ 

**Data Modality:**
* Text (PDB files) 
 
**Non-Audio, Image, Text Training Data Size:**  
* Between 0.5MB and 2MB per sample (PDB file)

**Data Collection Method by dataset:**  
Synthetic (AlphaFold predictions)

**Labeling Method by dataset:** N/A

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**  We use a subset of the AFDB subset of 1,300 structures as a validation set during training.

Extensive benchmarks and evaluations can be found in the associated paper, https://arxiv.org/pdf/2507.09466.

## Inference: 
**Acceleration Engine:** Pytorch <br>
**Test Hardware:** A100 <br>

**Test Hardware:** NVIDIA A100

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  

For more detailed information on ethical considerations for this model, please see the Model Card++ Bias, Explainability, Safety & Security, and Privacy Subcards.

Users are responsible for ensuring the physical properties of model-generated molecules are appropriately evaluated and comply with applicable safety regulations and ethical standards.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).