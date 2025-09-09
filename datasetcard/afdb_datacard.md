## Dataset Description:
The AlphaFold Protein Structure Database (AFDB) is a collection of protein structure predictions made using the machine learning model AlphaFold. 

We are releasing a subset of the AlphaFold Protein Structure Database. We use two subsets of the AFDB, one comprising 344,508 structures and the other one comprising 46,942,694.

This dataset is ready for commercial/non-commercial use.

## Dataset Owner(s):
AFDB is a collaborative project between Google DeepMind and the European Bioinformatics Institute (EMBL-EBI). La-Proteina release is re-distributing a subset of the AFDB.

## Dataset Creation Date:
February 20, 2025

## License/Terms of Use: 
[CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode)

## Intended Usage:
We use the dataset to train La-Proteina, a generative model for all-atom protein design. This dataset can be used by protein designers, structural biologists, bioengineers, computational biologists and protein engineers.

## Dataset Characterization
**Data Collection Method**<br>
* Synthetic <br>

**Labeling Method**<br>
* Automated

## Dataset Format
Text files (PDB format) containing protein sequence, atomic coordinates, and confidence scores.

## Dataset Quantification
We use two subsets of the AFDB, one comprising 344,508 PDB files and the other one comprising 46,942,694.


Each PDB file contains 3 features: protein sequence, atomic coordinates, and confidence score.

The total data storage is approximately 44TB (approximate 1MB per PDB file).


## References:
[AlphaFold Dataset Guide](https://github.com/google-deepmind/alphafold/blob/main/afdb/README.md)<br>
[AlphaFold Protein Structure Database FAQs](https://alphafold.ebi.ac.uk/faq)<br>
[Jumper, J et al. Highly accurate protein structure prediction with AlphaFold. Nature (2021).](https://www.nature.com/articles/s41586-021-03819-2)<br>
[Varadi, M et al. AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models. Nucleic Acids Research (2021).](https://academic.oup.com/nar/article/50/D1/D439/6430488)<br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.   

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).



