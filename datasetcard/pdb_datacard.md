## Dataset Description:
The RCSB Protein Data Bank (PDB) is a database for the three-dimensional structural data of large biological molecules such as proteins and nucleic acids.

We are releasing a subset of this dataset, with the following modification. To benchmark La-Proteina on the motif scaffolding task, we removed some residues from the PDB files to form a motif. Additionally, we also removed residues with unresolved atoms.

This dataset is ready for commercial/non-commercial use.

## Dataset Owner(s):
Worldwide Protein Data Bank (wwPDB) maintains the PDB. La-Proteina release is re-distributing a subset of the PDB. 

## Dataset Creation Date:
June 18, 2025.

## License/Terms of Use: 
[CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode)

ADDITIONAL INFORMATION: Attribution to the original authors of the PDB structure data. (https://www.wwpdb.org/about/cite-us) - see references below. 

## Intended Usage:
La-Proteina uses a subset of the PDB to create a benchmark for atomistic motif scaffolding. This dataset can be used by protein designers, structural biologists, bioengineers, computational biologists and protein engineers.

## Dataset Characterization
**Data Collection Method**<br>
* Human <br>

**Labeling Method**<br>
* Automated

## Dataset Format
PDB files use a text-based format to describe the 3D structures of biological macromolecules, containing protein sequence and atomic coordinates. PDB format consists of lines of information in a text file. Each line of information in the file is called a record. A file generally contains several different types of records, which are arranged in a specific order to describe a structure.

## Dataset Quantification
The entire PDB contains 241,345 protein structures. La-Proteina uses 19.
The total data storage is approximately 15MB.

## Reference(s):
[wwPDB](https://www.wwpdb.org/)<br>
H.M. Berman, K. Henrick, H. Nakamura (2003) Announcing the worldwide Protein Data Bank Nature Structural Biology 10 (12): 980<br>
H.M. Berman, K. Henrick, H.Nakamura, J.L. Markley (2007) The Worldwide Protein Data Bank (wwPDB): Ensuring a single, uniform archive of PDB data Nucleic Acids Res. 35 (Database issue): D301-3<br>
wwPDB consortium. (2019) Protein Data Bank: the single global archive for 3D macromolecular structure data. Nucleic Acids Res 47: D520-D528 doi: https://doi.org/10.1093/nar/gky949<br>
[RSCB PDB File Download Services](https://www.rcsb.org/docs/programmatic-access/file-download-services)

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.   

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).



