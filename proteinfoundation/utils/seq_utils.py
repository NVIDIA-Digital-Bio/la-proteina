from openfold.np.residue_constants import restype_3to1


def seq_batch_3to1(sequences3):
    """Converts a batch of protein sequences in 3-char format to the sequences
    in 1-char format, using the 3to1 mapping from openfold.

    Args:
        sequences3: list of lists, where each sub-list is a protein sequence in 3-char format.
            Different sequences may have different lengths. Each element in the sub-list
            is a string of length 3 representing an amino acid.

    Returns:
        sequences1: list of strings, where each string is the corresponding sequence in 1-char format.
    """
    seq_3to1 = lambda seq3: "".join([restype_3to1[c] for c in seq3])
    return [seq_3to1(seq3) for seq3 in sequences3]
