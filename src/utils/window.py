# helper functions for moving window across sequence
import torch
from torch.nn.functional import sigmoid


VOCABULARY = (
    'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C'
)
MEAN_LLPS_WINDOW = -2.197


def get_all_mutant_windows(sequence: str):
    mut_windows = []
    for i in range(len(sequence)):
        mut_windows_i = []
        for v in VOCABULARY:
            seq_v = sequence[max(0,(i-10)):i] + v + sequence[(i+1):min((i+11),len(sequence))]
            mut_windows_i.append(seq_v)
        mut_windows.append(mut_windows_i)
    return mut_windows


def get_all_saprot_mutant_windows(sequence: str):
    mut_windows = []
    for i in range(len(sequence)//2):
        mut_windows_i = []
        for v in VOCABULARY:
            seq_v = sequence[2*max(0,(i-10)):2*i] + v + '#' + sequence[2*(i+1):2*min((i+11),len(sequence))]
            mut_windows_i.append(seq_v)
        mut_windows.append(mut_windows_i)
    return mut_windows


def get_all_windows(sequence: str):
    """
    Get all moving windows across a sequence. Default length 21. 
    Windows at the two ends will be shorter.
    For example, "AGFQPQ" using window length 5 gives 
    gives ["AGF", "AGFQ", "AGFQP", "GFQPQ", "FQPQ", "QPQ"]

    :param sequence: input protein sequence
    :type sequence: str
    :return: list of all windows, the list has same length as the input sequence
    :rtype: list[str]
    """
    wt_windows = []
    for i in range(len(sequence)):
        seq = sequence[max(0,(i-10)):min((i+11),len(sequence))]
        wt_windows.append(seq)
    return wt_windows


def get_all_saprot_windows(sequence: str):
    """
    Get all moving windows across an SaProt sequence. Default length 21. 
    Windows at the two ends will be shorter.
    For example, "AdGdF#QvPvQd" using window length 5 gives 
    ["AdGdF#", "AdGdF#Qv", "AdGdF#QvPv", "GdF#QvPvQd", "F#QvPvQd", "QvPvQd"]

    :param sequence: input SaProt sequence
    :type sequence: str
    :return: list of all windows, the list has half of length of the input SaProt sequence
    :rtype: list[str]
    """
    wt_windows = []
    for i in range(len(sequence)//2):
        seq = sequence[2*max(0,(i-10)):2*min((i+11),len(sequence))]
        wt_windows.append(seq)
    return wt_windows


def get_window(sequence: str, pos_aa: int):
    """
    Get window of a sequence at a specific position. Default length 21.
    Windows at the two ends will be shorter.
    For example, "AGFQPQ" at position 2 using window length 5 gives "AGFQP"

    :param sequence: input protein sequence
    :type sequence: str
    :param pos_aa: position of the window, must be within range
    :type pos_aa: int
    :return: window at a specific position
    :rtype: str
    """
    return sequence[max(0,(pos_aa-10)):min((pos_aa+11),len(sequence))]


def get_saprot_window(sequence: str, pos_aa: int):
    """
    Get window of an SaProt sequence at a specific position. Default length 21.
    Windows at the two ends will be shorter.
    For example, "AdGdF#QvPvQd" at position 2 using window length 5 gives "AdGdF#QvPv"

    :param sequence: input protein sequence
    :type sequence: str
    :param pos_aa: position of the window, must be within range
    :type pos_aa: int
    :return: window at a specific position
    :rtype: str
    """
    return sequence[2*max(0,(pos_aa-10)):2*min((pos_aa+11),len(sequence))]


def flatten(sequence_list: list):
    """
    Flatten list of lists of sequences.
    """
    return [s for seq in sequence_list for s in seq]


def compute_llps_window(
        model, sequence: str, saprot_sequence: str, pos_aa: int, correction: float = MEAN_LLPS_WINDOW
    ):
    """
    Predict LLPS score for a given window of sequence.

    :param model: LLPS prediction model
    :type model: torch.nn.module
    :param sequence: input protein sequence
    :type sequence: str
    :param saprot_sequence: input SaProt sequence
    :type saprot_sequence: str
    :param pos_aa: position of the window, must be within range
    :type pos_aa: int
    :param correction: mean LLPS score across all windows subtracted before sigmoid,
        to centre window-based logits around 0 and normalize delta LLPS scores
    :type correction: float
    :return: LLPS score
    :rtype: float
    """
    wt_sequence = [
        get_window(sequence, pos) 
        for pos in range(pos_aa, pos_aa + 1)
    ]
    wt_saprot_sequence = [
        get_saprot_window(saprot_sequence, pos)
        for pos in range(pos_aa, pos_aa + 1)
    ]
    # LLPS score
    model.eval()
    with torch.no_grad():
        prob = sigmoid(
            model(wt_sequence, wt_saprot_sequence) - correction
        ).max().item()
    return prob


def compute_llps_mean(model, sequence: str, saprot_sequence: str):
    """
    Compute mean LLPS score over all windows across a sequence. 
    Gives an overall score for a protein. Can be used to normalize delta LLPS scores.

    :param model: LLPS prediction model
    :type model: torch.nn.module
    :param sequence: input protein sequence
    :type sequence: str
    :param saprot_sequence: input SaProt sequence
    :type saprot_sequence: str
    :return: LLPS score
    :rtype: float
    """
    wt_windows = get_all_windows(sequence)
    wt_saprot_windows = get_all_saprot_windows(saprot_sequence)
    # mean LLPS score across all windows of a sequence
    model.eval()
    with torch.no_grad():
        prob_list = [
            sigmoid(model([wt1], [wt2])[0]).item()
            for wt1, wt2 in zip(wt_windows, wt_saprot_windows)
        ]
    return sum(prob_list) / len(prob_list)


def compute_critical_region(model, sequence, saprot_sequence, correction=MEAN_LLPS_WINDOW):
    wt_windows = get_all_windows(sequence)
    wt_saprot_windows = get_all_saprot_windows(saprot_sequence)
    # LLPS score at residue level
    model.eval()
    with torch.no_grad():
        prob_list = [
            sigmoid(model([wt1], [wt2])[0] - correction).item()
            for wt1, wt2 in zip(wt_windows, wt_saprot_windows)
        ]
    return prob_list


def compute_mutagenesis(model, sequence, saprot_sequence, correction=MEAN_LLPS_WINDOW):
    wt_windows = get_all_windows(sequence)
    mut_windows = get_all_mutant_windows(sequence)
    wt_saprot_windows = get_all_saprot_windows(saprot_sequence)
    mut_saprot_windows = get_all_saprot_mutant_windows(saprot_sequence)
    # delta LLPS score for each mutant sequence
    all_diff = []
    model.eval()
    with torch.no_grad():
        prob_ref_list = [
            sigmoid(model([wt1], [wt2])[0] - correction).item() 
            for wt1, wt2 in zip(wt_windows, wt_saprot_windows)
        ]
        for seq1, seq2, prob_ref in zip(mut_windows, mut_saprot_windows, prob_ref_list):
            all_diff_i = []
            for s1, s2 in zip(seq1, seq2):
                prob = sigmoid(model([s1], [s2])[0] - correction)
                diff_i = (prob - prob_ref).item()
                all_diff_i.append(diff_i)
            all_diff.append(all_diff_i)
    return all_diff
