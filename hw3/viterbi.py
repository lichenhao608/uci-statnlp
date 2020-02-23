import numpy as np


def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    R = np.zeros((N, L))
    bp = np.zeros((N, L))
    R[0, :] = emission_scores[0, :] + start_scores
    bp[0, :] = -1  # -1 indicates the start

    for i in range(1, N):
        for l in range(L):
            score = emission_scores[i, l] + trans_scores[:, l] + R[i-1, :]
            R[i, l] = np.max(score)
            bp[i, l] = np.argmax(score)

    finalscore = R[-1, :] + end_scores

    best_score = np.max(finalscore)
    best_seq = np.zeros(N, dtype=int)
    best_seq[-1] = np.argmax(finalscore)
    for i in range(N - 2, -1, -1):
        last_best = best_seq[i + 1]
        best_seq[i] = bp[i + 1, last_best]

    return best_score, best_seq
