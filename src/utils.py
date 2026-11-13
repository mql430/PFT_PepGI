import logging

import dgl
import numpy as np
import torch

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25

def graph_collate_func(x):
    d, p, y, idx = zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p)), torch.tensor(y), list(idx)


def integer_label_protein(sequence, max_length=1200):
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding


def to_3d(x):
    return x.flatten(2).transpose(1, 2)


def to_4d(x, h, w):
    return x.transpose(1, 2).unflatten(2, (h, w))
