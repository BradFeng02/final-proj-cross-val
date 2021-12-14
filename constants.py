"""
A module with constants used by train and test
"""

from os import listdir

CLF_PATH = 'model.clf'

DATA_PATH = 'data/txt_sentoken/'

N_SAMP = 1000  # 1k pos, 1k neg

TRAIN_RATIO = .8
N_TRAIN = int(TRAIN_RATIO * N_SAMP)
N_TEST = N_SAMP - N_TRAIN

K_FOLDS = 8
assert N_TRAIN % K_FOLDS == 0  # should be divisible
FOLD_SIZE = int(N_TRAIN / K_FOLDS)
CV_N_TRAIN = FOLD_SIZE * (K_FOLDS - 1)

neg_data = sorted(listdir(DATA_PATH + 'neg'))
pos_data = sorted(listdir(DATA_PATH + 'pos'))
assert N_SAMP <= len(neg_data) and N_SAMP <= len(pos_data)

PROG_BAR_SIZE = 35
SHORT_PROG_SIZE = 5
TRAIN_PAD = 2 * len(str(N_TRAIN * 2)) + 1
TEST_PAD = 2 * len(str(N_TEST * 2)) + 1
CV_TRAIN_PAD = 2 * len(str(CV_N_TRAIN * 2)) + 1
CV_TEST_PAD = 2 * len(str(FOLD_SIZE * 2)) + 1