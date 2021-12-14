"""
train and validate with k fold validation
"""

from math import floor
import numpy as np
from sklearn import svm
from constants import (
    DATA_PATH, neg_data, pos_data,
    K_FOLDS, FOLD_SIZE, CV_N_TRAIN, N_TRAIN,
    PROG_BAR_SIZE, SHORT_PROG_SIZE, CV_TRAIN_PAD, CV_TEST_PAD
)
from features import make_input_arrs, find_feats, standardize_feats


def getNegData(index: int) -> str:
    neg_file = open(DATA_PATH + 'neg/' + neg_data[index], 'r')
    neg = neg_file.read()
    neg_file.close()
    return neg


def getPosData(index: int) -> str:
    pos_file = open(DATA_PATH + 'pos/' + pos_data[index], 'r')
    pos = pos_file.read()
    pos_file.close()
    return pos


# PROGRAM BEGIN
print('{}-fold cross validation\nfold size: {} (x2)\ntotal training data: {} (x2)\n'
      .format(K_FOLDS, FOLD_SIZE, N_TRAIN))

results = np.zeros((K_FOLDS, 3))  # neg, pos, all

for k in range(K_FOLDS):
    trainBeg = k * FOLD_SIZE
    trainEnd = (k + K_FOLDS - 1) * FOLD_SIZE  # wraps around

    # TRAINING
    prog = [' '] * (PROG_BAR_SIZE + 2)
    prog[0] = '['
    prog[-1] = ']'
    progTitle = 'training {}/{}:\t'.format(k, K_FOLDS - 1)
    print(progTitle, ''.join(prog), sep='', end='\r')

    samples, labels = make_input_arrs(CV_N_TRAIN)
    labels[CV_N_TRAIN:] = 1  # neg, ..., pos, ...

    for i in range(trainBeg, trainEnd):
        fileInd = i % N_TRAIN

        find_feats(getNegData(fileInd), samples, i - trainBeg)
        find_feats(getPosData(fileInd), samples, i - trainBeg + CV_N_TRAIN)

        p = floor((i - trainBeg) / CV_N_TRAIN * (PROG_BAR_SIZE + 1))
        if p > 0:
            prog[p] = '='
        print(progTitle, ''.join(prog),
              ' {}/{}'.format((i - trainBeg + 1) * 2, CV_N_TRAIN * 2)
              .rjust(CV_TRAIN_PAD + 1),
              sep='', end='\r')
    # endfor

    standardize_feats(samples, CV_N_TRAIN)
    clf = svm.SVC()
    clf.fit(samples, labels) # TODO replace with whatever model
    print('')


    # EVALUATING
    prog = [' '] * (SHORT_PROG_SIZE + 2)
    prog[0] = '['
    prog[-1] = ']'
    progTitle = 'evaluating {}:\t'.format(k)
    print(progTitle, ''.join(prog), sep='', end='\r')

    samples, labels = make_input_arrs(FOLD_SIZE)
    labels[FOLD_SIZE:] = 1  # neg, ..., pos, ...

    for i in range(trainEnd, trainEnd + FOLD_SIZE):
        fileInd = i % N_TRAIN

        find_feats(getNegData(fileInd), samples, i - trainEnd)
        find_feats(getPosData(fileInd), samples, i - trainEnd + FOLD_SIZE)

        p = floor((i - trainEnd) / FOLD_SIZE * (SHORT_PROG_SIZE + 1))
        if p > 0:
            prog[p] = '='
        print(progTitle, ''.join(prog),
              ' {}/{}'.format((i - trainEnd + 1) * 2, FOLD_SIZE * 2)
              .rjust(CV_TEST_PAD + 1),
              sep='', end='\r')
    # endfor

    standardize_feats(samples, FOLD_SIZE)
    predicts = clf.predict(samples)

    correct = neg_correct = pos_correct = 0
    for i in range(FOLD_SIZE):
        if labels[i] == predicts[i]:
            correct += 1
            neg_correct += 1
        if labels[i + FOLD_SIZE] == predicts[i + FOLD_SIZE]:
            correct += 1
            pos_correct += 1

    print('')
    results[k, 0] = neg_correct / FOLD_SIZE
    print('n', round(results[k, 0], 2), '({}/{})'.format(neg_correct, FOLD_SIZE), end='  |  ')
    results[k, 1] = pos_correct / FOLD_SIZE
    print('p', round(results[k, 1], 2), '({}/{})'.format(pos_correct, FOLD_SIZE), end='  |  ')
    results[k, 2] = correct / (2 * FOLD_SIZE)
    print('a', round(results[k, 2], 2), '({}/{})'.format(correct, 2 * FOLD_SIZE))
    print('')
#endfor

neg_avg = pos_avg = all_avg = 0.0
print('summary:')
print('fold', 'neg', 'pos', 'overall accuracy', sep='\t')
for k in range(K_FOLDS):
    print(k,
          str(round(results[k, 0], 4)).ljust(6),
          str(round(results[k, 1], 4)).ljust(6),
          str(round(results[k, 2], 4)).ljust(6),
          sep='\t')
    neg_avg += results[k, 0]
    pos_avg += results[k, 1]
    all_avg += results[k, 2]
print("\navg",
      str(round(neg_avg / K_FOLDS, 4)).ljust(6),
      str(round(pos_avg / K_FOLDS, 4)).ljust(6),
      str(round(all_avg / K_FOLDS, 4)).ljust(6),
      sep='\t')
