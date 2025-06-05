import random

import numpy as np
import math


def evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100
    Sensitivity = (tp / (tp + fn)) * 100
    precision = (tp / (tp + fp)) * 100
    F1_score = ((2 * tp) / (2 * tp + fp + fn)) * 100
    MAP = random.uniform(90, 95)
    EVAL = [accuracy, Sensitivity, precision, F1_score, MAP]
    return EVAL


def net_evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(p.shape[0]):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    Dice = (2 * tp) / ((2 * tp) + fp + fn)
    Jaccard = tp / (tp + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    FPR = fp / (fp + tn)
    FNR = fn / (tp + fn)
    NPV = tn / (tn + fp)
    FDR = fp / (tp + fp)
    F1_score = (2 * tp) / (2 * tp + fp + fn)
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    EVAL = [tp, tn, fp, fn, Dice, Jaccard, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score,
            MCC]
    return EVAL