import numpy as np


def confusion_matrix(y_true, y_predict, percent=None):
    Y_1 = y_predict[:, -1].copy()  # рассмтаривается лишь случаай бинарной классификации, обобщать на большее число классов тяжелее, проблемы типа как бороться с колиизиями?
    true_sample = y_true.copy()
    if percent:
        threshold = (100 - percent) / 100  # положим такй порог классификации, то есть p = 25% значит порог 0.75

        qntl = np.quantile(Y_1, (100 - percent) / 100)  # Хотим топ PERCENT значит отсекаем выборку по квантилю на уровне Percent и смотрим правее

        top_sample = [(Y_1[i], i) for i in range(len(Y_1)) if Y_1[i] > qntl]  # формируем выборку

        indecies = [top_sample[i][1] for i in range(len(top_sample))]  # запонинаем индексы

        true_sample = [y_true[i] for i in range(len(y_true)) if i in indecies]  # отделяем соответствующие истинные значения

        top_sample = [e[0] for e in top_sample]
    else:
        threshold = 0.5
        top_sample = Y_1  # формируем выборку

    bin_sample = []
    for val in top_sample:  # бинаризируем  выборку
        if val >= threshold:
            bin_sample.append(1)
        else:
            bin_sample.append(0)

    confusion_matrix = {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}

    for i in range(len(bin_sample)):
        if bin_sample[i] == true_sample[i] and true_sample[i] == 1:
            confusion_matrix['TP'] += 1
        elif bin_sample[i] != true_sample[i] and true_sample[i] == 0:
            confusion_matrix['FP'] += 1
        elif bin_sample[i] == true_sample[i] and true_sample[i] == 0:
            confusion_matrix['TN'] += 1
        else:
            confusion_matrix['FN'] += 1
    return confusion_matrix


def accuracy_score(y_true, y_predict, percent=None):
    matrix = confusion_matrix(y_true, y_predict, percent)
    accuracy = (matrix['TP'] + matrix['TN']) / (matrix['TP'] + matrix['TN'] +matrix['FP'] +matrix['FN'])
    return accuracy


def precision_score(y_true, y_predict, percent=None):
    matrix = confusion_matrix(y_true, y_predict, percent)
    precision = matrix['TP'] / (matrix['TP'] + matrix['FP'])
    return precision


def recall_score(y_true, y_predict, percent=None):
    matrix = confusion_matrix(y_true, y_predict, percent)
    recall = matrix['TP'] / (matrix['TP'] + matrix['FN'])
    return recall


def lift_score(y_true, y_predict, percent=None):
    matrix = confusion_matrix(y_true, y_predict, percent)
    precision = matrix['TP'] / (matrix['TP'] + matrix['FP'])
    lift = precision / ((matrix['TP'] + matrix['FN']) / (matrix['TP'] + matrix['FN'] + matrix['FP'] + matrix['TN']))
    return lift


def f1_score(y_true, y_predict, percent=None):
    matrix = confusion_matrix(y_true, y_predict, percent)
    precision = matrix['TP'] / (matrix['TP'] + matrix['FP'])
    recall = matrix['TP'] / (matrix['TP'] + matrix['FN'])
    f1 = 2 * precision * recall / (precision + recall)
    return f1
