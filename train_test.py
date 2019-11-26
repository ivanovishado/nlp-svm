#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ivan Vladimir Meza-Ruiz/ ivanvladimir at turing.iimas.unam.mx
# 2016/IIMAS/UNAM
# ----------------------------------------------------------------------

# Cargando librerias
import itertools
import argparse
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def create_vectors(filename):
    x = []
    y = []
    with open(filename, encoding="utf-8") as lines:
        for line in lines:
            line = line.strip()
            bits = line.rsplit(" ", 1)
            x.append(bits[0])
            y.append(int(bits[1]))
    return x, y


def serialize(filename, object_to_serialize):
    with open(filename, 'wb') as f:
        pickle.dump(object_to_serialize, f, protocol=pickle.HIGHEST_PROTOCOL)


def plot_weights(coefs):
    fig, axes = plt.subplots(4, 4)
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = coefs[0].min(), coefs[0].max()
    for coef, ax in zip(coefs[0].T, axes.ravel()):
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())

    # plt.show()


if __name__ == "__main__":
    # Command line options
    p = argparse.ArgumentParser("Documentos")
    p.add_argument('TRAINING', help="Archivos con datos de entreanmiento")
    p.add_argument('TESTING', help="Archivo con datos de prueba")
    p.add_argument("-v", "--verbose",
                   action="store_true", dest="verbose",
                   help="Modo verbose [Off]")
    p.add_argument('--version', action='version', version='create_segments 0.1')
    opts = p.parse_args()

    # Prepara función de verbose  -----------------------------------------
    if opts.verbose:
        def verbose(*args, **kargs):
            print(*args, **kargs)
    else:
        verbose = lambda *a: None

    X_train, Y_train = create_vectors(opts.TRAINING)

    X_test, Y_test = create_vectors(opts.TESTING)

    # Read stopwords
    with open('stopwords-es.txt', encoding="utf-8") as f:
        stop_words = f.read().splitlines()

    count_vect = CountVectorizer(stop_words=stop_words)
    X_train = count_vect.fit_transform(X_train)
    X_test = count_vect.transform(X_test)
    delattr(count_vect, 'stop_words')  # se eliminan las stop_words del CV para
                                       # que la serialización no sea tan pesada
    serialize('CV.pkl', count_vect)

    clf = SVC(probability=True, kernel='linear')
    clf.fit(X_train, Y_train)
    serialize('clf.pkl', clf)

    print('Coefs:')
    print(clf.coef_)

    Y_pred = clf.predict(X_test)
    print("Train score=", clf.score(X_train, Y_train))
    print("Test score=", clf.score(X_test, Y_test))
    print(classification_report(Y_test, Y_pred))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test, Y_pred)
    np.set_printoptions(precision=2)

    class_names = ['Non-violent news', 'Violent news']

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
