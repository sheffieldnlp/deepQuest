import numpy as np

from keras.utils import np_utils

import logging
import copy


class ECOC_Classifier(object):
    """
        Class for defining a classical ECOC classifier.
    """

    def __init__(self, table, distance='hamming'):
        """
            Creates an ECOC classifier defined by a coding 'table' (numpy matrix) and a 'distance' for computing the predictions.
            Available distances:
                - hamming
            Each row represents a classifier and each column represents a class.
        """
        self.table = np.array(table)
        self.setDistance(distance)

    def append(self, rows):
        """
            Appends new rows to the ECOC table which will define the code for new classifiers.
        """
        self.table += rows

    def setDistance(self, distance):
        """
            Changes the kind of distance used for computing the predictions.
        """
        f_dist = distance + 'Distance'
        if (hasattr(self, f_dist)):
            logging.info("<<< Building ECOC_Classifier with " + distance + " distance >>>")
        else:
            raise Exception('The distance "' + distance + '" is not implemented for an ECOC_Classifier.')

        self.distance = distance

    def setName(self, name, plots_path=None, models_path=None):
        self.name = name

    def softmax(self, similarities):
        """
            Applies the softmax function to a set of similarities for obtaining a normalized vector of probabilities.
        """
        n_classes = similarities.shape[1]
        exp = np.exp(similarities)
        sum_exp = np.transpose(np.tile(np.sum(exp, axis=1), [n_classes, 1]))
        return exp / sum_exp

    def dist2sim(self, dist):
        """
            Converts a set of distances into similarities
        """
        return 1 / (dist + 0.00001)

    # ------------------------------------------------------- #
    #       TEST/PREDICT functions
    # ------------------------------------------------------- #

    def predictOnBatch(self, X, in_name=None, out_name=None, expand=False):
        """
            Predicts the classes for a set of samples.
        """
        # Get desired input
        if (in_name):
            X = copy.copy(X[in_name])

        # Binarize input
        X = np.where(X > 0.5, 1, -1)

        #### Apply prediction
        # Calculate distances
        exec ('distances = self.' + self.distance + 'Distance(X)')
        # Convert distances into similarities
        similarities = self.dist2sim(distances)
        # Calculate probability
        predictions = self.softmax(similarities)

        # returns a vector of probabilities (one per class)
        return predictions

    def testOnBatch(self, X, Y, accuracy=True, out_name=None):
        """
            Applies a test on the samples provided and returns the resulting loss and accuracy (if True).
        """
        pred = self.predictOnBatch(X)
        [score, top_score] = self._getECOCAccuracy(Y, pred)
        return (0.0, score, top_score)

    def _getECOCAccuracy(self, GT, pred, topN=5):
        """
            Calculates the topN accuracy obtained from a set of samples on a ECOC_Classifier.
        """

        top_pred = np.argsort(pred, axis=1)[:, ::-1][:, :np.min([topN, pred.shape[1]])]
        pred = np_utils.categorical_probas_to_classes(pred)
        GT = np_utils.categorical_probas_to_classes(GT)

        # Top1 accuracy
        correct = [1 if pred[i] == GT[i] else 0 for i in range(len(pred))]
        accuracies = float(np.sum(correct)) / float(len(correct))

        # TopN accuracy
        top_correct = [1 if GT[i] in top_pred[i, :] else 0 for i in range(top_pred.shape[0])]
        top_accuracies = float(np.sum(top_correct)) / float(len(top_correct))

        return [accuracies, top_accuracies]

    # ------------------------------------------------------- #
    #       DISTANCES
    #           Available definitions of distance for an ECOC_Classifier
    # ------------------------------------------------------- #
    def hammingDistance(self, X):
        """
            Returns a matrix of dimensions (n_samples, n_classes) with the distance of each sample to each class.
        """
        n_samples = X.shape[0]
        n_classes = self.table.shape[1]

        distances = np.zeros((n_samples, n_classes))
        for s in range(n_samples):
            x_sample = np.transpose(np.tile(X[s, :, 0], [n_classes, 1]))
            distances[s, :] = np.sum((1 - x_sample * self.table) / 2.0, axis=0)

        return distances
