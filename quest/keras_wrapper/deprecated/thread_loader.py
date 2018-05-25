from keras.models import model_from_json

import threading
import sys


class ThreadDataLoader(threading.Thread):
    """
        Data loader based on threads (parallel execution)
    """

    def __init__(self, target, *args):
        self._target = target
        self._args = args
        threading.Thread.__init__(self)

    def run(self):
        [resultOK, exception, X, Y] = self._target(*self._args)
        self.X = X
        self.Y = Y
        self.resultOK = resultOK
        self.exception = exception


def retrieveXY(dataset, set_name, batchSize, normalization, meanSubstraction, dataAugmentation):
    """
        Retrieves a set of samples from the given dataset and the given set name
    """
    try:
        X_batch, Y_batch = dataset.getXY(set_name, batchSize, normalization=normalization,
                                         meanSubstraction=meanSubstraction, dataAugmentation=dataAugmentation)
        return [True, '', X_batch, Y_batch]
    except:
        return [False, sys.exc_info(), None, None]


class ThreadModelLoader(threading.Thread):
    """
        Model loader based on threads (parallel execution)
    """

    def __init__(self, target, *args):
        self._target = target
        self._args = args
        threading.Thread.__init__(self)

    def run(self):
        self.model = self._target(*self._args)


def retrieveModel(path_json, path_h5, path_pkl):
    """
        Loads a model using a parallel thread.
    """
    # Load model structure
    model = model_from_json(open(path_json).read())
    # Load model weights
    model.load_weights(path_h5)
    stage = pk.load(open(path_pkl, 'rb'))
    stage.model = model
    return stage
