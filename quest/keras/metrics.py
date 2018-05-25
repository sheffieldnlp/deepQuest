from __future__ import absolute_import
import six
from . import backend as K
from .losses import mean_squared_error
from .losses import mean_absolute_error
from .losses import mean_absolute_percentage_error
from .losses import mean_squared_logarithmic_error
from .losses import hinge
from .losses import logcosh
from .losses import squared_hinge
from .losses import categorical_crossentropy
from .losses import sparse_categorical_crossentropy
from .losses import binary_crossentropy
from .losses import kullback_leibler_divergence
from .losses import poisson
from .losses import cosine_proximity
from .utils.generic_utils import deserialize_keras_object


def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=-1)
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)), axis=-1)

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def pearson_corr(y_true, y_pred):
    fsp = y_pred - K.mean(y_pred)
    fst = y_true - K.mean(y_true)

    devP = K.std(y_pred)
    devT = K.std(y_true)

    return K.mean(fsp * fst) / (devP * devT)


def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)

def perplexity(y_true, y_pred, mask=None):
    ''' Calculates the perplexity
    '''
    if mask is not None:
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        mask = K.permute_dimensions(K.reshape(mask, y_true.shape[:-1]), (0, 1, 'x'))
        truth_mask = K.flatten(y_true*mask).nonzero()[0]  ### How do you do this on tensorflow?
        predictions = K.gather(y_pred.flatten(), truth_mask)
        return K.pow(2, K.mean(-K.log2(predictions)))
    else:
        return K.pow(2, K.mean(-K.log2(y_pred)))

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    # Aliases

mse = MSE = mean_squared_error
rmse = RMSE = root_mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity


def serialize(metric):
    return metric.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='metric function')


def get(identifier):
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'metric function identifier:', identifier)
