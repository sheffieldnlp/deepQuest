#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# bertlayer.py
#
# Copyright (C) 2019 Frederic Blain (feedoo) <f.blain@sheffield.ac.uk>
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Fred (feedoo) Blain wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a tomato juice or coffee in return
#

"""

"""

from numpy.random import seed
seed(42)

import random
random.seed(42)
del random

from keras.layers import *

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K
import keras_wrapper.extra.bert_tokenization as tokenization

class BertLayer(Layer):
    def __init__(self, max_seq_len=70, output_representation='sequence_output', trainable=True, **kwargs):
        self.bert = None
        super(BertLayer, self).__init__(**kwargs)

        self.bert_path="https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"
        self.output_representation = output_representation
        self.max_seq_len = max_seq_len
        self.trainable = trainable

    def build(self, input_shape):
        self.bert = hub.Module(self.bert_path,
                               trainable=True, name="{}_module".format(self.name))

        # Remove unused layers and set trainable parameters
        if self.output_representation in ['sequence_output', 'cls_output']:
            # self.trainable_weights += [var for var in self.bert.variables
                                       # if not "/cls/" in var.name and not "/pooler/" in var.name]
            self.trainable_weights += [var for var in self.bert.variables
                                       if "layer_9" in var.name or "layer_10" in var.name or "layer_11" in var.name or "layer_12" in var.name]
            # self.trainable_weights += [var for var in self.bert.variables
                                       # if "layer_5" in var.name or "layer_6" in var.name or "layer_7" in var.name or "layer_8" in var.name or "layer_9" in var.name or "layer_10" in var.name or "layer_11" in var.name or "layer_12" in var.name]
            self.trainable_weights += [var for var in self.bert.variables
                                       if "embeddings/word_embeddings" in var.name]
            # self.trainable_weights += [var for var in self.bert.variables
                                       # if "/layer_" in var.name]
            # self.trainable_weights += [var for var in self.bert.variables
                                       # if "embeddings" in var.name]
        else:
            self.trainable_weights += [var for var in self.bert.variables
                                       if not "/cls/" in var.name]
        super(BertLayer, self).build(input_shape)

    def call(self, x, mask=None):
        inputs = dict(input_ids=x[0], input_mask=x[1], segment_ids=x[2])

        if self.output_representation in ['sequence_output', 'cls_output']:
            outputs = self.bert(inputs, as_dict=True, signature='tokens')['sequence_output']
        else:
            outputs = self.bert(inputs, as_dict=True, signature='tokens')['pooled_output']

        if self.output_representation == 'cls_output':
            return K.tf.squeeze(outputs[:, 0:1, :], axis=1)
        else:
            return outputs

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if self.output_representation in ['pooled_output', 'cls_output']:
            return (None, 768)
        else:
            return (None, self.max_seq_len, 768)

# class BertLayer(tf.keras.layers.Layer):
# class BertLayer(Layer):
#     def __init__(
#         self,
#         n_fine_tune_layers=10,
#         pooling="mean",
#         # pooling="first",
#         # bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
#         bert_path="https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1",
#         max_seq_len = 70,
#         **kwargs
#     ):
#         self.n_fine_tune_layers = n_fine_tune_layers
#         self.trainable = False
#         self.output_size = 768
#         self.pooling = pooling
#         self.bert_path = bert_path
#         if self.pooling not in ["first", "mean"]:
#             raise NameError("Undefined pooling type (must be either first or mean, but is {})".format(self.pooling))
#
#
#         super(BertLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.bert = hub.Module(
#             "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", trainable=self.trainable)
#
#         # # Remove unused layers
#         # trainable_vars = self.bert.variables
#         # if self.pooling == "first":
#         #     trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
#         #     trainable_layers = ["pooler/dense"]
#         #
#         # elif self.pooling == "mean":
#         #     trainable_vars = [
#         #         var
#         #         for var in trainable_vars
#         #         if not "/cls/" in var.name and not "/pooler/" in var.name
#         #     ]
#         #     trainable_layers = []
#         # else:
#         #     raise NameError("Undefined pooling type (must be either first or mean, but is {}".format(self.pooling))
#         #
#         # # Select how many layers to fine tune
#         # for i in range(self.n_fine_tune_layers):
#         #     trainable_layers.append("encoder/layer_{str(11 - i)}")
#         #
#         # # Update trainable vars to contain only the specified layers
#         # trainable_vars = [
#         #     var
#         #     for var in trainable_vars
#         #     if any([l in var.name for l in trainable_layers])
#         # ]
#
#         # # Add to trainable weights
#         # for var in trainable_vars:
#         #     self._trainable_weights.append(var)
#         #
#         # for var in self.bert.variables:
#         #     if var not in self._trainable_weights:
#         #         self._non_trainable_weights.append(var)
#
#         super(BertLayer, self).build(input_shape)
#
#     def call(self, inputs):
#         inputs = [K.cast(x, dtype="int32") for x in inputs]
#         input_ids, input_mask, segment_ids = inputs
#
#         bert_inputs = dict(
#             input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
#         )
#         # if self.pooling == "first":
#         #     pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
#         #         "pooled_output"
#         #     ]
#         # elif self.pooling == "mean":
#         #     result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
#         #         "sequence_output"
#         #     ]
#         #
#             # mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
#             # masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
#             #         tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
#             # input_mask = tf.cast(input_mask, tf.float32)
#             # pooled = masked_reduce_mean(result, input_mask)
#             # pooled = result
#         # else:
#         #     raise NameError("Undefined pooling type (must be either first or mean, but is {}".format(self.pooling))
#
#         # return pooled
#
#         result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
#                 # "pooled_output"
#                 "sequence_output"
#                 ]
#         import ipdb; ipdb.set_trace()
#         result = K.temporal_padding(result, padding=(0, 70))
#         import ipdb; ipdb.set_trace()
#         return result
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_size)
#
#
