# -*- coding: utf-8 -*-
import cPickle as pk
import copy
import fnmatch
import logging
import math
import ntpath
import os
import random
import re
import sys
import threading
from collections import Counter
from operator import add
import time
import codecs
import numpy as np
from PIL import Image as pilimage
from scipy import misc
from scipy import ndimage

from extra.read_write import create_dir_if_not_exists
from keras.utils import np_utils
from .utils import bbox


# ------------------------------------------------------- #
#       SAVE/LOAD
#           External functions for saving and loading Dataset instances
# ------------------------------------------------------- #

def saveDataset(dataset, store_path):
    """
    Saves a backup of the current Dataset object.

    :param dataset: Dataset object to save
    :param store_path: Saving path
    :return: None
    """
    create_dir_if_not_exists(store_path)
    store_path = store_path + '/Dataset_' + dataset.name + '.pkl'
    if not dataset.silence:
        logging.info("<<< Saving Dataset instance to " + store_path + " ... >>>")

    pk.dump(dataset, open(store_path, 'wb'), protocol=pk.HIGHEST_PROTOCOL)

    if not dataset.silence:
        logging.info("<<< Dataset instance saved >>>")


def loadDataset(dataset_path):
    """
    Loads a previously saved Dataset object.

    :param dataset_path: Path to the stored Dataset to load
    :return: Loaded Dataset object
    """

    logging.info("<<< Loading Dataset instance from " + dataset_path + " ... >>>")

    dataset = pk.load(open(dataset_path, 'rb'))

    logging.info("<<< Dataset instance loaded >>>")
    return dataset


# ------------------------------------------------------- #
#       DATA BATCH GENERATOR CLASS
# ------------------------------------------------------- #


class Data_Batch_Generator(object):
    """
    Batch generator class. Retrieves batches of data.
    """

    def __init__(self,
                 set_split,
                 net,
                 dataset,
                 num_iterations,
                 batch_size=50,
                 normalization=False,
                 data_augmentation=True,
                 mean_substraction=True,
                 predict=False,
                 random_samples=-1,
                 shuffle=True,
                 temporally_linked=False,
                 init_sample=-1,
                 final_sample=-1):
        """
        Initializes the Data_Batch_Generator
        :param set_split: Split (train, val, test) to retrieve data
        :param net: Net which use the data
        :param dataset: Dataset instance
        :param num_iterations: Maximum number of iterations
        :param batch_size: Size of the minibatch
        :param normalization: Switches on/off the normalization of images
        :param data_augmentation: Switches on/off the data augmentation of the input
        :param mean_substraction: Switches on/off the mean substraction for images
        :param predict: Whether we are predicting or training
        :param random_samples: Retrieves this number of training samples
        :param shuffle: Shuffle the training dataset
        :param temporally_linked: Indicates if we are using a temporally-linked model
        """
        self.set_split = set_split
        self.dataset = dataset
        self.net = net
        self.predict = predict
        self.temporally_linked = temporally_linked
        self.first_idx = -1
        self.init_sample = init_sample
        self.final_sample = final_sample

        # Several parameters
        self.params = {'batch_size': batch_size,
                       'data_augmentation': data_augmentation,
                       'mean_substraction': mean_substraction,
                       'normalization': normalization,
                       'num_iterations': num_iterations,
                       'random_samples': random_samples,
                       'shuffle': shuffle}

    def generator(self):
        """
        Gets and processes the data
        :return: generator with the data
        """

        if self.set_split == 'train' and not self.predict:
            data_augmentation = self.params['data_augmentation']
        else:
            data_augmentation = False

        it = 0
        while 1:
            if self.set_split == 'train' and it % self.params['num_iterations'] == 0 and \
                    not self.predict and self.params['random_samples'] == -1 and self.params['shuffle']:
                silence = self.dataset.silence
                self.dataset.silence = True
                self.dataset.shuffleTraining()
                self.dataset.silence = silence
            if it % self.params['num_iterations'] == 0 and self.params['random_samples'] == -1:
                self.dataset.resetCounters(set_name=self.set_split)
            it += 1

            # Checks if we are finishing processing the data split
            init_sample = (it - 1) * self.params['batch_size']
            final_sample = it * self.params['batch_size']
            batch_size = self.params['batch_size']
            n_samples_split = eval("self.dataset.len_" + self.set_split)
            if final_sample >= n_samples_split:
                final_sample = n_samples_split
                batch_size = final_sample - init_sample
                it = 0

            # Recovers a batch of data
            if self.params['random_samples'] > 0:
                num_retrieve = min(self.params['random_samples'], self.params['batch_size'])
                if self.temporally_linked:
                    if self.first_idx == -1:
                        self.first_idx = np.random.randint(0, n_samples_split - self.params['random_samples'], 1)[0]
                        self.next_idx = self.first_idx
                    indices = range(self.next_idx, self.next_idx + num_retrieve)
                    self.next_idx += num_retrieve
                else:
                    indices = np.random.randint(0, n_samples_split, num_retrieve)
                self.params['random_samples'] -= num_retrieve

                # At sampling from train/val, we always have Y
                if self.predict:
                    X_batch = self.dataset.getX_FromIndices(self.set_split,
                                                            indices,
                                                            normalization=self.params['normalization'],
                                                            meanSubstraction=self.params['mean_substraction'],
                                                            dataAugmentation=data_augmentation)
                    data = self.net.prepareData(X_batch, None)[0]

                else:
                    X_batch, Y_batch = self.dataset.getXY_FromIndices(self.set_split,
                                                                      indices,
                                                                      normalization=self.params['normalization'],
                                                                      meanSubstraction=self.params['mean_substraction'],
                                                                      dataAugmentation=data_augmentation)
                    data = self.net.prepareData(X_batch, Y_batch)

            elif self.init_sample > -1 and self.final_sample > -1:
                indices = range(self.init_sample, self.final_sample)
                if self.predict:
                    X_batch = self.dataset.getX_FromIndices(self.set_split,
                                                            indices,
                                                            normalization=self.params['normalization'],
                                                            meanSubstraction=self.params['mean_substraction'],
                                                            dataAugmentation=data_augmentation)
                    data = self.net.prepareData(X_batch, None)[0]

                else:
                    X_batch, Y_batch = self.dataset.getXY_FromIndices(self.set_split,
                                                                      indices,
                                                                      normalization=self.params['normalization'],
                                                                      meanSubstraction=self.params['mean_substraction'],
                                                                      dataAugmentation=data_augmentation)
                    data = self.net.prepareData(X_batch, Y_batch)

            else:
                if self.predict:
                    X_batch = self.dataset.getX(self.set_split,
                                                init_sample,
                                                final_sample,
                                                normalization=self.params['normalization'],
                                                meanSubstraction=self.params['mean_substraction'],
                                                dataAugmentation=False)
                    data = self.net.prepareData(X_batch, None)[0]
                else:
                    X_batch, Y_batch = self.dataset.getXY(self.set_split,
                                                          batch_size,
                                                          normalization=self.params['normalization'],
                                                          meanSubstraction=self.params['mean_substraction'],
                                                          dataAugmentation=data_augmentation)
                    data = self.net.prepareData(X_batch, Y_batch)
            yield (data)


class Homogeneous_Data_Batch_Generator(object):
    """
    Batch generator class. Retrieves batches of data.
    """

    def __init__(self,
                 set_split,
                 net,
                 dataset,
                 num_iterations,
                 batch_size=50,
                 joint_batches=20,
                 normalization=False,
                 data_augmentation=True,
                 mean_substraction=True,
                 predict=False,
                 random_samples=-1,
                 shuffle=True):
        """
        Initializes the Data_Batch_Generator
        :param set_split: Split (train, val, test) to retrieve data
        :param net: Net which use the data
        :param dataset: Dataset instance
        :param num_iterations: Maximum number of iterations
        :param batch_size: Size of the minibatch
        :param normalization: Switches on/off the normalization of images
        :param data_augmentation: Switches on/off the data augmentation of the input
        :param mean_substraction: Switches on/off the mean substraction for images
        :param predict: Whether we are predicting or training
        :param random_samples: Retrieves this number of training samples
        :param shuffle: Shuffle the training dataset
        :param temporally_linked: Indicates if we are using a temporally-linked model
        """
        self.set_split = set_split
        self.dataset = dataset
        self.net = net
        self.predict = predict
        self.first_idx = -1
        self.batch_size = batch_size
        self.it = 0

        # Several parameters
        self.params = {'data_augmentation': data_augmentation,
                       'mean_substraction': mean_substraction,
                       'normalization': normalization,
                       'num_iterations': num_iterations,
                       'random_samples': random_samples,
                       'shuffle': shuffle,
                       'joint_batches': joint_batches}
        self.reset()

    def retrieve_batch(self):

        if self.set_split == 'train' and not self.predict:
            data_augmentation = self.params['data_augmentation']
        else:
            data_augmentation = False

        if self.set_split == 'train' and self.it % self.params['num_iterations'] == 0 and \
                not self.predict and self.params['random_samples'] == -1 and self.params['shuffle']:
            silence = self.dataset.silence
            self.dataset.silence = True
            self.dataset.shuffleTraining()
            self.dataset.silence = silence
        if self.it % self.params['num_iterations'] == 0 and self.params['random_samples'] == -1:
            self.dataset.resetCounters(set_name=self.set_split)
        self.it += 1

        # Checks if we are finishing processing the data split
        init_sample = (self.it - 1) * self.batch_size
        final_sample = self.it * self.batch_size
        batch_size = self.batch_size
        joint_batches = self.params['joint_batches']
        n_samples_split = eval("self.dataset.len_" + self.set_split)
        if final_sample >= n_samples_split:
            final_sample = n_samples_split
            batch_size = final_sample - init_sample
            self.it = 0

        X_batch, Y_batch = self.dataset.getXY(self.set_split,
                                              batch_size * joint_batches,
                                              normalization=self.params['normalization'],
                                              meanSubstraction=self.params['mean_substraction'],
                                              dataAugmentation=data_augmentation)

        self.X_batch = X_batch
        self.Y_batch = Y_batch

    def reset(self):
        self.retrieve_batch()
        text_Y_batch = self.Y_batch[0][1]  # just use mask
        batch_lengths = np.asarray([int(np.sum(cc)) for cc in text_Y_batch])
        self.tidx = batch_lengths.argsort()
        self.curr_idx = 0

    def get_data(self):
        new_X = []
        new_Y = []
        next_idx = min(self.curr_idx + self.batch_size, len(self.tidx))
        self.batch_tidx = self.tidx[self.curr_idx:next_idx]

        for x_input_idx in range(len(self.X_batch)):
            x_to_add = [self.X_batch[x_input_idx][i] for i in self.batch_tidx]
            new_X.append(np.asarray(x_to_add))
        for y_input_idx in range(len(self.Y_batch)):
            Y_batch_ = []
            for data_mask_idx in range(len(self.Y_batch[y_input_idx])):
                y_to_add = np.asarray([self.Y_batch[y_input_idx][data_mask_idx][i] for i in self.batch_tidx])
                Y_batch_.append(y_to_add)
            new_Y.append(tuple(Y_batch_))

        data = self.net.prepareData(new_X, new_Y)
        self.curr_idx = next_idx
        if self.curr_idx >= len(self.tidx):
            self.reset()
        return data

    def generator(self):
        """
        Gets and processes the data
        :return: generator with the data
        """
        while True:
            yield self.get_data()


# ------------------------------------------------------- #
#       MAIN CLASS
# ------------------------------------------------------- #
class Dataset(object):
    """
    Class for defining instances of databases adapted for Keras. It includes several utility functions for
    easily managing data splits, image loading, mean calculation, etc.
    """

    def __init__(self, name, path, silence=False, vocabulary = dict(), vocabulary_len = dict(), doc_size=0):
        """
        Dataset initializer
        :param name: Dataset name
        :param path: Path to the folder where the images are stored
        :param silence: Verbosity
        """
        # Dataset name
        self.name = name
        self.doc_size = doc_size
        # Path to the folder where the images are stored
        self.path = path

        # If silence = False, some informative sentences will be printed while using the "Dataset" object instance
        self.silence = silence

        # Variable for storing external extra variables
        self.extra_variables = dict()

        # Data loading parameters
        # Lock for threads synchronization
        self.__lock_read = threading.Lock()

        # Indicators for knowing if the data [X, Y] has been loaded for each data split
        self.loaded_train = [False, False]
        self.loaded_val = [False, False]
        self.loaded_test = [False, False]
        self.len_train = 0
        self.len_val = 0
        self.len_test = 0

        # Initialize dictionaries of samples
        self.X_train = dict()
        self.X_val = dict()
        self.X_test = dict()
        self.Y_train = dict()
        self.Y_val = dict()
        self.Y_test = dict()


        # Optionally, we point to the raw files. Note that these are not inputs/outputs of the dataset.
        # That means, we won't pre/post process the content of these files in the Dataset class.
        self.loaded_raw_train = [False, False]
        self.loaded_raw_val = [False, False]
        self.loaded_raw_test = [False, False]

        self.X_raw_train = dict()
        self.X_raw_val = dict()
        self.X_raw_test = dict()
        self.Y_raw_train = dict()
        self.Y_raw_val = dict()
        self.Y_raw_test = dict()

        #################################################

        # Parameters for managing all the inputs and outputs
        # List of identifiers for the inputs and outputs and their respective types
        # (which will define the preprocessing applied)
        self.ids_inputs = []
        self.types_inputs = []  # see accepted types in self.__accepted_types_inputs
        self.inputs_data_augmentation_types = dict()  # see accepted types in self._available_augm_<input_type>
        self.optional_inputs = []

        self.ids_outputs = []
        self.types_outputs = []  # see accepted types in self.__accepted_types_outputs
        self.sample_weights = dict()  # Choose whether we should compute output masks or not

        # List of implemented input and output data types
        self.__accepted_types_inputs = ['raw-image', 'image-features',
                                        'video', 'video-features',
                                        'text',
                                        'id', 'ghost', 'file-name', 'doc']
        self.__accepted_types_outputs = ['categorical', 'binary',
                                         'real',
                                         'text',
                                         '3DLabel', '3DSemanticLabel',
                                         'id', 'file-name']
        #    inputs/outputs with type 'id' are only used for storing external identifiers for your data
        #    they will not be used in any way. IDs must be stored in text files with a single id per line

        # List of implemented input normalization functions
        self.__available_norm_im_vid = ['0-1']  # 'image' and 'video' only
        self.__available_norm_feat = ['L2']  # 'image-features' and 'video-features' only

        # List of implemented input data augmentation functions
        self.__available_augm_vid_feat = ['random_selection', 'noise']  # 'video-features' only
        #################################################

        # Parameters used for inputs/outputs of type 'text'
        self.extra_words = {'<pad>': 0, '<unk>': 1, '<null>': 2}  # extra words introduced in all vocabularies
        self.extra_pad = {'<pad>': 0}
        self.vocabulary = vocabulary  # vocabularies (words2idx and idx2words)
        self.max_text_len = dict()  # number of words accepted in a 'text' sample
        self.vocabulary_len = vocabulary_len  # number of words in the vocabulary
        self.text_offset = dict()  # number of timesteps that the text is shifted (to the right)
        self.fill_text = dict()  # text padding mode
        self.pad_on_batch = dict()  # text padding mode: If pad_on_batch, the sample will have the maximum length
        # of the current batch. Else, it will have a fixed length (max_text_len)
        self.words_so_far = dict()  # if True, each sample will be represented as the complete set of words until
        # the point defined by the timestep dimension
        # (e.g. t=0 'a', t=1 'a dog', t=2 'a dog is', etc.)
        self.mapping = dict()  # Source -- Target predefined word mapping
        self.BPE = None  # Byte Pair Encoding instance
        self.BPE_separator = None
        self.BPE_built = False
        #################################################

        # Parameters used for inputs of type 'video' or 'video-features'
        self.counts_frames = dict()
        self.paths_frames = dict()
        self.max_video_len = dict()
        #################################################

        # Parameters used for inputs of type 'image-features' or 'video-features'
        self.features_lengths = dict()
        #################################################

        # Parameters used for inputs of type 'raw-image'
        # Image resize dimensions used for all the returned images
        self.img_size = dict()
        # Image crop dimensions for the returned images
        self.img_size_crop = dict()
        # Training mean image
        self.train_mean = dict()
        # Whether they are RGB images (or grayscale)
        self.use_RGB = dict()
        #################################################

        # Parameters used for outputs of type 'categorical', '3DLabels' or '3DSemanticLabel'
        self.classes = dict()
        self.dic_classes = dict()
        #################################################

        # Parameters used for outputs of type '3DLabels' or '3DSemanticLabel'
        self.id_in_3DLabel = dict()
        self.num_poolings_model = dict()
        #################################################

        # Parameters used for outputs of type '3DSemanticLabel'
        self.semantic_classes = dict()
        #################################################

        # Parameters used for outputs of type 'sparse'
        self.sparse_binary = dict()
        #################################################

        # Reset counters to start loading data in batches
        self.resetCounters()

    def shuffleTraining(self):
        """
        Applies a random shuffling to the training samples.
        """
        if not self.silence:
            logging.info("Shuffling training samples.")

        # Shuffle
        num = self.len_train
        shuffled_order = random.sample([i for i in range(num)], num)

        # Process each input sample
        for id in self.X_train.keys():
            self.X_train[id] = [self.X_train[id][s] for s in shuffled_order]
        # Process each output sample
        for id in self.Y_train.keys():
            self.Y_train[id] = [self.Y_train[id][s] for s in shuffled_order]

        if not self.silence:
            logging.info("Shuffling training done.")

    def keepTopOutputs(self, set_name, id_out, n_top):
        self.__checkSetName(set_name)

        if id_out not in self.ids_outputs:
            raise Exception("The parameter 'id_out' must specify a valid id for an output of the dataset.\n"
                            "Error produced because parameter %s was not in %s" % (id_out, self.ids_outputs))

        logging.info('Keeping top ' + str(n_top) + ' outputs from the ' + set_name + ' set and removing the rest.')

        # Sort outputs by number of occurrences
        samples = None
        exec ('samples = self.Y_' + set_name)
        count = Counter(samples[id_out])
        most_frequent = sorted(count.items(), key=lambda x: x[1], reverse=True)[:n_top]
        most_frequent = [m[0] for m in most_frequent]

        # Select top samples
        kept = []
        for i, s in enumerate(samples[id_out]):
            if s in most_frequent:
                kept.append(i)

        # Remove non-top samples
        # Inputs
        ids = None
        exec ('ids = self.X_' + set_name + '.keys()')
        for id in ids:
            exec ('self.X_' + set_name + '[id] = [self.X_' + set_name + '[id][k] for k in kept]')
        # Outputs
        exec ('ids = self.Y_' + set_name + '.keys()')
        for id in ids:
            exec ('self.Y_' + set_name + '[id] = [self.Y_' + set_name + '[id][k] for k in kept]')
        new_len = len(samples[id_out])
        exec ('self.len_' + set_name + ' = new_len')

        self.__checkLengthSet(set_name)

        logging.info(str(new_len) + ' samples remaining after removal.')

    # ------------------------------------------------------- #
    #       GENERAL SETTERS
    #           classes list, train, val and test set, etc.
    # ------------------------------------------------------- #

    def resetCounters(self, set_name="all"):
        """
        Resets some basic counter indices for the next samples to read.
        """
        if set_name == "all":
            self.last_train = 0
            self.last_val = 0
            self.last_test = 0
        else:
            self.__checkSetName(set_name)
            exec ('self.last_' + set_name + '=0')

    def setSilence(self, silence):
        """
        Changes the silence mode of the 'Dataset' instance.
        """
        self.silence = silence

    def setListGeneral(self, path_list, split=[0.8, 0.1, 0.1], shuffle=True, type='raw-image', id='image'):
        """
            Deprecated
        """
        logging.info("WARNING: The method setListGeneral() is deprecated, consider using setInput() instead.")
        self.setInput(path_list, split, type=type, id=id)

    def setList(self, path_list, set_name, type='raw-image', id='image'):
        """
            DEPRECATED
        """
        logging.info("WARNING: The method setList() is deprecated, consider using setInput() instead.")
        self.setInput(path_list, set_name, type, id)

    def setRawInput(self, path_list, set_name, type='file-name', id='raw-text', overwrite_split=False):
        """
            Loads a list which can contain all samples from either the 'train', 'val', or
            'test' set splits (specified by set_name).

            # General parameters

            :param path_list: can either be a path to a text file containing the paths to the images or a python list of paths
            :param set_name: identifier of the set split loaded ('train', 'val' or 'test')
            :param type: identifier of the type of input we are loading (accepted types can be seen in self.__accepted_types_inputs)
            :param id: identifier of the input data loaded
            :param repeat_set: repeats the inputs given (useful when we have more outputs than inputs). Int or array of ints.
            :param required: flag for optional inputs
        """
        self.__checkSetName(set_name)

        # Insert type and id of input data
        keys_X_set = eval('self.X_raw_' + set_name + '.keys()')
        if id not in self.ids_inputs or overwrite_split:
            self.ids_inputs.append(id)
            self.types_inputs.append(type)
            if id not in self.optional_inputs:
                self.optional_inputs.append(id)  # This is always optional
        elif id in keys_X_set and not overwrite_split:
            raise Exception('An input with id "' + id + '" is already loaded into the Database.')

        if type not in self.__accepted_types_inputs:
            raise NotImplementedError(
                'The input type "' + type + '" is not implemented. The list of valid types are the following: ' + str(
                    self.__accepted_types_inputs))

        exec ('self.X_raw_' + set_name + '[id] = path_list')
        exec ('self.loaded_raw_' + set_name + '[0] = True')
        if not self.silence:
            logging.info('Loaded "' + set_name + '" set inputs of type "' + type + '" with id "' + id + '".')

    def setInput(self, path_list, set_name, type='raw-image', id='image', repeat_set=1, required=True,
                 overwrite_split=False, normalization_types=None, data_augmentation_types=None,
                 add_additional=False,
                 img_size=[256, 256, 3], img_size_crop=[227, 227, 3], use_RGB=True,
                 # 'raw-image' / 'video'   (height, width, depth)
                 max_text_len=35, tokenization='tokenize_none', offset=0, fill='end', min_occ=0,  # 'text'
                 pad_on_batch=True, build_vocabulary=False, max_words=0, words_so_far=False,  # 'text'
                 bpe_codes=None, separator='@@', # 'text'
                 feat_len=1024,  # 'image-features' / 'video-features'
                 max_video_len=26  # 'video'
                 ):
        """
            Loads a list which can contain all samples from either the 'train', 'val', or
            'test' set splits (specified by set_name).

            # General parameters

            :param path_list: can either be a path to a text file containing the paths to the images or a python list of paths
            :param set_name: identifier of the set split loaded ('train', 'val' or 'test')
            :param type: identifier of the type of input we are loading (accepted types can be seen in self.__accepted_types_inputs)
            :param id: identifier of the input data loaded
            :param repeat_set: repeats the inputs given (useful when we have more outputs than inputs). Int or array of ints.
            :param required: flag for optional inputs
            :param overwrite_split: indicates that we want to overwrite the data with id that was already declared in the dataset
            :param normalization_types: type of normalization applied to the current input if we activate the data normalization while loading
            :param data_augmentation_types: type of data augmentation applied to the current input if we activate the data augmentation while loading
            :param add_additional: adds additional data to an already existent input ID


            # 'raw-image'-related parameters

            :param img_size: size of the input images (any input image will be resized to this)
            :param img_size_crop: size of the cropped zone (when dataAugmentation=False the central crop will be used)


            # 'text'-related parameters

            :param tokenization: type of tokenization applied (must be declared as a method of this class) (only applicable when type=='text').
            :param build_vocabulary: whether a new vocabulary will be built from the loaded data or not (only applicable when type=='text'). A previously calculated vocabulary will be used if build_vocabulary is an 'id' from a previously loaded input/output
            :param max_text_len: maximum text length, the rest of the data will be padded with 0s (only applicable if the output data is of type 'text').
            :param max_words: a maximum of 'max_words' words from the whole vocabulary will be chosen by number or occurrences
            :param offset: number of timesteps that the text is shifted to the right (for sequential conditional models, which take as input the previous output)
            :param fill: select whether padding before or after the sequence
            :param min_occ: minimum number of occurrences allowed for the words in the vocabulary. (default = 0)
            :param pad_on_batch: the batch timesteps size will be set to the length of the largest sample +1 if True, max_len will be used as the fixed length otherwise
            :param words_so_far: if True, each sample will be represented as the complete set of words until the point defined by the timestep dimension (e.g. t=0 'a', t=1 'a dog', t=2 'a dog is', etc.)
            :param bpe_codes: Codes used for applying BPE encoding.
            :param separator: BPE encoding separator.

            # 'image-features' and 'video-features'- related parameters

            :param feat_len: size of the feature vectors for each dimension. We must provide a list if the features are not vectors.


            # 'video'-related parameters

            :param max_video_len: maximum video length, the rest of the data will be padded with 0s (only applicable if the input data is of type 'video' or video-features').
        """
        self.__checkSetName(set_name)

        # Insert type and id of input data
        keys_X_set = eval('self.X_' + set_name + '.keys()')
        if id not in self.ids_inputs:
            self.ids_inputs.append(id)
            self.types_inputs.append(type)
        elif id in keys_X_set and not overwrite_split and not add_additional:
            raise Exception('An input with id "' + id + '" is already loaded into the Database.')

        if not required and id not in self.optional_inputs:
            self.optional_inputs.append(id)

        if type not in self.__accepted_types_inputs:
            raise NotImplementedError(
                'The input type "' + type + '" is not implemented. The list of valid types are the following: ' + str(
                    self.__accepted_types_inputs))

        # Proprocess the input data depending on its type
        if type == 'raw-image':
            data = self.preprocessImages(path_list, id, set_name, img_size, img_size_crop, use_RGB)
        elif type == 'video':
            data = self.preprocessVideos(path_list, id, set_name, max_video_len, img_size, img_size_crop)
        elif type == 'text':
            if self.max_text_len.get(id) is None:
                self.max_text_len[id] = dict()
            data = self.preprocessText(path_list, id, set_name, tokenization, build_vocabulary, max_text_len,
                                       max_words, offset, fill, min_occ, pad_on_batch, words_so_far,
                                       bpe_codes=bpe_codes, separator=separator)
        elif type == 'doc':
            if self.max_text_len.get(id) is None:
                self.max_text_len[id] = dict()
            data = self.preprocessDoc(path_list, id, set_name, tokenization, build_vocabulary, max_text_len,
                                       max_words, offset, fill, min_occ, pad_on_batch, words_so_far,
                                       bpe_codes=bpe_codes, separator=separator)

        elif type == 'image-features':
            data = self.preprocessFeatures(path_list, id, set_name, feat_len)
        elif type == 'video-features':
            # Check if the chosen data augmentation types exists
            if data_augmentation_types is not None:
                for da in data_augmentation_types:
                    if da not in self.__available_augm_vid_feat:
                        raise NotImplementedError(
                            'The chosen data augmentation type ' + da + ' is not implemented for the type "video-features".')
            self.inputs_data_augmentation_types[id] = data_augmentation_types
            data = self.preprocessVideoFeatures(path_list, id, set_name, max_video_len, img_size, img_size_crop,
                                                feat_len)
        elif type == 'id':
            data = self.preprocessIDs(path_list, id, set_name)
        elif type == 'ghost':
            data = []

        if isinstance(repeat_set, list) or isinstance(repeat_set, (np.ndarray, np.generic)) or repeat_set > 1:
            data = list(np.repeat(data, repeat_set))

        self.__setInput(data, set_name, type, id, overwrite_split, add_additional)

    def __setInput(self, set, set_name, type, id, overwrite_split, add_additional):
        #print("type set of __setInput:", len(set), "id", id) #Shu
        if add_additional:
            exec ('self.X_' + set_name + '[id] += set')
        else:
            exec ('self.X_' + set_name + '[id] = set')
        exec ('self.loaded_' + set_name + '[0] = True')
        #print("optional inputs", self.optional_inputs) #Shu
        if id not in self.optional_inputs:
            #print("Not optional inputs", id) #Shu
            exec ('self.len_' + set_name + ' = len(self.X_' + set_name + '[id])')
            if not overwrite_split and not add_additional:
                self.__checkLengthSet(set_name)

        if not self.silence:
            logging.info(
                'Loaded "' + set_name + '" set inputs of type "' + type + '" with id "' + id + '" and length ' + str(
                    eval('self.len_' + set_name)) + '.')

    def removeInput(self, set_name, id='label', type='categorical'):
        # Ensure that the output exists before removing it
        keys_X_set = eval('self.X_' + set_name + '.keys()')
        if id in self.ids_inputs:
            ind_remove = self.ids_inputs.index(id)
            del self.ids_inputs[ind_remove]
            del self.types_inputs[ind_remove]
            exec ('del self.X_' + set_name + '[id]')
        elif id not in keys_X_set:
            raise Exception('An input with id "' + id + '" does not exist in the Database.')
        if not self.silence:
            logging.info('Removed "' + set_name + '" set input of type "' + type + '" with id "' + id + '.')

    def setLabels(self, labels_list, set_name, type='categorical', id='label'):
        """
            DEPRECATED
        """
        logging.info("WARNING: The method setLabels() is deprecated, consider using setOutput() instead.")
        self.setOutput(labels_list, set_name, type=type, id=id)

    def setRawOutput(self, path_list, set_name, type='file-name', id='raw-text', overwrite_split=False, add_additional=False):
        """
            Loads a list which can contain all samples from either the 'train', 'val', or
            'test' set splits (specified by set_name).

            # General parameters

            :param path_list: can either be a path to a text file containing the paths to the images or a python list of paths
            :param set_name: identifier of the set split loaded ('train', 'val' or 'test')
            :param type: identifier of the type of input we are loading (accepted types can be seen in self.__accepted_types_inputs)
            :param id: identifier of the input data loaded
            :param repeat_set: repeats the inputs given (useful when we have more outputs than inputs). Int or array of ints.
            :param required: flag for optional inputs
        """
        self.__checkSetName(set_name)

        # Insert type and id of input data
        keys_Y_set = eval('self.Y_raw_' + set_name + '.keys()')
        if id not in self.ids_inputs:
            self.ids_inputs.append(id)
            self.types_inputs.append(type)
            if id not in self.optional_inputs:
                self.optional_inputs.append(id)  # This is always optional

        elif id in keys_Y_set and (not overwrite_split or not add_additional):
            raise Exception('An input with id "' + id + '" is already loaded into the Database.')

        if type not in self.__accepted_types_inputs:
            raise NotImplementedError(
                'The input type "' + type + '" is not implemented. The list of valid types are the following: ' + str(
                    self.__accepted_types_inputs))

        exec ('self.Y_raw_' + set_name + '[id] = path_list')
        exec ('self.loaded_raw_' + set_name + '[1] = True')

        if not self.silence:
            logging.info('Loaded "' + set_name + '" set inputs of type "' + type + '" with id "' + id + '".')

    def setOutput(self, path_list, set_name, type='categorical', id='label', repeat_set=1, overwrite_split=False, add_additional=False,
                  sample_weights=False,
                  tokenization='tokenize_none', max_text_len=0, offset=0, fill='end', min_occ=0,  # 'text'
                  pad_on_batch=True, words_so_far=False, build_vocabulary=False, max_words=0,  # 'text'
                  bpe_codes=None, separator='@@', # 'text'
                  associated_id_in=None, num_poolings=None,  # '3DLabel' or '3DSemanticLabel'
                  sparse=False, # 'binary'
                  ):
        """
            Loads a set of output data, usually (type=='categorical') referencing values in self.classes (starting from 0)

            # General parameters

            :param path_list: can either be a path to a text file containing the labels or a python list of labels.
            :param set_name: identifier of the set split loaded ('train', 'val' or 'test').
            :param type: identifier of the type of input we are loading (accepted types can be seen in self.__accepted_types_outputs).
            :param id: identifier of the input data loaded.
            :param repeat_set: repeats the outputs given (useful when we have more inputs than outputs). Int or array of ints.
            :param overwrite_split: indicates that we want to overwrite the data with id that was already declared in the dataset
            :param add_additional: adds additional data to an already existent output ID
            :param sample_weights: switch on/off sample weights usage for the current output

            # 'text'-related parameters

            :param tokenization: type of tokenization applied (must be declared as a method of this class) (only applicable when type=='text').
            :param build_vocabulary: whether a new vocabulary will be built from the loaded data or not (only applicable when type=='text').
            :param max_text_len: maximum text length, the rest of the data will be padded with 0s (only applicable if the output data is of type 'text') Set to 0 if the whole sentence will be used as an output class.
            :param max_words: a maximum of 'max_words' words from the whole vocabulary will be chosen by number or occurrences
            :param offset: number of timesteps that the text is shifted to the right (for sequential conditional models, which take as input the previous output)
            :param fill: select whether padding before or after the sequence
            :param min_occ: minimum number of occurrences allowed for the words in the vocabulary. (default = 0)
            :param pad_on_batch: the batch timesteps size will be set to the length of the largest sample +1 if True, max_len will be used as the fixed length otherwise
            :param words_so_far: if True, each sample will be represented as the complete set of words until the point defined by the timestep dimension (e.g. t=0 'a', t=1 'a dog', t=2 'a dog is', etc.)
            :param bpe_codes: Codes used for applying BPE encoding.
            :param separator: BPE encoding separator.

            # '3DLabel' or '3DSemanticLabel'-related parameters

            :param associated_id_in: id of the input 'raw-image' associated to the inputted 3DLabels or 3DSemanticLabel
            :param num_poolings: number of pooling layers used in the model (used for calculating output dimensions)

            # 'binary'-related parameters

            :param sparse: indicates if the data is stored as a list of lists with class indices, e.g. [[4, 234],[87, 222, 4568],[3],...]

        """
        self.__checkSetName(set_name)

        # Insert type and id of output data
        keys_Y_set = eval('self.Y_' + set_name + '.keys()')
        if id not in self.ids_outputs:
            self.ids_outputs.append(id)
            self.types_outputs.append(type)
        elif id in keys_Y_set and not overwrite_split and not add_additional:
            raise Exception('An output with id "' + id + '" is already loaded into the Database.')

        if type not in self.__accepted_types_outputs:
            raise NotImplementedError(
                'The output type "' + type + '" is not implemented. The list of valid types are the following: ' + str(
                    self.__accepted_types_outputs))

        # Preprocess the output data depending on its type
        if type == 'categorical':
            self.setClasses(path_list, id)
            data = self.preprocessCategorical(path_list, id,
                                              sample_weights=True if sample_weights and set_name == 'train' else False)
        elif type == 'text':
            if self.max_text_len.get(id) is None:
                self.max_text_len[id] = dict()
            data = self.preprocessText(path_list, id, set_name, tokenization, build_vocabulary, max_text_len,
                                       max_words, offset, fill, min_occ, pad_on_batch, words_so_far,
                                       bpe_codes=bpe_codes, separator=separator, use_extra_words=True, use_pad=True)
        elif type == 'binary':
            data = self.preprocessBinary(path_list, id, sparse)
        elif type == 'real':
            data = self.preprocessReal(path_list)
        elif type == 'id':
            data = self.preprocessIDs(path_list, id)
        elif (type == '3DLabel'):
            data = self.preprocess3DLabel(path_list, id, associated_id_in, num_poolings)
        elif (type == '3DSemanticLabel'):
            data = self.preprocess3DSemanticLabel(path_list, id, associated_id_in, num_poolings)

        if isinstance(repeat_set, list) or isinstance(repeat_set, (np.ndarray, np.generic)) or repeat_set > 1:
            data = list(np.repeat(data, repeat_set))
        if self.sample_weights.get(id) is None:
            self.sample_weights[id] = dict()
        if id in sample_weights:
            self.sample_weights[id][set_name] = sample_weights[id]
        self.__setOutput(data, set_name, type, id, overwrite_split, add_additional)



    def __setOutput(self, labels, set_name, type, id, overwrite_split, add_additional):
        if add_additional:
            exec('self.Y_' + set_name + '[id] += labels')
        else:
            exec('self.Y_' + set_name + '[id] = labels')
        exec ('self.loaded_' + set_name + '[1] = True')
        exec ('self.len_' + set_name + ' = len(self.Y_' + set_name + '[id])')
        if not overwrite_split and not add_additional:
            self.__checkLengthSet(set_name)

        if not self.silence:
            logging.info(
                'Loaded "' + set_name + '" set outputs of type "' + type + '" with id "' + id + '" and length ' + str(
                    eval('self.len_' + set_name)) + '.')

    def removeOutput(self, set_name, id='label', type='categorical'):
        # Ensure that the output exists before removing it
        keys_Y_set = eval('self.Y_' + set_name + '.keys()')
        if id in self.ids_outputs:
            ind_remove = self.ids_outputs.index(id)
            del self.ids_outputs[ind_remove]
            del self.types_outputs[ind_remove]
            exec ('del self.Y_' + set_name + '[id]')
        elif id not in keys_Y_set:
            raise Exception('An output with id "' + id + '" does not exist in the Database.')
        if not self.silence:
            logging.info('Removed "' + set_name + '" set outputs of type "' + type + '" with id "' + id + '.')

    # ------------------------------------------------------- #
    #       TYPE 'categorical' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #

    def setClasses(self, path_classes, id):
        """
        Loads the list of classes of the dataset.
        Each line must contain a unique identifier of the class.

        :param path_classes: Path to a text file with the classes or an instance of the class list.
        :param id: Dataset id

        :return: None
        """

        if isinstance(path_classes, str) and os.path.isfile(path_classes):
            classes = []
            with open(path_classes, 'r') as list_:
                for line in list_:
                    classes.append(line.rstrip('\n'))
            self.classes[id] = classes
        elif isinstance(path_classes, list):
            self.classes[id] = path_classes
        else:
            raise Exception('Wrong type for "path_classes".'
                            ' It must be a path to a text file with the classes or an instance of the class list.\n'
                            'It currently is: %s' % str(path_classes))

        self.dic_classes[id] = dict()
        for c in range(len(self.classes[id])):
            self.dic_classes[id][self.classes[id][c]] = c

        if not self.silence:
            logging.info('Loaded classes list with ' + str(len(self.dic_classes[id])) + " different labels.")

    def preprocessCategorical(self, labels_list, id, sample_weights=False):
        """
        Preprocesses categorical data.

        :param labels_list: Label list. Given as a path to a file or as an instance of the class list.

        :return: Preprocessed labels.
        """

        if isinstance(labels_list, str) and os.path.isfile(labels_list):
            labels = []
            with open(labels_list, 'r') as list_:
                for line in list_:
                    labels.append(int(line.rstrip('\n').split(' ')))
        elif isinstance(labels_list, list):
            labels = labels_list
        else:
            raise Exception('Wrong type for "labels_list". '
                            'It must be a path to a text file with the labels or an instance of the class list.\n'
                            'It currently is: %s' % str(labels_list))

        if sample_weights:
            n_classes = len(set(labels))
            counts_per_class = np.zeros((n_classes,))
            for lab in labels:
                counts_per_class[lab] += 1

            # Apply balanced weights per class
            inverse_counts_per_class = [sum(counts_per_class) - c_i for c_i in counts_per_class]
            weights_per_class = [float(c_i) / sum(inverse_counts_per_class) for c_i in inverse_counts_per_class]
            self.extra_variables['class_weights_' + id] = weights_per_class

        return labels


    def loadCategorical(self, y_raw, nClasses):
        y = np_utils.to_categorical(y_raw, nClasses).astype(np.uint8)
        return y

    # ------------------------------------------------------- #
    #       TYPE 'binary' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #

    def preprocessBinary(self, labels_list, id, sparse):
        """
        Preprocesses binary classes.

        :param labels_list: Binary label list given as an instance of the class list.
        :param sparse: indicates if the data is stored as a list of lists with class indices, e.g. [[4, 234],[87, 222, 4568],[3],...]

        :return: Preprocessed labels.
        """
        if not isinstance(labels_list, list):
            raise Exception('Wrong type for "path_list". It must be an instance of the class list.')

        if sparse:
            labels = labels_list
        else: # convert to sparse representation
            labels = [[str(i) for i, x in enumerate(y) if x == 1] for y in labels_list]
        self.sparse_binary[id] = True

        unique_label_set = []
        for sample in labels:
            if sample not in unique_label_set:
                unique_label_set.append(sample)
        y_vocab = ['::'.join(sample) for sample in unique_label_set]

        self.build_vocabulary(y_vocab, id, split_symbol='::', use_extra_words=False)

        return labels

    def loadBinary(self, y_raw, id):

        try:
            sparse = self.sparse_binary[id]
        except: # allows retrocompatibility
            sparse = False

        if sparse: # convert sparse into numpy array
            n_samples = len(y_raw)
            voc = self.vocabulary[id]['words2idx']
            num_words = len(voc.keys())
            y = np.zeros((n_samples, num_words), dtype=np.uint8)
            for i, y_ in enumerate(y_raw):
                for elem in y_:
                    y[i,voc[elem]] = 1
        else:
            y = np.array(y_raw).astype(np.uint8)

        return y

    # ------------------------------------------------------- #
    #       TYPE 'real' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #

    def preprocessReal(self, labels_list):
        """
        Preprocesses real classes.

        :param labels_list: Label list. Given as a path to a file or as an instance of the class list.

        :return: Preprocessed labels.
        """
        if isinstance(labels_list, str) and os.path.isfile(labels_list):
            labels = []
            with open(labels_list, 'r') as list_:
                for line in list_:
                    labels.append(float(line.rstrip('\n')))
        elif isinstance(labels_list, list):
            labels = labels_list
        else:
            raise Exception('Wrong type for "labels_list". '
                            'It must be a path to a text file with real values or an instance of the class list.\n'
                            'It currently is: %s' % str(labels_list))


        return labels

    # ------------------------------------------------------- #
    #       TYPE 'features' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #

    def preprocessFeatures(self, path_list, id, set_name, feat_len):
        """
        Preprocesses features. We should give a path to a text file where each line must contain a path to a .npy file storing a feature vector.
        Alternatively "path_list" can be an instance of the class list.

        :param path_list: Path to a text file where each line must contain a path to a .npy file storing a feature vector. Alternatively, instance of the class list.
        :param id: Dataset id
        :param set_name: Used?
        :param feat_len: Length of features. If all features have the same length, given as a number. Otherwise, list.

        :return: Preprocessed features
        """
        # file with a list, each line being a path to a .npy file with a feature vector
        #print("Shu test of visual features", path_list) #Shu
        if isinstance(path_list, str) and os.path.isfile(path_list):
            data = []
            if id == "visual_feature":
                import pandas as pd
                data_csv = pd.read_csv(path_list, header=None, sep=" ")
                data = data_csv.values.tolist()
            else:
                with open(path_list, 'r') as list_:
                    for line in list_:
                        # data.append(np.fromstring(line.rstrip('\n'), sep=','))
                        data.append(line.rstrip('\n'))
        elif isinstance(path_list, list):
            data = path_list
        else:
            raise Exception('Wrong type for "path_list". It must be a path to a text file. Each line must contain a path'
                ' to a .npy file storing a feature vector. Alternatively "path_list"'
                ' can be an instance of the class list.\n'
                'Currently it is: %s .' % str(path_list))

        if not isinstance(feat_len, list):
            feat_len = [feat_len]
        self.features_lengths[id] = feat_len
        #print("Feature length", len(data[0])) # Shu add
        return data

    def loadFeatures(self, X, feat_len, normalization_type='L2', normalization=False, loaded=False, external=False,
                     data_augmentation=True):
        """
        Loads and normalizes features.

        :param X: Features to load.
        :param feat_len: Length of the features.
        :param normalization_type: Normalization to perform to the features (see: self.__available_norm_feat)
        :param normalization: Whether to normalize or not the features.
        :param loaded: Flag that indicates if these features have been already loaded.
        :param external: Boolean indicating if the paths provided in 'X' are absolute paths to external images
        :param data_augmentation: Perform data augmentation (with mean=0.0, std_dev=0.01)

        :return: Loaded features as numpy array
        """
        #print("test Shu load features (visual features)", "normalisation", normalization, "data_aug", data_augmentation) #Shu
        #print("X in loadFeatures", type(X)) #Shu
        #print("length example of X visual feature", len(X[0])) #Shu
        if normalization and normalization_type not in self.__available_norm_feat:
            raise NotImplementedError(
                'The chosen normalization type ' + normalization_type + ' is not implemented for the type "image-features" and "video-features".')

        n_batch = len(X)
        #print("n_batch", n_batch) #Shu
        features = np.zeros(tuple([n_batch] + feat_len)) # row: n_batch; column: feat_len

        for i, feat in enumerate(X):
            if isinstance(feat, list):
                #print("feat0 printed", feat[0]) #Shu
                feat = np.asarray(feat)

            else:
                if not external:
                    feat = self.path + '/' + feat

                feat = np.load(feat)

            if data_augmentation:
                noise_mean = 0.0
                noise_dev = 0.01
                noise = np.random.normal(noise_mean, noise_dev, feat.shape)
                feat += noise

            if normalization:
                if normalization_type == 'L2':
                    feat = feat / np.linalg.norm(feat, ord=2)

            features[i] = feat

        return np.array(features)

    # ------------------------------------------------------- #
    #       TYPE 'text' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #

    def preprocessDoc(self, annotations_list, id, set_name, tokenization, build_vocabulary, max_text_len,
                       max_words, offset, fill, min_occ, pad_on_batch, words_so_far, bpe_codes=None, separator='@@',
                       use_extra_words=True, use_pad=False):
        """
        Preprocess 'text' data type: Builds vocabulary (if necessary) and preprocesses the sentences.
        Also sets Dataset parameters.

        :param annotations_list: Path to the sentences to process.
        :param id: Dataset id of the data.
        :param set_name: Name of the current set ('train', 'val', 'test')
        :param tokenization: Tokenization to perform.
        :param build_vocabulary: Whether we should build a vocabulary for this text or not.
        :param max_text_len: Maximum length of the text. If max_text_len == 0, we treat the full sentence as a class.
        :param max_words: Maximum number of words to include in the dictionary.
        :param offset: Text shifting.
        :param fill: Whether we path with zeros at the beginning or at the end of the sentences.
        :param min_occ: Minimum occurrences of each word to be included in the dictionary.
        :param pad_on_batch: Whether we get sentences with length of the maximum length of the minibatch or sentences with a fixed (max_text_length) length.
        :param words_so_far: Experimental feature. Should be ignored.
        :param bpe_codes: Codes used for applying BPE encoding.
        :param separator: BPE encoding separator.

        :return: Preprocessed sentences.
        """
        #print("Preprocessing doc Shu, path:", annotations_list) #Shu
        sentences = []
        sentences2d = []
        if isinstance(annotations_list, str) and os.path.isfile(annotations_list):
            with open(annotations_list, 'r') as list_:
                sentences_doc = []
                sent_counter = 0
                for line in list_:
                    new_line = line.rstrip('\n')
                    if (new_line == '#doc#'):
                        if sent_counter < self.doc_size:
                            for i in range(self.doc_size - sent_counter):
                                sentences_doc.append('<pad> <pad> <pad>')
                        sentences2d.append(sentences_doc)
                        sentences_doc = []
                        sent_counter = 0
                    else:
                        if sent_counter < self.doc_size:
                            sentences.append(new_line)
                            sentences_doc.append(new_line)
                            sent_counter += 1

        elif isinstance(annotations_list, list):
            sentences = annotations_list
        else:
            raise Exception(
                'Wrong type for "annotations_list". It must be a path to a text file with the sentences or a list of sentences. '
                'It currently is: %s' % (str(annotations_list)))

        # Check if tokenization method exists
        if hasattr(self, tokenization):
            if 'bpe' in tokenization.lower():
                assert bpe_codes is not None, 'bpe_codes must be specified when applying a BPE tokenization.'
                self.build_bpe(bpe_codes, separator)
            tokfun = eval('self.' + tokenization)
            if not self.silence:
                logging.info('\tApplying tokenization function: "' + tokenization + '".')
        else:
            raise Exception('Tokenization procedure "' + tokenization + '" is not implemented.')

        # Tokenize sentences
        if max_text_len != 0:  # will only tokenize if we are not using the whole sentence as a class
            for i in range(len(sentences)):
                sentences[i] = tokfun(sentences[i])

        # Build vocabulary
        error_vocab = False
        if build_vocabulary == True:
            self.build_vocabulary(sentences, id, max_text_len != 0, min_occ=min_occ, n_words=max_words,
                                  use_extra_words=use_extra_words, use_pad=use_pad)
        elif isinstance(build_vocabulary, str):
            if build_vocabulary in self.vocabulary:
                self.vocabulary[id] = self.vocabulary[build_vocabulary]
                if not self.silence:
                    logging.info('\tReusing vocabulary named "' + build_vocabulary + '" for data with id "' + id + '".')
            else:
                raise Exception('The parameter "build_vocabulary" must be a boolean '
                                'or a str containing an id of the vocabulary we want to copy.\n'
                                'It currently is: %s' % str(build_vocabulary))

        elif isinstance(build_vocabulary, dict):
            self.vocabulary[id] = build_vocabulary
            if not self.silence:
                logging.info('\tReusing vocabulary from dictionary for data with id "' + id + '".')

        if not id in self.vocabulary:
            raise Exception('The dataset must include a vocabulary with'
                            ' id "' + id + '" in order to process the type "text" data. '
                                           'Set "build_vocabulary" to True if you want '
                                           'to use the current data for building the vocabulary.')

        # Store max text len
        self.max_text_len[id][set_name] = max_text_len
        self.text_offset[id] = offset
        self.fill_text[id] = fill
        self.pad_on_batch[id] = pad_on_batch
        self.words_so_far[id] = words_so_far

        return sentences2d

    def preprocessText(self, annotations_list, id, set_name, tokenization, build_vocabulary, max_text_len,
                       max_words, offset, fill, min_occ, pad_on_batch, words_so_far, bpe_codes=None, separator='@@', use_extra_words=True, use_pad=False):
        """
        Preprocess 'text' data type: Builds vocabulary (if necessary) and preprocesses the sentences.
        Also sets Dataset parameters.

        :param annotations_list: Path to the sentences to process.
        :param id: Dataset id of the data.
        :param set_name: Name of the current set ('train', 'val', 'test')
        :param tokenization: Tokenization to perform.
        :param build_vocabulary: Whether we should build a vocabulary for this text or not.
        :param max_text_len: Maximum length of the text. If max_text_len == 0, we treat the full sentence as a class.
        :param max_words: Maximum number of words to include in the dictionary.
        :param offset: Text shifting.
        :param fill: Whether we path with zeros at the beginning or at the end of the sentences.
        :param min_occ: Minimum occurrences of each word to be included in the dictionary.
        :param pad_on_batch: Whether we get sentences with length of the maximum length of the minibatch or sentences with a fixed (max_text_length) length.
        :param words_so_far: Experimental feature. Should be ignored.
        :param bpe_codes: Codes used for applying BPE encoding.
        :param separator: BPE encoding separator.

        :return: Preprocessed sentences.
        """
        sentences = []
        if isinstance(annotations_list, str) and os.path.isfile(annotations_list):
            with codecs.open(annotations_list, 'r', 'utf-8') as list_:
                for line in list_:
                    sentences.append(line.rstrip('\n'))
        elif isinstance(annotations_list, list):
            sentences = annotations_list
        else:
            raise Exception(
                'Wrong type for "annotations_list". It must be a path to a text file with the sentences or a list of sentences. '
                'It currently is: %s' % (str(annotations_list)))

        # Check if tokenization method exists
        if hasattr(self, tokenization):
            if 'bpe' in tokenization.lower():
                assert bpe_codes is not None, 'bpe_codes must be specified when applying a BPE tokenization.'
                self.build_bpe(bpe_codes, separator)
            tokfun = eval('self.' + tokenization)
            if not self.silence:
                logging.info('\tApplying tokenization function: "' + tokenization + '".')
        else:
            raise Exception('Tokenization procedure "' + tokenization + '" is not implemented.')

        # Tokenize sentences
        if max_text_len != 0:  # will only tokenize if we are not using the whole sentence as a class
            for i in range(len(sentences)):
                sentences[i] = tokfun(sentences[i])

        # Build vocabulary
        error_vocab = False
        if build_vocabulary == True:
            self.build_vocabulary(sentences, id, max_text_len != 0, min_occ=min_occ, n_words=max_words, use_extra_words=use_extra_words, use_pad=use_pad)
        elif isinstance(build_vocabulary, str):
            if build_vocabulary in self.vocabulary:
                self.vocabulary[id] = self.vocabulary[build_vocabulary]
                if not self.silence:
                    logging.info('\tReusing vocabulary named "' + build_vocabulary + '" for data with id "' + id + '".')
            else:
                raise Exception('The parameter "build_vocabulary" must be a boolean '
                                'or a str containing an id of the vocabulary we want to copy.\n'
                                'It currently is: %s' % str(build_vocabulary))

        elif isinstance(build_vocabulary, dict):
            self.vocabulary[id] = build_vocabulary
            if not self.silence:
                logging.info('\tReusing vocabulary from dictionary for data with id "' + id + '".')

        if not id in self.vocabulary:
            raise Exception('The dataset must include a vocabulary with'
                            ' id "' + id + '" in order to process the type "text" data. '
                            'Set "build_vocabulary" to True if you want '
                            'to use the current data for building the vocabulary.')

        # Store max text len
        self.max_text_len[id][set_name] = max_text_len
        self.text_offset[id] = offset
        self.fill_text[id] = fill
        self.pad_on_batch[id] = pad_on_batch
        self.words_so_far[id] = words_so_far


        return sentences

    def build_vocabulary(self, captions, id, tokfun=None, do_split=True, min_occ=0, n_words=0, split_symbol=' ', use_pad=False, use_extra_words=True):
        """
        Vocabulary builder for data of type 'text'

        :param captions: Corpus sentences
        :param id: Dataset id of the text
        :param tokfun: Tokenization function. (used?)
        :param do_split: Split sentence by words or use the full sentence as a class.
        :param split_symbol: symbol used for separating the elements in each sentence
        :param min_occ: Minimum occurrences of each word to be included in the dictionary.
        :param n_words: Maximum number of words to include in the dictionary.
        :return: None.
        """
        if not self.silence:
            logging.info("Creating vocabulary for data with id '" + id + "'.")

        counters = []
        sentence_counts = []
        counter = Counter()
        sentence_count = 0
        for line in captions:
            if do_split:
                words = line.strip().split(split_symbol)
                counter.update(words)
            else:
                counter.update([line])
            sentence_count += 1

        if not do_split and not self.silence:
            logging.info('Using whole sentence as a single word.')

        counters.append(counter)
        sentence_counts.append(sentence_count)
        # logging.info("\t %d unique words in %d sentences with a total of %d words." %
        #      (len(counter), sentence_count, sum(counter.values())))

        combined_counter = reduce(add, counters)
        if not self.silence:
            logging.info("\t Total: %d unique words in %d sentences with a total of %d words." %
                         (len(combined_counter), sum(sentence_counts), sum(combined_counter.values())))

        # keep only words with less than 'min_occ' occurrences
        if min_occ > 1:
            removed = 0
            for k in combined_counter.keys():
                if combined_counter[k] < min_occ:
                    del combined_counter[k]
                    removed += 1
            if not self.silence:
                logging.info("\t Removed %d words with less than %d occurrences. New total: %d." %
                             (removed, min_occ, len(combined_counter)))

        # keep only top 'n_words'
        if n_words > 0:
            if use_extra_words:
                vocab_count = combined_counter.most_common(n_words - len(self.extra_words))
            else:
                vocab_count = combined_counter.most_common(n_words)
            if not self.silence:
                logging.info("Creating dictionary of %s most common words, covering "
                             "%2.1f%% of the text."
                             % (n_words,
                                100.0 * sum([count for word, count in vocab_count]) /
                                sum(combined_counter.values())))
        else:
            if not self.silence:
                logging.info("Creating dictionary of all words")
            vocab_count = counter.most_common()

        dictionary = {}
        for i, (word, count) in enumerate(vocab_count):
            dictionary[word] = i
            if use_extra_words:
                dictionary[word] += len(self.extra_words)
            # if use_pad:
            #     dictionary[word] += len(self.extra_pad)

        if use_extra_words:
            for w, k in self.extra_words.iteritems():
                dictionary[w] = k

        # if use_pad:
        #     for w, k in self.extra_pad.iteritems():
        #         dictionary[w] = k


        # Store dictionary and append to previously existent if needed.
        if id not in self.vocabulary:
            self.vocabulary[id] = dict()
            self.vocabulary[id]['words2idx'] = dictionary
            inv_dictionary = {v: k for k, v in dictionary.items()}
            self.vocabulary[id]['idx2words'] = inv_dictionary

            self.vocabulary_len[id] = len(vocab_count)
            if use_extra_words:
                self.vocabulary_len[id] += len(self.extra_words)

        else:
            old_keys = self.vocabulary[id]['words2idx'].keys()
            new_keys = dictionary.keys()
            added = 0
            for key in new_keys:
                if key not in old_keys:
                    self.vocabulary[id]['words2idx'][key] = self.vocabulary_len[id]
                    self.vocabulary_len[id] += 1
                    added += 1

            inv_dictionary = {v: k for k, v in self.vocabulary[id]['words2idx'].items()}
            self.vocabulary[id]['idx2words'] = inv_dictionary

            if not self.silence:
                logging.info('Appending ' + str(added) + ' words to dictionary with id "' + id + '".')
                logging.info('\tThe new total is ' + str(self.vocabulary_len[id]) + '.')

    def merge_vocabularies(self, ids):
        '''
        Merges the vocabularies from a set of text inputs/outputs into a single one.

        :param ids: identifiers of the inputs/outputs whose vocabularies will be merged
        :return: None
        '''
        assert isinstance(ids, list), 'ids must be a list of inputs/outputs identifiers of type text'
        if not self.silence:
            logging.info('Merging vocabularies of the following ids: ' + str(ids))

        # Pick the first vocabulary as reference
        vocab_ref = self.vocabulary[ids[0]]['words2idx']
        next_idx = max(vocab_ref.values()) + 1

        # Merge all vocabularies to the reference
        for i in range(1, len(ids)):
            id = ids[i]
            vocab = self.vocabulary[id]['words2idx']
            for w in vocab.keys():
                if w not in vocab_ref.keys():
                    vocab_ref[w] = next_idx
                    next_idx += 1

        # Also build idx2words
        self.vocabulary[ids[0]]['words2idx'] = vocab_ref
        inv_dictionary = {v: k for k, v in vocab_ref.items()}
        self.vocabulary[ids[0]]['idx2words'] = inv_dictionary
        self.vocabulary_len[ids[0]] = len(self.vocabulary[ids[0]]['words2idx'].keys())

        # Insert in all ids
        for i in range(1, len(ids)):
            self.vocabulary[ids[i]]['words2idx'] = self.vocabulary[ids[0]]['words2idx']
            self.vocabulary[ids[i]]['idx2words'] = self.vocabulary[ids[0]]['idx2words']
            self.vocabulary_len[ids[i]] = self.vocabulary_len[ids[0]]

        if not self.silence:
            logging.info('\tThe new total is ' + str(self.vocabulary_len[ids[0]]) + '.')

    def build_bpe(self, codes, separator='@@', vocabulary=None, glossaries=None):
        """
        Constructs a BPE encoder instance. Currently, vocabulary and glossaries options are not implemented.
        :param codes: File with BPE codes (created by learn_bpe.py)
        :param separator: Separator between non-final subword units (default: '@@'))
        :param vocabulary: Vocabulary file. If provided, this script reverts any merge operations that produce an OOV.
        :param glossaries: The strings provided in glossaries will not be affected
                           by the BPE (i.e. they will neither be broken into subwords,
                           nor concatenated with other subwords.
        :return: None
        """
        from keras_wrapper.extra.external import BPE
        with open(codes, 'r') as cods:
            self.BPE = BPE(cods, separator, vocabulary, glossaries)
        self.BPE_separator = separator
        self.BPE_built = True

    def load3DLabels(self, bbox_list, nClasses, dataAugmentation, daRandomParams, img_size, size_crop, image_list):
        '''
        Loads a set of outputs of the type 3DLabel (used for detection)

        :param bbox_list: list of bboxes, labels and original sizes
        :param nClasses: number of different classes to be detected
        :param dataAugmentation: are we applying data augmentation?
        :param daRandomParams: random parameters applied on data augmentation (vflip, hflip and random crop)
        :param img_size: resized applied to input images
        :param size_crop: crop size applied to input images
        :param image_list: list of input images used as identifiers to 'daRandomParams'
        :return: 3DLabels with shape (batch_size, width*height, classes)
        '''

        n_samples = len(bbox_list)
        h, w, d = img_size
        h_crop, w_crop, d_crop = size_crop
        labels = np.zeros((n_samples, nClasses, h_crop, w_crop), dtype=np.float32)

        for i in range(n_samples):
            line = bbox_list[i]
            arrayLine = line.split(';')
            arrayBndBox = arrayLine[:-1]
            w_original, h_original, d_original = eval(arrayLine[-1])

            label3D = np.zeros((nClasses, h_original, w_original), dtype=np.float32)

            for array in arrayBndBox:
                bndbox = eval(array)[0]
                idxclass = eval(array)[1]

                # bndbox_ones = np.ones((bndbox[3] - bndbox[1] + 1, bndbox[2] - bndbox[0] + 1))
                # label3D[idxclass, bndbox[1] - 1:bndbox[3], bndbox[0] - 1:bndbox[2]] = bndbox_ones

                bndbox_ones = np.ones((bndbox[2] - bndbox[0] + 1, bndbox[3] - bndbox[1] + 1))
                label3D[idxclass, bndbox[0] - 1:bndbox[2], bndbox[1] - 1:bndbox[3]] = bndbox_ones

            if not dataAugmentation or daRandomParams == None:
                # Resize 3DLabel to crop size.
                for j in range(nClasses):
                    label2D = misc.imresize(label3D[j], (h_crop, w_crop))
                    maxval = np.max(label2D)
                    if maxval > 0: label2D /= maxval
                    labels[i, j] = label2D
            else:
                label3D_rs = np.zeros((nClasses, h_crop, w_crop), dtype=np.float32)
                # Crop the labels (random crop)
                for j in range(nClasses):
                    label2D = misc.imresize(label3D[j], (h, w))
                    maxval = np.max(label2D)
                    if maxval > 0: label2D /= maxval
                    randomParams = daRandomParams[image_list[i]]
                    # Take random crop
                    left = randomParams["left"]
                    right = np.add(left, size_crop[0:2])

                    label2D = label2D[left[0]:right[0], left[1]:right[1]]

                    # Randomly flip (with a certain probability)
                    flip = randomParams["hflip"]
                    prob_flip_horizontal = randomParams["prob_flip_horizontal"]
                    if flip < prob_flip_horizontal:  # horizontal flip
                        label2D = np.fliplr(label2D)
                    flip = randomParams["vflip"]
                    prob_flip_vertical = randomParams["prob_flip_vertical"]
                    if flip < prob_flip_vertical:  # vertical flip
                        label2D = np.flipud(label2D)

                    label3D_rs[j] = label2D

                labels[i] = label3D_rs

        # Reshape labels to (batch_size, width*height, classes) before returning
        labels = np.reshape(labels, (n_samples, nClasses, w_crop * h_crop))
        labels = np.transpose(labels, (0, 2, 1))

        return labels

    def load3DSemanticLabels(self, labeled_images_list, nClasses, classes_to_colour, dataAugmentation, daRandomParams,
                             img_size, size_crop, image_list):
        '''
        Loads a set of outputs of the type 3DSemanticLabel (used for semantic segmentation TRAINING)

        :param labeled_images_list: list of labeled images
        :param nClasses: number of different classes to be detected
        :param classes_to_colour: dictionary relating each class id to their corresponding colour in the labeled image
        :param dataAugmentation: are we applying data augmentation?
        :param daRandomParams: random parameters applied on data augmentation (vflip, hflip and random crop)
        :param img_size: resized applied to input images
        :param size_crop: crop size applied to input images
        :param image_list: list of input images used as identifiers to 'daRandomParams'
        :return: 3DSemanticLabels with shape (batch_size, width*height, classes)
        '''

        n_samples = len(labeled_images_list)
        h, w, d = img_size
        h_crop, w_crop, d_crop = size_crop
        labels = np.zeros((n_samples, nClasses, h_crop, w_crop), dtype=np.float32)

        for i in range(n_samples):
            line = labeled_images_list[i].rstrip('\n')

            ### Load labeled GT image
            labeled_im = self.path + '/' + line
            # Check if the filename includes the extension
            [path, filename] = ntpath.split(labeled_im)
            [filename, ext] = os.path.splitext(filename)
            # If it doesn't then we find it
            if not ext:
                filename = fnmatch.filter(os.listdir(path), filename + '*')
                if not filename:
                    raise Exception('Non existent image ' + labeled_im)
                else:
                    labeled_im = path + '/' + filename[0]
            # Read image
            try:
                logging.disable(logging.CRITICAL)
                labeled_im = pilimage.open(labeled_im)
                labeled_im = np.asarray(labeled_im)
                logging.disable(logging.NOTSET)
                labeled_im = misc.imresize(labeled_im, (h, w))
            except:
                logging.warning("WARNING!")
                logging.warning("Can't load image " + labeled_im)
                labeled_im = np.zeros((h, w))

            label3D = np.zeros((nClasses, h, w), dtype=np.float32)

            # Insert 1s in the corresponding positions for each class
            for class_id, colour in classes_to_colour.iteritems():
                # indices = np.where(np.all(labeled_im == colour, axis=-1))
                indices = np.where(labeled_im == class_id)
                num_vals = len(indices[0])
                if num_vals > 0:
                    for idx_pos in range(num_vals):
                        x, y = indices[0][idx_pos], indices[1][idx_pos]
                        label3D[class_id, x, y] = 1.

            if not dataAugmentation or daRandomParams == None:
                # Resize 3DLabel to crop size.
                for j in range(nClasses):
                    label2D = misc.imresize(label3D[j], (h_crop, w_crop))
                    maxval = np.max(label2D)
                    if maxval > 0: label2D /= maxval
                    labels[i, j] = label2D
            else:
                label3D_rs = np.zeros((nClasses, h_crop, w_crop), dtype=np.float32)
                # Crop the labels (random crop)
                for j in range(nClasses):
                    label2D = misc.imresize(label3D[j], (h, w))
                    maxval = np.max(label2D)
                    if maxval > 0: label2D /= maxval
                    randomParams = daRandomParams[image_list[i]]
                    # Take random crop
                    left = randomParams["left"]
                    right = np.add(left, size_crop[0:2])

                    label2D = label2D[left[0]:right[0], left[1]:right[1]]

                    # Randomly flip (with a certain probability)
                    flip = randomParams["hflip"]
                    prob_flip_horizontal = randomParams["prob_flip_horizontal"]
                    if flip < prob_flip_horizontal:  # horizontal flip
                        label2D = np.fliplr(label2D)
                    flip = randomParams["vflip"]
                    prob_flip_vertical = randomParams["prob_flip_vertical"]
                    if flip < prob_flip_vertical:  # vertical flip
                        label2D = np.flipud(label2D)

                    label3D_rs[j] = label2D

                labels[i] = label3D_rs

        # Reshape labels to (batch_size, width*height, classes) before returning
        labels = np.reshape(labels, (n_samples, nClasses, w_crop * h_crop))
        labels = np.transpose(labels, (0, 2, 1))

        return labels

    def loadText(self, X, vocabularies, max_len, offset, fill, pad_on_batch, words_so_far, loading_X=False):
        """
        Text encoder: Transforms samples from a text representation into a numerical one. It also masks the text.

        :param X: Text to encode.
        :param vocabularies: Mapping word -> index
        :param max_len: Maximum length of the text.
        :param offset: Shifts the text to the right, adding null symbol at the start
        :param fill: 'start': the resulting vector will be filled with 0s at the beginning, 'end': it will be filled with 0s at the end, 'center': the vector will be surrounded by 0s, both at beginning and end
        :param pad_on_batch: Whether we get sentences with length of the maximum length of the minibatch or sentences with a fixed (max_text_length) length.
        :param words_so_far: Experimental feature. Use with caution.
        :param loading_X: Whether we are loading an input or an output of the model
        :return: Text as sequence of number. Mask for each sentence.
        """
        vocab = vocabularies['words2idx']
        n_batch = len(X)
        if max_len == 0:  # use whole sentence as class
            X_out = np.zeros(n_batch).astype('int32')
            for i in range(n_batch):
                w = X[i]
                X_out[i] = vocab.get(w, vocab['<unk>'])
            if loading_X:
                X_out = (X_out, None)  # This None simulates a mask
        else:  # process text as a sequence of words
            if pad_on_batch:
                max_len_batch = min(max([len(x.split(' ')) for x in X]) + 1, max_len)
            else:
                max_len_batch = max_len

            if words_so_far:
                X_out = np.ones((n_batch, max_len_batch, max_len_batch)).astype('int32') * self.extra_words['<pad>']
                X_mask = np.zeros((n_batch, max_len_batch, max_len_batch)).astype('int8')
                null_row = np.ones((1, max_len_batch)).astype('int32') * self.extra_words['<pad>']
                zero_row = np.zeros((1, max_len_batch)).astype('int8')
                if offset > 0:
                    null_row[0] = np.append([vocab['<null>']] * offset, null_row[0, :-offset])
            else:
                X_out = np.ones((n_batch, max_len_batch)).astype('int32') * self.extra_words['<pad>']
                X_mask = np.zeros((n_batch, max_len_batch)).astype('int8')

            if max_len_batch == max_len:
                max_len_batch -= 1  # always leave space for <eos> symbol
            # fills text vectors with each word (fills with 0s or removes remaining words w.r.t. max_len)
            for i in range(n_batch):
                x = X[i].strip().split(' ')
                len_j = len(x)
                if fill == 'start':
                    offset_j = max_len_batch - len_j - 1
                elif fill == 'center':
                    offset_j = (max_len_batch - len_j) / 2
                    len_j = len_j + offset_j
                else:
                    offset_j = 0
                    len_j = min(len_j, max_len_batch)
                if offset_j < 0:
                    len_j = len_j + offset_j
                    offset_j = 0

                if words_so_far:
                    for j, w in zip(range(len_j), x[:len_j]):
                        next_w = vocab.get(w, next_w = vocab['<unk>'])
                        for k in range(j, len_j):
                            X_out[i, k + offset_j, j + offset_j] = next_w
                            X_mask[i, k + offset_j, j + offset_j] = 1  # fill mask
                        X_mask[i, j + offset_j, j + 1 + offset_j] = 1  # add additional 1 for the <eos> symbol

                else:
                    for j, w in zip(range(len_j), x[:len_j]):
                        X_out[i, j + offset_j] = vocab.get(w, vocab['<unk>'])
                        X_mask[i, j + offset_j] = 1  # fill mask
                    X_mask[i, len_j + offset_j] = 1  # add additional 1 for the <eos> symbol

                if offset > 0:  # Move the text to the right -> null symbol
                    if words_so_far:
                        for k in range(len_j):
                            X_out[i, k] = np.append([vocab['<null>']] * offset, X_out[i, k, :-offset])
                            X_mask[i, k] = np.append([0] * offset, X_mask[i, k, :-offset])
                        X_out[i] = np.append(null_row, X_out[i, :-offset], axis=0)
                        X_mask[i] = np.append(zero_row, X_mask[i, :-offset], axis=0)
                    else:
                        X_out[i] = np.append([vocab['<null>']] * offset, X_out[i, :-offset])
                        X_mask[i] = np.append([0] * offset, X_mask[i, :-offset])
                if offset < 0:
                    X_out[i] = np.append(X_out[i, 1:],[vocab['<pad>']] * -offset)
                    X_mask[i] = np.append(X_mask[i, 1:],[0] * -offset)
            X_out = (X_out, X_mask)

        return X_out

    def loadTextOneHot(self, X, vocabularies, vocabulary_len, max_len, offset, fill, pad_on_batch, words_so_far,
                       sample_weights=False, loading_X=False):

        """
        Text encoder: Transforms samples from a text representation into a one-hot. It also masks the text.

        :param X: Text to encode.
        :param vocabularies: Mapping word -> index
        :param max_len: Maximum length of the text.
        :param offset: Shifts the text to the right, adding null symbol at the start
        :param fill: 'start': the resulting vector will be filled with 0s at the beginning, 'end': it will be filled with 0s at the end, 'center': the vector will be surrounded by 0s, both at beginning and end
        :param pad_on_batch: Whether we get sentences with length of the maximum length of the minibatch or sentences with a fixed (max_text_length) length.
        :param words_so_far: Experimental feature. Use with caution.
        :param loading_X: Whether we are loading an input or an output of the model
        :return: Text as sequence of number. Mask for each sentence.
        """

        y = self.loadText(X, vocabularies, max_len, offset, fill, pad_on_batch,
                          words_so_far, loading_X=loading_X)
        # Use whole sentence as class (classifier model)
        if max_len == 0:
            y_aux = np_utils.to_categorical(y, vocabulary_len).astype(np.uint8)
        # Use words separately (generator model)
        else:
            y_aux = np.zeros(list(y[0].shape) + [vocabulary_len]).astype(np.uint8)
            for idx in range(y[0].shape[0]):
                y_aux[idx] = np_utils.to_categorical(y[0][idx], vocabulary_len).astype(
                    np.uint8)
            if sample_weights:
                y_aux = (y_aux, y[1])  # join data and mask
        return y_aux

    def loadMapping(self, path_list):
        """
        Loads a mapping of Source -- Target words.
        :param path_list: Pickle object with the mapping
        :return: None
        """
        if not self.silence:
            logging.info("Loading source -- target mapping.")
        self.mapping = pk.load(open(path_list, 'rb'))
        if not self.silence:
            logging.info("Source -- target mapping loaded with a total of %d words." % len(self.mapping.keys()))

    # ------------------------------------------------------- #
    #       Tokenization functions
    # ------------------------------------------------------- #

    def tokenize_basic(self, caption, lowercase=True):
        """
        Basic tokenizer for the input/output data of type 'text':
           * Splits punctuation
           * Optional lowercasing

        :param caption: String to tokenize
        :param lowercase: Whether to lowercase the caption or not
        :return: Tokenized version of caption
        """

        punct = ['.', ';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',',
                 '?', '!']

        def processPunctuation(inText):
            outText = inText
            for p in punct:
                outText = outText.replace(p, ' ' + p + ' ')
            return outText

        resAns = caption.lower() if lowercase else caption
        resAns = resAns.replace('\n', ' ')
        resAns = resAns.replace('\t', ' ')
        resAns = processPunctuation(resAns)
        resAns = resAns.replace('  ', ' ')
        return resAns

    def tokenize_aggressive(self, caption, lowercase=True):
        """
        Aggressive tokenizer for the input/output data of type 'text':
        * Removes punctuation
        * Optional lowercasing

        :param caption: String to tokenize
        :param lowercase: Whether to lowercase the caption or not
        :return: Tokenized version of caption
        """
        punct = ['.', ';', r"/", '[', ']', '"', '{', '}', '(', ')',
                 '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!',
                 '¿', '¡', '\n', '\t', '\r']

        def processPunctuation(inText):
            outText = inText
            for p in punct:
                outText = outText.replace(p, '')
            return outText

        resAns = caption.lower() if lowercase else caption
        resAns = processPunctuation(resAns)
        resAns = re.sub('[  ]+', ' ', resAns)
        resAns = resAns.strip()
        return resAns

    def tokenize_icann(self, caption):
        """
        Tokenization used for the icann paper:
        * Removes some punctuation (. , ")
        * Lowercasing

        :param caption: String to tokenize
        :return: Tokenized version of caption
        """
        tokenized = re.sub('[.,"\n\t]+', '', caption)
        tokenized = re.sub('[  ]+', ' ', tokenized)
        tokenized = map(lambda x: x.lower(), tokenized.split())
        tokenized = " ".join(tokenized)
        return tokenized

    def tokenize_montreal(self, caption):
        """
        Similar to tokenize_icann
            * Removes some punctuation
            * Lowercase

        :param caption: String to tokenize
        :return: Tokenized version of caption
        """
        tokenized = re.sub('[.,"\n\t]+', '', caption.strip())
        tokenized = re.sub('[\']+', " '", tokenized)
        tokenized = re.sub('[  ]+', ' ', tokenized)
        tokenized = map(lambda x: x.lower(), tokenized.split())
        tokenized = " ".join(tokenized)
        return tokenized

    def tokenize_soft(self, caption, lowercase=True):
        """
        Tokenization used for the icann paper:
            * Removes very little punctuation
            * Lowercase
        :param caption: String to tokenize
        :param lowercase: Whether to lowercase the caption or not
        :return: Tokenized version of caption
        """
        tokenized = re.sub('[\n\t]+', '', caption.strip())
        tokenized = re.sub('[\.]+', ' . ', tokenized)
        tokenized = re.sub('[,]+', ' , ', tokenized)
        tokenized = re.sub('[!]+', ' ! ', tokenized)
        tokenized = re.sub('[?]+', ' ? ', tokenized)
        tokenized = re.sub('[\{]+', ' { ', tokenized)
        tokenized = re.sub('[\}]+', ' } ', tokenized)
        tokenized = re.sub('[\(]+', ' ( ', tokenized)
        tokenized = re.sub('[\)]+', ' ) ', tokenized)
        tokenized = re.sub('[\[]+', ' [ ', tokenized)
        tokenized = re.sub('[\]]+', ' ] ', tokenized)
        tokenized = re.sub('["]+', ' " ', tokenized)
        tokenized = re.sub('[\']+', " ' ", tokenized)
        tokenized = re.sub('[  ]+', ' ', tokenized)
        tokenized = map(lambda x: x.lower(), tokenized.split())
        tokenized = " ".join(tokenized)
        return tokenized

    def tokenize_none(self, caption):
        """
        Does not tokenizes the sentences. Only performs a stripping

        :param caption: String to tokenize
        :return: Tokenized version of caption
        """
        tokenized = re.sub('[\n\t]+', '', caption.strip())
        return tokenized

    def tokenize_none_char(self, caption):
        """
        Character-level tokenization. Respects all symbols. Separates chars. Inserts <space> sybmol for spaces.
        If found an escaped char, "&apos;" symbol, it is converted to the original one
        # List of escaped chars (by moses tokenizer)
        & ->  &amp;
        | ->  &#124;
        < ->  &lt;
        > ->  &gt;
        ' ->  &apos;
        " ->  &quot;
        [ ->  &#91;
        ] ->  &#93;
        :param caption: String to tokenize
        :return: Tokenized version of caption
        """

        def convert_chars(x):
            if x == ' ':
                return '<space>'
            else:
                return x.encode('utf-8')

        tokenized = re.sub('[\n\t]+', '', caption.strip())
        tokenized = re.sub('&amp;', ' & ', tokenized)
        tokenized = re.sub('&#124;', ' | ', tokenized)
        tokenized = re.sub('&gt;', ' > ', tokenized)
        tokenized = re.sub('&lt;', ' < ', tokenized)
        tokenized = re.sub('&apos;', " ' ", tokenized)
        tokenized = re.sub('&quot;', ' " ', tokenized)
        tokenized = re.sub('&#91;', ' [ ', tokenized)
        tokenized = re.sub('&#93;', ' ] ', tokenized)
        tokenized = re.sub('[  ]+', ' ', tokenized)
        tokenized = [convert_chars(char) for char in tokenized.decode('utf-8')]
        tokenized = " ".join(tokenized)
        return tokenized

    def tokenize_CNN_sentence(self, caption):
        """
        Tokenization employed in the CNN_sentence package
        (https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py#L97).
        :param caption: String to tokenize
        :return: Tokenized version of caption
        """
        tokenized = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", caption)
        tokenized = re.sub(r"\'s", " \'s", tokenized)
        tokenized = re.sub(r"\'ve", " \'ve", tokenized)
        tokenized = re.sub(r"n\'t", " n\'t", tokenized)
        tokenized = re.sub(r"\'re", " \'re", tokenized)
        tokenized = re.sub(r"\'d", " \'d", tokenized)
        tokenized = re.sub(r"\'ll", " \'ll", tokenized)
        tokenized = re.sub(r",", " , ", tokenized)
        tokenized = re.sub(r"!", " ! ", tokenized)
        tokenized = re.sub(r"\(", " \( ", tokenized)
        tokenized = re.sub(r"\)", " \) ", tokenized)
        tokenized = re.sub(r"\?", " \? ", tokenized)
        tokenized = re.sub(r"\s{2,}", " ", tokenized)
        return tokenized.strip().lower()

    def tokenize_questions(self, caption):
        """
        Basic tokenizer for VQA questions:
            * Lowercasing
            * Splits contractions
            * Removes punctuation
            * Numbers to digits

        :param caption: String to tokenize
        :return: Tokenized version of caption
        """
        contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                        "couldnt": "couldn't",
                        "couldn'tve": "couldn’t’ve", "couldnt’ve": "couldn’t’ve", "didnt": "didn’t",
                        "doesnt": "doesn’t",
                        "dont": "don’t", "hadnt": "hadn’t", "hadnt’ve": "hadn’t’ve", "hadn'tve": "hadn’t’ve",
                        "hasnt": "hasn’t", "havent": "haven’t", "hed": "he’d", "hed’ve": "he’d’ve", "he’dve": "he’d’ve",
                        "hes": "he’s", "howd": "how’d", "howll": "how’ll", "hows": "how’s", "Id’ve": "I’d’ve",
                        "I’dve": "I’d’ve", "Im": "I’m", "Ive": "I’ve", "isnt": "isn’t", "itd": "it’d",
                        "itd’ve": "it’d’ve",
                        "it’dve": "it’d’ve", "itll": "it’ll", "let’s": "let’s", "maam": "ma’am", "mightnt": "mightn’t",
                        "mightnt’ve": "mightn’t’ve", "mightn’tve": "mightn’t’ve", "mightve": "might’ve",
                        "mustnt": "mustn’t",
                        "mustve": "must’ve", "neednt": "needn’t", "notve": "not’ve", "oclock": "o’clock",
                        "oughtnt": "oughtn’t",
                        "ow’s’at": "’ow’s’at", "’ows’at": "’ow’s’at", "’ow’sat": "’ow’s’at", "shant": "shan’t",
                        "shed’ve": "she’d’ve", "she’dve": "she’d’ve", "she’s": "she’s", "shouldve": "should’ve",
                        "shouldnt": "shouldn’t", "shouldnt’ve": "shouldn’t’ve", "shouldn’tve": "shouldn’t’ve",
                        "somebody’d": "somebodyd", "somebodyd’ve": "somebody’d’ve", "somebody’dve": "somebody’d’ve",
                        "somebodyll": "somebody’ll", "somebodys": "somebody’s", "someoned": "someone’d",
                        "someoned’ve": "someone’d’ve", "someone’dve": "someone’d’ve", "someonell": "someone’ll",
                        "someones": "someone’s", "somethingd": "something’d", "somethingd’ve": "something’d’ve",
                        "something’dve": "something’d’ve", "somethingll": "something’ll", "thats": "that’s",
                        "thered": "there’d", "thered’ve": "there’d’ve", "there’dve": "there’d’ve",
                        "therere": "there’re",
                        "theres": "there’s", "theyd": "they’d", "theyd’ve": "they’d’ve", "they’dve": "they’d’ve",
                        "theyll": "they’ll", "theyre": "they’re", "theyve": "they’ve", "twas": "’twas",
                        "wasnt": "wasn’t",
                        "wed’ve": "we’d’ve", "we’dve": "we’d’ve", "weve": "we've", "werent": "weren’t",
                        "whatll": "what’ll",
                        "whatre": "what’re", "whats": "what’s", "whatve": "what’ve", "whens": "when’s", "whered":
                            "where’d", "wheres": "where's", "whereve": "where’ve", "whod": "who’d",
                        "whod’ve": "who’d’ve",
                        "who’dve": "who’d’ve", "wholl": "who’ll", "whos": "who’s", "whove": "who've", "whyll": "why’ll",
                        "whyre": "why’re", "whys": "why’s", "wont": "won’t", "wouldve": "would’ve",
                        "wouldnt": "wouldn’t",
                        "wouldnt’ve": "wouldn’t’ve", "wouldn’tve": "wouldn’t’ve", "yall": "y’all",
                        "yall’ll": "y’all’ll",
                        "y’allll": "y’all’ll", "yall’d’ve": "y’all’d’ve", "y’alld’ve": "y’all’d’ve",
                        "y’all’dve": "y’all’d’ve",
                        "youd": "you’d", "youd’ve": "you’d’ve", "you’dve": "you’d’ve", "youll": "you’ll",
                        "youre": "you’re", "youve": "you’ve"}
        punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
                 '_', '-', '>', '<', '@', '`', ',', '?', '!']
        commaStrip = re.compile("(\d)(\,)(\d)")
        periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        manualMap = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                     'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
        articles = ['a', 'an', 'the']

        def processPunctuation(inText):
            outText = inText
            for p in punct:
                if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
                    outText = outText.replace(p, '')
                else:
                    outText = outText.replace(p, ' ')
            outText = periodStrip.sub("", outText, re.UNICODE)
            return outText

        def processDigitArticle(inText):
            outText = []
            tempText = inText.lower().split()
            for word in tempText:
                word = manualMap.setdefault(word, word)
                if word not in articles:
                    outText.append(word)
                else:
                    pass
            for wordId, word in enumerate(outText):
                if word in contractions:
                    outText[wordId] = contractions[word]
            outText = ' '.join(outText)
            return outText

        resAns = caption.lower()
        resAns = resAns.replace('\n', ' ')
        resAns = resAns.replace('\t', ' ')
        resAns = resAns.strip()
        resAns = processPunctuation(resAns.decode("utf-8").encode("utf-8"))
        resAns = processDigitArticle(resAns)

        return resAns

    def tokenize_bpe(self, caption):
        """
        Applies BPE segmentation (https://github.com/rsennrich/subword-nmt)
        :param caption: Caption to detokenize.
        :return: Encoded version of caption.
        """
        if not self.BPE_built:
            raise Exception, 'Prior to use the "tokenize_bpe" method, you should invoke "build_BPE"'
        tokenized = re.sub('[\n\t]+', '', caption.strip())
        tokenized = self.BPE.segment(tokenized).strip()
        return tokenized

    def detokenize_none(self, caption):
        """
        Dummy function: Keeps the caption as it is.
        :param caption: String to de-tokenize.
        :return: Same caption.
        """
        return caption


    def detokenize_bpe(self, caption, separator='@@'):
        """
        Reverts BPE segmentation (https://github.com/rsennrich/subword-nmt)
        :param caption: Caption to detokenize.
        :param separator: BPE separator.
        :return: Detokenized version of caption.
        """
        bpe_detokenization = re.compile('(' + separator + ' )|(' + separator + ' ?$)')
        detokenized = bpe_detokenization.sub("", str(caption).strip())
        return detokenized

    def detokenize_none_char(self, caption):
        """
        Character-level detokenization. Respects all symbols. Joins chars into words. Words are delimited by
        the <space> token. If found an special character is converted to the escaped char.
        # List of escaped chars (by moses tokenizer)
            & ->  &amp;
            | ->  &#124;
            < ->  &lt;
            > ->  &gt;
            ' ->  &apos;
            " ->  &quot;
            [ ->  &#91;
            ] ->  &#93;
            :param caption: String to de-tokenize.
            :return: Detokenized version of caption.
        """

        def deconvert_chars(x):
            if x == '<space>':
                return ' '
            else:
                return x.encode('utf-8')

        detokenized = re.sub(' & ', ' &amp; ', str(caption).strip())
        detokenized = re.sub(' \| ', ' &#124; ', detokenized)
        detokenized = re.sub(' > ', ' &gt; ', detokenized)
        detokenized = re.sub(' < ', ' &lt; ', detokenized)
        detokenized = re.sub("' ", ' &apos; ', detokenized)
        detokenized = re.sub('" ', ' &quot; ', detokenized)
        detokenized = re.sub('\[ ', ' &#91; ', detokenized)
        detokenized = re.sub('\] ', ' &#93; ', detokenized)
        detokenized = re.sub(' ', '', detokenized)
        detokenized = re.sub('<space>', ' ', detokenized)
        return detokenized

    # ------------------------------------------------------- #
    #       TYPE 'video' and 'video-features' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #

    def preprocessVideos(self, path_list, id, set_name, max_video_len, img_size, img_size_crop):

        if isinstance(path_list, list) and len(path_list) == 2:
            # path to all images in all videos
            data = []
            with open(path_list[0], 'r') as list_:
                for line in list_:
                    data.append(line.rstrip('\n'))
            # frame counts
            counts_frames = []
            with open(path_list[1], 'r') as list_:
                for line in list_:
                    counts_frames.append(int(line.rstrip('\n')))

            if id not in self.paths_frames:
                self.paths_frames[id] = dict()
            self.paths_frames[id][set_name] = data
            self.max_video_len[id] = max_video_len
            self.img_size[id] = img_size
            self.img_size_crop[id] = img_size_crop
        else:
            raise Exception('Wrong type for "path_list". It must be a list containing two paths: '
                            'a path to a text file with the paths to all images in all videos in '
                            '[0] and a path to another text file with the number of frames of '
                            'each video in each line in [1] (which will index the paths in the first file).\n'
                            'It currently is: %s' % str(path_list))

        return counts_frames

    def preprocessVideoFeatures(self, path_list, id, set_name, max_video_len, img_size, img_size_crop, feat_len):

        if isinstance(path_list, list) and len(path_list) == 2:
            if isinstance(path_list[0], str):
                # path to all images in all videos
                paths_frames = []
                with open(path_list[0], 'r') as list_:
                    for line in list_:
                        paths_frames.append(line.rstrip('\n'))
            elif isinstance(path_list[0], list):
                paths_frames = path_list[0]
            else:
                raise Exception('Wrong type for "path_frames". It must be a path to a file containing a'
                                ' list of frames or directly a list of frames.\n'
                                'It currently is: %s' % str(path_list[0]))

            if isinstance(path_list[1], str):
                # frame counts
                counts_frames = []
                with open(path_list[1], 'r') as list_:
                    for line in list_:
                        counts_frames.append(int(line.rstrip('\n')))
            elif isinstance(path_list[1], list):
                counts_frames = path_list[1]
            else:
                raise Exception('Wrong type for "counts_frames".'
                                ' It must be a path to a file containing a list of counts or directly a list of counts.\n'
                                'It currently is: %s' % str(path_list[1]))

            # video indices
            video_indices = range(len(counts_frames))

            if id not in self.paths_frames:
                self.paths_frames[id] = dict()
            if id not in self.counts_frames:
                self.counts_frames[id] = dict()

            self.paths_frames[id][set_name] = paths_frames
            self.counts_frames[id][set_name] = counts_frames
            self.max_video_len[id] = max_video_len
            self.img_size[id] = img_size
            self.img_size_crop[id] = img_size_crop
        else:
            raise Exception('Wrong type for "path_list". '
                            'It must be a list containing two paths: a path to a text file with the paths to all '
                            'images in all videos in [0] and a path to another text file with the number of frames '
                            'of each video in each line in [1] (which will index the paths in the first file).'
                            'It currently is: %s' % str(path_list[1]))

        if feat_len is not None:
            if not isinstance(feat_len, list):
                feat_len = [feat_len]
            self.features_lengths[id] = feat_len

        return video_indices

    def loadVideos(self, n_frames, id, last, set_name, max_len, normalization_type, normalization, meanSubstraction,
                   dataAugmentation):
        """
         Loads a set of videos from disk. (Untested!)

        :param n_frames: Number of frames per video
        :param id: Id to load
        :param last: Last video loaded
        :param set_name:  'train', 'val', 'test'
        :param max_len: Maximum length of videos
        :param normalization_type:  Type of normalization applied
        :param normalization: Whether we apply a 0-1 normalization to the images
        :param meanSubstraction:  Whether we are removing the training mean
        :param dataAugmentation:  Whether we are applying dataAugmentatino (random cropping and horizontal flip)
        """

        n_videos = len(n_frames)
        V = np.zeros((n_videos, max_len * 3, self.img_size_crop[id][0], self.img_size_crop[id][1]))

        idx = [0 for i in range(n_videos)]
        # recover all indices from image's paths of all videos
        for v in range(n_videos):
            this_last = last + v
            if this_last >= n_videos:
                v = this_last % n_videos
                this_last = v
            idx[v] = int(sum(eval('self.X_' + set_name + '[id][:this_last]')))

        # load images from each video
        for enum, (n, i) in enumerate(zip(n_frames, idx)):
            paths = self.paths_frames[id][set_name][i:i + n]
            daRandomParams = None
            if dataAugmentation:
                daRandomParams = self.getDataAugmentationRandomParams(paths, id)
            # returns numpy array with dimensions (batch, channels, height, width)
            images = self.loadImages(paths, id, normalization_type, normalization, meanSubstraction, dataAugmentation,
                                     daRandomParams)
            # fills video matrix with each frame (fills with 0s or removes remaining frames w.r.t. max_len)
            len_j = images.shape[0]
            offset_j = max_len - len_j
            if offset_j < 0:
                len_j = len_j + offset_j
                offset_j = 0
            for j in range(len_j):
                V[enum, (j + offset_j) * 3:(j + offset_j + 1) * 3] = images[j]

        return V

    def loadVideoFeatures(self, idx_videos, id, set_name, max_len, normalization_type, normalization, feat_len,
                          external=False, data_augmentation=True):
        """

        :param idx_videos: indices of the videos in the complete list of the current set_name
        :param id: identifier of the input/output that we are loading
        :param set_name: 'train', 'val' or 'test'
        :param max_len: maximum video length (number of frames)
        :param normalization_type: type of data normalization applied
        :param normalization: Switch on/off data normalization
        :param feat_len: length of the features about to load
        :param external: Switch on/off data loading from external dataset (not sharing self.path)
        :param data_augmentation: Switch on/off data augmentation
        :return:
        """

        n_videos = len(idx_videos)
        if isinstance(feat_len, list):
            feat_len = feat_len[0]
        features = np.zeros((n_videos, max_len, feat_len))

        selected_frames = self.getFramesPaths(idx_videos, id, set_name, max_len, data_augmentation)
        data_augmentation_types = self.inputs_data_augmentation_types[id]

        # load features from selected paths
        for i, vid_paths in enumerate(selected_frames):
            for j, feat in enumerate(vid_paths):
                if not external:
                    feat = self.path + '/' + feat

                # Check if the filename includes the extension
                feat = np.load(feat)

                if data_augmentation and 'noise' in data_augmentation_types:
                    noise_mean = 0.0
                    noise_dev = 0.01
                    noise = np.random.normal(noise_mean, noise_dev, feat.shape)
                    feat += noise

                if normalization:
                    if normalization_type == 'L2':
                        feat = feat / np.linalg.norm(feat, ord=2)

                features[i, j] = feat

        return np.array(features)

    def getFramesPaths(self, idx_videos, id, set_name, max_len, data_augmentation):
        """
        Recovers the paths from the selected video frames.
        """

        # recover chosen data augmentation types
        data_augmentation_types = self.inputs_data_augmentation_types[id]
        if data_augmentation_types is None:
            data_augmentation_types = []

        n_frames = [self.counts_frames[id][set_name][i_idx_vid] for i_idx_vid in idx_videos]

        n_videos = len(idx_videos)
        idx = [0 for i_nvid in range(n_videos)]
        # recover all initial indices from image's paths of all videos
        for v in range(n_videos):
            last_idx = idx_videos[v]
            idx[v] = int(sum(self.counts_frames[id][set_name][:last_idx]))

        # select subset of max_len from n_frames[i]
        selected_frames = [0 for i_nvid in range(n_videos)]
        for enum, (n, i) in enumerate(zip(n_frames, idx)):
            paths = self.paths_frames[id][set_name][i:i + n]

            if data_augmentation and 'random_selection' in data_augmentation_types:  # apply random frames selection
                selected_idx = sorted(random.sample(range(n), min(max_len, n)))
            else:  # apply equidistant frames selection
                selected_idx = np.round(np.linspace(0, n - 1, min(max_len, n)))
                # splits = np.array_split(range(n), min(max_len, n))
                # selected_idx = [s[0] for s in splits]

            selected_paths = [paths[int(idx)] for idx in selected_idx]
            selected_frames[enum] = selected_paths

        return selected_frames

    def loadVideosByIndex(self, n_frames, id, indices, set_name, max_len, normalization_type, normalization,
                          meanSubstraction, dataAugmentation):
        n_videos = len(indices)
        V = np.zeros((n_videos, max_len * 3, self.img_size_crop[id][0], self.img_size_crop[id][1]))

        idx = [0 for i in range(n_videos)]
        # recover all indices from image's paths of all videos
        for v in range(n_videos):
            idx[v] = int(sum(eval('self.X_' + set_name + '[id][indices[v]]')))

        # load images from each video
        for enum, (n, i) in enumerate(zip(n_frames, idx)):
            paths = self.paths_frames[id][set_name][i:i + n]
            daRandomParams = None
            if dataAugmentation:
                daRandomParams = self.getDataAugmentationRandomParams(paths, id)
            # returns numpy array with dimensions (batch, channels, height, width)
            images = self.loadImages(paths, id, normalization_type, normalization, meanSubstraction, dataAugmentation,
                                     daRandomParams)
            # fills video matrix with each frame (fills with 0s or removes remaining frames w.r.t. max_len)
            len_j = images.shape[0]
            offset_j = max_len - len_j
            if offset_j < 0:
                len_j = len_j + offset_j
                offset_j = 0
            for j in range(len_j):
                V[enum, (j + offset_j) * 3:(j + offset_j + 1) * 3] = images[j]

        return V

    # ------------------------------------------------------- #
    #       TYPE 'id' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #

    def preprocessIDs(self, path_list, id, set_name):

        logging.info('WARNING: inputs or outputs with type "id" will not be treated in any way by the dataset.')
        if isinstance(path_list, str) and os.path.isfile(path_list):  # path to list of IDs
            data = []
            with open(path_list, 'r') as list_:
                for line in list_:
                    data.append(line.rstrip('\n'))
        elif isinstance(path_list, list):
            data = path_list
        else:
            raise Exception('Wrong type for "path_list". '
                            'It must be a path to a text file with an id in each line'
                            ' or an instance of the class list with an id in each position.'
                            'It currently is: %s' % str(path_list))


        return data

    # ------------------------------------------------------- #
    #       TYPE '3DSemanticLabel' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #

    def getImageFromPrediction_3DSemanticLabel(self):
        raise NotImplementedError('ToDo: implement from read_write.py')

    def preprocess3DSemanticLabel(self, path_list, id, associated_id_in, num_poolings):
        return self.preprocess3DLabel(path_list, id, associated_id_in, num_poolings)

    def setSemanticClasses(self, path_classes, id):
        """
        Loads the list of semantic classes of the dataset together with their corresponding colours in the GT image.
        Each line must contain a unique identifier of the class and its associated RGB colour representation separated by commas.

        :param path_classes: Path to a text file with the classes and their colours.
        :param id: input/output id

        :return: None
        """
        if isinstance(path_classes, str) and os.path.isfile(path_classes):
            semantic_classes = dict()
            with open(path_classes, 'r') as list_:
                for line in list_:
                    line = line.rstrip('\n').split(',')
                    if len(line) != 4:
                        raise Exception('Wrong format for semantic classes.'
                                        ' Must contain a class name followed by the '
                                        'RGB colour values separated by commas.'
                                        'It currently has a line of length: %s' % str(len(line)))

                    class_id = self.dic_classes[id][line[0]]
                    semantic_classes[int(class_id)] = [int(line[1]), int(line[2]), int(line[3])]
            self.semantic_classes[id] = semantic_classes
        else:
            raise Exception('Wrong type for "path_classes".'
                            ' It must be a path to a text file with the classes '
                            'and their associated colour in the GT image.'
                            'It currently is: %s' % str(path_classes))

        if not self.silence:
            logging.info('Loaded semantic classes list for data with id: ' + id)

    def load_GT_3DSemanticLabels(self, gt, id):
        '''
        Loads a GT list of 3DSemanticLabels in a 2D matrix and reshapes them to an Nx1 array (EVALUATION)

        :param gt: list of Dataset output of type 3DSemanticLabels
        :param id: id of the input/output we are processing
        :return: out_list: containing a list of label images reshaped as an Nx1 array
        '''
        out_list = []

        assoc_id_in = self.id_in_3DLabel[id]
        classes_to_colour = self.semantic_classes[id]
        nClasses = len(classes_to_colour.keys())
        img_size = self.img_size[assoc_id_in]
        size_crop = self.img_size_crop[assoc_id_in]
        num_poolings = self.num_poolings_model[id]

        n_samples = len(gt)
        h, w, d = img_size
        h_crop, w_crop, d_crop = size_crop

        # Modify output dimensions depending on number of poolings applied
        if num_poolings is not None:
            h_crop = int(np.floor(h_crop / np.power(2, num_poolings)))
            w_crop = int(np.floor(w_crop / np.power(2, num_poolings)))

        for i in range(n_samples):
            pre_labels = np.zeros((nClasses, h_crop, w_crop), dtype=np.float32)
            #labels = np.zeros((h_crop, w_crop), dtype=np.uint8)
            line = gt[i]

            ### Load labeled GT image
            labeled_im = self.path + '/' + line
            # Check if the filename includes the extension
            [path, filename] = ntpath.split(labeled_im)
            [filename, ext] = os.path.splitext(filename)
            # If it doesn't then we find it
            if not ext:
                filename = fnmatch.filter(os.listdir(path), filename + '*')
                if not filename:
                    raise Exception('Non existent image ' + labeled_im)
                else:
                    labeled_im = path + '/' + filename[0]
            # Read image
            try:
                logging.disable(logging.CRITICAL)
                labeled_im = pilimage.open(labeled_im)
                labeled_im = np.asarray(labeled_im)
                logging.disable(logging.NOTSET)
                labeled_im = misc.imresize(labeled_im, (h, w))
            except:
                logging.warning("WARNING!")
                logging.warning("Can't load image " + labeled_im)
                labeled_im = np.zeros((h, w))

            label3D = np.zeros((nClasses, h, w), dtype=np.float32)

            # Insert 1s in the corresponding positions for each class
            for class_id, colour in classes_to_colour.iteritems():
                # indices = np.where(np.all(labeled_im == colour, axis=-1))
                indices = np.where(labeled_im == class_id)
                num_vals = len(indices[0])
                if num_vals > 0:
                    for idx_pos in range(num_vals):
                        x, y = indices[0][idx_pos], indices[1][idx_pos]
                        label3D[class_id, x, y] = 1.

            # Resize 3DLabel to crop size.
            for j in range(nClasses):
                label2D = misc.imresize(label3D[j], (h_crop, w_crop))
                maxval = np.max(label2D)
                if maxval > 0: label2D /= maxval
                pre_labels[j] = label2D

            # Convert to single matrix with class IDs
            labels = np.argmax(pre_labels, axis=0)
            labels = np.reshape(labels, (w_crop * h_crop))

            out_list.append(labels)

        return out_list

    def resize_semantic_output(self, predictions, ids_out):
        out_pred = []

        for pred, id_out in zip(predictions, ids_out):

            assoc_id_in = self.id_in_3DLabel[id_out]
            in_size = self.img_size_crop[assoc_id_in]
            out_size = self.img_size[assoc_id_in]
            n_classes = len(self.classes[id_out])

            pred = np.transpose(pred, [1, 0])
            pred = np.reshape(pred, (-1, in_size[0], in_size[1]))

            new_pred = np.zeros(tuple([n_classes] + out_size[0:2]))
            for pos, p in enumerate(pred):
                new_pred[pos] = misc.imresize(p, tuple(out_size[0:2]))

            new_pred = np.reshape(new_pred, (-1, out_size[0] * out_size[1]))
            new_pred = np.transpose(new_pred, [1, 0])

            out_pred.append(new_pred)

        return out_pred

    # ------------------------------------------------------- #
    #       TYPE '3DLabel' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #

    def preprocess3DLabel(self, path_list, id, associated_id_in, num_poolings):
        if (isinstance(path_list, str) and os.path.isfile(path_list)):
            path_list_3DLabel = []
            with open(path_list, 'r') as list_:
                for line in list_:
                    path_list_3DLabel.append(line.strip())
        else:
            raise Exception('Wrong type for "path_list". '
                            'It must be a path to a text file with the path to 3DLabel files.'
                            'It currently is: %s' % str(path_list))

        self.num_poolings_model[id] = num_poolings
        self.id_in_3DLabel[id] = associated_id_in

        return path_list_3DLabel

    def convert_3DLabels_to_bboxes(self, predictions, original_sizes, threshold=0.5, idx_3DLabel=0,
                                   size_restriction=0.001):
        """
        Converts a set of predictions of type 3DLabel to their corresponding bounding boxes.

        :param predictions: 3DLabels predicted by Model_Wrapper. If type is list it will be assumed that position 0 corresponds to 3DLabels
        :param original_sizes: original sizes of the predicted images width and height
        :param threshold: minimum overlapping threshold for considering a prediction valid
        :return: predicted_bboxes, predicted_Y, predicted_scores for each image
        """
        out_list = []

        # if type is list it will be assumed that position 0 corresponds to 3DLabels
        if isinstance(predictions, list):
            predict_3dLabels = predictions[idx_3DLabel]
        else:
            predict_3dLabels = predictions

        # Reshape from (n_samples, width*height, nClasses) to (n_samples, nClasses, width, height)
        n_samples, wh, n_classes = predict_3dLabels.shape
        w, h, d = self.img_size_crop[self.id_in_3DLabel[self.ids_outputs[idx_3DLabel]]]
        predict_3dLabels = np.transpose(predict_3dLabels, (0, 2, 1))
        predict_3dLabels = np.reshape(predict_3dLabels, (n_samples, n_classes, w, h))

        # list of hypotheses with the following info [predicted_bboxes, predicted_Y, predicted_scores]
        for s in range(n_samples):
            bboxes = []
            Y = []
            scores = []
            orig_h, orig_w = original_sizes[s]
            wratio = float(orig_w) / w
            hratio = float(orig_h) / h
            for c in range(n_classes):
                map = predict_3dLabels[s][c]

                # Compute binary selected region
                binary_heat = map
                binary_heat = np.where(binary_heat >= threshold, 255, 0)

                # Get biggest connected component
                min_size = map.shape[0] * map.shape[1] * size_restriction
                labeled, nr_objects = ndimage.label(binary_heat)  # get connected components
                [objects, counts] = np.unique(labeled, return_counts=True)  # count occurrences
                # biggest_components = np.argsort(counts[1:])[::-1]
                # selected_components = [1 if counts[i+1] >= min_size else 0 for i in biggest_components] # check minimum size restriction
                # selected_components = [1 for i in range(len(objects))]
                # biggest_components = biggest_components[:min([np.sum(selected_components), 9999])] # get all bboxes


                for obj in objects[1:]:
                    current_obj = np.where(labeled == obj, 255, 0)  # get the biggest

                    # Draw bounding box on original image
                    box = list(bbox(current_obj))
                    current_obj = np.nonzero(current_obj)
                    if len(current_obj) > min_size:  # filter too small bboxes

                        # expand box before final detection
                        # x_exp = box[2]# * box_expansion
                        # y_exp = box[3]# * box_expansion
                        # box[0] = max([0, box[0]-x_exp/2])
                        # box[1] = max([0, box[1]-y_exp/2])
                        # change width and height by xmax and ymax
                        # box[2] += box[0]
                        # box[3] += box[1]
                        # box[2] = min([new_reshape_size[1]-1, box[2] + x_exp])
                        # box[3] = min([new_reshape_size[0]-1, box[3] + y_exp])

                        # get score of the region
                        score = np.mean([map[point[0], point[1]] for point in current_obj])

                        # convert bbox to original size
                        box = np.array([box[0] * wratio, box[1] * hratio, box[2] * wratio, box[3] * hratio])
                        box = box.astype(int)

                        bboxes.append(box)
                        Y.append(c)
                        scores.append(score)

            out_list.append([bboxes, Y, scores])

        return out_list

    def convert_GT_3DLabels_to_bboxes(self, gt):
        '''
        Converts a GT list of 3DLabels to a set of bboxes.

        :param gt: list of Dataset output of type 3DLabels
        :return: [out_list, original_sizes], where out_list contains a list of samples with the following info [GT_bboxes, GT_Y], and original_sizes contains the original width and height for each image
        '''
        out_list = []
        original_sizes = []
        # extra_vars[split]['references'] - list of samples with the following info [GT_bboxes, GT_Y]

        n_samples = len(gt)
        for i in range(n_samples):
            bboxes = []
            Y = []

            line = gt[i]
            arrayLine = line.split(';')
            arrayBndBox = arrayLine[:-1]
            w_original, h_original, d_original = eval(arrayLine[-1])
            original_sizes.append([h_original, w_original])

            for array in arrayBndBox:
                bndbox = eval(array)[0]
                # bndbox = [bndbox[1], bndbox[0], bndbox[3], bndbox[2]]
                idxclass = eval(array)[1]
                Y.append(idxclass)
                bboxes.append(bndbox)
                # bndbox_ones = np.ones((bndbox[2] - bndbox[0] + 1, bndbox[3] - bndbox[1] + 1))
                # label3D[idxclass, bndbox[0] - 1:bndbox[2], bndbox[1] - 1:bndbox[3]] = bndbox_ones

            out_list.append([bboxes, Y])

        return [out_list, original_sizes]

    # ------------------------------------------------------- #
    #       TYPE 'raw-image' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #

    def preprocessImages(self, path_list, id, set_name, img_size, img_size_crop, use_RGB):
        if isinstance(path_list, str) and os.path.isfile(path_list):  # path to list of images' paths
            data = []
            with open(path_list, 'r') as list_:
                for line in list_:
                    data.append(line.rstrip('\n'))
        elif isinstance(path_list, list):
            data = path_list
        else:
            raise Exception('Wrong type for "path_list". It must be a path to a text file with an image '
                'path in each line or an instance of the class list with an image path in each position.'
                            'It currently is: %s' % str(path_list))

        self.img_size[id] = img_size
        self.img_size_crop[id] = img_size_crop
        self.use_RGB[id] = use_RGB

        # Tries to load a train_mean file from the dataset folder if exists
        mean_file_path = self.path + '/train_mean'
        for s in range(len(self.img_size[id])):
            mean_file_path += '_' + str(self.img_size[id][s])
        mean_file_path += '_' + id + '_.jpg'
        if os.path.isfile(mean_file_path):
            self.setTrainMean(mean_file_path, id)

        return data

    def setTrainMean(self, mean_image, id, normalization=False):
        """
            Loads a pre-calculated training mean image, 'mean_image' can either be:

            - numpy.array (complete image)
            - list with a value per channel
            - string with the path to the stored image.

            :param id: identifier of the type of input whose train mean is being introduced.
        """
        if isinstance(mean_image, str):
            if not self.silence:
                logging.info("Loading train mean image from file.")
            mean_image = misc.imread(mean_image)
        elif isinstance(mean_image, list):
            mean_image = np.array(mean_image)
        self.train_mean[id] = mean_image.astype(np.float32)

        if normalization:
            self.train_mean[id] = self.train_mean[id] / 255.0

        if self.train_mean[id].shape != tuple(self.img_size_crop[id]):
            if len(self.train_mean[id].shape) == 1 and self.train_mean[id].shape[0] == self.img_size_crop[id][2]:
                if not self.silence:
                    logging.info("Converting input train mean pixels into mean image.")
                mean_image = np.zeros(tuple(self.img_size_crop[id]))
                for c in range(self.img_size_crop[id][2]):
                    mean_image[:, :, c] = self.train_mean[id][c]
                self.train_mean[id] = mean_image
            else:
                logging.warning(
                    "The loaded training mean size does not match the desired images size.\nChange the images size with setImageSize(size) or recalculate the training mean with calculateTrainMean().")

    def calculateTrainMean(self, id):
        """
            Calculates the mean of the data belonging to the training set split in each channel.
        """
        calculate = False
        if not id in self.train_mean or not isinstance(self.train_mean[id], np.ndarray):
            calculate = True
        elif self.train_mean[id].shape != tuple(self.img_size[id]):
            calculate = True
            if not self.silence:
                logging.warning(
                    "The loaded training mean size does not match the desired images size. Recalculating mean...")

        if calculate:
            if not self.silence:
                logging.info("Start training set mean calculation...")

            I_sum = np.zeros(self.img_size_crop[id], dtype=np.float64)

            # Load images in batches and sum all of them
            init = 0
            batch = 200
            for final in range(batch, self.len_train, batch):
                I = self.getX('train', init, final, meanSubstraction=False)[self.ids_inputs.index(id)]
                for im in I:
                    I_sum += im
                if not self.silence:
                    sys.stdout.write('\r')
                    sys.stdout.write("Processed %d/%d images..." % (final, self.len_train))
                    sys.stdout.flush()
                init = final
            I = self.getX('train', init, self.len_train, meanSubstraction=False)[self.ids_inputs.index(id)]
            for im in I:
                I_sum += im
            if not self.silence:
                sys.stdout.write('\r')
                sys.stdout.write("Processed %d/%d images..." % (final, self.len_train))
                sys.stdout.flush()

            # Mean calculation
            self.train_mean[id] = I_sum / self.len_train

            # Store the calculated mean
            mean_name = '/train_mean'
            for s in range(len(self.img_size[id])):
                mean_name += '_' + str(self.img_size[id][s])
            mean_name += '_' + id + '_.jpg'
            store_path = self.path + '/' + mean_name
            misc.imsave(store_path, self.train_mean[id])

            # self.train_mean[id] = self.train_mean[id].astype(np.float32)/255.0

            if not self.silence:
                logging.info("Image mean stored in " + store_path)

        # Return the mean
        return self.train_mean[id]

    def loadImages(self, images, id, normalization_type='0-1',
                   normalization=False, meanSubstraction=True,
                   dataAugmentation=True, daRandomParams=None,
                   external=False, loaded=False):
        """
            Loads a set of images from disk.

            :param images : list of image string names or list of matrices representing images (only if loaded==True)
            :param id : identifier in the Dataset object of the data we are loading
            :param normalization_type: type of normalization applied
            :param normalization : whether we applying a 0-1 normalization to the images
            :param meanSubstraction : whether we are removing the training mean
            :param dataAugmentation : whether we are applying dataAugmentatino (random cropping and horizontal flip)
            :param daRandomParams : dictionary with results of random data augmentation provided by self.getDataAugmentationRandomParams()
            :param external : if True the images will be loaded from an external database, in this case the list of images must be absolute paths
            :param loaded : set this option to True if images is a list of matricies instead of a list of strings
        """
        # Check if the chosen normalization type exists
        if normalization and normalization_type not in self.__available_norm_im_vid:
            raise NotImplementedError(
                'The chosen normalization type ' + normalization_type + ' is not implemented for the type "raw-image" and "video".')

        # Prepare the training mean image
        if meanSubstraction:  # remove mean

            if id not in self.train_mean:
                raise Exception('Training mean is not loaded or calculated yet for the input with id "' + id + '".')
            train_mean = copy.copy(self.train_mean[id])
            train_mean = misc.imresize(train_mean, self.img_size_crop[id][0:2])

            # Transpose dimensions
            if len(self.img_size[id]) == 3:  # if it is a 3D image
                # Convert RGB to BGR
                if self.img_size[id][2] == 3:  # if has 3 channels
                    train_mean = train_mean[:, :, ::-1]
                train_mean = train_mean.transpose(2, 0, 1)

            # Also normalize training mean image if we are applying normalization to images
            if normalization:
                if normalization_type == '0-1':
                    train_mean = train_mean / 255.0

        nImages = len(images)

        type_imgs = np.float32
        if len(self.img_size[id]) == 3:
            I = np.zeros([nImages] + [self.img_size_crop[id][2]] + self.img_size_crop[id][0:2], dtype=type_imgs)
        else:
            I = np.zeros([nImages] + self.img_size_crop[id], dtype=type_imgs)

        ''' Process each image separately '''
        for i in range(nImages):
            im = images[i]

            if not loaded:
                if not external:
                    im = self.path + '/' + im

                # Check if the filename includes the extension
                [path, filename] = ntpath.split(im)
                [filename, ext] = os.path.splitext(filename)

                # If it doesn't then we find it
                if not ext:
                    filename = fnmatch.filter(os.listdir(path), filename + '*')
                    if not filename:
                        raise Exception('Non existent image ' + im)
                    else:
                        im = path + '/' + filename[0]
                imname = im

                # Read image
                try:
                    logging.disable(logging.CRITICAL)
                    im = pilimage.open(im)
                    logging.disable(logging.NOTSET)

                except:
                    logging.warning("WARNING!")
                    logging.warning("Can't load image " + im)
                    im = np.zeros(tuple(self.img_size[id]))

            # Convert to RGB
            if not type(im).__module__ == np.__name__:
                if self.use_RGB[id]:
                    im = im.convert('RGB')
                else:
                    im = im.convert('L')

            # Data augmentation
            if not dataAugmentation:
                # Use whole image
                # im = np.asarray(im, dtype=type_imgs)
                # im = misc.imresize(im, (self.img_size_crop[id][1], self.img_size_crop[id][0]))
                im = im.resize((self.img_size_crop[id][1], self.img_size_crop[id][0]))
                im = np.asarray(im, dtype=type_imgs)
            else:
                randomParams = daRandomParams[images[i]]
                # Resize
                # im = np.asarray(im, dtype=type_imgs)
                # im = misc.imresize(im, (self.img_size[id][1], self.img_size[id][0]))
                im = im.resize((self.img_size[id][1], self.img_size[id][0]))
                im = np.asarray(im, dtype=type_imgs)

                # Take random crop
                left = randomParams["left"]
                right = np.add(left, self.img_size_crop[id][0:2])
                if self.use_RGB[id]:
                    try:
                        im = im[left[0]:right[0], left[1]:right[1], :]
                    except:
                        print '------- ERROR -------'
                        print left
                        print right
                        print im.shape
                        print imname
                        raise Exception('Error with image ' + imname)
                else:
                    im = im[left[0]:right[0], left[1]:right[1]]

                # Randomly flip (with a certain probability)
                flip = randomParams["hflip"]
                prob_flip_horizontal = randomParams["prob_flip_horizontal"]
                if flip < prob_flip_horizontal:  # horizontal flip
                    im = np.fliplr(im)
                prob_flip_vertical = randomParams["prob_flip_vertical"]
                flip = randomParams["vflip"]
                if flip < prob_flip_vertical:  # vertical flip
                    im = np.flipud(im)

            # Normalize
            if normalization:
                if normalization_type == '0-1':
                    im = im / 255.0

            # Permute dimensions
            if len(self.img_size[id]) == 3:
                # Convert RGB to BGR
                if self.img_size[id][2] == 3:  # if has 3 channels
                    im = im[:, :, ::-1]
                im = im.transpose(2, 0, 1)
            else:
                pass

            # Substract training images mean
            if meanSubstraction:  # remove mean
                im = im - train_mean

            I[i] = im

        return I

    def getDataAugmentationRandomParams(self, images, id, prob_flip_horizontal=0.5, prob_flip_vertical=0.0):

        daRandomParams = dict()

        for i in range(len(images)):
            # Random crop
            margin = [self.img_size[id][0] - self.img_size_crop[id][0],
                      self.img_size[id][1] - self.img_size_crop[id][1]]
            if margin[0] > 0:
                left = random.sample([k_ for k_ in range(margin[0])], 1)
            else:
                left = [0]
            if margin[1] > 0:
                left += random.sample([k for k in range(margin[1])], 1)
            else:
                left += [0]

            # Randomly flip (with a certain probability)
            hflip = random.random()
            vflip = random.random()

            randomParams = dict()
            randomParams["left"] = left
            randomParams["hflip"] = hflip
            randomParams["vflip"] = vflip
            randomParams["prob_flip_horizontal"] = prob_flip_horizontal
            randomParams["prob_flip_vertical"] = prob_flip_vertical

            daRandomParams[images[i]] = randomParams

        return daRandomParams

    def getClassID(self, class_name, id):
        """
            :return: the class id (int) for a given class string.
        """
        return self.dic_classes[id][class_name]

    # ------------------------------------------------------- #
    #       GETTERS
    #           [X,Y] pairs or X only
    # ------------------------------------------------------- #

    def getX(self, set_name, init, final, normalization_type='0-1', normalization=False,
             meanSubstraction=True, dataAugmentation=True, debug=False):
        """
            Gets all the data samples stored between the positions init to final

            :param set_name: 'train', 'val' or 'test' set
            :param init: initial position in the corresponding set split. Must be bigger or equal than 0 and smaller than final.
            :param final: final position in the corresponding set split.
            :param debug: if True all data will be returned without preprocessing


            # 'raw-image', 'video', 'image-features' and 'video-features'-related parameters

            :param normalization: indicates if we want to normalize the data.


            # 'image-features' and 'video-features'-related parameters

            :param normalization_type: indicates the type of normalization applied. See available types in self.__available_norm_im_vid for 'raw-image' and 'video' and self.__available_norm_feat for 'image-features' and 'video-features'.


            # 'raw-image' and 'video'-related parameters

            :param meanSubstraction: indicates if we want to substract the training mean from the returned images (only applicable if normalization=True)
            :param dataAugmentation: indicates if we want to apply data augmentation to the loaded images (random flip and cropping)

            :return: X, list of input data variables from sample 'init' to 'final' belonging to the chosen 'set_name'
        """
        self.__checkSetName(set_name)
        self.__isLoaded(set_name, 0)

        if final > eval('self.len_' + set_name):
            raise Exception('"final" index must be smaller than the number of samples in the set.')
        if init < 0:
            raise Exception('"init" index must be equal or greater than 0.')
        if init >= final:
            raise Exception('"init" index must be smaller than "final" index.')

        X = []
        for id_in, type_in in zip(self.ids_inputs, self.types_inputs):
            ghost_x = False
            if id_in in self.optional_inputs:
                try:
                    x = eval('self.X_' + set_name + '[id_in][init:final]')
                    assert len(x) == (final - init)
                except:
                    x = [[]] * (final - init)
                    ghost_x = True
            else:
                x = eval('self.X_' + set_name + '[id_in][init:final]')

            if not debug and not ghost_x:
                if type_in == 'raw-image':
                    daRandomParams = None
                    if dataAugmentation:
                        daRandomParams = self.getDataAugmentationRandomParams(x, id_in)
                    x = self.loadImages(x, id_in, normalization_type, normalization, meanSubstraction, dataAugmentation,
                                        daRandomParams)
                elif type_in == 'video':
                    x = self.loadVideos(x, id_in, final, set_name, self.max_video_len[id_in],
                                        normalization_type, normalization, meanSubstraction, dataAugmentation)
                elif type_in == 'text':
                    x = self.loadText(x, self.vocabulary[id_in],
                                      self.max_text_len[id_in][set_name], self.text_offset[id_in],
                                      fill=self.fill_text[id_in], pad_on_batch=self.pad_on_batch[id_in],
                                      words_so_far=self.words_so_far[id_in], loading_X=True)[0]
                elif type_in == 'doc':
                    x_t = []
                    counter = 0
                    doc_size = 0
                    sent_len = 0
                    for i in x:
                        x_i = self.loadText(i, self.vocabulary[id_in],
                                                           self.max_text_len[id_in][set_name], self.text_offset[id_in],
                                                           fill=self.fill_text[id_in],
                                                           pad_on_batch=self.pad_on_batch[id_in],
                                                           words_so_far=self.words_so_far[id_in], loading_X=True)[0]
                        x_t.append(x_i)
                        doc_size = x_i.shape[0]
                        sent_len = x_i.shape[1]
                        counter += 1
                    x_fin = np.empty((len(x), doc_size, sent_len), dtype=object)
                    counter = 0
                    for i in x_t:
                        x_fin[counter,:,:] = i
                        counter += 1
                    x = x_fin

                elif type_in == 'image-features':
                    #print("Correctly asked the image-features with getX") #Shu
                    x = self.loadFeatures(x, self.features_lengths[id_in], normalization_type, normalization,
                                          data_augmentation=dataAugmentation)
                elif type_in == 'video-features':
                    x = self.loadVideoFeatures(x, id_in, set_name, self.max_video_len[id_in],
                                               normalization_type, normalization, self.features_lengths[id_in],
                                               data_augmentation=dataAugmentation)
            X.append(x)

        return X

    def getXY(self, set_name, k, normalization_type='0-1', normalization=False, meanSubstraction=True,
              dataAugmentation=True, debug=False):
        """
            Gets the [X,Y] pairs for the next 'k' samples in the desired set.

            :param set_name: 'train', 'val' or 'test' set
            :param k: number of consecutive samples retrieved from the corresponding set.
            :param sorted_batches: If True, it will pick data of the same size
            :param debug: if True all data will be returned without preprocessing


            # 'raw-image', 'video', 'image-features' and 'video-features'-related parameters

            :param normalization: indicates if we want to normalize the data.


            # 'image-features' and 'video-features'-related parameters

            :param normalization_type: indicates the type of normalization applied. See available types in self.__available_norm_im_vid for 'image' and 'video' and self.__available_norm_feat for 'image-features' and 'video-features'.


            # 'raw-image' and 'video'-related parameters

            :param meanSubstraction: indicates if we want to substract the training mean from the returned images (only applicable if normalization=True)
            :param dataAugmentation: indicates if we want to apply data augmentation to the loaded images (random flip and cropping)

            :return: [X,Y], list of input and output data variables of the next 'k' consecutive samples belonging to the chosen 'set_name'
            :return: [X, Y, [new_last, last, surpassed]] if debug==True
        """
        self.__checkSetName(set_name)
        self.__isLoaded(set_name, 0)
        self.__isLoaded(set_name, 1)

        [new_last, last, surpassed] = self.__getNextSamples(k, set_name)

        # Recover input samples
        X = []
        for id_in, type_in in zip(self.ids_inputs, self.types_inputs):
            if id_in in self.optional_inputs:
                try:
                    if surpassed:
                        x = eval('self.X_' + set_name + '[id_in][last:]') + eval(
                            'self.X_' + set_name + '[id_in][0:new_last]')
                    else:
                        x = eval('self.X_' + set_name + '[id_in][last:new_last]')
                except:
                    x = []
            else:
                if surpassed:
                    x = eval('self.X_' + set_name + '[id_in][last:]') + eval(
                        'self.X_' + set_name + '[id_in][0:new_last]')
                else:
                    x = eval('self.X_' + set_name + '[id_in][last:new_last]')

            # Pre-process inputs
            if not debug:
                if type_in == 'raw-image':
                    daRandomParams = None
                    if dataAugmentation:
                        daRandomParams = self.getDataAugmentationRandomParams(x, id_in)
                    x = self.loadImages(x, id_in, normalization_type, normalization, meanSubstraction, dataAugmentation,
                                        daRandomParams)
                elif type_in == 'video':
                    x = self.loadVideos(x, id_in, last, set_name, self.max_video_len[id_in],
                                        normalization_type, normalization, meanSubstraction, dataAugmentation)
                elif type_in == 'text':
                    x = self.loadText(x, self.vocabulary[id_in],
                                      self.max_text_len[id_in][set_name], self.text_offset[id_in],
                                      fill=self.fill_text[id_in], pad_on_batch=self.pad_on_batch[id_in],
                                      words_so_far=self.words_so_far[id_in], loading_X=True)[0]
                elif type_in == 'doc':
                    x_t = []
                    counter = 0
                    doc_size = 0
                    sent_len = 0
                    for i in x:
                        x_i = self.loadText(i, self.vocabulary[id_in],
                                       self.max_text_len[id_in][set_name], self.text_offset[id_in],
                                       fill=self.fill_text[id_in], pad_on_batch=self.pad_on_batch[id_in],
                                       words_so_far=self.words_so_far[id_in], loading_X=True)[0]
                        doc_size = x_i.shape[0]
                        sent_len = x_i.shape[1]
                        x_t.append(x_i)
                        counter += 1
                    x_fin = np.empty((len(x), doc_size, sent_len), dtype=object)
                    counter = 0
                    for i in x_t:
                        x_fin[counter,:,:] = i
                        counter += 1
                    x = x_fin

                elif type_in == 'image-features':
                    #print("Correctly asked for the image-features of getXY SHU") #Shu
                    x = self.loadFeatures(x, self.features_lengths[id_in], normalization_type, normalization,
                                          data_augmentation=dataAugmentation)
                elif type_in == 'video-features':
                    x = self.loadVideoFeatures(x, id_in, set_name, self.max_video_len[id_in], normalization_type,
                                               normalization, self.features_lengths[id_in],
                                               data_augmentation=dataAugmentation)
            X.append(x)

        # Recover output samples
        Y = []
        for id_out, type_out in zip(self.ids_outputs, self.types_outputs):
            if surpassed:
                y = eval('self.Y_' + set_name + '[id_out][last:]') + eval('self.Y_' + set_name + '[id_out][0:new_last]')
            else:
                y = eval('self.Y_' + set_name + '[id_out][last:new_last]')

            # Pre-process outputs
            if not debug:
                if type_out == 'categorical':
                    nClasses = len(self.dic_classes[id_out])
                    #load_sample_weights = self.sample_weights[id_out][set_name]
                    y = self.loadCategorical(y, nClasses)
                elif type_out == 'binary':
                    y = self.loadBinary(y, id_out)
                elif type_out == 'real':
                    y = np.array(y).astype(np.float32)
                elif (type_out == '3DLabel'):
                    nClasses = len(self.classes[id_out])
                    assoc_id_in = self.id_in_3DLabel[id_out]
                    if surpassed:
                        imlist = eval('self.X_' + set_name + '[assoc_id_in][last:]') + eval(
                            'self.X_' + set_name + '[assoc_id_in][0:new_last]')
                    else:
                        imlist = eval('self.X_' + set_name + '[assoc_id_in][last:new_last]')

                    y = self.load3DLabels(y, nClasses, dataAugmentation, daRandomParams,
                                          self.img_size[assoc_id_in], self.img_size_crop[assoc_id_in],
                                          imlist)
                elif (type_out == '3DSemanticLabel'):
                    nClasses = len(self.classes[id_out])
                    classes_to_colour = self.semantic_classes[id_out]
                    assoc_id_in = self.id_in_3DLabel[id_out]
                    if surpassed:
                        imlist = eval('self.X_' + set_name + '[assoc_id_in][last:]') + eval(
                            'self.X_' + set_name + '[assoc_id_in][0:new_last]')
                    else:
                        imlist = eval('self.X_' + set_name + '[assoc_id_in][last:new_last]')

                    y = self.load3DSemanticLabels(y, nClasses, classes_to_colour, dataAugmentation, daRandomParams,
                                                  self.img_size[assoc_id_in], self.img_size_crop[assoc_id_in],
                                                  imlist)
                elif type_out == 'text':
                    y = self.loadText(y, self.vocabulary[id_out],
                                      self.max_text_len[id_out][set_name], self.text_offset[id_out],
                                      fill=self.fill_text[id_out], pad_on_batch=self.pad_on_batch[id_out],
                                      words_so_far=self.words_so_far[id_out], loading_X=False)
                    # Use whole sentence as class (classifier model)
                    if self.max_text_len[id_out][set_name] == 0:
                        y_aux = np_utils.to_categorical(y, self.vocabulary_len[id_out]).astype(np.uint8)
                    # Use words separately (generator model)
                    else:
                        y_aux = np.zeros(list(y[0].shape) + [self.vocabulary_len[id_out]]).astype(np.uint8)
                        for idx in range(y[0].shape[0]):
                            y_aux[idx] = np_utils.to_categorical(y[0][idx], self.vocabulary_len[id_out]).astype(
                                np.uint8)

                        if set_name in self.sample_weights[id_out]:

                            new_dict = dict()
                            for key in self.vocabulary[id_out]['idx2words']:
                                value = self.vocabulary[id_out]['idx2words'][key]
                                if value in self.sample_weights[id_out][set_name]:
                                    new_dict[key] = self.sample_weights[id_out][set_name][value]

                            for i in range(y_aux.shape[0]):
                                for j in range(y_aux[i].shape[0]):
                                    for new_key in new_dict:
                                        if y_aux[i][j][new_key] == 1:
                                            y[1][i][j] = new_dict[new_key]
                            y_aux = (y_aux, y[1])

                        # elif self.sample_weights[id_out][set_name]:
                        #     y_aux = (y_aux, y[1])  # join data and mask
                    y = y_aux
            Y.append(y)

        if debug:
            return [X, Y, [new_last, last, surpassed]]

        return [X, Y]

    def getXY_FromIndices(self, set_name, k, normalization_type='0-1', normalization=False, meanSubstraction=True,
                          dataAugmentation=True, debug=False):
        """
            Gets the [X,Y] pairs for the samples in positions 'k' in the desired set.

            :param set_name: 'train', 'val' or 'test' set
            :param k: positions of the desired samples
            :param sorted_batches: If True, it will pick data of the same size
            :param debug: if True all data will be returned without preprocessing


            # 'raw-image', 'video', 'image-features' and 'video-features'-related parameters

            :param normalization: indicates if we want to normalize the data.


            # 'image-features' and 'video-features'-related parameters

            :param normalization_type: indicates the type of normalization applied. See available types in self.__available_norm_im_vid for 'raw-image' and 'video' and self.__available_norm_feat for 'image-features' and 'video-features'.


            # 'raw-image' and 'video'-related parameters

            :param meanSubstraction: indicates if we want to substract the training mean from the returned images (only applicable if normalization=True)
            :param dataAugmentation: indicates if we want to apply data augmentation to the loaded images (random flip and cropping)

            :return: [X,Y], list of input and output data variables of the samples identified by the indices in 'k' samples belonging to the chosen 'set_name'
        """

        self.__checkSetName(set_name)
        self.__isLoaded(set_name, 0)
        self.__isLoaded(set_name, 1)

        # Recover input samples
        X = []
        for id_in, type_in in zip(self.ids_inputs, self.types_inputs):
            ghost_x = False
            if id_in in self.optional_inputs:
                try:
                    x = [eval('self.X_' + set_name + '[id_in][index]') for index in k]
                except:
                    x = [[]] * len(k)
                    ghost_x = True
            else:
                x = [eval('self.X_' + set_name + '[id_in][index]') for index in k]

            # if(set_name=='val'):
            #    logging.info(x)

            # Pre-process inputs
            if not debug and not ghost_x:
                if type_in == 'raw-image':
                    daRandomParams = None
                    if dataAugmentation:
                        daRandomParams = self.getDataAugmentationRandomParams(x, id_in)
                    x = self.loadImages(x, id_in, normalization_type, normalization, meanSubstraction, dataAugmentation,
                                        daRandomParams)
                elif type_in == 'video':
                    x = self.loadVideosByIndex(x, id_in, k, set_name, self.max_video_len[id_in],
                                               normalization_type, normalization, meanSubstraction, dataAugmentation)
                elif type_in == 'text':
                    x = self.loadText(x, self.vocabulary[id_in],
                                      self.max_text_len[id_in][set_name], self.text_offset[id_in],
                                      fill=self.fill_text[id_in], pad_on_batch=self.pad_on_batch[id_in],
                                      words_so_far=self.words_so_far[id_in], loading_X=True)[0]
                elif type_in == 'image-features':
                    x = self.loadFeatures(x, self.features_lengths[id_in], normalization_type, normalization,
                                          data_augmentation=dataAugmentation)
                elif type_in == 'video-features':
                    x = self.loadVideoFeatures(x, id_in, set_name, self.max_video_len[id_in],
                                               normalization_type, normalization, self.features_lengths[id_in],
                                               data_augmentation=dataAugmentation)
            X.append(x)

        # Recover output samples
        Y = []
        for id_out, type_out in zip(self.ids_outputs, self.types_outputs):
            y = [eval('self.Y_' + set_name + '[id_out][index]') for index in k]

            # if(set_name=='val'):
            #    logging.info(y)

            # Pre-process outputs
            if not debug:
                if type_out == 'categorical':
                    nClasses = len(self.dic_classes[id_out])
                    load_sample_weights = self.sample_weights[id_out][set_name]
                    y = self.loadCategorical(y, nClasses, id_out, load_sample_weights)
                elif type_out == 'binary':
                    y = self.loadBinary(y, id_out)
                elif type_out == 'real':
                    y = np.array(y).astype(np.float32)
                elif (type_out == '3DLabel'):
                    nClasses = len(self.classes[id_out])
                    assoc_id_in = self.id_in_3DLabel[id_out]
                    imlist = [eval('self.X_' + set_name + '[assoc_id_in][index]') for index in k]
                    y = self.load3DLabels(y, nClasses, dataAugmentation, daRandomParams,
                                          self.img_size[assoc_id_in], self.img_size_crop[assoc_id_in],
                                          imlist)
                elif (type_out == '3DSemanticLabel'):
                    nClasses = len(self.classes[id_out])
                    classes_to_colour = self.semantic_classes[id_out]
                    assoc_id_in = self.id_in_3DLabel[id_out]
                    imlist = [eval('self.X_' + set_name + '[assoc_id_in][index]') for index in k]
                    y = self.load3DSemanticLabels(y, nClasses, classes_to_colour, dataAugmentation, daRandomParams,
                                                  self.img_size[assoc_id_in], self.img_size_crop[assoc_id_in],
                                                  imlist)
                elif type_out == 'text':
                    y = self.loadText(y, self.vocabulary[id_out],
                                      self.max_text_len[id_out][set_name], self.text_offset[id_out],
                                      fill=self.fill_text[id_out], pad_on_batch=self.pad_on_batch[id_out],
                                      words_so_far=self.words_so_far[id_out], loading_X=False)

                    # Use whole sentence as class (classifier model)
                    if self.max_text_len[id_out][set_name] == 0:
                        y_aux = np_utils.to_categorical(y, self.vocabulary_len[id_out]).astype(np.uint8)
                    # Use words separately (generator model)
                    else:
                        y_aux = np.zeros(list(y[0].shape) + [self.vocabulary_len[id_out]]).astype(np.uint8)
                        for idx in range(y[0].shape[0]):
                            y_aux[idx] = np_utils.to_categorical(y[0][idx], self.vocabulary_len[id_out]).astype(
                                np.uint8)
                        if set_name in self.sample_weights[id_out]:

                            new_dict = dict()
                            for key in self.vocabulary[id_out]['idx2words']:
                                value = self.vocabulary[id_out]['idx2words'][key]
                                if value in self.sample_weights[id_out][set_name]:
                                    new_dict[key] = self.sample_weights[id_out][set_name][value]

                            for i in range(y_aux.shape[0]):
                                for j in range(y_aux[i].shape[0]):
                                    for new_key in new_dict:
                                        if y_aux[i][j][new_key] == 1:
                                            y[1][i][j] = new_dict[new_key]
                            y_aux = (y_aux, y[1])

                        # if self.sample_weights[id_out][set_name]:
                        #     y_aux = (y_aux, y[1])  # join data and mask
                    y = y_aux
            Y.append(y)

        return [X, Y]

    def getX_FromIndices(self, set_name, k, normalization_type='0-1', normalization=False, meanSubstraction=True,
                         dataAugmentation=True, debug=False):
        """
            Gets the [X,Y] pairs for the samples in positions 'k' in the desired set.

            :param set_name: 'train', 'val' or 'test' set
            :param k: positions of the desired samples
            :param sorted_batches: If True, it will pick data of the same size
            :param debug: if True all data will be returned without preprocessing


            # 'raw-image', 'video', 'image-features' and 'video-features'-related parameters

            :param normalization: indicates if we want to normalize the data.


            # 'image-features' and 'video-features'-related parameters

            :param normalization_type: indicates the type of normalization applied. See available types in self.__available_norm_im_vid for 'raw-image' and 'video' and self.__available_norm_feat for 'image-features' and 'video-features'.


            # 'raw-image' and 'video'-related parameters

            :param meanSubstraction: indicates if we want to substract the training mean from the returned images (only applicable if normalization=True)
            :param dataAugmentation: indicates if we want to apply data augmentation to the loaded images (random flip and cropping)

            :return: [X,Y], list of input and output data variables of the samples identified by the indices in 'k' samples belonging to the chosen 'set_name'
        """

        self.__checkSetName(set_name)
        self.__isLoaded(set_name, 0)

        # Recover input samples
        X = []
        for id_in, type_in in zip(self.ids_inputs, self.types_inputs):
            #print("getX_FromIndices id_in", id_in, "type_in", type_in) #Shu
            ghost_x = False
            if id_in in self.optional_inputs:
                try:
                    x = [eval('self.X_' + set_name + '[id_in][index]') for index in k]
                except:
                    x = [[]] * len(k)
                    ghost_x = True
            else:
                x = [eval('self.X_' + set_name + '[id_in][index]') for index in k]

            # if(set_name=='val'):
            #    logging.info(x)

            # Pre-process inputs
            if not debug and not ghost_x:
                if type_in == 'raw-image':
                    daRandomParams = None
                    if dataAugmentation:
                        daRandomParams = self.getDataAugmentationRandomParams(x, id_in)
                    x = self.loadImages(x, id_in, normalization_type, normalization, meanSubstraction, dataAugmentation,
                                        daRandomParams)
                elif type_in == 'video':
                    x = self.loadVideosByIndex(x, id_in, k, set_name, self.max_video_len[id_in],
                                               normalization_type, normalization, meanSubstraction, dataAugmentation)
                elif type_in == 'text':
                    x = self.loadText(x, self.vocabulary[id_in],
                                      self.max_text_len[id_in][set_name], self.text_offset[id_in],
                                      fill=self.fill_text[id_in], pad_on_batch=self.pad_on_batch[id_in],
                                      words_so_far=self.words_so_far[id_in], loading_X=True)[0]
                elif type_in == 'image-features':
                    x = self.loadFeatures(x, self.features_lengths[id_in], normalization_type, normalization,
                                          data_augmentation=dataAugmentation)
                elif type_in == 'video-features':
                    x = self.loadVideoFeatures(x, id_in, set_name, self.max_video_len[id_in],
                                               normalization_type, normalization, self.features_lengths[id_in],
                                               data_augmentation=dataAugmentation)
            X.append(x)

        return X

    def getY(self, set_name, init, final, normalization_type='0-1', normalization=False, meanSubstraction=True,
             dataAugmentation=True, debug=False):
        """
            Gets the [Y] samples for the FULL dataset

            :param set_name: 'train', 'val' or 'test' set
            :param init: initial position in the corresponding set split. Must be bigger or equal than 0 and smaller than final.
            :param final: final position in the corresponding set split.
            :param debug: if True all data will be returned without preprocessing


            # 'raw-image', 'video', 'image-features' and 'video-features'-related parameters

            :param normalization: indicates if we want to normalize the data.


            :param normalization_type: indicates the type of normalization applied. See available types in self.__available_norm_im_vid for 'raw-image' and 'video' and self.__available_norm_feat for 'image-features' and 'video-features'.


            # 'raw-image' and 'video'-related parameters

            :param meanSubstraction: indicates if we want to substract the training mean from the returned images (only applicable if normalization=True)
            :param dataAugmentation: indicates if we want to apply data augmentation to the loaded images (random flip and cropping)

            :return: Y, list of output data variables from sample 'init' to 'final' belonging to the chosen 'set_name'
        """
        self.__checkSetName(set_name)
        self.__isLoaded(set_name, 1)

        if final > eval('self.len_' + set_name):
            raise Exception('"final" index must be smaller than the number of samples in the set.')
        if init < 0:
            raise Exception('"init" index must be equal or greater than 0.')
        if init >= final:
            raise Exception('"init" index must be smaller than "final" index.')

        # Recover output samples
        Y = []
        for id_out, type_out in zip(self.ids_outputs, self.types_outputs):
            y = eval('self.Y_' + set_name + '[id_out][init:final]')

            # Pre-process outputs
            if not debug:
                if type_out == 'categorical':
                    nClasses = len(self.dic_classes[id_out])
                    load_sample_weights = self.sample_weights[id_out][set_name]
                    y = self.loadCategorical(y, nClasses, id_out, load_sample_weights)
                elif type_out == 'binary':
                    y = self.loadBinary(y, id_out)
                elif type_out == 'real':
                    y = np.array(y).astype(np.float32)
                elif (type_out == '3DLabel'):
                    nClasses = len(self.classes[id_out])
                    assoc_id_in = self.id_in_3DLabel[id_out]
                    imlist = eval('self.X_' + set_name + '[assoc_id_in][init:final]')
                    y = self.load3DLabels(y, nClasses, dataAugmentation, None,
                                          self.img_size[assoc_id_in], self.img_size_crop[assoc_id_in],
                                          imlist)
                elif (type_out == '3DSemanticLabel'):
                    nClasses = len(self.classes[id_out])
                    classes_to_colour = self.semantic_classes[id_out]
                    assoc_id_in = self.id_in_3DLabel[id_out]
                    imlist = eval('self.X_' + set_name + '[assoc_id_in][init:final]')
                    y = self.load3DSemanticLabels(y, nClasses, classes_to_colour, dataAugmentation, None,
                                                  self.img_size[assoc_id_in], self.img_size_crop[assoc_id_in],
                                                  imlist)
                elif type_out == 'text':
                    y = self.loadText(y, self.vocabulary[id_out],
                                      self.max_text_len[id_out][set_name], self.text_offset[id_out],
                                      fill=self.fill_text[id_out], pad_on_batch=self.pad_on_batch[id_out],
                                      words_so_far=self.words_so_far[id_out], loading_X=False)

                    # Use whole sentence as class (classifier model)
                    if self.max_text_len[id_out][set_name] == 0:
                        y_aux = np_utils.to_categorical(y, self.vocabulary_len[id_out]).astype(np.uint8)
                    # Use words separately (generator model)
                    else:
                        y_aux = np.zeros(list(y[0].shape) + [self.vocabulary_len[id_out]]).astype(np.uint8)
                        for idx in range(y[0].shape[0]):
                            y_aux[idx] = np_utils.to_categorical(y[0][idx], self.vocabulary_len[id_out]).astype(
                                np.uint8)
                        if self.sample_weights[id_out][set_name]:
                            y_aux = (y_aux, y[1])  # join data and mask

                    y = y_aux
            Y.append(y)

        return Y

    # ------------------------------------------------------- #
    #       AUXILIARY FUNCTIONS
    #
    # ------------------------------------------------------- #


    def __str__(self):
        """
        Prints the basic input-output information of the Dataset instance.

        :return: String representation of the Dataset.
        """

        str_ = '---------------------------------------------\n'
        str_ += '\tDataset ' + self.name + '\n'
        str_ += '---------------------------------------------\n'
        str_ += 'store path: ' + self.path + '\n'
        str_ += 'data length: ' + '\n'
        str_ += '\ttrain - ' + str(self.len_train) + '\n'
        str_ += '\tval   - ' + str(self.len_val) + '\n'
        str_ += '\ttest  - ' + str(self.len_test) + '\n'

        str_ += '\n'
        str_ += '[ INPUTS ]\n'
        for id_in, type_in in zip(self.ids_inputs, self.types_inputs):
            str_ += type_in + ': ' + id_in + '\n'

        str_ += '\n'
        str_ += '[ OUTPUTS ]\n'
        for id_out, type_out in zip(self.ids_outputs, self.types_outputs):
            str_ += type_out + ': ' + id_out + '\n'

        str_ += '---------------------------------------------\n'
        return str_

    def __isLoaded(self, set_name, pos):
        """
        Checks if the data from set_name at pos is already loaded
        :param set_name:
        :param pos:
        :return:
        """
        if eval('not self.loaded_' + set_name + '[pos]'):
            if pos == 0:
                raise Exception('Set ' + set_name + ' samples are not loaded yet.')
            elif pos == 1:
                raise Exception('Set ' + set_name + ' labels are not loaded yet.')
        return

    def __checkSetName(self, set_name):
        """
        Checks name of a split.
        Only "train", "val" or "test" are valid set names.
        :param set_name: Split name
        :return: Boolean specifying the validity of the name
        """
        if set_name != 'train' and set_name != 'val' and set_name != 'test':
            raise Exception(
                'Incorrect set_name specified "' + set_name + '"\nOnly "train", "val" or "test" are valid set names.')
        return

    def __checkLengthSet(self, set_name):
        """
        Check that the length of the inputs and outputs match. Only checked if the input is not optional.
        :param set_name:
        :return:
        """
        if eval('self.loaded_' + set_name + '[0] and self.loaded_' + set_name + '[1]'):
            lengths = []
            plot_ids_in = []
            for id_in in self.ids_inputs:
                #print("id_in", id_in, set_name) #Shu
                if id_in not in self.optional_inputs:
                    plot_ids_in.append(id_in)
                    exec ('lengths.append(len(self.X_' + set_name + '[id_in]))')
            for id_out in self.ids_outputs:
                #print("id_out", id_out, set_name) #Shu
                exec ('lengths.append(len(self.Y_' + set_name + '[id_out]))')
            if lengths[1:] != lengths[:-1]:
                raise Exception('Inputs and outputs size '
                                '(' + str(lengths) + ') for "' + set_name + '" set do not match.\n'
                                                                            '\t Inputs:' + str(plot_ids_in) + ''
                                                                            '\t Outputs:' + str(
                    self.ids_outputs))

    def __getNextSamples(self, k, set_name):
        """
            Gets the indices to the next K samples we are going to read.
        """
        self.__lock_read.acquire()  # LOCK (for avoiding reading the same samples by different threads)

        new_last = eval('self.last_' + set_name + '+k')
        last = eval('self.last_' + set_name)
        length = eval('self.len_' + set_name)
        if new_last > length:
            new_last = new_last - length
            surpassed = True
        else:
            surpassed = False
        exec ('self.last_' + set_name + '= new_last')

        self.__lock_read.release()  # UNLOCK

        return [new_last, last, surpassed]

    def __getstate__(self):
        """
            Behavour applied when pickling a Dataset instance.
        """
        obj_dict = self.__dict__.copy()
        del obj_dict['_Dataset__lock_read']
        return obj_dict

    def __setstate__(self, dict):
        """
            Behavour applied when unpickling a Dataset instance.
        """
        dict['_Dataset__lock_read'] = threading.Lock()
        self.__dict__ = dict
