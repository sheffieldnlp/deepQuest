import numpy as np
from keras.models import Sequential
from keras.utils import np_utils

from keras_wrapper.cnn_model import CNN_Model


class Stage(CNN_Model):
    """
        Class for defining a single stage from a Staged_Network. 
        This class is only intended to be used in conjunction with the Staged_Network class.
    """

    def __init__(self, nInput, nOutput, input_shape, output_shape, type='basic_model', silence=False,
                 structure_path=None, weights_path=None, model_name=None, plots_path=None, models_path=None):
        """
            Basic class constructor. See CNN_Model parameters for additional details.
            
            :param nInput: number of different classes used in the whole Stage_Network instance.
            :param nOutput: number of output values (classes or neuron values) in the current Stage.
            :param input_shape: array with 3 integers which define the images' input shape [height, width, channels]
            :param output_shape: array with integers which define the network's output shape [height, width, channels]
            :param structure_path: path to a Keras' model json file. If we speficy this parameter then 'type' will be only an informative parameter.
            :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
            :param model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
            :param plots_path: path to the folder where the plots will be stored during training
            :param models_path: path to the folder where the temporal model packups will be stored
        """
        super(Stage, self).__init__(nOutput, type, silence, input_shape, structure_path, weights_path, model_name,
                                    plots_path, models_path)
        self._CNN_Model__toprint += ['nInput', 'nOutput', 'mask', 'mapping']

        # List of input samples awaiting for the next forward pass
        self.__emptyTrainingQueues()
        self.nInput = nInput
        self.nOutput = nOutput

        # (default) Input classes mapping for the Stage instance.
        mapping = dict()
        for input in range(nInput):
            mapping[input] = input
        self.defineClassMapping(mapping)

        # (default) Output mask for the Stage instance.
        self.output_shape = output_shape
        self.mask = None

        # Class weight or balanced training
        self.class_weight = dict()
        for i in range(self.nOutput):
            self.class_weight[i] = 1

    def defineClassMapping(self, mapping):
        """
            Defines an input mapping from all the classes available in a Staged_Network to the ones used in this particular Stage instance.
            
            :param mapping: dictionary with all the classes in the Staged_Network as 'keys' and the corresponding mapped inputs to this Stage as 'values'. If some 'key' is not used, its 'value' should be set to None.
        """
        self.mapping = mapping

    def defineOutputMask(self, mask):
        """
            Defines an output mask for redirecting this stage's output to the following stage once this stage has been trained (on test mode).
                
            :param mask: dictionary with all the Stage's output values indices as 'keys' and the corresponding mapped indices on the final test's output as 'values'. If some 'key' is not used, its 'value' should be set to None. E.g. if we have output_shape = [1,1] then mask = {'[0]': [0], '[1]': None} If we want to disable the mask, we can set it to None. The mask is disabled by default.
        """
        self.mask = mask

    def applyClassMapping(self, Y):
        """
            Returns the corresponding integer identifiers for the current Stage's mapping given a set of categorical arrays Y.
        """
        # Get labels from Keras' categorical representation
        labels = np_utils.categorical_probas_to_classes(Y)

        # Map labels for this stage
        return [self.mapping[l] for l in labels]

    def applyMask(self, prediction):
        """
            Returns a prediction matrix after applying the defined output mask.
        """
        # prediction = prediction[0]
        if (not self.mask):
            # return [prediction]
            return prediction

        nSamples = prediction.shape[0]
        out = np.zeros(tuple([nSamples] + self.output_shape));
        for s in range(nSamples):
            for i, o in self.mask.iteritems():
                if (o):
                    o = str(o)
                    exec ('out[s,' + o[1:-1] + '] = prediction[s,' + i[1:-1] + ']')

        # return [out]
        return out

    # ------------------------------------------------------- #
    #       TRAINING/TEST
    #           Methods for train and testing on the current Stage
    # ------------------------------------------------------- #

    def predictOnBatch(self, X, in_name=None, out_name=None, expand=False):
        """
            Applies a forward pass and returns the predicted values after being processed by the output mask. 
            
            :param in_name: name of the input we are asking for (from the previous stage). Only applicable to Graph models.
            :param out_name: name of the output used for prediction (from the current stage).
            :param expand: indicates if we want to expand the input dimensions to 4
        """
        predictions = CNN_Model.predictOnBatch(self, X, in_name=in_name, out_name=out_name, expand=expand)

        if (not self.mask):
            return predictions

        return self.applyMask(predictions)

    def testOnBatch(self, X, Y, accuracy=True, out_name=None):
        """
            Applies a test on the samples provided and returns the resulting loss and accuracy (if True).
            It selects only the samples that are valid for the current Stage (self.mapping[labels] != None).
            
            :param out_name: name of the output we are asking for. Only applicable to Graph models.
        """
        # Map labels for this stage
        mapped_labels = self.applyClassMapping(Y)

        # Choose only the valid labels for this stage
        selected_samples = [[i, l] for i, l in enumerate(mapped_labels) if l != None]

        # Return False if none of the provided samples are valid for the current stage
        if (not selected_samples):
            return False

        n_selected = len(selected_samples)
        # Gather the selected ones
        selected_X = np.zeros(tuple([n_selected] + list(X.shape[1:])))
        selected_Y = np.zeros(tuple([n_selected, self.nOutput]))
        for i, [s, l] in enumerate(selected_samples):
            selected_X[i] = X[s]
            selected_Y[i] = np_utils.to_categorical([l], self.nOutput).astype(np.uint8)

        # Evaluate
        return CNN_Model.testOnBatch(self, selected_X, selected_Y, accuracy=accuracy, out_name=out_name)

    def trainOnBatch_DEPRECATED_class_weight(self, X, Y, batch_size, balanced=True, out_name=None):
        """
            Trains the current stage on the last 'batch_size' samples received. It makes sure the mapping of the
            introduced samples is balanced among all the defined classes. If it isn't it keeps storing them for
            a latter forward-backward pass.
            
            :param X: numpy array with a set of loaded images ready to use for Keras
            :param Y: numpy array with a set of categorical labels ready to use for keras
            :param batch_size: number of samples that we want to use to train on the next batch
            :param balanced: indicates if we want to train on a set of balanced samples (same number of samples from each class)
            :param out_name: name of the output node that will be used to evaluate the network accuracy. Only applicable for Graph models.
        """
        # Map labels for this stage
        mapped_labels = self.applyClassMapping(Y)

        # Choose only the valid labels for this stage
        n_valid = len(mapped_labels) - mapped_labels.count(None)
        if (n_valid > 0):
            batch = dict()
            batch['X'] = np.zeros(tuple([n_valid] + list(X.shape[1:])))
            batch['Y'] = np.zeros(tuple([n_valid, self.nOutput]))
            total_counts = 0
            for i, l in enumerate(mapped_labels):
                if l != None:
                    batch['X'][total_counts] = X[i]
                    # batch['Y'][total_counts] = Y[i]
                    batch['Y'][total_counts] = np_utils.to_categorical([l], self.nOutput).astype(np.uint8)
                    total_counts += 1

            # Train on batch
            if (isinstance(self.model, Sequential)):
                # Balanced training
                if (balanced):
                    class_weight = self.class_weight
                else:
                    class_weight = None

                loss = self.model.train_on_batch(batch['X'], batch['Y'], class_weight=class_weight, accuracy=False)
                loss = loss[0]
                [score, top_score] = self._getSequentialAccuracy(batch['Y'], self.model.predict_on_batch(batch['X'])[0])
            else:
                [data, last_output] = self._prepareGraphData(batch['X'], batch['Y'])
                # Balanced training
                if (balanced):
                    if (out_name):
                        out = out_name
                    else:
                        out = last_output
                    class_weight = {out: self.class_weight}
                else:
                    class_weight = {}

                loss = self.model.train_on_batch(data, class_weight=class_weight)
                loss = loss[0]
                score = self._getGraphAccuracy(data, self.model.predict_on_batch(data))
                top_score = score[1]
                score = score[0]
                if (out_name):
                    score = score[out_name]
                    top_score = top_score[out_name]
                else:
                    score = score[last_output]
                    top_score = top_score[last_output]

            # If we have been able to train, we return the current training loss and score and the number of samples in the batch
            return [loss, score, top_score, n_valid]
        return False

    def trainOnBatch(self, X, Y, batch_size, balanced=True, out_name=None):
        """
            Trains the current stage on the last 'batch_size' samples received. It makes sure the mapping of the
            introduced samples is balanced among all the defined classes. If it isn't it keeps storing them for
            a latter forward-backward pass.
            
            :param X: numpy array with a set of loaded images ready to use for Keras
            :param Y: numpy array with a set of categorical labels ready to use for keras
            :param batch_size: number of samples that we want to use to train on the next batch
            :param balanced: indicates if we want to train on a set of balanced samples (same number of samples from each class)
            :param out_name: name of the output node that will be used to evaluate the network accuracy. Only applicable for Graph models.
        """
        # Map labels for this stage
        mapped_labels = self.applyClassMapping(Y)

        # Choose only the valid labels for this stage
        n_valid = len(mapped_labels) - mapped_labels.count(None)
        if (n_valid > 0):
            batch = dict()
            batch['X'] = np.zeros(tuple([n_valid] + list(X.shape[1:])))
            batch['Y'] = np.zeros(tuple([n_valid, self.nOutput]))
            total_counts = 0
            counts = [0 for i in range(self.nOutput)]
            labels = [None for i in range(n_valid)]
            for i, l in enumerate(mapped_labels):
                if l != None:
                    batch['X'][total_counts] = X[i]
                    batch['Y'][total_counts] = np_utils.to_categorical([l], self.nOutput).astype(np.uint8)
                    counts[l] += 1
                    labels[total_counts] = l
                    total_counts += 1

            # Calculate samples weight
            sw = [counts[l] for l in labels]
            sw = 1.0 / np.array(sw)

            # Train on batch
            if (isinstance(self.model, Sequential)):
                # Balanced training
                if (balanced):
                    sample_weight = sw
                else:
                    sample_weight = None

                loss = self.model.train_on_batch(batch['X'], batch['Y'], sample_weight=sample_weight, accuracy=False)
                loss = loss[0]
                [score, top_score] = self._getSequentialAccuracy(batch['Y'], self.model.predict_on_batch(batch['X'])[0])
            else:
                [data, last_output] = self._prepareGraphData(batch['X'], batch['Y'])

                # Balanced training
                if (balanced):
                    if (out_name):
                        out = out_name
                    else:
                        out = last_output
                    sample_weight = {out: sw}
                else:
                    sample_weight = {}

                loss = self.model.train_on_batch(data, sample_weight=sample_weight)
                loss = loss[0]
                score = self._getGraphAccuracy(data, self.model.predict_on_batch(data))
                top_score = score[1]
                score = score[0]
                if (out_name):
                    score = score[out_name]
                    top_score = top_score[out_name]
                else:
                    score = score[last_output]
                    top_score = top_score[last_output]

            # If we have been able to train, we return the current training loss and score and the number of samples in the batch
            return [loss, score, top_score, n_valid]
        return False

    def trainOnBatch_DEPRECATED_lists(self, X, Y, batch_size, balanced=True, out_name=None):
        """
            Trains the current stage on the last 'batch_size' samples received. It makes sure the mapping of the
            introduced samples is balanced among all the defined classes. If it isn't it keeps storing them for
            a latter forward-backward pass.
            
            :param X: numpy array with a set of loaded images ready to use for Keras
            :param Y: numpy array with a set of categorical labels ready to use for keras
            :param batch_size: number of samples that we want to use to train on the next batch
            :param balanced: indicates if we want to train on a set of balanced samples (same number of samples from each class)
            :param out_name: name of the output node that will be used to evaluate the network accuracy. Only applicable for Graph models.
        """
        # Map labels for this stage
        mapped_labels = self.applyClassMapping(Y)

        # Choose only the valid labels for this stage
        for i, l in enumerate(mapped_labels):
            if l != None:
                # Insert Xs
                self.__X_queue.append(X[i])
                #                if(len(X.shape) == 4):
                #                    self.__X_queue.append(X[i,:,:,:])
                #                elif(len(X.shape) == 3):
                #                    self.__X_queue.append(X[i,:,:])
                #                else:
                #                    self.__X_queue.append(X[i,:])
                # Insert Ys
                self.__Y_queue.append(np_utils.to_categorical([l], self.nOutput).astype(np.uint8))
                # Insert integer labels
                self.__label_queue.append(l)

        # Get the next set of samples
        [ready, batch] = self.isReadyToTrainOnBatch(batch_size, balanced)
        if (ready):
            # Train on batch
            if (isinstance(self.model, Sequential)):
                loss = self.model.train_on_batch(batch['X'], batch['Y'], accuracy=False)
                loss = loss[0]
                [score, top_score] = self._getSequentialAccuracy(batch['Y'], self.model.predict_on_batch(batch['X'])[0])
            else:
                [data, last_output] = self._prepareGraphData(batch['X'], batch['Y'])
                loss = self.model.train_on_batch(data)
                loss = loss[0]
                score = self._getGraphAccuracy(data, self.model.predict_on_batch(data))
                top_score = score[1]
                score = score[0]
                if (out_name):
                    score = score[out_name]
                    top_score = top_score[out_name]
                else:
                    score = score[last_output]
                    top_score = top_score[last_output]

            # If we have been able to train, we return the current training loss and score
            return [loss, score, top_score, batch_size]

        else:
            counts = batch
            # If there are too many samples stored in the queues, then we empty them
            if (len(self.__label_queue) > batch_size * 2):

                #                #############################################
                #                logging.debug("Removing samples in stage...")
                #                logging.debug(str(self.mapping))
                #                labels_list = set(self.__label_queue)
                #                for l in labels_list:
                #                    logging.debug("%s: %s samples" % (str(l), str(self.__label_queue.count(l))))
                #                #############################################

                # Remove any sample whose total count is above the mean
                mean_count = np.mean(counts)
                to_remove = batch_size
                i = 0
                while (to_remove > 0):
                    l = self.__label_queue[i]
                    if (counts[l] >= mean_count):
                        self.__label_queue.pop(i)
                        self.__X_queue.pop(i)
                        self.__Y_queue.pop(i)
                        counts[l] -= 1
                        to_remove -= 1
                    else:
                        mean_count = np.mean(counts)
                        i += 1

                    #                #############################################
                    #                logging.debug("Showing remaining samples after removal...")
                    #                labels_list = set(self.__label_queue)
                    #                for l in labels_list:
                    #                    logging.debug("%s: %s samples" % (str(l), str(self.__label_queue.count(l))))
                    #                #############################################

        # If we do not have enough samples yet, then it returns False
        return False

    def isReadyToTrainOnBatch(self, batch_size, balanced):
        """
            Checks if the input samples lists have enough samples to train on a batch. 
            In this case returns the first 'batch_size' samples, else returns False.
            
            :param batch_size: number of samples retrieved for the next training batch
            :param balanced: wether we want a set of balanced samples or not
        """
        # Check if we have enough samples for the chosen batch_size
        if (len(self.__label_queue) < batch_size):
            return [False, []]

        # Counts #samples per class
        counts = [0 for i in range(self.nOutput)]
        for l in self.__label_queue:
            counts[l] += 1

        # Get minimum samples per class (if balanced)
        if (balanced):
            min_num = np.ceil(float(batch_size) / float(self.nOutput))
        else:
            min_num = 0

        # Check if we have enough samples for each class
        not_ok = [c < min_num for c in counts]

        # We do not have enough balanced samples
        if (any(not_ok)):
            return [False, counts]

        # Recover and return samples
        counts = [0 for i in range(self.nOutput)]
        i = 0
        total_counts = 0
        batch = dict()

        # Initialize matrices 
        if (self.__X_queue[0].shape[0] == 1 and len(self.__X_queue[0].shape) == 2):
            batch['X'] = np.zeros(tuple([batch_size] + list(self.__X_queue[0].shape[1:])))
        else:
            batch['X'] = np.zeros(tuple([batch_size] + list(self.__X_queue[0].shape)))
        if (self.__Y_queue[0].shape[0] == 1 and len(self.__Y_queue[0].shape) == 2):
            batch['Y'] = np.zeros(tuple([batch_size] + list(self.__Y_queue[0].shape[1:])))
        else:
            batch['Y'] = np.zeros(tuple([batch_size] + list(self.__Y_queue[0].shape)))

        # Insert samples one by one
        while (total_counts < batch_size):
            l = self.__label_queue[i]
            if (counts[l] == 0 or counts[l] < min_num or not balanced):

                # Recover chosen sample
                x = self.__X_queue.pop(i)
                y = self.__Y_queue.pop(i)
                self.__label_queue.pop(i)

                # Insert into this batch
                batch['X'][total_counts] = x
                batch['Y'][total_counts] = y
                #                if(len(x.shape) == 3):
                #                    batch['X'][total_counts,:,:,:] = x
                #                elif(len(x.shape) == 2):
                #                    batch['X'][total_counts,:,:] = x
                #                batch['Y'][total_counts,:] = y

                counts[l] += 1
                total_counts += 1
            else:
                i += 1

        return [True, batch]

    def __emptyTrainingQueues(self):
        """
            Removes all the elements stored in the queues of awaiting samples.
        """

        self.__X_queue = list()
        self.__Y_queue = list()  # Y categorical representation for the current Stage (after mapping)
        self.__label_queue = list()  # integer representation of the __Y_queue
