import matplotlib as mpl
from keras.models import Sequential, model_from_json
from keras.utils import np_utils

from keras_wrapper.cnn_model import saveModel
from keras_wrapper.deprecated.thread_loader import ThreadDataLoader, retrieveXY

mpl.use('Agg')  # run matplotlib without X server (GUI)

import numpy as np
import cPickle as pk

import time
import os
import math
import copy
import itertools
import logging
import shutil


# def Stage_from_CNN_Model(model):
#    stage = model
#    stage.__class__ = Stage

# ------------------------------------------------------- #
#       SAVE/LOAD
#           External functions for saving and loading CNN_Model instances
# ------------------------------------------------------- #

def saveStagedModel(staged_network, path=None):
    """
        Saves a backup of the current Staged_Network object.
    """

    if (not path):
        path = staged_network.model_path

    if (not staged_network.silence):
        logging.info("<<< Saving Staged_Network model to " + path + " ... >>>")

    # Create models dir
    if (not os.path.isdir(path)):
        os.makedirs(path)

    # Process each stage
    for i in range(staged_network.getNumStages()):
        stage = staged_network.getStage(i)
        path_stage = path + '/Stage_' + str(i)
        if (not os.path.isdir(path_stage)):
            os.makedirs(path_stage)
        if (isinstance(stage, list)):
            for j, s in enumerate(stage):
                path_branch = path_stage + '/Branch_' + str(j)
                if (not os.path.isdir(path_branch)):
                    os.makedirs(path_branch)
                if (hasattr(s, 'model')):
                    # Save model structure
                    json_string = s.model.to_json()
                    open(path_branch + '/Stage_structure.json', 'w').write(json_string)
                    # Save model weights
                    s.model.save_weights(path_branch + '/Stage_weights.h5', overwrite=True)
                # Save additional information
                pk.dump(s, open(path_branch + '/Stage_instance.pkl', 'wb'))
        else:
            if (hasattr(stage, 'model')):
                # Save model structure
                json_string = stage.model.to_json()
                open(path_stage + '/Stage_structure.json', 'w').write(json_string)
                # Save model weights
                stage.model.save_weights(path_stage + '/Stage_weights.h5', overwrite=True)
            # Save additional information
            pk.dump(stage, open(path_stage + '/Stage_instance.pkl', 'wb'))

    # Save additional information
    pk.dump(staged_network, open(path + '/Staged_Network.pkl', 'wb'))

    if (not staged_network.silence):
        logging.info("<<< Staged_Network model saved >>>")


def loadStagedModel(model_path):  # , parallel_loaders=10):
    """
        Loads a previously saved Staged_Network object.
    """
    logging.info("<<< Loading Staged_Network model from " + model_path + "/Staged_Network.pkl ... >>>")

    t = time.time()

    # Load additional information
    staged_network = pk.load(open(model_path + '/Staged_Network.pkl', 'rb'))

    # Get all stages
    stages = next(os.walk(model_path))[1]
    stages_list = [0 for i in range(len(stages))]

    # Load each stage
    for i, stage in enumerate(stages):
        stage_num = stage.split('_')
        stage_num = int(stage_num[1])
        path_stage = model_path + '/' + stage

        b = next(os.walk(path_stage))[1]
        branches = list()
        for b_ in b:
            if ('Branch_' in b_):
                branches.append(b_)

        nBranches = len(branches)
        list_stages = [0 for i in range(nBranches)]
        # Load each branch if exists more than one
        if (nBranches > 1):

            for s in branches:
                branch_num = s.split('_')
                branch_num = int(branch_num[1])
                path_branch = path_stage + '/' + s

                if os.path.exists(path_branch + '/Stage_structure.json'):
                    # Load model structure
                    model = model_from_json(open(path_branch + '/Stage_structure.json').read())
                    # Load model weights
                    model.load_weights(path_branch + '/Stage_weights.h5')

                stage = pk.load(open(path_branch + '/Stage_instance.pkl', 'rb'))

                if os.path.exists(path_branch + '/Stage_structure.json'):
                    stage.model = model

                list_stages[branch_num] = stage
                # list_stages.append(stage)

            stages_list[stage_num] = list_stages
            # staged_network.addStage(list_stages, reloading_model=True)
        # Load the Stage instance in this folder otherwise
        else:
            if os.path.exists(path_stage + '/Stage_structure.json'):
                # Load model structure
                model = model_from_json(open(path_stage + '/Stage_structure.json').read())
                # Load model weights
                model.load_weights(path_stage + '/Stage_weights.h5')

            stage = pk.load(open(path_stage + '/Stage_instance.pkl', 'rb'))

            if os.path.exists(path_stage + '/Stage_structure.json'):
                stage.model = model

            stages_list[stage_num] = stage
            # staged_network.addStage(stage, reloading_model=True)

    # Add all stages
    for s in stages_list:
        staged_network.addStage(s, reloading_model=True)

    logging.info("<<< Staged_Network model loaded in %0.6s seconds. >>>" % str(time.time() - t))
    return staged_network


class Staged_Network(object):
    """
        Builds and manages a set of CNN_Model instances that will be trained and executed in ordered stages.
    """

    def __init__(self, silence=False, model_name=None, plots_path=None, models_path=None):
        """
            Builds a Staged_Network empty instance.
            
            :param silence: set to True if you don't want the model to output informative messages
            :param model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
            :param plots_path: path to the folder where the plots will be stored during training
            :param models_path: path to the folder where the temporal model packups will be stored
        """
        self.silence = silence
        self.testing_parameters = list()

        # Set Network name
        self.setName(model_name, plots_path, models_path)

        # List of stages, which should be Stage instances or lists of Stage instances
        self.__stages = list()
        # List indicating the axis where we want to apply the outputs fusion for the parallel stages
        self.__joinOnAxis = list()
        # List of strings indicating the name of the outputs valid for evaluation (if the stages are Sequential, then None).
        self.__outNames = list()
        # List of strings indicating the name of the previous stage outputs that will be inputted to the current stage 
        # for forward propagation (if the previous stages are Sequential, then None).
        self.__inNames = list()
        # List of booleans indicating if the training of a certain stage/branch is enabled
        self.__trainingIsEnabled = list()
        # List of booleans indicating if we want to expand the dimensions (to 4) of the input data for a specific stage/branch
        self.__expandDimensions = list()
        # List of booleans indicating if we want to apply a balanced training on a certain stage/branch
        self.__balancedTraining = list()

    def addStage(self, stage, axis=0, out_name=None, in_name=None, reloading_model=False,
                 training_is_enabled=True, expand_dimensions=True, balanced=True):
        """
            Add either a Stage or CNN_Model object to the list of stages. 
            If the current stage is a parallel stage, it must be a list of Stages and the parameter axis must be specified.
            
            :param stage: single Stage instance or multi-stage (parallalel) list of Stage instances.
            :param axis: indicates the axis where we want to apply the join of the outputs of all the branches in this stage. Only has to be specified for the parallel stages.
            :param out_name: name of the output node used for evaluation. If stage is a list, then we must provide a list of output identifiers. Only has to be specified for the Graph stages
            :param in_name: name of the previous stage output that will be used for propagation to the next stage. If stage is a list, then we must provide a list of output identifiers. Only has to be specified for the Graph stages
            :param reloading_model: only should be set to 'True' when the Staged_Network model is being reloaded from memory
            :param training_is_enabled: indicates if the training is enabled for the current stage. If stage is a list, then input a list of booleans if only some branches' training will be enabled.
            :param expand_dimensions: indicates if we want to expand the dimensions (to 4) of the inputted data to this stage. If stage is a list, then input a list of booleans if only some branches' input dimensions will be expanded.
            :param balanced: indicates if we want to perform a balanced training (equal number of samples per class). If stage is a list, then input a list of booleans if only some branches will be trained in a balanced manner.
        """
        if (not reloading_model):
            if (not self.silence):
                logging.info("<<< Adding stage " + str(len(self.__stages)) + " to model >>>")

            # Change model name and plot/save location
            nStages = len(self.__stages)
            if (isinstance(stage, list)):
                nBranches = len(stage)

                # Create list of out_name
                if (not isinstance(out_name, list)):
                    out_ = out_name
                    out_name = list()
                    for b in range(nBranches):
                        out_name.append(out_)

                # Create list of in_name
                if (not isinstance(in_name, list)):
                    in_ = in_name
                    in_name = list()
                    for b in range(nBranches):
                        in_name.append(in_)

                # Create list of training enabled/disabled
                if (not isinstance(training_is_enabled, list)):
                    enabled = training_is_enabled
                    training_is_enabled = list()
                    for b in range(nBranches):
                        training_is_enabled.append(enabled)

                # Create list of dimensions expanding enabled/disabled
                if (not isinstance(expand_dimensions, list)):
                    expand = expand_dimensions
                    expand_dimensions = list()
                    for b in range(nBranches):
                        expand_dimensions.append(expand)

                # Create list of balanced training enabled/disabled
                if (not isinstance(balanced, list)):
                    balanced_enabled = balanced
                    balanced = list()
                    for b in range(nBranches):
                        balanced.append(balanced_enabled)

                # Set names to all branches
                for b in range(nBranches):
                    model_name = 'Stage_' + str(nStages) + '/Branch_' + str(b)
                    stage[b].setName(model_name, plots_path=self.plot_path + '/' + model_name,
                                     models_path=self.model_path + '/' + model_name)
            else:
                training_is_enabled = [training_is_enabled]
                expand_dimensions = [expand_dimensions]
                balanced = [balanced]
                model_name = 'Stage_' + str(nStages) + '/Branch_0'
                stage.setName(model_name, plots_path=self.plot_path + '/' + model_name,
                              models_path=self.model_path + '/' + model_name)

            self.__stages.append(stage)
            self.__joinOnAxis.append(axis)
            self.__outNames.append(out_name)
            self.__inNames.append(in_name)
            self.__trainingIsEnabled.append(training_is_enabled)
            self.__expandDimensions.append(expand_dimensions)
            self.__balancedTraining.append(balanced)
        else:
            self.__stages.append(stage)

    def addBranch(self, branch, stage_id, axis=0, out_name=None, in_name=None,
                  training_is_enabled=True, expand_dimensions=True, balanced=True):
        """
            Adds a new branch to a specific stage.
                
            :param stage_id: id position of the stage where the new branch will be added.
            :param axis: indicates the axis where we want to apply the join of the outputs.
            :param out_name: name of the output node used for evaluation. If stage is a list, then we must provide a list of output identifiers. Only has to be specified for the Graph stages
            :param in_name: name of the previous stage output that will be used for propagation to the next stage. If stage is a list, then we must provide a list of output identifiers. Only has to be specified for the Graph stages
            :param training_is_enabled: indicates if the training is enabled for the input branches. If stage is a list, then input a list of booleans if only some branches' training will be enabled.
            :param expand_dimensions: indicates if we want to expand the dimensions (to 4) of the inputted data to this stage. If stage is a list, then input a list of booleans if only some branches' input dimensions will be expanded.
            :param balanced: indicates if we want to perform a balanced training (equal number of samples per class). If stage is a list, then input a list of booleans if only some branches will be trained in a balanced manner.
        """
        # Recover the indicated stage
        stage = self.getStage(stage_id)
        if (not stage):
            raise Exception("The current number of existing stages is smaller than the defined 'stage_id'.")

        if (not self.silence):
            logging.info("<<< Adding branches to stage " + str(stage_id) + " >>>")

        # Process new branches
        if (isinstance(branch, list)):
            nBranches = len(branch)

            # Create list of out_name
            if (not isinstance(out_name, list)):
                out_ = out_name
                out_name = list()
                for b in range(nBranches):
                    out_name.append(out_)

            # Create list of in_name
            if (not isinstance(in_name, list)):
                in_ = in_name
                in_name = list()
                for b in range(nBranches):
                    in_name.append(in_)

            # Create list of training enabled/disabled
            if (not isinstance(training_is_enabled, list)):
                enabled = training_is_enabled
                training_is_enabled = list()
                for b in range(nBranches):
                    training_is_enabled.append(enabled)

                    # Create list of dimensions expanding enabled/disabled
            if (not isinstance(expand_dimensions, list)):
                expand = expand_dimensions
                expand_dimensions = list()
                for b in range(nBranches):
                    expand_dimensions.append(expand)

            # Create list of balanced training enabled/disabled
            if (not isinstance(balanced, list)):
                balanced_enabled = balanced
                balanced = list()
                for b in range(nBranches):
                    balanced.append(balanced_enabled)
        else:
            nBranches = 1
            branch = [branch]
            training_is_enabled = [training_is_enabled]
            out_name = [out_name]
            in_name = [in_name]
            expand_dimensions = [expand_dimensions]
            balanced = [balanced]

        # Process the previously introduced branches
        if (isinstance(stage, list)):  # The stage was already a list
            nPrevBranches = len(stage)
            prev_out_name = self.__outNames[stage_id]
            prev_in_name = self.__inNames[stage_id]
        else:  # The stage was not a list, yet
            nPrevBranches = 1
            stage = [stage]
            prev_out_name = [self.__outNames[stage_id]]
            prev_in_name = [self.__inNames[stage_id]]

        # Set names to all new branches
        for b in range(nBranches):
            model_name = 'Stage_' + str(stage_id) + '/Branch_' + str(b + nPrevBranches)
            branch[b].setName(model_name, plots_path=self.plot_path + '/' + model_name,
                              models_path=self.model_path + '/' + model_name)

        # Merge information
        self.__stages[stage_id] = stage + branch
        self.__joinOnAxis[stage_id] = axis
        self.__outNames[stage_id] = prev_out_name + out_name
        self.__inNames[stage_id] = prev_in_name + in_name
        self.__trainingIsEnabled[stage_id] += training_is_enabled
        self.__expandDimensions[stage_id] += expand_dimensions
        self.__balancedTraining[stage_id] += balanced

    def enableTraining(self, stage_id, training_is_enabled):
        """
            Replaces the trainingIsEnabled list from the selected stage.
        """
        stage = self.getStage(stage_id)
        if (not stage):
            raise Exception("The current number of existing stages is smaller than the defined 'stage_id'.")
        self.__trainingIsEnabled[stage_id] = training_is_enabled

    def popStage(self):
        """
            Removes the last stage on a Staged_Network
        """
        stage_data = []
        stage_data.append(self.__stages.pop())
        stage_data.append(self.__joinOnAxis.pop())
        stage_data.append(self.__outNames.pop())
        stage_data.append(self.__inNames.pop())
        stage_data.append(self.__trainingIsEnabled.pop())
        stage_data.append(self.__expandDimensions.pop())
        stage_data.append(self.__balancedTraining.pop())

        return stage_data

    def replaceStage(self, stage, position):
        """
            Replaces a Stage object on a certain position
        """
        nStages = self.getNumStages()
        if (nStages > position):
            old_stage = self.__stages[position]
            if (isinstance(stage, list)):
                nBranches = len(stage)
                for b in range(nBranches):
                    model_name = 'Stage_' + str(nStages - 1) + '/Branch_' + str(b)
                    stage[b].setName(model_name, plots_path=self.plot_path + '/' + model_name,
                                     models_path=self.model_path + '/' + model_name)
            else:
                model_name = 'Stage_' + str(nStages - 1) + '/Branch_0'
                stage.setName(model_name, plots_path=self.plot_path + '/' + model_name,
                              models_path=self.model_path + '/' + model_name)
            self.__stages[position] = stage
            return old_stage
        else:
            raise Exception("The current number of existing stages is smaller than the defined replace position.")

    def removeBranches(self, branch_ids, stage_id):
        """
            Removes a specific set of branches from a specific stage.
        """
        # Perform some initial checks
        stage = self.getStage(stage_id)
        if (not stage):
            raise Exception("The current number of existing stages is smaller than the defined 'stage_id'.")

        if (not isinstance(branch_ids, list)):
            branch_ids = []
        branch_ids.sort()
        branch_ids = branch_ids[::-1]

        if (not isinstance(stage, list)):
            raise Exception("The defined 'stage_id' must be a list of branches.")

        if (len(stage) - 1 < branch_ids[0]):
            raise Exception("All branch_ids must be smaller than the number of branches in the stage.")

        # Start removal
        for b in branch_ids:
            self.__stages[stage_id].pop(b)
            self.__outNames[stage_id].pop(b)
            self.__inNames[stage_id].pop(b)
            self.__trainingIsEnabled[stage_id].pop(b)
            self.__expandDimensions[stage_id].pop(b)
            self.__balancedTraining[stage_id].pop(b)

        # Reset names of the remaining branches
        for b in range(len(self.__stages)):
            model_name = 'Stage_' + str(stage_id) + '/Branch_' + str(b)
            self.__stages[stage_id][b].setName(model_name, plots_path=self.plot_path + '/' + model_name,
                                               models_path=self.model_path + '/' + model_name)

    def __str__(self):
        """
            Plot Staged_Network.
        """
        obj_str = '#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n'
        class_name = self.__class__.__name__
        obj_str += '\t\t' + class_name + ' instance\n'
        obj_str += '#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n'

        obj_str += 'Number of stages: ' + str(self.getNumStages()) + '\n\n'

        for i, s in enumerate(self.__stages):
            obj_str += '::: Stage ' + str(i) + '\n'
            if (isinstance(s, list)):
                n_branches = len(s)
                plot_out_dim = True;
            else:
                plot_out_dim = False;
                n_branches = 1
            obj_str += '\tNumber of branches: ' + str(n_branches) + '\n'
            obj_str += '\tjoinOnAxis: ' + str(self.__joinOnAxis[i]) + '\n'
            obj_str += '\toutNames: ' + str(self.__outNames[i]) + '\n'
            obj_str += '\tinNames: ' + str(self.__inNames[i]) + '\n'
            obj_str += '\ttrainingIsEnabled: ' + str(self.__trainingIsEnabled[i]) + '\n'
            obj_str += '\texpandDimensions: ' + str(self.__expandDimensions[i]) + '\n'
            obj_str += '\tbalancedTraining: ' + str(self.__balancedTraining[i]) + '\n'
            if (plot_out_dim):
                axis = self.getJoinOnAxis(i)

                if (isinstance(s[0].model, Sequential)):
                    shape = list(s[0].model.layers[-1].output_shape)
                elif (isinstance(s[0].model, Graph)):
                    shape = list(s[0].model.outputs[self.__outNames[i][0]].output_shape)

                for i_net in range(1, n_branches):
                    if (isinstance(s[i_net].model, Sequential)):
                        next_shape = s[i_net].model.layers[-1].output_shape
                    elif (isinstance(s[i_net].model, Graph)):
                        next_shape = s[i_net].model.outputs[self.__outNames[i][i_net]].output_shape
                    shape[axis + 1] = shape[axis + 1] + next_shape[axis + 1]

                obj_str += '\tOutput dimensions: ' + str(tuple(shape)) + '\n'
            obj_str += '\n'

        obj_str += '#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n'

        return obj_str

    # ------------------------------------------------------- #
    #       SETTERS/GETTERS
    # ------------------------------------------------------- #
    def setName(self, model_name, plots_path=None, models_path=None, clear_dirs=True):
        """
            Changes the name (identifier) of the Staged_Network instance.
        """
        if (not model_name):
            self.name = time.strftime("%Y-%m-%d") + '_' + time.strftime("%X")
            create_dirs = False
        else:
            self.name = model_name
            create_dirs = True

        if (not plots_path):
            self.plot_path = 'Plots/' + self.name
        else:
            self.plot_path = plots_path

        if (not models_path):
            self.model_path = 'Models/' + self.name
        else:
            self.model_path = models_path

        # Remove directories if existed
        if (clear_dirs):
            if (os.path.isdir(self.model_path)):
                shutil.rmtree(self.model_path)
            if (os.path.isdir(self.plot_path)):
                shutil.rmtree(self.plot_path)

        # Create new ones
        if (create_dirs):
            if (not os.path.isdir(self.model_path)):
                os.makedirs(self.model_path)
            if (not os.path.isdir(self.plot_path)):
                os.makedirs(self.plot_path)

    def changeAllNames(self, model_name, clear_dirs=True):
        """
            Changes all the model names and plot/save locations from all the stages and braches in the current Staged_Network. 
            Only the base 'model_name' must be given, the standard stage/branch names, plot and model locations will be used.
        """
        self.name = model_name
        self.plot_path = 'Plots/' + self.name
        self.model_path = 'Models/' + self.name

        for s_, s in enumerate(self.__stages):
            if (isinstance(s, list)):
                for b_, b in enumerate(s):
                    model_name = 'Stage_' + str(s_) + '/Branch_' + str(b_)
                    b.setName(model_name, plots_path=self.plot_path + '/' + model_name,
                              models_path=self.model_path + '/' + model_name, clear_dirs=clear_dirs)
            else:
                model_name = 'Stage_' + str(s_) + '/Branch_0'
                s.setName(model_name, plots_path=self.plot_path + '/' + model_name,
                          models_path=self.model_path + '/' + model_name, clear_dirs=clear_dirs)

    def getNumStages(self):
        """
            Gets the current number of defined Stages.
        """
        return len(self.__stages)

    def getStage(self, position):
        """
            Returns the Stage object on a certain position.
        """
        if (self.getNumStages() > position):
            return self.__stages[position]
        return False

    def getJoinOnAxis(self, position):
        """
            Returns the axis on a certain position.
        """
        if (self.getNumStages() > position):
            return self.__joinOnAxis[position]
        return False

    def checkParameters(self, input_params, default_params):
        """
            Validates a set of input parameters and uses the default ones if not specified.
        """
        valid_params = [key for key in default_params]
        params = dict()

        # Check input parameters' validity
        for key, val in input_params.iteritems():
            if key in valid_params:
                params[key] = val
            else:
                raise Exception("Parameter '" + key + "' is not a valid parameter.")

        # Use default parameters if not provided
        for key, default_val in default_params.iteritems():
            if key not in params:
                params[key] = default_val

        return params

    # ------------------------------------------------------- #
    #       TRAINING/TEST
    #           Methods for train and testing
    # ------------------------------------------------------- #

    def testNet(self, ds, stage_id=None, parameters=dict()):
        """
            Applies a complete round of tests using the test set in the provided Dataset instance. 
            If stage_id=None the test will be applied on the whole staged network.
            The available (optional) testing parameters are the following ones:
            
            :param batch_size: size of the batch (number of images) applied on each interation
                
            ####    Data processing parameters
            
            :param n_parallel_loaders: number of parallel data loaders allowed to work at the same time 
            :param normalize_images: boolean indicating if we want to 0-1 normalize the image pixel values
            :param mean_substraction: boolean indicating if we want to substract the training mean
        """
        # Check input parameters and recover default values if needed
        default_params = {'batch_size': 50, 'n_parallel_loaders': 8, 'normalize_images': False,
                          'mean_substraction': True};
        params = self.checkParameters(parameters, default_params)
        self.testing_parameters.append(copy.copy(params))

        logging.info("<<< Testing model >>>")

        numIterationsTest = int(math.ceil(float(ds.len_test) / params['batch_size']))

        if (stage_id == None):
            id_last_stage = self.getNumStages() - 1
        else:
            id_last_stage = stage_id

        # Pick the tested stage
        stage = self.getStage(id_last_stage)
        if (not stage):
            raise Exception("The current number of existing stages is smaller than the defined 'stage_id'.")

        if (not isinstance(stage, list)):
            stage = [stage]

        # Initialize results lists
        scores = []
        scores_top = []
        losses = []
        counts_samples = []
        for i in range(len(stage)):
            scores.append([])
            scores_top.append([])
            losses.append([])
            counts_samples.append([])

        # Initialize queue of data loaders
        t_test_queue = []
        for t_ind in range(numIterationsTest):
            t = ThreadDataLoader(retrieveXY, ds, 'test', params['batch_size'],
                                 params['normalize_images'], params['mean_substraction'], False)
            if (t_ind < params['n_parallel_loaders']):
                t.start()
            t_test_queue.append(t)

        # Start test
        for it_test in range(numIterationsTest):

            t_test = t_test_queue[it_test]
            t_test.join()
            if (t_test.resultOK):
                X_test = t_test.X
                Y_test = t_test.Y
            else:
                exc_type, exc_obj, exc_trace = t.exception
                # deal with the exception
                print exc_type, exc_obj
                print exc_trace
                raise Exception('Exception occurred in ThreadLoader.')
            t_test_queue[it_test] = None
            if (it_test + params['n_parallel_loaders'] < numIterationsTest):
                t_test = t_test_queue[it_test + params['n_parallel_loaders']]
                t_test.start()

            # Apply forward pass on all stages
            X_test = self.forwardUntilStage(X_test, id_last_stage)

            # Apply test on the last stage
            for i_net, net in enumerate(stage):
                if (len(stage) == 1):  # only one model
                    # Select input
                    if (self.__inNames[id_last_stage]):
                        X_in = copy.copy(X_test[self.__inNames[id_last_stage]])
                    else:
                        X_in = copy.copy(X_test)

                    # Expand dimensions
                    if (self.__expandDimensions[id_last_stage][0]):
                        while (len(X_in.shape) < 4):
                            X_in = np.expand_dims(X_in, axis=1)

                    result = net.testOnBatch(X_in, Y_test, accuracy=True, out_name=self.__outNames[id_last_stage])
                else:  # branched model
                    # Select input
                    if (self.__inNames[id_last_stage][i_net]):
                        X_in = copy.copy(X_test[self.__inNames[id_last_stage][i_net]])
                    else:
                        X_in = copy.copy(X_test)

                    # Expand dimensions
                    if (self.__expandDimensions[id_last_stage][i_net]):
                        while (len(X_in.shape) < 4):
                            X_in = np.expand_dims(X_in, axis=1)

                    result = net.testOnBatch(X_in, Y_test, accuracy=True,
                                             out_name=self.__outNames[id_last_stage][i_net])
                if (result):
                    (loss, score, score_top, count_samples) = result
                    losses[i_net].append(float(loss))
                    scores[i_net].append(float(score))
                    scores_top[i_net].append(float(score_top))
                    counts_samples[i_net].append(float(count_samples))

        ds.resetCounters(set_name='test')

        # Plot result for each branch in the stage
        for i_net in range(len(stage)):
            n_report = np.sum(counts_samples[i_net])
            loss = np.sum(np.array(losses[i_net]) * np.array(counts_samples[i_net])) / n_report
            score = np.sum(np.array(scores[i_net]) * np.array(counts_samples[i_net])) / n_report
            score_top = np.sum(np.array(scores_top[i_net]) * np.array(counts_samples[i_net])) / n_report

            logging.info("Stage " + str(id_last_stage) + " - Net " + str(i_net))
            logging.info("\tTest loss: " + str(loss))
            logging.info("\tTest accuracy: " + str(score))
            logging.info("\tTest accuracy top-5: " + str(score_top))

    def trainNet(self, ds, stage_id, parameters=dict()):
        """
            Trains stage defined by stage_id on the given dataset 'ds'. 
            The available (optional) training parameters are the following ones:
            
            ####    Visualization parameters
            
            :param report_iter: number of iterations between each loss report
            :param iter_for_val: number of interations between each validation test
            :param num_iterations_val: number of iterations applied on the validation dataset for computing the average performance (if None then all the validation data will be tested)
                
            ####    Learning parameters
            
            :param n_epochs: number of epochs that will be applied during training
            :param batch_size: size of the batch (number of images) applied on each interation by the SGD optimization
            :param lr_decay: number of iterations passed for decreasing the learning rate 
            :param lr_gamma: proportion of learning rate kept at each decrease.  It can also be a set of rules defined by a list, e.g. lr_gamma = [[3000, 0.9], ..., [None, 0.8]] means 0.9 until iteration 3000, ..., 0.8 until the end.

            ####    Data processing parameters
            
            :param n_parallel_loaders: number of parallel data loaders allowed to work at the same time 
            :param normalize_images: boolean indicating if we want to 0-1 normalize the image pixel values
            :param mean_substraction: boolean indicating if we want to substract the training mean
            :param data_augmentation: boolean indicating if we want to perform data augmentation (always False on validation)
                
            ####    Other parameters
            
            :param save_model: number of iterations between each model backup
        """
        # Recover the indicated stage
        stage = self.getStage(stage_id)
        if (not stage):
            raise Exception("The current number of existing stages is smaller than the defined 'stage_id'.")

        # Check input parameters and recover default values if needed
        default_params = {'n_epochs': 1, 'batch_size': 50, 'report_iter': 50, 'iter_for_val': 1000,
                          'lr_decay': 1000, 'lr_gamma': 0.1, 'save_model': 5000, 'num_iterations_val': None,
                          'n_parallel_loaders': 8, 'normalize_images': False, 'mean_substraction': True,
                          'data_augmentation': True};

        logging.info("<<< Training Stage " + str(stage_id) + " >>>")

        # Check if it is a single-network stage or a multi-network stage
        if (isinstance(stage, list)):
            params = stage[0].checkParameters(parameters, default_params)
            stage[0].training_parameters.append(copy.copy(params))
        else:
            params = stage.checkParameters(parameters, default_params)
            stage.training_parameters.append(copy.copy(params))
            stage = [stage]

        self._trainStage(ds, params, stage, stage_id, state=dict())

        logging.info("<<< Finished training Stage " + str(stage_id) + " >>>")

    def resumeTrainNet(self, ds, stage_id, parameters=dict()):
        """
            Resumes the last training state of a stored stage keeping also its training parameters. 
            If we introduce any parameter through the argument 'parameters', it will be replaced by the old one.
        """
        # Recover the indicated stage
        stage = self.getStage(stage_id)
        if (not stage):
            raise Exception("The current number of existing stages is smaller than the defined 'stage_id'.")

        logging.info("<<< Resuming training Stage " + str(stage_id) + " >>>")

        # Check if it is a single-network stage or a multi-network stage
        if (isinstance(stage, list)):
            # Recovers the old training parameters (replacing them by the new ones if any)
            default_params = stage[0].training_parameters[-1]
            params = stage[0].checkParameters(parameters, default_params)
            stage[0].training_parameters.append(copy.copy(params))

            # Recovers the last training state
            state = stage[0].training_state

        else:
            # Recovers the old training parameters (replacing them by the new ones if any)
            default_params = stage.training_parameters[-1]
            params = stage.checkParameters(parameters, default_params)
            stage.training_parameters.append(copy.copy(params))

            # Recovers the last training state
            state = stage.training_state
            stage = [stage]

        self._trainStage(ds, params, stage, stage_id, state)

        logging.info("<<< Finished training Stage " + str(stage_id) + " >>>")

    def _trainStage(self, ds, params, stage, stage_id, state=dict()):
        """
            Trains a stage composed of a series of parallel layers.
        """
        training_is_enabled = self.__trainingIsEnabled[stage_id]
        # Return if training is disabled on all branches
        if (not any(training_is_enabled)):
            if (not self.silence):
                logging.info("Training is disabled for all branches in stage " + str(stage_id))
            return

        logging.info("Training parameters: " + str(params))

        is_first_save = True
        scores_train = []
        losses_train = []
        top_scores_train = []
        counts_batch = []

        for i in range(len(stage)):
            scores_train.append([])
            losses_train.append([])
            top_scores_train.append([])
            counts_batch.append([])

        # Calculate how many interations are we going to perform
        if (not state.has_key('n_iterations_per_epoch')):
            state['n_iterations_per_epoch'] = int(math.ceil(float(ds.len_train) / params['batch_size']))
            state['count_iteration'] = 0
            state['epoch'] = 0
            state['it'] = -1
        else:
            state['count_iteration'] -= 1
            state['it'] -= 1

        # Calculate how many validation interations are we going to perform per test
        if (params['num_iterations_val'] == None):
            params['num_iterations_val'] = int(math.ceil(float(ds.len_val) / params['batch_size']))

        # Apply params['n_epochs'] for training
        for state['epoch'] in range(state['epoch'], params['n_epochs']):
            logging.info("<<< Starting epoch " + str(state['epoch'] + 1) + "/" + str(params['n_epochs']) + " >>>")

            # Shuffle the training samples before each epoch
            ds.shuffleTraining()

            # Initialize queue of parallel data loaders
            t_queue = []
            for t_ind in range(state['n_iterations_per_epoch']):
                t = ThreadDataLoader(retrieveXY, ds, 'train', params['batch_size'],
                                     params['normalize_images'], params['mean_substraction'],
                                     params['data_augmentation'])
                if (t_ind > state['it'] and t_ind < params['n_parallel_loaders'] + state['it'] + 1):
                    t.start()
                t_queue.append(t)

            for state['it'] in range(state['it'] + 1, state['n_iterations_per_epoch']):
                state['count_iteration'] += 1

                # Recovers a pre-loaded batch of data
                t = t_queue[state['it']]
                t.join()
                if (t.resultOK):
                    X_batch = t.X
                    Y_batch = t.Y
                else:
                    exc_type, exc_obj, exc_trace = t.exception
                    # deal with the exception
                    print exc_type, exc_obj
                    print exc_trace
                    raise Exception('Exception occurred in ThreadLoader.')
                t_queue[state['it']] = None
                if (state['it'] + params['n_parallel_loaders'] < state['n_iterations_per_epoch']):
                    t = t_queue[state['it'] + params['n_parallel_loaders']]
                    t.start()

                # Get output result from the previous stages
                X_batch = self.forwardUntilStage(X_batch, stage_id)

                # Forward and backward passes on the current batch
                for i_net, net in enumerate(stage):

                    # Check if training is enabled
                    if (training_is_enabled[i_net]):
                        if (len(stage) == 1):  # single stage model
                            # Select input
                            if (self.__inNames[stage_id]):
                                X_in = copy.copy(X_batch[self.__inNames[stage_id]])
                            else:
                                X_in = copy.copy(X_batch)

                            # Expand dimensions
                            if (self.__expandDimensions[stage_id][0]):
                                while (len(X_in.shape) < 4):
                                    X_in = np.expand_dims(X_in, axis=1)

                            result = net.trainOnBatch(X_in, Y_batch, batch_size=params['batch_size'],
                                                      out_name=self.__outNames[stage_id],
                                                      balanced=self.__balancedTraining[stage_id][0])
                        else:  # branched model
                            # Select input
                            if (self.__inNames[stage_id][i_net]):
                                X_in = copy.copy(X_batch[self.__inNames[stage_id][i_net]])
                            else:
                                X_in = copy.copy(X_batch)

                            # Expand dimensions
                            if (self.__expandDimensions[stage_id][i_net]):
                                while (len(X_in.shape) < 4):
                                    X_in = np.expand_dims(X_in, axis=1)

                            result = net.trainOnBatch(X_in, Y_batch, batch_size=params['batch_size'],
                                                      out_name=self.__outNames[stage_id][i_net],
                                                      balanced=self.__balancedTraining[stage_id][i_net])
                        if (result):
                            losses_train[i_net].append(float(result[0]))
                            scores_train[i_net].append(float(result[1]))
                            top_scores_train[i_net].append(float(result[2]))
                            counts_batch[i_net].append(float(result[3]))

                        # Report train info
                        if (state['count_iteration'] % params['report_iter'] == 0 and losses_train[
                            i_net]):  # only plot if we have some data
                            # loss = np.mean(losses_train[i_net])
                            # score = np.mean(scores_train[i_net])
                            # top_score = np.mean(top_scores_train[i_net])
                            n_report = np.sum(counts_batch[i_net])
                            loss = np.sum(np.array(losses_train[i_net]) * np.array(counts_batch[i_net])) / n_report
                            score = np.sum(np.array(scores_train[i_net]) * np.array(counts_batch[i_net])) / n_report
                            top_score = np.sum(
                                np.array(top_scores_train[i_net]) * np.array(counts_batch[i_net])) / n_report

                            logging.info("Stage " + str(stage_id) + " - Net " + str(i_net))
                            logging.info("Train - Iteration: " + str(state['count_iteration']) + "   (" + str(
                                state['count_iteration'] * params['batch_size']) + " samples seen)")
                            logging.info("\tTrain loss: " + str(loss))
                            logging.info("\tTrain accuracy: " + str(score))
                            logging.info("\tTrain accuracy top-5: " + str(top_score))

                            net.log('train', 'iteration', state['count_iteration'])
                            net.log('train', 'loss', loss)
                            net.log('train', 'accuracy', score)
                            net.log('train', 'accuracy top-5', top_score)

                            losses_train[i_net] = []
                            scores_train[i_net] = []
                            top_scores_train[i_net] = []
                            counts_batch[i_net] = []

                # Test network on validation set
                if (state['count_iteration'] > 0 and state['count_iteration'] % params['iter_for_val'] == 0):
                    logging.info("Applying validation...")
                    scores = []
                    losses = []
                    scores_top = []
                    counts_samples = []
                    for i in range(len(stage)):
                        scores.append([])
                        losses.append([])
                        scores_top.append([])
                        counts_samples.append([])

                    t_val_queue = []
                    for t_ind in range(params['num_iterations_val']):
                        t = ThreadDataLoader(retrieveXY, ds, 'val', params['batch_size'],
                                             params['normalize_images'], params['mean_substraction'], False)
                        if (t_ind < params['n_parallel_loaders']):
                            t.start()
                        t_val_queue.append(t)

                    for it_val in range(params['num_iterations_val']):

                        # Recovers a pre-loaded batch of data
                        t_val = t_val_queue[it_val]
                        t_val.join()
                        if (t_val.resultOK):
                            X_val = t_val.X
                            Y_val = t_val.Y
                        else:
                            exc_type, exc_obj, exc_trace = t.exception
                            # deal with the exception
                            print exc_type, exc_obj
                            print exc_trace
                            raise Exception('Exception occurred in ThreadLoader.')
                        t_val_queue[it_val] = None
                        if (it_val + params['n_parallel_loaders'] < params['num_iterations_val']):
                            t_val = t_val_queue[it_val + params['n_parallel_loaders']]
                            t_val.start()

                        # Get output result from the previous stages
                        X_val = self.forwardUntilStage(X_val, stage_id)

                        # Forward prediction pass
                        for i_net, net in enumerate(stage):
                            # Only validate if training is enabled
                            if (training_is_enabled[i_net]):
                                if (len(stage) == 1):  # only one model
                                    # Select input
                                    if (self.__inNames[stage_id]):
                                        X_in = copy.copy(X_val[self.__inNames[stage_id]])
                                    else:
                                        X_in = copy.copy(X_val)

                                    # Expand dimensions
                                    if (self.__expandDimensions[stage_id][0]):
                                        while (len(X_in.shape) < 4):
                                            X_in = np.expand_dims(X_in, axis=1)

                                    result = net.testOnBatch(X_in, Y_val, accuracy=True,
                                                             out_name=self.__outNames[stage_id])
                                else:  # branched model
                                    # Select input
                                    if (self.__inNames[stage_id][i_net]):
                                        X_in = copy.copy(X_val[self.__inNames[stage_id][i_net]])
                                    else:
                                        X_in = copy.copy(X_val)

                                    # Expand dimensions
                                    if (self.__expandDimensions[stage_id][i_net]):
                                        while (len(X_in.shape) < 4):
                                            X_in = np.expand_dims(X_in, axis=1)

                                    result = net.testOnBatch(X_in, Y_val, accuracy=True,
                                                             out_name=self.__outNames[stage_id][i_net])
                                if (result):
                                    losses[i_net].append(float(result[0]))
                                    scores[i_net].append(float(result[1]))
                                    scores_top[i_net].append(float(result[2]))
                                    counts_samples[i_net].append(float(result[3]))

                    ds.resetCounters(set_name='val')
                    for i_net, net in enumerate(stage):
                        # Only report and plot if training is enabled
                        if (training_is_enabled[i_net]):
                            n_report = np.sum(counts_samples[i_net])
                            loss = np.sum(np.array(losses[i_net]) * np.array(counts_samples[i_net])) / n_report
                            score = np.sum(np.array(scores[i_net]) * np.array(counts_samples[i_net])) / n_report
                            score_top = np.sum(np.array(scores_top[i_net]) * np.array(counts_samples[i_net])) / n_report

                            logging.info("Stage " + str(stage_id) + " - Net " + str(i_net))
                            logging.info("Val - Iteration: " + str(state['count_iteration']))
                            logging.info("\tValidation loss: " + str(loss))
                            logging.info("\tValidation accuracy: " + str(score))
                            logging.info("\tValidation accuracy top-5: " + str(score_top))

                            net.log('val', 'iteration', state['count_iteration'])
                            net.log('val', 'loss', loss)
                            net.log('val', 'accuracy', score)
                            net.log('val', 'accuracy top-5', score_top)

                            net.plot()

                # Save the model
                if (state['count_iteration'] % params['save_model'] == 0):
                    stage[0].training_state = state
                    for i_net, net in enumerate(stage):
                        # Only save stage if training is enabled
                        if (training_is_enabled[i_net] or is_first_save):
                            saveModel(net, state['count_iteration'])
                    saveStagedModel(self)
                    is_first_save = False

                # Decrease the current learning rate
                if (state['count_iteration'] % params['lr_decay'] == 0):
                    # Check if we have a set of rules
                    if (isinstance(params['lr_gamma'], list)):
                        # Check if the current lr_gamma rule is still valid
                        if (params['lr_gamma'][0][0] == None or params['lr_gamma'][0][0] > state['count_iteration']):
                            lr_gamma = params['lr_gamma'][0][1]
                        else:
                            # Find next valid lr_gamma
                            while (params['lr_gamma'][0][0] != None and params['lr_gamma'][0][0] <= state[
                                'count_iteration']):
                                params['lr_gamma'].pop(0)
                            lr_gamma = params['lr_gamma'][0][1]
                    # Else, we have a single lr_gamma for the whole training
                    else:
                        lr_gamma = params['lr_gamma']

                    for i_net, net in enumerate(stage):
                        # Only change lr if training is enabled
                        if (training_is_enabled[i_net]):
                            lr = net.lr * lr_gamma
                            momentum = 1 - lr
                            net.setOptimizer(lr, momentum)

            state['it'] = -1  # start again from the first iteration of the next epoch

    def forwardUntilStage(self, X, stage_id):
        """
            Applies a forward pass on all the stages until 'stage_id' (not included).
        """
        if (stage_id == 0):
            return X

        # Forward on each stage
        for s in range(stage_id):
            stage = self.getStage(s)

            ## FORWARD PASS

            # Brached stage
            if (isinstance(stage, list)):
                axis = self.getJoinOnAxis(s)
                nStages = len(stage)
                # Get prediction for each parallel stage separately
                out = list()
                for i, net in enumerate(stage):
                    if (isinstance(net.model, Graph)):
                        o = net.predictOnBatch(X, in_name=self.__inNames[s][i], out_name=self.__outNames[s][i],
                                               expand=self.__expandDimensions[s][i])
                    else:
                        o = net.predictOnBatch(X, in_name=self.__inNames[s][i], expand=self.__expandDimensions[s][i])
                    out.append(o)

                # Prepare joint result matrix
                shape = list(out[0].shape)
                for i_net in range(1, nStages):
                    shape[axis + 1] = shape[axis + 1] + out[i_net].shape[axis + 1]
                X = np.zeros(tuple(shape))

                # Insert final result
                offset = 0
                for i, o in enumerate(out):
                    str_idx = '[:'
                    for s in range(1, len(shape)):
                        if s == axis + 1:
                            str_idx += ',' + str(offset) + ':' + str(offset + o.shape[s])
                        else:
                            str_idx += ',:'
                    str_idx += ']'
                    exec ('X' + str_idx + ' = o')
                    offset += o.shape[axis + 1]

            # Single model stage
            else:
                if (hasattr(stage, 'model') and isinstance(stage.model, Graph)):
                    if (
                        s == self.getNumStages() - 1):  # only apply the out_name if we are at the final stage of the Staged_Network
                        out_name = self.__outNames[s]
                    else:
                        out_name = None
                    X = stage.predictOnBatch(X, in_name=self.__inNames[s], out_name=out_name,
                                             expand=self.__expandDimensions[s][0])
                else:
                    X = stage.predictOnBatch(X, in_name=self.__inNames[s], expand=self.__expandDimensions[s][0])

        return X

    def valWorsePairs(self, ds, n_pairs=5, parameters=dict(), avoid_pairs=[]):
        """
            Applies a complete round of tests using the validation ('val') set in the provided Dataset instance for finding the set
            of 'n_pairs' pairs of classes that have a higher intra-error. The set of pairs of classes provided as tuples (#classA, #classB)
            will not be selected.
            The available (optional) testing parameters are the following ones:
            
            :param batch_size: size of the batch (number of images) applied on each interation
                
            ####    Data processing parameters
            
            :param n_parallel_loaders: number of parallel data loaders allowed to work at the same time 
            :param normalize_images: boolean indicating if we want to 0-1 normalize the image pixel values
            :param mean_substraction: boolean indicating if we want to substract the training mean
        """
        # Check input parameters and recover default values if needed
        default_params = {'batch_size': 50, 'n_parallel_loaders': 8, 'normalize_images': False,
                          'mean_substraction': True};
        params = self.checkParameters(parameters, default_params)

        logging.info("<<< Validating model to find top " + str(n_pairs) + " worse scoring pairs of classes >>>")

        n_classes = len(ds.classes)
        confusion_matrix = np.zeros((n_classes, n_classes))
        numIterationsTest = int(math.ceil(float(ds.len_val) / params['batch_size']))

        # Initialize queue of data loaders
        t_test_queue = []
        for t_ind in range(numIterationsTest):
            t = ThreadDataLoader(retrieveXY, ds, 'val', params['batch_size'],
                                 params['normalize_images'], params['mean_substraction'], False)
            if (t_ind < params['n_parallel_loaders']):
                t.start()
            t_test_queue.append(t)

        # Start test
        for it_test in range(numIterationsTest):

            t_test = t_test_queue[it_test]
            t_test.join()
            if (t_test.resultOK):
                X_test = t_test.X
                Y_test = t_test.Y
            else:
                exc_type, exc_obj, exc_trace = t.exception
                # deal with the exception
                print exc_type, exc_obj
                print exc_trace
                raise Exception('Exception occurred in ThreadLoader.')
            t_test_queue[it_test] = None
            if (it_test + params['n_parallel_loaders'] < numIterationsTest):
                t_test = t_test_queue[it_test + params['n_parallel_loaders']]
                t_test.start()

            # Apply forward pass on all stages until the current one (included)
            predicted_classes = self.predictClassesOnBatch(X_test, topN=1)
            # Get GT classes
            gt_classes = np_utils.categorical_probas_to_classes(Y_test)

            # Store counters in confusion matrix
            for p, gt in zip(predicted_classes, gt_classes):
                confusion_matrix[gt, p] += 1

        # Get accuracy
        accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))

        # Save confusion matrix for its manual analysis
        mat_name = 'confusion_matrix_' + time.strftime("%Y-%m-%d") + '_' + time.strftime("%X")
        if (not self.silence):
            logging.info("\tSaving confusion matrix to " + self.model_path + '/' + mat_name + '.npy')
        np.save(self.model_path + '/' + mat_name + '.npy', np.array(confusion_matrix))

        ds.resetCounters(set_name='val')
        return [self._getWorsePairs(confusion_matrix, n_pairs, avoid_pairs), accuracy]

    def removeWorseClassifiers(self, ds, stage_id, min_accuracy=0.7, parameters=dict()):
        """
            Applies a complete round of tests using the validation ('val') set in the provided Dataset instance for finding the set
            of recently trained classifiers in 'stage_id' that have not been properly trained. If they do not accomplish a minimum 
            accuracy 'min_accuracy' they are removed.
            
            :param batch_size: size of the batch (number of images) applied on each interation
                
            ####    Data processing parameters
            
            :param n_parallel_loaders: number of parallel data loaders allowed to work at the same time 
            :param normalize_images: boolean indicating if we want to 0-1 normalize the image pixel values
            :param mean_substraction: boolean indicating if we want to substract the training mean
        """
        # Check input parameters and recover default values if needed
        default_params = {'batch_size': 50, 'n_parallel_loaders': 8, 'normalize_images': False,
                          'mean_substraction': True};

        # Recover the indicated stage
        stage = self.getStage(stage_id)
        if (not stage):
            raise Exception("The current number of existing stages is smaller than the defined 'stage_id'.")
        training_is_enabled = self.__trainingIsEnabled[stage_id]

        logging.info("<<< Validating model to remove non-learning classifiers >>>")

        # Check if it is a single-network stage or a multi-network stage
        if (isinstance(stage, list)):
            params = stage[0].checkParameters(parameters, default_params)
        else:
            params = stage.checkParameters(parameters, default_params)
            stage = [stage]

            # Prepare validation accuracy variables
        n_classes = len(ds.classes)
        numIterationsTest = int(math.ceil(float(ds.len_val) / params['batch_size']))
        to_remove = []  # indicates which classifiers will be removed

        scores = []
        counts_samples = []
        for i in range(len(stage)):
            scores.append([])
            counts_samples.append([])

        # Initialize queue of data loaders
        t_test_queue = []
        for t_ind in range(numIterationsTest):
            t = ThreadDataLoader(retrieveXY, ds, 'val', params['batch_size'],
                                 params['normalize_images'], params['mean_substraction'], False)
            if (t_ind < params['n_parallel_loaders']):
                t.start()
            t_test_queue.append(t)

        # Start test
        for it_test in range(numIterationsTest):

            t_test = t_test_queue[it_test]
            t_test.join()
            if (t_test.resultOK):
                X_val = t_test.X
                Y_val = t_test.Y
            else:
                exc_type, exc_obj, exc_trace = t.exception
                # deal with the exception
                print exc_type, exc_obj
                print exc_trace
                raise Exception('Exception occurred in ThreadLoader.')
            t_test_queue[it_test] = None
            if (it_test + params['n_parallel_loaders'] < numIterationsTest):
                t_test = t_test_queue[it_test + params['n_parallel_loaders']]
                t_test.start()

            # Get output result from the previous stages
            X_val = self.forwardUntilStage(X_val, stage_id)

            # Forward prediction pass
            for i_net, net in enumerate(stage):
                # Only validate if training is enabled
                if (training_is_enabled[i_net]):
                    if (len(stage) == 1):  # only one model
                        # Select input
                        if (self.__inNames[stage_id]):
                            X_in = copy.copy(X_val[self.__inNames[stage_id]])
                        else:
                            X_in = copy.copy(X_val)

                        # Expand dimensions
                        if (self.__expandDimensions[stage_id][0]):
                            while (len(X_in.shape) < 4):
                                X_in = np.expand_dims(X_in, axis=1)

                        result = net.testOnBatch(X_in, Y_val, accuracy=True, out_name=self.__outNames[stage_id])
                    else:  # branched model
                        # Select input
                        if (self.__inNames[stage_id][i_net]):
                            X_in = copy.copy(X_val[self.__inNames[stage_id][i_net]])
                        else:
                            X_in = copy.copy(X_val)

                        # Expand dimensions
                        if (self.__expandDimensions[stage_id][i_net]):
                            while (len(X_in.shape) < 4):
                                X_in = np.expand_dims(X_in, axis=1)

                        result = net.testOnBatch(X_in, Y_val, accuracy=True, out_name=self.__outNames[stage_id][i_net])
                    if (result):
                        scores[i_net].append(float(result[1]))
                        counts_samples[i_net].append(float(result[3]))

        ds.resetCounters(set_name='val')

        # Check which classifiers are below the minimum accuracy
        for i_net, net in enumerate(stage):
            # Only report and plot if training is enabled
            if (training_is_enabled[i_net]):
                n_report = np.sum(counts_samples[i_net])
                score = np.sum(np.array(scores[i_net]) * np.array(counts_samples[i_net])) / n_report

                if (score < min_accuracy):
                    to_remove.append(i_net)

        # Remove the classifiers which have not reached the minimum
        if (to_remove):
            self.removeBranches(to_remove, stage_id)

        return to_remove  # returns IDs of the removed branches

    def predictOnBatch(self, X):
        """
            Applies a forward pass along all the Staged_Network and returns the predicted values. 
        """
        return self.forwardUntilStage(X, self.getNumStages())

    def predictClassesOnBatch(self, X, topN=5):
        """
            Applies a forward pass along all the Staged_Network and returns the topN predicted classes sorted. 
        """
        predictions = self.predictOnBatch(X)
        return np.argsort(predictions, axis=1)[:, ::-1][:, :np.min([topN, predictions.shape[1]])]

    def _getWorsePairs(self, conf_mat, N, avoid_pairs):
        """
            Returns the N pairs of classes with a worse intra-error w.r.t. the confusion matrix 'conf_mat'. All the pairs of classes
            provided in 'avoid_pairs' will not be included in the result.
        """
        # Get all pairs
        n_classes = conf_mat.shape[0]
        labels_list = [l for l in range(n_classes)]
        pairs = tuple(itertools.combinations(labels_list, 2))

        # Normalize conf_mat
        norm_conf_mat = conf_mat / np.repeat(np.transpose(np.expand_dims(np.sum(conf_mat, axis=1), 0)), n_classes,
                                             axis=1).astype(np.float32)

        # Pick worse pairs
        n_pairs = len(pairs)
        errors = [0 for i in range(n_pairs)]
        for i, p in enumerate(pairs):
            errors[i] = norm_conf_mat[p[0], p[1]] + norm_conf_mat[p[1], p[0]]

        worse_pairs = np.argsort(errors)[::-1]
        worse_pairs = [pairs[p] for p in worse_pairs]

        worse_chosen = []
        i = 0
        while (len(worse_chosen) < N and i < n_pairs):
            if (worse_pairs[i] not in avoid_pairs):
                worse_chosen.append(worse_pairs[i])
            i += 1
        return worse_chosen

    # ------------------------------------------------------- #
    #       SAVE/LOAD
    #           Auxiliary methods for saving and loading the model.
    # ------------------------------------------------------- #

    def __getstate__(self):
        """
            Behavour applied when pickling a Staged_Network instance.
        """
        obj_dict = self.__dict__.copy()
        obj_dict['_Staged_Network__stages'] = list()
        return obj_dict
