import copy
import itertools
import logging
import time

import numpy as np

from keras.layers.convolutional import ZeroPadding2D


def bbox(img, mode='max'):
    """
    Returns a bounding box covering all the non-zero area in the image.

    :param img: Image on which print the bounding box
    :param mode:  "width_height" returns width in [2] and height in [3], "max" returns xmax in [2] and ymax in [3]
    :return:
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y, ymax = np.where(rows)[0][[0, -1]]
    x, xmax = np.where(cols)[0][[0, -1]]

    if mode == 'width_height':
        return x, y, xmax - x, ymax - y
    elif mode == 'max':
        return x, y, xmax, ymax


def build_OneVsOneECOC_Stage(n_classes_ecoc, input_shape, ds, stage1_lr=0.01, ecoc_version=2):
    """

    :param n_classes_ecoc:
    :param input_shape:
    :param ds:
    :param stage1_lr:
    :param ecoc_version:
    :return:
    """
    n_classes = len(ds.classes)
    labels_list = [str(l) for l in range(n_classes)]

    combs = tuple(itertools.combinations(labels_list, n_classes_ecoc))
    stage = list()
    outputs_list = list()

    count = 0
    n_combs = len(combs)
    for c in combs:
        t = time.time()

        # Create each one_vs_one classifier of the intermediate stage
        if ecoc_version == 1:
            s = Stage(nInput=n_classes, nOutput=n_classes_ecoc, input_shape=input_shape, output_shape=[1, 2],
                      type='One_vs_One_Inception', silence=True)
        elif ecoc_version == 2:
            s = Stage(nInput=n_classes, nOutput=n_classes_ecoc, input_shape=input_shape, output_shape=[1, 2],
                      type='One_vs_One_Inception_v2', silence=True)
        # Build input mapping
        input_mapping = dict()
        for i in range(n_classes):
            i_str = str(i)
            if i_str in c:
                input_mapping[i] = c.index(i_str)
            else:
                input_mapping[i] = None
        # Build output mask
        # output_mask = {'[0]': [0], '[1]': None}
        s.defineClassMapping(input_mapping)
        # s.defineOutputMask(output_mask)
        s.setOptimizer(lr=stage1_lr)
        s.silence = False
        stage.append(s)

        outputs_list.append('loss_OnevsOne/output')

        logging.info('Built model %s/%s for classes %s in %0.5s seconds.' % (
            str(count + 1), str(n_combs), c, str(time.time() - t)))
        count += 1

    return [stage, outputs_list]


def build_OneVsAllECOC_Stage(n_classes_ecoc, input_shape, ds, stage1_lr):
    """

    :param n_classes_ecoc:
    :param input_shape:
    :param ds:
    :param stage1_lr:
    :return:
    """
    n_classes = len(ds.classes)

    stage = list()
    outputs_list = list()

    count = 0
    for c in range(n_classes):
        t = time.time()

        # Create each one_vs_one classifier of the intermediate stage
        s = Stage(nInput=n_classes, nOutput=n_classes_ecoc, input_shape=input_shape, output_shape=[1],
                  type='One_vs_One_Inception', silence=True)
        # Build input mapping
        input_mapping = dict()
        for i in range(n_classes):
            if i == c:
                input_mapping[i] = 0
            else:
                input_mapping[i] = 1
        # Build output mask
        output_mask = {'[0]': [0], '[1]': None}
        s.defineClassMapping(input_mapping)
        s.defineOutputMask(output_mask)
        s.setOptimizer(lr=stage1_lr)
        s.silence = False
        stage.append(s)

        outputs_list.append('loss_OnevsOne/output')

        logging.info('Built model %s/%s for classes %s in %0.5s seconds.' % (
            str(count + 1), str(n_classes), '(' + str(c) + ' vs All)', str(time.time() - t)))
        count += 1

    return [stage, outputs_list]


def build_Specific_OneVsOneECOC_Stage(pairs, input_shape, ds, lr, ecoc_version=2):
    """

    :param pairs:
    :param input_shape:
    :param ds:
    :param lr:
    :param ecoc_version:
    :return:
    """
    n_classes = len(ds.classes)

    stage = list()
    outputs_list = list()

    count = 0
    n_pairs = len(pairs)
    logging.info("Building " + str(n_pairs) + " classifiers...")

    for c in pairs:
        t = time.time()

        # Create each one_vs_one classifier of the intermediate stage
        if ecoc_version == 1:
            s = Stage(nInput=n_classes, nOutput=2, input_shape=input_shape, output_shape=[2],
                      type='One_vs_One_Inception', silence=True)
        elif ecoc_version == 2:
            s = Stage(nInput=n_classes, nOutput=2, input_shape=input_shape, output_shape=[2],
                      type='One_vs_One_Inception_v2', silence=True)
        # Build input mapping
        input_mapping = dict()
        for i in range(n_classes):
            if i in c:
                input_mapping[i] = c.index(i)
            else:
                input_mapping[i] = None
        # Build output mask
        # output_mask = {'[0]': [0], '[1]': None}
        s.defineClassMapping(input_mapping)
        # s.defineOutputMask(output_mask)
        s.setOptimizer(lr=lr)
        s.silence = False
        stage.append(s)

        outputs_list.append('loss_OnevsOne/output')

        logging.info('Built model %s/%s for classes %s = %s in %0.5s seconds.' % (
            str(count + 1), str(n_pairs), c, (ds.classes[c[0]], ds.classes[c[1]]), str(time.time() - t)))
        count += 1

    return [stage, outputs_list]


def build_Specific_OneVsOneVsRestECOC_Stage(pairs, input_shape, ds, lr, ecoc_version=2):
    """

    :param pairs:
    :param input_shape:
    :param ds:
    :param lr:
    :param ecoc_version:
    :return:
    """
    n_classes = len(ds.classes)

    stage = list()
    outputs_list = list()

    count = 0
    n_pairs = len(pairs)
    for c in pairs:
        t = time.time()

        # Create each one_vs_one classifier of the intermediate stage
        if ecoc_version == 1:
            s = Stage(nInput=n_classes, nOutput=3, input_shape=input_shape, output_shape=[3],
                      type='One_vs_One_Inception', silence=True)
        elif ecoc_version == 2:
            s = Stage(nInput=n_classes, nOutput=3, input_shape=input_shape, output_shape=[3],
                      type='One_vs_One_Inception_v2', silence=True)
        # Build input mapping
        input_mapping = dict()
        for i in range(n_classes):
            if i in c:
                input_mapping[i] = c.index(i)
            else:
                input_mapping[i] = 2
        # Build output mask
        # output_mask = {'[0]': [0], '[1]': None}
        s.defineClassMapping(input_mapping)
        # s.defineOutputMask(output_mask)
        s.setOptimizer(lr=lr)
        s.silence = False
        stage.append(s)

        outputs_list.append('loss_OnevsOne/output')

        logging.info('Built model %s/%s for classes %s = %s in %0.5s seconds.' % (
            str(count + 1), str(n_pairs), c, (ds.classes[c[0]], ds.classes[c[1]]), str(time.time() - t)))
        count += 1

    return [stage, outputs_list]


def build_Specific_OneVsOneECOC_loss_Stage(net, input, input_shape, classes, ecoc_version=3, pairs=None,
                                           functional_api=False, activations=['softmax', 'softmax']):
    """

    :param net:
    :param input:
    :param input_shape:
    :param classes:
    :param ecoc_version:
    :param pairs:
    :param functional_api:
    :param activations:
    :return:
    """
    n_classes = len(classes)
    if pairs is None:  # generate any possible combination of two classes
        pairs = tuple(itertools.combinations(range(n_classes), 2))

    outputs_list = list()
    n_pairs = len(pairs)
    ecoc_table = np.zeros((n_classes, n_pairs, 2))

    logging.info("Building " + str(n_pairs) + " OneVsOne structures...")

    for i, c in enumerate(pairs):
        # t = time.time()

        # Insert 1s in the corresponding positions of the ecoc table
        ecoc_table[c[0], i, 0] = 1
        ecoc_table[c[1], i, 1] = 1

        # Create each one_vs_one classifier of the intermediate stage
        if functional_api == False:
            if ecoc_version == 1:
                output_name = net.add_One_vs_One_Inception(input, input_shape, i, nOutput=2, activation=activations[0])
            elif ecoc_version == 2:
                output_name = net.add_One_vs_One_Inception_v2(input, input_shape, i, nOutput=2,
                                                              activation=activations[0])
            else:
                raise NotImplementedError
        else:
            if ecoc_version == 1:
                output_name = net.add_One_vs_One_Inception_Functional(input, input_shape, i, nOutput=2,
                                                                      activation=activations[0])
            elif ecoc_version == 2:
                raise NotImplementedError()
            elif ecoc_version == 3 or ecoc_version == 4 or ecoc_version == 5 or ecoc_version == 6:
                if ecoc_version == 3:
                    nkernels = 16
                elif ecoc_version == 4:
                    nkernels = 64
                elif ecoc_version == 5:
                    nkernels = 128
                elif ecoc_version == 6:
                    nkernels = 256
                else:
                    raise NotImplementedError()
                if i == 0:
                    in_node = net.model.get_layer(input).output
                    padding_node = ZeroPadding2D(padding=(1, 1), name='3x3/ecoc_padding')(in_node)
                output_name = net.add_One_vs_One_3x3_Functional(padding_node, input_shape, i, nkernels, nOutput=2,
                                                                activation=activations[0])
            elif ecoc_version == 7:
                if i == 0:
                    in_node = net.model.get_layer(input).output
                    padding_node = ZeroPadding2D(padding=(1, 1), name='3x3/ecoc_padding')(in_node)
                output_name = net.add_One_vs_One_3x3_double_Functional(padding_node, input_shape, i, nOutput=2,
                                                                       activation=activations[0])
            else:
                raise NotImplementedError()
        outputs_list.append(output_name)

        # logging.info('Built model %s/%s for classes %s = %s in %0.5s seconds.'%(str(i+1),
        #  str(n_pairs), c, (classes[c[0]], classes[c[1]]), str(time.time()-t)))

    ecoc_table = np.reshape(ecoc_table, [n_classes, 2 * n_pairs])

    # Build final Softmax layer
    if not functional_api:
        output_names = net.add_One_vs_One_Merge(outputs_list, n_classes, activation=activations[1])
    else:
        output_names = net.add_One_vs_One_Merge_Functional(outputs_list, n_classes, activation=activations[1])
    logging.info('Built ECOC merge layers.')

    return [ecoc_table, output_names]


def prepareECOCLossOutputs(net, ds, ecoc_table, input_name, output_names, splits=['train', 'val', 'test']):
    """

    :param net:
    :param ds:
    :param ecoc_table:
    :param input_name:
    :param output_names:
    :param splits:
    :return:
    """
    # Insert ecoc_table in net
    if 'additional_data' not in net.__dict__.keys():
        net.additional_data = dict()
    net.additional_data['ecoc_table'] = ecoc_table

    # Retrieve labels' id and images' id in dataset
    id_labels = ds.ids_outputs[ds.types_outputs.index('categorical')]
    id_labels_ecoc = 'labels_ecoc'

    # Insert ecoc-loss labels for each data split
    for s in splits:
        labels_ecoc = []
        exec ('labels = ds.Y_' + s + '[id_labels]')
        n = len(labels)
        for i in range(n):
            labels_ecoc.append(ecoc_table[labels[i]])
        ds.setOutput(labels_ecoc, s, type='binary', id=id_labels_ecoc)

    # Set input and output mappings from dataset to network
    pos_images = ds.types_inputs.index('image')
    pos_labels = ds.types_outputs.index('categorical')
    pos_labels_ecoc = ds.types_outputs.index('binary')

    inputMapping = {input_name: pos_images}
    # inputMapping = {0: pos_images}
    net.setInputsMapping(inputMapping)

    outputMapping = {output_names[0]: pos_labels_ecoc, output_names[1]: pos_labels}
    # outputMapping = {0: pos_labels_ecoc, 1: pos_labels}
    net.setOutputsMapping(outputMapping, acc_output=output_names[1])


def loadGoogleNetForFood101(nClasses=101, load_path='/media/HDD_2TB/CNN_MODELS/GoogleNet'):
    """

    :param nClasses:
    :param load_path:
    :return:
    """
    logging.info('Loading GoogLeNet...')

    # Build model (loading the previously converted Caffe's model)
    googLeNet = Stage(nClasses, nClasses, [224, 224, 3], [nClasses], type='GoogleNet',
                      model_name='GoogleNet_Food101_retrained',
                      structure_path=load_path + '/Keras_model_structure.json',
                      weights_path=load_path + '/Keras_model_weights.h5')

    return googLeNet


def prepareGoogleNet_Food101(model_wrapper):
    """
    Prepares the GoogleNet model after its conversion from Caffe
    :param model_wrapper:
    :return:
    """
    # Remove unnecessary intermediate optimizers
    layers_to_delete = ['loss2/ave_pool', 'loss2/conv', 'loss2/relu_conv', 'loss2/fc_flatten', 'loss2/fc',
                        'loss2/relu_fc', 'loss2/drop_fc', 'loss2/classifier', 'output_loss2/loss',
                        'loss1/ave_pool', 'loss1/conv', 'loss1/relu_conv', 'loss1/fc_flatten', 'loss1/fc',
                        'loss1/relu_fc', 'loss1/drop_fc', 'loss1/classifier', 'output_loss1/loss']
    model_wrapper.removeLayers(layers_to_delete)
    model_wrapper.removeOutputs(['loss1/loss', 'loss2/loss'])


def prepareGoogleNet_Food101_ECOC_loss(model_wrapper):
    """
    Prepares the GoogleNet model for inserting an ECOC structure after removing the last part of the net
    :param model_wrapper:
    :return:
    """
    # Remove all last layers (from 'inception_5a' included)
    layers_to_delete = ['inception_5a/1x1', 'inception_5a/relu_1x1', 'inception_5a/3x3_reduce',
                        'inception_5a/relu_3x3_reduce',
                        'inception_5a/3x3_zeropadding', 'inception_5a/3x3', 'inception_5a/relu_3x3',
                        'inception_5a/5x5_reduce',
                        'inception_5a/relu_5x5_reduce', 'inception_5a/5x5_zeropadding', 'inception_5a/5x5',
                        'inception_5a/relu_5x5',
                        'inception_5a/pool_zeropadding', 'inception_5a/pool', 'inception_5a/pool_proj',
                        'inception_5a/relu_pool_proj', 'inception_5a/output', 'inception_5b/1x1',
                        'inception_5b/relu_1x1', 'inception_5b/3x3_reduce', 'inception_5b/relu_3x3_reduce',
                        'inception_5b/3x3_zeropadding', 'inception_5b/3x3', 'inception_5b/relu_3x3',
                        'inception_5b/5x5_reduce',
                        'inception_5b/relu_5x5_reduce', 'inception_5b/5x5_zeropadding', 'inception_5b/5x5',
                        'inception_5b/relu_5x5',
                        'inception_5b/pool_zeropadding', 'inception_5b/pool', 'inception_5b/pool_proj',
                        'inception_5b/relu_pool_proj',
                        'inception_5b/output', 'pool5/7x7_s1', 'pool5/drop_7x7_s1',
                        'loss3/classifier_foodrecognition_flatten',
                        'loss3/classifier_foodrecognition']
    [layers, params] = model_wrapper.removeLayers(copy.copy(layers_to_delete))
    # Remove softmax output
    model_wrapper.removeOutputs(['loss3/loss3'])

    return ['pool4/3x3_s2', [832, 7, 7]]  # returns the name of the last layer and its output shape
    # Adds a new output after the layer 'pool4/3x3_s2'
    # model_wrapper.model.add_output(name='pool4', input='pool4/3x3_s2')


def prepareGoogleNet_Food101_Stage1(model_wrapper):
    """
    Prepares the GoogleNet model for serving as the first Stage of a Staged_Netork
    :param model_wrapper:
    :return:
    """
    # Adds a new output after the layer 'pool4/3x3_s2'
    model_wrapper.model.add_output(name='pool4', input='pool4/3x3_s2')


def prepareGoogleNet_Stage2(stage1, stage2):
    """
    Removes the second part of the GoogleNet for inserting it into the second stage.
    :param stage1:
    :param stage2:
    :return:
    """
    # Remove all last layers (from 'inception_5a' included)
    layers_to_delete = ['inception_5a/1x1', 'inception_5a/relu_1x1', 'inception_5a/3x3_reduce',
                        'inception_5a/relu_3x3_reduce',
                        'inception_5a/3x3_zeropadding', 'inception_5a/3x3', 'inception_5a/relu_3x3',
                        'inception_5a/5x5_reduce',
                        'inception_5a/relu_5x5_reduce', 'inception_5a/5x5_zeropadding', 'inception_5a/5x5',
                        'inception_5a/relu_5x5',
                        'inception_5a/pool_zeropadding', 'inception_5a/pool', 'inception_5a/pool_proj',
                        'inception_5a/relu_pool_proj',
                        'inception_5a/output', 'inception_5b/1x1', 'inception_5b/relu_1x1', 'inception_5b/3x3_reduce',
                        'inception_5b/relu_3x3_reduce',
                        'inception_5b/3x3_zeropadding', 'inception_5b/3x3', 'inception_5b/relu_3x3',
                        'inception_5b/5x5_reduce',
                        'inception_5b/relu_5x5_reduce', 'inception_5b/5x5_zeropadding', 'inception_5b/5x5',
                        'inception_5b/relu_5x5',
                        'inception_5b/pool_zeropadding', 'inception_5b/pool', 'inception_5b/pool_proj',
                        'inception_5b/relu_pool_proj',
                        'inception_5b/output', 'pool5/7x7_s1', 'pool5/drop_7x7_s1',
                        'loss3/classifier_foodrecognition_flatten',
                        'loss3/classifier_foodrecognition', 'output_loss3/loss3']
    [layers, params] = stage1.removeLayers(copy.copy(layers_to_delete))
    # Remove softmax output
    stage1.removeOutputs(['loss3/loss3'])

    layers_to_delete_2 = ["conv1/7x7_s2_zeropadding", "conv1/7x7_s2", "conv1/relu_7x7", "pool1/3x3_s2_zeropadding",
                          "pool1/3x3_s2", "pool1/norm1", "conv2/3x3_reduce", "conv2/relu_3x3_reduce",
                          "conv2/3x3_zeropadding", "conv2/3x3", "conv2/relu_3x3", "conv2/norm2",
                          "pool2/3x3_s2_zeropadding", "pool2/3x3_s2", "inception_3a/1x1", "inception_3a/relu_1x1",
                          "inception_3a/3x3_reduce", "inception_3a/relu_3x3_reduce", "inception_3a/3x3_zeropadding",
                          "inception_3a/3x3", "inception_3a/relu_3x3", "inception_3a/5x5_reduce",
                          "inception_3a/relu_5x5_reduce", "inception_3a/5x5_zeropadding", "inception_3a/5x5",
                          "inception_3a/relu_5x5", "inception_3a/pool_zeropadding", "inception_3a/pool",
                          "inception_3a/pool_proj", "inception_3a/relu_pool_proj", "inception_3a/output",
                          "inception_3b/1x1", "inception_3b/relu_1x1", "inception_3b/3x3_reduce",
                          "inception_3b/relu_3x3_reduce", "inception_3b/3x3_zeropadding", "inception_3b/3x3",
                          "inception_3b/relu_3x3", "inception_3b/5x5_reduce", "inception_3b/relu_5x5_reduce",
                          "inception_3b/5x5_zeropadding", "inception_3b/5x5", "inception_3b/relu_5x5",
                          "inception_3b/pool_zeropadding", "inception_3b/pool", "inception_3b/pool_proj",
                          "inception_3b/relu_pool_proj", "inception_3b/output", "pool3/3x3_s2_zeropadding",
                          "pool3/3x3_s2", "inception_4a/1x1", "inception_4a/relu_1x1", "inception_4a/3x3_reduce",
                          "inception_4a/relu_3x3_reduce", "inception_4a/3x3_zeropadding", "inception_4a/3x3",
                          "inception_4a/relu_3x3", "inception_4a/5x5_reduce", "inception_4a/relu_5x5_reduce",
                          "inception_4a/5x5_zeropadding", "inception_4a/5x5", "inception_4a/relu_5x5",
                          "inception_4a/pool_zeropadding", "inception_4a/pool", "inception_4a/pool_proj",
                          "inception_4a/relu_pool_proj", "inception_4a/output", "inception_4b/1x1",
                          "inception_4b/relu_1x1", "inception_4b/3x3_reduce", "inception_4b/relu_3x3_reduce",
                          "inception_4b/3x3_zeropadding", "inception_4b/3x3", "inception_4b/relu_3x3",
                          "inception_4b/5x5_reduce", "inception_4b/relu_5x5_reduce", "inception_4b/5x5_zeropadding",
                          "inception_4b/5x5", "inception_4b/relu_5x5", "inception_4b/pool_zeropadding",
                          "inception_4b/pool", "inception_4b/pool_proj", "inception_4b/relu_pool_proj",
                          "inception_4b/output", "inception_4c/1x1", "inception_4c/relu_1x1", "inception_4c/3x3_reduce",
                          "inception_4c/relu_3x3_reduce", "inception_4c/3x3_zeropadding", "inception_4c/3x3",
                          "inception_4c/relu_3x3", "inception_4c/5x5_reduce", "inception_4c/relu_5x5_reduce",
                          "inception_4c/5x5_zeropadding", "inception_4c/5x5", "inception_4c/relu_5x5",
                          "inception_4c/pool_zeropadding", "inception_4c/pool", "inception_4c/pool_proj",
                          "inception_4c/relu_pool_proj", "inception_4c/output", "inception_4d/1x1",
                          "inception_4d/relu_1x1", "inception_4d/3x3_reduce", "inception_4d/relu_3x3_reduce",
                          "inception_4d/3x3_zeropadding", "inception_4d/3x3", "inception_4d/relu_3x3",
                          "inception_4d/5x5_reduce", "inception_4d/relu_5x5_reduce", "inception_4d/5x5_zeropadding",
                          "inception_4d/5x5", "inception_4d/relu_5x5", "inception_4d/pool_zeropadding",
                          "inception_4d/pool", "inception_4d/pool_proj", "inception_4d/relu_pool_proj",
                          "inception_4d/output", "inception_4e/1x1", "inception_4e/relu_1x1", "inception_4e/3x3_reduce",
                          "inception_4e/relu_3x3_reduce", "inception_4e/3x3_zeropadding", "inception_4e/3x3",
                          "inception_4e/relu_3x3", "inception_4e/5x5_reduce", "inception_4e/relu_5x5_reduce",
                          "inception_4e/5x5_zeropadding", "inception_4e/5x5", "inception_4e/relu_5x5",
                          "inception_4e/pool_zeropadding", "inception_4e/pool", "inception_4e/pool_proj",
                          "inception_4e/relu_pool_proj", "inception_4e/output", "pool4/3x3_s2_zeropadding",
                          "pool4/3x3_s2"]

    # Remove initial layers
    [layers_, params_] = stage2.removeLayers(copy.copy(layers_to_delete_2))
    # Remove previous input
    stage2.removeInputs(['input_data'])
    # Add new input
    stage2.model.add_input(name='input_data', input_shape=(832, 7, 7))
    stage2.model.nodes[layers_to_delete[0]].previous = stage2.model.inputs['input_data']

    ## Insert layers into stage
    # stage2.model = Graph()
    ## Input
    # stage2.model.add_input(name='input_data', input_shape=(832,7,7))
    # for l_name,l,p in zip(layers_to_delete, layers, params):
    #    stage2.model.namespace.add(l_name)
    #    stage2.model.nodes[l_name] = l
    #    stage2.model.node_config.append(p)
    ##input = stage2.model.input # keep input
    ## Connect first layer with input
    # stage2.model.node_config[0]['input'] = 'input_data'
    # stage2.model.nodes[layers_to_delete[0]].previous = stage2.model.inputs['input_data']
    # stage2.model.input_config[0]['input_shape'] = [832,7,7]
    #    
    ## Output
    # stage2.model.add_output(name='loss3/loss3', input=layers_to_delete[-1])
    ##stage2.model.add_output(name='loss3/loss3_', input=layers_to_delete[-1])
    ##stage2.model.input = input # recover input

def simplifyDataset(ds, id_classes, n_classes=50):
    """

    :param ds:
    :param id_classes:
    :param n_classes:
    :return:
    """
    logging.info("Simplifying %s from %d to %d classes." % (str(ds.name), len(ds.classes), n_classes))
    ds.classes[id_classes] = ds.classes[id_classes][:n_classes]

    id_labels = ds.ids_outputs[ds.types_outputs.index('categorical')]

    # reduce each data split
    for s in ['train', 'val', 'test']:
        kept_Y = dict()
        kept_X = dict()
        exec ('labels_set = ds.Y_' + s + '[id_labels]')
        for i, y in enumerate(labels_set):
            if y < n_classes:
                for id_out in ds.ids_outputs:
                    exec ('sample = ds.Y_' + s + '[id_out][i]')
                    try:
                        kept_Y[id_out].append(sample)
                    except:
                        kept_Y[id_out] = []
                        kept_Y[id_out].append(sample)
                for id_in in ds.ids_inputs:
                    exec ('sample = ds.X_' + s + '[id_in][i]')
                    try:
                        kept_X[id_in].append(sample)
                    except:
                        kept_X[id_in] = []
                        kept_X[id_in].append(sample)
        exec ('ds.X_' + s + ' = copy.copy(kept_X)')
        exec ('ds.Y_' + s + ' = copy.copy(kept_Y)')
        exec ('ds.len_' + s + ' = len(kept_Y[id_labels])')


# Text-related utils
def one_hot_2_indices(preds, pad_sequences=True, verbose=0):
    """
    Converts a one-hot codification into a index-based one
    :param preds: Predictions codified as one-hot vectors.
    :param pad_sequences: Whether we should pad sequence or not
    :param verbose: Verbosity level, by default 0.
    :return: List of convertedpredictions
    """
    if verbose > 0:
        logging.info('Converting one hot prediction into indices...')
    preds = map(lambda x: np.nonzero(x)[1], preds)
    if pad_sequences:
        preds = [pred[:sum([int(elem > 0) for elem in pred]) + 1] for pred in preds]
    return preds


def indices_2_one_hot(indices, n):
    """
    Converts a list of indices into one hot codification

    :param indices: list of indices
    :param n: integer. Size of the vocabulary
    :return: numpy array with shape (len(indices), n)
    """
    one_hot = np.zeros((len(indices), n), dtype=np.int)
    for i in range(len(indices)):
        if indices[i] >= n:
            raise ValueError("Index out of bounds when converting to one hot")
        one_hot[i, indices[i]] = 1

    return one_hot


# ------------------------------------------------------- #
#       DECODING FUNCTIONS
#           Functions for decoding predictions
# ------------------------------------------------------- #

def decode_predictions_one_hot(preds, index2word, verbose=0):
    """
    Decodes predictions following a one-hot codification.
    :param preds: Predictions codified as one-hot vectors.
    :param index2word: Mapping from word indices into word characters.
    :param verbose: Verbosity level, by default 0.
    :return: List of decoded predictions
    """
    if verbose > 0:
        logging.info('Decoding one hot prediction ...')
    preds = map(lambda prediction: np.nonzero(prediction)[1], preds)
    PAD = '<pad>'
    flattened_answer_pred = [map(lambda index: index2word[index], pred) for pred in preds]
    answer_pred_matrix = np.asarray(flattened_answer_pred)
    answer_pred = []

    for a_no in answer_pred_matrix:
        end_token_pos = [j for j, x in enumerate(a_no) if x == PAD]
        end_token_pos = None if len(end_token_pos) == 0 else end_token_pos[0]
        tmp = ' '.join(a_no[:end_token_pos])
        answer_pred.append(tmp)
    return answer_pred


def decode_predictions(preds, temperature, index2word, sampling_type, verbose=0):
    """
    Decodes predictions
    :param preds: Predictions codified as the output of a softmax activation function.
    :param temperature: Temperature for sampling.
    :param index2word: Mapping from word indices into word characters.
    :param sampling_type: 'max_likelihood' or 'multinomial'.
    :param verbose: Verbosity level, by default 0.
    :return: List of decoded predictions.
    """

    if verbose > 0:
        logging.info('Decoding prediction ...')

    answer_pred_matrix = []

    for pred in preds:
        flattened_preds = pred.reshape(-1, pred.shape[-1])
        flattened_answer_pred = map(lambda index: index2word[index], sampling(scores=flattened_preds,
                                                                              sampling_type=sampling_type,
                                                                              temperature=temperature))
        answer_pred_matrix.append(np.asarray(flattened_answer_pred).reshape(pred.shape[:-1]))

    # flattened_preds = preds.reshape(-1, preds.shape[-1])
    # flattened_answer_pred = map(lambda index: index2word[index], sampling(scores=flattened_preds,
    #                                                                       sampling_type=sampling_type,
    #                                                                       temperature=temperature))
    # answer_pred_matrix = np.asarray(flattened_answer_pred).reshape(preds.shape[:-1])

    answer_pred = []
    EOS = '<eos>'
    PAD = '<pad>'

    # for a_no in answer_pred_matrix:
    #     if len(a_no.shape) > 1:  # only process word by word if our prediction has more than one output
    #         init_token_pos = 0
    #         end_token_pos = [j for j, x in enumerate(a_no) if x == EOS or x == PAD]
    #         end_token_pos = None if len(end_token_pos) == 0 else end_token_pos[0]
    #         tmp = ' '.join(a_no[init_token_pos:end_token_pos])
    #     else:
    #         tmp = a_no
    #     answer_pred.append(tmp)

    for a_no in answer_pred_matrix:
        if a_no.ndim == 1:
            if len(a_no.shape) > 1:  # only process word by word if our prediction has more than one output
                init_token_pos = 0
                end_token_pos = [j for j, x in enumerate(a_no) if x == EOS or x == PAD]
                end_token_pos = None if len(end_token_pos) == 0 else end_token_pos[0]
                tmp = ' '.join(a_no[init_token_pos:end_token_pos])
            else:
                tmp = a_no
            answer_pred.append(tmp)
        else:
            for i in range(a_no.shape[0]):
                # if len(a_no.shape) > 1:  # only process word by word if our prediction has more than one output
                init_token_pos = 0
                end_token_pos = [j for j, x in enumerate(a_no[i]) if x == EOS or x == PAD]
                end_token_pos = None if len(end_token_pos) == 0 else end_token_pos[0]
                tmp = ' '.join(a_no[i][init_token_pos:end_token_pos])
                # else:
                #    tmp = a_no
                answer_pred.append(tmp)


    return answer_pred


def decode_multilabel(preds, index2word, min_val=0.5, get_probs=False, verbose=0):
    """
    Decodes predictions
    :param preds: Predictions codified as the output of a softmax activation function.
    :param index2word: Mapping from word indices into word characters.
    :param min_val: Minimum value needed for considering a positive prediction.
    :param get_probs: additionally return probability for each predicted label
    :param verbose: Verbosity level, by default 0.
    :return: List of decoded predictions.
    """

    if verbose > 0:
        logging.info('Decoding prediction ...')

    answer_pred = []
    probs_pred = []
    for pred in preds:
        current_pred = []
        current_probs = []
        for ind, word in enumerate(pred):
            if word >= min_val:
                current_pred.append(index2word[ind])
                current_probs.append(word)
        answer_pred.append(current_pred)
        probs_pred.append(current_probs)

    if get_probs:
        return answer_pred, probs_pred
    else:
        return answer_pred


def replace_unknown_words(src_word_seq, trg_word_seq, hard_alignment, unk_symbol,
                          heuristic=0, mapping=None, verbose=0):
    """
    Replaces unknown words from the target sentence according to some heuristic.
    Borrowed from: https://github.com/sebastien-j/LV_groundhog/blob/master/experiments/nmt/replace_UNK.py
    :param src_word_seq: Source sentence words
    :param trg_word_seq: Hypothesis words
    :param hard_alignment: Target-Source alignments
    :param unk_symbol: Symbol in trg_word_seq to replace
    :param heuristic: Heuristic (0, 1, 2)
    :param mapping: External alignment dictionary
    :param verbose: Verbosity level
    :return: trg_word_seq with replaced unknown words
    """
    trans_words = trg_word_seq
    new_trans_words = []
    if verbose > 2:
        print "Input sentence:", src_word_seq
        print "Hard alignments", hard_alignment
    for j in xrange(len(trans_words)):
        if trans_words[j] == unk_symbol:
            UNK_src = src_word_seq[hard_alignment[j]]
            if heuristic == 0:  # Copy (ok when training with large vocabularies on en->fr, en->de)
                new_trans_words.append(UNK_src)
            elif heuristic == 1:
                # Use the most likely translation (with t-table). If not found, copy the source word.
                # Ok for small vocabulary (~30k) models
                if mapping.get(UNK_src) is not None:
                    new_trans_words.append(mapping[UNK_src])
                else:
                    new_trans_words.append(UNK_src)
            elif heuristic == 2:
                # Use t-table if the source word starts with a lowercase letter. Otherwise copy
                # Sometimes works better than other heuristics
                if mapping.get(UNK_src) is not None and UNK_src.decode('utf-8')[0].islower():
                    new_trans_words.append(mapping[UNK_src])
                else:
                    new_trans_words.append(UNK_src)
        else:
            new_trans_words.append(trans_words[j])

    return new_trans_words


def decode_predictions_beam_search(preds, index2word, alphas=None, heuristic=0,
                                   x_text=None, unk_symbol='<unk>', pad_sequences=False,
                                   mapping=None, verbose=0):
    """
    Decodes predictions from the BeamSearch method.

    :param preds: Predictions codified as word indices.
    :param index2word: Mapping from word indices into word characters.
    :param alphas: Attention model weights
    :param heuristic: Replace unknown words heuristic (0, 1 or 2)
    :param x_text: Source text (for unk replacement)
    :param unk_symbol: Unknown words symbol
    :param pad_sequences: Whether we should make a zero-pad on the input sequence.
    :param mapping: Source-target dictionary (for unk_replace heuristics 1 and 2)
    :param verbose: Verbosity level, by default 0.
    :return: List of decoded predictions
    """
    if verbose > 0:
        logging.info('Decoding beam search prediction ...')

    if alphas is not None:
        assert x_text is not None, 'When using POS_UNK, you must provide the input ' \
                                   'text to decode_predictions_beam_search!'
        if verbose > 0:
            logging.info('Using heuristic %d' % heuristic)
    if pad_sequences:
        preds = [pred[:sum([int(elem > 0) for elem in pred]) + 1] for pred in preds]
    flattened_answer_pred = [map(lambda x: index2word[x], pred) for pred in preds]
    answer_pred = []

    if alphas is not None:
        x_text = map(lambda x: x.split(), x_text)
        hard_alignments = map(
            lambda alignment, x_sentence: np.argmax(alignment[:, :max(1, len(x_sentence))], axis=1),
            alphas, x_text)
        for i, a_no in enumerate(flattened_answer_pred):
            if unk_symbol in a_no:
                if verbose > 1:
                    print unk_symbol, "at sentence number", i
                    print "hypothesis:", a_no
                    if verbose > 2:
                        print "alphas:", alphas[i]

                a_no = replace_unknown_words(x_text[i],
                                             a_no,
                                             hard_alignments[i],
                                             unk_symbol,
                                             heuristic=heuristic,
                                             mapping=mapping,
                                             verbose=verbose)
                if verbose > 1:
                    print "After unk_replace:", a_no
            tmp = ' '.join(a_no[:-1])
            answer_pred.append(tmp)
    else:
        for a_no in flattened_answer_pred:
            tmp = ' '.join(a_no[:-1])
            answer_pred.append(tmp)
    return answer_pred


def sample(a, temperature=1.0):
    """
    Helper function to sample an index from a probability array
    :param a: Probability array
    :param temperature: The higher, the flatter probabilities. Hence more random outputs.
    :return:
    """
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def sampling(scores, sampling_type='max_likelihood', temperature=1.0):
    """
    Sampling words (each sample is drawn from a categorical distribution).
    Or picks up words that maximize the likelihood.
    :param scores: array of size #samples x #classes;
    every entry determines a score for sample i having class j
    :param sampling_type:
    :param temperature: Predictions temperature. The higher, the flatter probabilities. Hence more random outputs.
    :return: set of indices chosen as output, a vector of size #samples
    """
    if isinstance(scores, dict):
        scores = scores['output']

    if sampling_type == 'multinomial':
        preds = np.asarray(scores).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    elif sampling_type == 'max_likelihood':
        return np.argmax(scores, axis=-1)
    else:
        raise NotImplementedError()


# Data structures-related utils
def flatten_list_of_lists(list_of_lists):
    """
    Flattens a list of lists
    :param list_of_lists: List of lists
    :return: Flatten list of lists
    """
    return [item for sublist in list_of_lists for item in sublist]


def flatten(l):
    """
    Flatten a list (more general than flatten_list_of_lists, but also more inefficient
    :param l:
    :return:
    """
    if not l:
        return l
    return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
