# -*- coding: utf-8 -*-

import copy
import os
import numpy as np
from scipy import misc
from skimage.transform import resize
from scipy import ndimage


################################################################################
#
#    Utility functions for performing object localization.
#
################################################################################


def prepareCAM(snet):
    ''' Prepares the network for generating Class Activation Mappings '''

    # Adds the output for heatmap generation
    snet.getStage(1).model.add_output(name='GAP/conv', input='CAM_conv/relu')
    snet.getStage(1).setOptimizer()

    # Get weights (position 0 -> no food, positions 1 -> food)
    W = snet.getStage(1).model.get_weights()[-2]
    b = snet.getStage(1).model.get_weights()[-1]  # recover bias although it will not be used

    return W


def loadImagesDataset(ds, init, final, load_original=True):
    '''
        Loads a list of images and their pre-processed representations "X" ready for applying a forward pass.
        The images loaded are stored in the Dataset object "test" division.
    '''

    X = ds.getX('test', init, final, normalization=False, meanSubstraction=True, dataAugmentation=False)
    if (load_original):
        images = np.transpose(
            ds.getX('test', init, final, normalization=False, meanSubstraction=False, dataAugmentation=False),
            (0, 2, 3, 1))

        images_ = copy.copy(images)
        images[:, :, :, 0] = images_[:, :, :, 2]
        images[:, :, :, 2] = images_[:, :, :, 0]

        return [images, X]
    return X


def loadImagesExternal(ds, list_imgs, load_original=True):
    '''
        Loads a list of images and their pre-processed representations "X" ready for applying a forward pass.
        The images loaded are external to the Dataset object.
    '''

    X = ds.loadImages(list_imgs, False, True, False, external=True)
    if (load_original):
        images = np.transpose(ds.loadImages(list_imgs, False, False, False, external=True), (0, 2, 3, 1))

        images_ = copy.copy(images)
        images[:, :, :, 0] = images_[:, :, :, 2]
        images[:, :, :, 2] = images_[:, :, :, 0]

        return [images, X]
    return X


def applyForwardPass(snet, X):
    '''
        Applies a forward pass through the GAP network on the pre-processed "X" images.
    '''
    # Apply forward pass
    # X = snet.forwardUntilStage(X,1)['inception_4e']
    X = snet.forwardUntilStage(X, 1)[snet._Staged_Network__inNames[1]]
    predictions = np.argmax(snet.getStage(1).predictOnBatch(X, out_name='GAP/softmax'), axis=1)
    X = snet.getStage(1).predictOnBatch(X, out_name='GAP/conv')

    return [X, predictions]


# def computeCAM(snet, X, W, reshape_size=[256, 256]):
#    '''
#        Applies a forward pass of the pre-processed samples "X" in the GAP net "snet" and generates the resulting 
#        CAM "maps" using the GAP weights "W" with the defined size "reshape_size".
#    '''
#    
#    # Apply forward pass in GAP model
#    [X, predictions] = applyForwardPass(snet, X)
#    
#    # Compute heatmaps (CAMs) for each class [n_samples, n_classes, height, width]
#    maps = np.zeros((X.shape[0], W.shape[1], reshape_size[0], reshape_size[1]))
#    for s in range(X.shape[0]):
#        weighted_activation = np.dot(np.transpose(W), np.reshape(X[s], (W.shape[0], X.shape[2]*X.shape[3])))
#        map = np.reshape(weighted_activation, (W.shape[1], X.shape[2], X.shape[3]))
#        maps[s] = resize(map, tuple([W.shape[1]]+reshape_size), order=1, preserve_range=True)
#        
#    return [maps, predictions]


def computeCAM(snet, X, W, reshape_size=[256, 256], n_top_convs=20):
    '''
        Applies a forward pass of the pre-processed samples "X" in the GAP net "snet" and generates the resulting 
        CAM "maps" using the GAP weights "W" with the defined size "reshape_size".
        Additionally, it returns the best "n_top_convs" convolutional features for each of the classes. The ranking is 
        computed considering the weight Wi assigned to the i-th feature map.
    '''

    # Apply forward pass in GAP model
    [X, predictions] = applyForwardPass(snet, X)

    # Get indices of best convolutional features for each class
    ind_best = np.zeros((W.shape[1], n_top_convs))
    for c in range(W.shape[1]):
        ind_best[c, :] = np.argsort(W[:, c])[::-1][:n_top_convs]

    # Compute heatmaps (CAMs) for each class [n_samples, n_classes, height, width]
    maps = np.zeros((X.shape[0], W.shape[1], reshape_size[0], reshape_size[1]))
    # Store top convolutional features
    convs = np.zeros((X.shape[0], W.shape[1], n_top_convs, reshape_size[0], reshape_size[1]))

    for s in range(X.shape[0]):
        weighted_activation = np.dot(np.transpose(W), np.reshape(X[s], (W.shape[0], X.shape[2] * X.shape[3])))
        map = np.reshape(weighted_activation, (W.shape[1], X.shape[2], X.shape[3]))
        maps[s] = resize(map, tuple([W.shape[1]] + reshape_size), order=1, preserve_range=True)

        for c in range(W.shape[1]):
            for enum_conv, i_conv in enumerate(ind_best[c]):
                convs[s, c, enum_conv] = resize(X[s, i_conv], reshape_size, order=1, preserve_range=True)

    return [maps, predictions, convs]


# def getBestConvFeatures(snet, X, W, reshape_size=[256, 256], n_top_convs=20):
#    '''
#        Returns the best "n_top_convs" convolutional features for each of the classes. The ranking is 
#        computed considering the weight Wi assigned to the i-th feature map.
#    '''
#    # Apply forward pass in GAP model
#    [X, predictions] = applyForwardPass(snet, X)
#    
#    # Get indices of best convolutional features for each class
#    ind_best = np.zeros((W.shape[1], n_top_convs))
#    for c in range(W.shape[1]):
#        ind_best[c,:] = np.argsort(W[:,c])[::-1][:20]
#
#    # Store top convolutional features
#    convs = np.zeros((X.shape[0], W.shape[1], n_top_convs, reshape_size[0], reshape_size[1]))
#    for s in range(X.shape[0]):
#        for c in range(W.shape[1]):
#            for enum_conv, i_conv in enumerate(ind_best[c]):
#                convs[s,c,enum_conv] = resize(X[s,i_conv], reshape_size, order=1, preserve_range=True)
#        
#    return convs



def bbox(img, mode='width_height'):
    '''
        Returns a bounding box covering all the non-zero area in the image.
        "mode" : "width_height" returns width in [2] and height in [3], "max" returns xmax in [2] and ymax in [3]
    '''
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y, ymax = np.where(rows)[0][[0, -1]]
    x, xmax = np.where(cols)[0][[0, -1]]

    if (mode == 'width_height'):
        return x, y, xmax - x, ymax - y
    elif (mode == 'max'):
        return x, y, xmax, ymax


def computeIoU(GT, pred):
    '''
        Calculates the Intersectino over Union value of two bounding boxes.
    '''
    intersection = max(0, min(GT[2], pred[2]) - max(GT[0], pred[0])) * max(0, min(GT[3], pred[3]) - max(GT[1], pred[1]))
    gt_area = (GT[2] - GT[0]) * float((GT[3] - GT[1]))
    pred_area = (pred[2] - pred[0]) * float((pred[3] - pred[1]))
    union = gt_area + pred_area - intersection
    return intersection / union


def getBBoxesFromCAMs(CAMs, reshape_size=[256, 256], percentage_heat=0.4, size_restriction=0.1, box_expansion=0.2,
                      use_gpu=True):
    '''
    Reference:
        Bolaños, Marc, and Petia Radeva. "Simultaneous Food Localization and Recognition." arXiv preprint arXiv:1604.07953 (2016).

    Description:
        Extracts a set of bounding boxes from the generated CAMs which contain food instances.
        This functions should only be called if the current image has been predicted as Food by the GAP FoodvsNon-food detector!

    Arguments:
        :param CAMs: list of class activation maps generated by the CAM network
        :param reshape_size: reshape proportions used for transorming the CAM for extracting bounding boxes
        :param percentage_heat: minimum percentage allowed for considering a detection (aka 't' in reference paper)
        :param size_restriction: remove all regions covering less than a certain percentage size of the original image (aka 's' in reference paper)
        :param box_expansion: expand the bounding boxes by a certain percentage (aka 'e' in reference paper)
        :param use_gpu: boolean indicating if we want to use the GPU for applying NMS
        :return: [predicted_bboxes, predicted_scores], containing a list of bboxes coordinates on the first position
                and a list of their corresponding scores on the second position
    '''
    try:
        from nms.gpu_nms import gpu_nms
        from nms.cpu_nms import cpu_nms
    except:
        raise Exception(
            "Cython is required for running this function:\npip install cython\nRun the following command inside "
            "kernel_wrapper/extra/nms after its installation:\npython setup.py build_ext --inplace")

    predicted_bboxes = []
    predicted_scores = []

    # Get all computed maps (if we are also using convolutional features)
    all_maps = CAMs

    for map in all_maps:

        # map = misc.imread(maps_dir[dataset]+'/'+samples_detection[dataset]['all_ids'][s]+'_CAM.jpg') # CAM only
        # map = misc.imread(map_path)  # CAM and convolutional features
        new_reshape_size = reshape_size

        # Resize map to original size
        map = resize(map, tuple(new_reshape_size), order=1, preserve_range=True)

        # Detect regions above a certain percentage of the max heat
        bb_thres = np.max(map) * percentage_heat

        # Compute binary selected region
        binary_heat = map
        binary_heat = np.where(binary_heat > bb_thres, 255, 0)

        # Get biggest connected component
        min_size = new_reshape_size[0] * new_reshape_size[1] * size_restriction
        labeled, nr_objects = ndimage.label(binary_heat)  # get connected components
        [objects, counts] = np.unique(labeled, return_counts=True)  # count occurrences
        biggest_components = np.argsort(counts[1:])[::-1]
        selected_components = [1 if counts[i + 1] >= min_size else 0 for i in
                               biggest_components]  # check minimum size restriction
        biggest_components = biggest_components[:min([np.sum(selected_components), 9999])]  # get all bboxes

        # Extract each component (which will become a bbox prediction)
        map = map / 255.0  # normalize map

        # Get bboxes
        for selected, comp in zip(selected_components, biggest_components):
            if (selected):
                max_heat = np.where(labeled == comp + 1, 255, 0)  # get the biggest

                # Draw bounding box on original image
                box = list(bbox(max_heat))

                # expand box before final detection
                x_exp = box[2] * box_expansion
                y_exp = box[3] * box_expansion
                box[0] = max([0, box[0] - x_exp / 2])
                box[1] = max([0, box[1] - y_exp / 2])
                # change width and height by xmax and ymax
                box[2] += box[0]
                box[3] += box[1]
                box[2] = min([new_reshape_size[1] - 1, box[2] + x_exp])
                box[3] = min([new_reshape_size[0] - 1, box[3] + y_exp])

                predicted_bboxes.append(box)

                # Get score for current bbox
                score = np.mean(map[box[1]:box[3], box[0]:box[2]])  # use mean CAM value of the bbox as a score
                predicted_scores.append(score)

    # Now apply NMS on all the obtained bboxes
    nms_threshold = 0.3
    # logging.info('bboxes before NMS: '+str(len(predicted_scores)))
    if (len(predicted_scores) > 0):
        dets = np.hstack((np.array(predicted_bboxes), np.array(predicted_scores)[:, np.newaxis])).astype(np.float32)
        if (use_gpu):
            keep = gpu_nms(dets, nms_threshold, device_id=0)
        else:
            keep = cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        predicted_bboxes = []
        predicted_scores = []
        for idet in range(dets.shape[0]):
            predicted_bboxes.append(dets[idet, :4])
            predicted_scores.append(dets[idet, -1])
            # logging.info('bboxes after NMS: '+str(len(predicted_scores)))

    return [predicted_bboxes, predicted_scores]


def recognizeBBoxes(img_path, predicted_bboxes, recognition_net, ds, remove_non_food=None):
    '''
    Description:
        Apply food recognition on a set of bounding boxes provided.

    Arguments:
        :param img_path: path to the image.
        :param predicted_bboxes: bounding box coordinates from the original image (see getBBoxesFromCAMs(...))
        :param recognition_net: CNN_Model instance of the network used for food recognition
        :param ds: Dataset instance used for pre-processing images
        :param remove_non_food: if not None then all bounding boxes predicted as class 'remove_non_food' will be removed from the detections
        :return: [final_bboxes, predicted_scores, predicted_Y], containing a list of bboxes coordinates on the first position,
                a list of their corresponding scores on the second position and a list of class ids on the last position.
    '''
    predicted_Y = []
    predicted_scores = []
    final_bboxes = []

    # Apply prediction on bounding boxes
    if (len(predicted_bboxes) > 0):
        # Load crops
        im = misc.imread(img_path)
        images_list = []
        for b in predicted_bboxes:
            images_list.append(im[b[1]:b[3], b[0]:b[2]])

        # Forward pass
        X = ds.loadImages(images_list, normalization=False, meanSubstraction=True,
                          dataAugmentation=False, loaded=True)
        predictions_rec = recognition_net.predictOnBatch(X)['loss3/loss3']

        # Store prediction
        max_pred = np.argmax(predictions_rec, axis=1)
        for im in range(predictions_rec.shape[0]):
            # Remove bounding box prediction if we consider it is "NoFood"
            if (remove_non_food is None or max_pred[im] != remove_non_food):
                predicted_Y.append(max_pred[im])
                predicted_scores.append(predictions_rec[im][max_pred[im]])
                final_bboxes.append(predicted_bboxes[im])

    return [final_bboxes, predicted_scores, predicted_Y]
