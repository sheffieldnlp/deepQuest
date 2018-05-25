# -*- coding: utf-8 -*-

import copy
import logging
import math
import sys
import time

import numpy as np

from keras_wrapper.dataset import Data_Batch_Generator
from keras_wrapper.utils import one_hot_2_indices


class BeamSearchEnsemble:
    def __init__(self, models, dataset, params_prediction, n_best=False, verbose=0):
        """

        :param models:
        :param dataset:
        :param params_prediction:
        """
        self.models = models
        self.dataset = dataset
        self.params = params_prediction
        self.optimized_search = params_prediction.get('optimized_search', False)
        self.return_alphas = params_prediction.get('coverage_penalty', False) or params_prediction.get('pos_unk', False)
        self.n_best = n_best
        self.verbose = verbose
        if self.verbose > 0:
            logging.info('<<< "Optimized search: %s >>>' % str(self.optimized_search))

    # PREDICTION FUNCTIONS: Functions for making prediction on input samples
    def predict_cond(self, models, X, states_below, params, ii, prev_outs=None):
        """
        Call the prediction functions of all models, according to their inputs
        :param models: List of models in the ensemble
        :param X: Input data
        :param states_below: Previously generated words (in case of conditional models)
        :param params: Model parameters
        :param ii: Decoding time-step
        :param prev_outs: Only for optimized models. Outputs from the previous time-step.
        :return: Combined outputs from the ensemble
        """

        probs_list = []
        prev_outs_list = []
        alphas_list = []
        for i, model in enumerate(models):
            if self.optimized_search:
                [model_probs, next_outs] = model.predict_cond_optimized(X, states_below, params,
                                                                        ii, prev_out=prev_outs[i])
                probs_list.append(model_probs)
                if self.return_alphas:
                    alphas_list.append(next_outs[-1][0])  # Shape: (k, n_steps)
                    next_outs = next_outs[:-1]
                prev_outs_list.append(next_outs)
            else:
                probs_list.append(model.predict_cond(X, states_below, params, ii))
        probs = sum(probs_list[i] for i in xrange(len(models))) / float(len(models))

        if self.return_alphas:
            alphas = np.asarray(sum(alphas_list[i] for i in xrange(len(models))))
        else:
            alphas = None
        if self.optimized_search:
            return probs, prev_outs_list, alphas
        else:
            return probs

    def beam_search(self, X, params, eos_sym=0, null_sym=2):
        """
        Beam search method for Cond models.
        (https://en.wikibooks.org/wiki/Artificial_Intelligence/Search/Heuristic_search/Beam_search)
        The algorithm in a nutshell does the following:

        1. k = beam_size
        2. open_nodes = [[]] * k
        3. while k > 0:

            3.1. Given the inputs, get (log) probabilities for the outputs.

            3.2. Expand each open node with all possible output.

            3.3. Prune and keep the k best nodes.

            3.4. If a sample has reached the <eos> symbol:

                3.4.1. Mark it as final sample.

                3.4.2. k -= 1

            3.5. Build new inputs (state_below) and go to 1.

        4. return final_samples, final_scores

        :param X: Model inputs
        :param params: Search parameters
        :param eos_sym: <end-of-sentence> symbol
        :param null_sym: <null> symbol
        :return: UNSORTED list of [k_best_samples, k_best_scores] (k: beam size)
        """
        k = params['beam_size']
        samples = []
        sample_scores = []
        pad_on_batch = params['pad_on_batch']
        dead_k = 0  # samples that reached eos
        live_k = 1  # samples that did not yet reached eos
        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')
        if self.return_alphas:
            sample_alphas = []
            hyp_alphas = [[]] * live_k

        maxlen = int(len(X[params['dataset_inputs'][0]][0]) * params['output_max_length_depending_on_x_factor']) if \
            params['output_max_length_depending_on_x'] else params['maxlen']

        minlen = int(len(X[params['dataset_inputs'][0]][0]) /
                     params['output_min_length_depending_on_x_factor'] + 1e-7) \
            if params['output_min_length_depending_on_x'] else 0

        # we must include an additional dimension if the input for each timestep are all the generated "words_so_far"
        if params['words_so_far']:
            if k > maxlen:
                raise NotImplementedError("BEAM_SIZE can't be higher than MAX_OUTPUT_TEXT_LEN!")
            state_below = np.asarray([[null_sym]] * live_k) \
                if pad_on_batch else np.asarray([np.zeros((maxlen, maxlen))] * live_k)
        else:
            state_below = np.asarray([null_sym] * live_k) \
                if pad_on_batch else np.asarray([np.zeros(maxlen)] * live_k)
        prev_outs = [None] * len(self.models)
        for ii in xrange(maxlen):
            # for every possible live sample calc prob for every possible label
            if self.optimized_search:  # use optimized search model if available
                [probs, prev_outs, alphas] = self.predict_cond(self.models, X, state_below, params, ii,
                                                               prev_outs=prev_outs)
            else:
                probs = self.predict_cond(self.models, X, state_below, params, ii)

            if minlen > 0 and ii < minlen:
                probs[:, eos_sym] = -np.inf

            # total score for every sample is sum of -log of word prb
            cand_scores = np.array(hyp_scores)[:, None] - np.log(probs)
            cand_flat = cand_scores.flatten()
            # Find the best options by calling argsort of flatten array
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]
            # Decypher flatten indices
            voc_size = probs.shape[1]
            trans_indices = ranks_flat / voc_size  # index of row
            word_indices = ranks_flat % voc_size  # index of col
            costs = cand_flat[ranks_flat]
            best_cost = costs[0]
            # Form a beam for the next iteration
            new_hyp_samples = []
            new_trans_indices = []
            new_hyp_scores = np.zeros(k - dead_k).astype('float32')
            if self.return_alphas:
                new_hyp_alphas = []
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                if params['search_pruning']:
                    if costs[idx] < k * best_cost:
                        new_hyp_samples.append(hyp_samples[ti] + [wi])
                        new_trans_indices.append(ti)
                        new_hyp_scores[idx] = copy.copy(costs[idx])
                        if self.return_alphas:
                            new_hyp_alphas.append(hyp_alphas[ti] + [alphas[ti]])
                    else:
                        dead_k += 1
                else:
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_trans_indices.append(ti)
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    if self.return_alphas:
                        new_hyp_alphas.append(hyp_alphas[ti] + [alphas[ti]])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_alphas = []
            indices_alive = []
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == eos_sym:  # finished sample
                    samples.append(new_hyp_samples[idx])
                    sample_scores.append(new_hyp_scores[idx])
                    if self.return_alphas:
                        sample_alphas.append(new_hyp_alphas[idx])
                    dead_k += 1
                else:
                    indices_alive.append(new_trans_indices[idx])
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    if self.return_alphas:
                        hyp_alphas.append(new_hyp_alphas[idx])
            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break
            state_below = np.asarray(hyp_samples, dtype='int64')

            # we must include an additional dimension if the input for each timestep are all the generated words so far
            if pad_on_batch:
                state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym, state_below))
                if params['words_so_far']:
                    state_below = np.expand_dims(state_below, axis=0)
            else:
                state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64'), state_below,
                                         np.zeros((state_below.shape[0],
                                                   max(maxlen - state_below.shape[1] - 1, 0)),
                                                  dtype='int64')))

                if params['words_so_far']:
                    state_below = np.expand_dims(state_below, axis=0)
                    state_below = np.hstack((state_below,
                                             np.zeros((state_below.shape[0], maxlen - state_below.shape[1],
                                                       state_below.shape[2]))))

            if self.optimized_search and ii > 0:
                for n_model in range(len(self.models)):
                    # filter next search inputs w.r.t. remaining samples
                    for idx_vars in range(len(prev_outs[n_model])):
                        prev_outs[n_model][idx_vars] = prev_outs[n_model][idx_vars][indices_alive]

        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                samples.append(hyp_samples[idx])
                sample_scores.append(hyp_scores[idx])
                if self.return_alphas:
                    sample_alphas.append(hyp_alphas[idx])
        if self.return_alphas:
            return samples, sample_scores, sample_alphas
        else:
            return samples, sample_scores, None

    def predictBeamSearchNet(self):
        """
        Approximates by beam search the best predictions of the net on the dataset splits chosen.
        Params from config that affect the sarch process:
            * max_batch_size: size of the maximum batch loaded into memory
            * n_parallel_loaders: number of parallel data batch loaders
            * normalization: apply data normalization on images/features or not (only if using images/features as input)
            * mean_substraction: apply mean data normalization on images or not (only if using images as input)
            * predict_on_sets: list of set splits for which we want to extract the predictions ['train', 'val', 'test']
            * optimized_search: boolean indicating if the used model has the optimized Beam Search implemented
             (separate self.model_init and self.model_next models for reusing the information from previous timesteps).

        The following attributes must be inserted to the model when building an optimized search model:

            * ids_inputs_init: list of input variables to model_init (must match inputs to conventional model)
            * ids_outputs_init: list of output variables of model_init (model probs must be the first output)
            * ids_inputs_next: list of input variables to model_next (previous word must be the first input)
            * ids_outputs_next: list of output variables of model_next (model probs must be the first output and
                                the number of out variables must match the number of in variables)
            * matchings_init_to_next: dictionary from 'ids_outputs_init' to 'ids_inputs_next'
            * matchings_next_to_next: dictionary from 'ids_outputs_next' to 'ids_inputs_next'

        :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        """

        # Check input parameters and recover default values if needed
        default_params = {'max_batch_size': 50,
                          'n_parallel_loaders': 8,
                          'beam_size': 5,
                          'normalize': False,
                          'mean_substraction': True,
                          'predict_on_sets': ['val'],
                          'maxlen': 20,
                          'n_samples': -1,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'pos_unk': False,
                          'state_below_index': -1,
                          'search_pruning': False,
                          'normalize_probs': False,
                          'alpha_factor': 0.0,
                          'coverage_penalty': False,
                          'length_penalty': False,
                          'length_norm_factor': 0.0,
                          'coverage_norm_factor': 0.0,
                          'output_max_length_depending_on_x': False,
                          'output_max_length_depending_on_x_factor': 3,
                          'output_min_length_depending_on_x': False,
                          'output_min_length_depending_on_x_factor': 2
                          }
        params = self.checkParameters(self.params, default_params)
        predictions = dict()
        for s in params['predict_on_sets']:
            logging.info("\n <<< Predicting outputs of " + s + " set >>>")
            assert len(params['model_inputs']) > 0, 'We need at least one input!'
            if not params['optimized_search']:  # use optimized search model if available
                assert not params['pos_unk'], 'PosUnk is not supported with non-optimized beam search methods'
            params['pad_on_batch'] = self.dataset.pad_on_batch[params['dataset_inputs'][-1]]
            # Calculate how many interations are we going to perform
            if params['n_samples'] < 1:
                n_samples = eval("self.dataset.len_" + s)
                num_iterations = int(math.ceil(float(n_samples)))  # / params['batch_size']))

                # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                # TODO: We prepare data as model 0... Different data preparators for each model?
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                batch_size=1,
                                                normalization=params['normalize'],
                                                data_augmentation=False,
                                                mean_substraction=params['mean_substraction'],
                                                predict=True).generator()
            else:
                n_samples = params['n_samples']
                num_iterations = int(math.ceil(float(n_samples)))  # / params['batch_size']))

                # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                batch_size=1,
                                                normalization=params['normalize'],
                                                data_augmentation=False,
                                                mean_substraction=params['mean_substraction'],
                                                predict=False,
                                                random_samples=n_samples).generator()
            if params['n_samples'] > 0:
                references = []
                sources_sampling = []
            best_samples = []
            if params['pos_unk']:
                best_alphas = []
                sources = []

            total_cost = 0
            sampled = 0
            start_time = time.time()
            eta = -1
            if self.n_best:
                n_best_list = []
            for j in range(num_iterations):
                data = data_gen.next()
                X = dict()
                if params['n_samples'] > 0:
                    s_dict = {}
                    for input_id in params['model_inputs']:
                        X[input_id] = data[0][input_id]
                        s_dict[input_id] = X[input_id]
                    sources_sampling.append(s_dict)

                    Y = dict()
                    for output_id in params['model_outputs']:
                        Y[output_id] = data[1][output_id]
                else:
                    s_dict = {}
                    for input_id in params['model_inputs']:
                        X[input_id] = data[input_id]
                        if params['pos_unk']:
                            s_dict[input_id] = X[input_id]
                    if params['pos_unk']:
                        sources.append(s_dict)

                for i in range(len(X[params['model_inputs'][0]])):
                    sampled += 1
                    sys.stdout.write('\r')
                    sys.stdout.write("Sampling %d/%d  -  ETA: %ds " % (sampled, n_samples, int(eta)))
                    sys.stdout.flush()
                    x = dict()
                    for input_id in params['model_inputs']:
                        x[input_id] = np.asarray([X[input_id][i]])
                    samples, scores, alphas = self.beam_search(x, params, null_sym=self.dataset.extra_words['<null>'])

                    if params['length_penalty'] or params['coverage_penalty']:
                        if params['length_penalty']:
                            length_penalties = [((5 + len(sample)) ** params['length_norm_factor']
                                                 / (5 + 1) ** params['length_norm_factor'])
                                                # this 5 is a magic number by Google...
                                                for sample in samples]
                        else:
                            length_penalties = [1.0 for _ in len(samples)]

                        if params['coverage_penalty']:
                            coverage_penalties = []
                            for k, sample in enumerate(samples):
                                # We assume that source sentences are at the first position of x
                                x_sentence = x[params['model_inputs'][0]][0]
                                alpha = np.asarray(alphas[k])
                                cp_penalty = 0.0
                                for cp_i in range(len(x_sentence)):
                                    att_weight = 0.0
                                    for cp_j in range(len(sample)):
                                        att_weight += alpha[cp_j, cp_i]
                                    cp_penalty += np.log(min(att_weight, 1.0))
                                coverage_penalties.append(params['coverage_norm_factor'] * cp_penalty)
                        else:
                            coverage_penalties = [0.0 for _ in len(samples)]
                        scores = [co / lp + cp for co, lp, cp in zip(scores, length_penalties, coverage_penalties)]

                    elif params['normalize_probs']:
                        counts = [len(sample) ** params['alpha_factor'] for sample in samples]
                        scores = [co / cn for co, cn in zip(scores, counts)]

                    if self.n_best:
                        n_best_indices = np.argsort(scores)
                        n_best_scores = np.asarray(scores)[n_best_indices]
                        n_best_samples = np.asarray(samples)[n_best_indices]
                        if alphas is not None:
                            n_best_alphas = [np.stack(alphas[i]) for i in n_best_indices]
                        else:
                            n_best_alphas = [None] * len(n_best_indices)
                        n_best_list.append([n_best_samples, n_best_scores, n_best_alphas])
                    best_score = np.argmin(scores)
                    best_sample = samples[best_score]
                    best_samples.append(best_sample)
                    if params['pos_unk']:
                        best_alphas.append(np.asarray(alphas[best_score]))
                    total_cost += scores[best_score]
                    eta = (n_samples - sampled) * (time.time() - start_time) / sampled
                    if params['n_samples'] > 0:
                        for output_id in params['model_outputs']:
                            references.append(Y[output_id][i])

            sys.stdout.write('Total cost of the translations: %f \t '
                             'Average cost of the translations: %f\n' % (total_cost, total_cost / n_samples))
            sys.stdout.write('The sampling took: %f secs (Speed: %f sec/sample)\n' %
                             ((time.time() - start_time), (time.time() - start_time) / n_samples))

            sys.stdout.flush()
            if self.n_best:
                if params['pos_unk']:
                    predictions[s] = (np.asarray(best_samples), np.asarray(best_alphas), sources), n_best_list
                else:
                    predictions[s] = np.asarray(best_samples), n_best_list
            else:
                if params['pos_unk']:
                    predictions[s] = (np.asarray(best_samples), np.asarray(best_alphas), sources)
                else:
                    predictions[s] = np.asarray(best_samples)

        if params['n_samples'] < 1:
            return predictions
        else:
            return predictions, references, sources_sampling

    def sample_beam_search(self, src_sentence):
        """

        :param src_sentence:
        :return:
        """
        # Check input parameters and recover default values if needed
        default_params = {'max_batch_size': 50,
                          'n_parallel_loaders': 8,
                          'beam_size': 5,
                          'normalize': False,
                          'mean_substraction': True,
                          'predict_on_sets': ['val'],
                          'maxlen': 20,
                          'n_samples': 1,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'state_below_index': -1,
                          'output_text_index': 0,
                          'search_pruning': False,
                          'pos_unk': False,
                          'normalize_probs': False,
                          'alpha_factor': 0.0,
                          'coverage_penalty': False,
                          'length_penalty': False,
                          'length_norm_factor': 0.0,
                          'coverage_norm_factor': 0.0,
                          'output_max_length_depending_on_x': False,
                          'output_max_length_depending_on_x_factor': 3,
                          'output_min_length_depending_on_x': False,
                          'output_min_length_depending_on_x_factor': 2
                          }
        params = self.checkParameters(self.params, default_params)
        params['pad_on_batch'] = self.dataset.pad_on_batch[params['dataset_inputs'][-1]]
        params['n_samples'] = 1
        if self.n_best:
            n_best_list = []
        X = dict()
        for input_id in params['model_inputs']:
            X[input_id] = src_sentence
        x = dict()
        for input_id in params['model_inputs']:
            x[input_id] = np.asarray([X[input_id]])
        samples, scores, alphas = self.beam_search(x,
                                                   params,
                                                   null_sym=self.dataset.extra_words['<null>'])

        if params['length_penalty'] or params['coverage_penalty']:
            if params['length_penalty']:
                length_penalties = [((5 + len(sample)) ** params['length_norm_factor']
                                     / (5 + 1) ** params['length_norm_factor'])  # this 5 is a magic number by Google...
                                    for sample in samples]
            else:
                length_penalties = [1.0 for _ in len(samples)]

            if params['coverage_penalty']:
                coverage_penalties = []
                for k, sample in enumerate(samples):
                    # We assume that source sentences are at the first position of x
                    x_sentence = x[params['model_inputs'][0]][0]
                    alpha = np.asarray(alphas[k])
                    cp_penalty = 0.0
                    for cp_i in range(len(x_sentence)):
                        att_weight = 0.0
                        for cp_j in range(len(sample)):
                            att_weight += alpha[cp_j, cp_i]
                        cp_penalty += np.log(min(att_weight, 1.0))
                    coverage_penalties.append(params['coverage_norm_factor'] * cp_penalty)
            else:
                coverage_penalties = [0.0 for _ in len(samples)]
            scores = [co / lp + cp for co, lp, cp in zip(scores, length_penalties, coverage_penalties)]

        elif params['normalize_probs']:
            counts = [len(sample) ** params['alpha_factor'] for sample in samples]
            scores = [co / cn for co, cn in zip(scores, counts)]

        if self.n_best:
            n_best_indices = np.argsort(scores)
            n_best_scores = np.asarray(scores)[n_best_indices]
            n_best_samples = np.asarray(samples)[n_best_indices]
            if alphas is not None:
                n_best_alphas = [np.stack(alphas[i]) for i in n_best_indices]
            else:
                n_best_alphas = [None] * len(n_best_indices)
            n_best_list.append([n_best_samples, n_best_scores, n_best_alphas])

        best_score_idx = np.argmin(scores)
        best_sample = samples[best_score_idx]
        if params['pos_unk']:
            best_alphas = np.asarray(alphas[best_score_idx])
        else:
            best_alphas = None
        if self.n_best:
            return (np.asarray(best_sample), scores[best_score_idx], np.asarray(best_alphas)), n_best_list
        else:
            return np.asarray(best_sample), scores[best_score_idx], np.asarray(best_alphas)

    def score_cond_model(self, X, Y, params, null_sym=2):
        """
        Beam search method for Cond models.
        (https://en.wikibooks.org/wiki/Artificial_Intelligence/Search/Heuristic_search/Beam_search)
        The algorithm in a nutshell does the following:

        1. k = beam_size
        2. open_nodes = [[]] * k
        3. while k > 0:

            3.1. Given the inputs, get (log) probabilities for the outputs.

            3.2. Expand each open node with all possible output.

            3.3. Prune and keep the k best nodes.

            3.4. If a sample has reached the <eos> symbol:

                3.4.1. Mark it as final sample.

                3.4.2. k -= 1

            3.5. Build new inputs (state_below) and go to 1.

        4. return final_samples, final_scores

        :param X: Model inputs
        :param params: Search parameters
        :param null_sym: <null> symbol
        :return: UNSORTED list of [k_best_samples, k_best_scores] (k: beam size)
        """
        # we must include an additional dimension if the input for each timestep are all the generated "words_so_far"
        pad_on_batch = params['pad_on_batch']
        score = 0.0
        if self.return_alphas:
            all_alphas = []
        else:
            all_alphas = None
        if params['words_so_far']:
            state_below = np.asarray([[null_sym]]) \
                if pad_on_batch else np.asarray([np.zeros((params['maxlen'], params['maxlen']))])
        else:
            state_below = np.asarray([null_sym]) \
                if pad_on_batch else np.asarray([np.zeros(params['maxlen'])])

        prev_outs = [None] * len(self.models)
        for ii in xrange(len(Y)):
            # for every possible live sample calc prob for every possible label
            if self.optimized_search:  # use optimized search model if available
                [probs, prev_outs, alphas] = self.predict_cond(self.models, X, state_below, params, ii,
                                                               prev_outs=prev_outs)
            else:
                probs = self.predict_cond(self.models, X, state_below, params, ii)
            # total score for every sample is sum of -log of word prb
            score -= np.log(probs[0, int(Y[ii])])
            state_below = np.asarray([Y[:ii]], dtype='int64')
            if self.return_alphas:
                all_alphas.append(alphas[0])
            # we must include an additional dimension if the input for each timestep are all the generated words so far
            if pad_on_batch:
                state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym, state_below))
                if params['words_so_far']:
                    state_below = np.expand_dims(state_below, axis=0)
            else:
                state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64'), state_below,
                                         np.zeros((state_below.shape[0],
                                                   max(params['maxlen'] - state_below.shape[1] - 1, 0)),
                                                  dtype='int64')))

                if params['words_so_far']:
                    state_below = np.expand_dims(state_below, axis=0)
                    state_below = np.hstack((state_below,
                                             np.zeros((state_below.shape[0], params['maxlen'] - state_below.shape[1],
                                                       state_below.shape[2]))))

            if self.optimized_search and ii > 0:
                for n_model in range(len(self.models)):
                    # filter next search inputs w.r.t. remaining samples
                    for idx_vars in range(len(prev_outs[n_model])):
                        prev_outs[n_model][idx_vars] = prev_outs[n_model][idx_vars]

        return score, all_alphas

    def scoreNet(self):
        """
        Approximates by beam search the best predictions of the net on the dataset splits chosen.
        Params from config that affect the sarch process:
            * max_batch_size: size of the maximum batch loaded into memory
            * n_parallel_loaders: number of parallel data batch loaders
            * normalization: apply data normalization on images/features or not (only if using images/features as input)
            * mean_substraction: apply mean data normalization on images or not (only if using images as input)
            * predict_on_sets: list of set splits for which we want to extract the predictions ['train', 'val', 'test']
            * optimized_search: boolean indicating if the used model has the optimized Beam Search implemented
             (separate self.model_init and self.model_next models for reusing the information from previous timesteps).

        The following attributes must be inserted to the model when building an optimized search model:

            * ids_inputs_init: list of input variables to model_init (must match inputs to conventional model)
            * ids_outputs_init: list of output variables of model_init (model probs must be the first output)
            * ids_inputs_next: list of input variables to model_next (previous word must be the first input)
            * ids_outputs_next: list of output variables of model_next (model probs must be the first output and
                                the number of out variables must match the number of in variables)
            * matchings_init_to_next: dictionary from 'ids_outputs_init' to 'ids_inputs_next'
            * matchings_next_to_next: dictionary from 'ids_outputs_next' to 'ids_inputs_next'

        :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        """

        # Check input parameters and recover default values if needed
        default_params = {'max_batch_size': 50,
                          'n_parallel_loaders': 8,
                          'beam_size': 5,
                          'normalize': False,
                          'mean_substraction': True,
                          'predict_on_sets': ['val'],
                          'maxlen': 20,
                          'n_samples': -1,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'state_below_index': -1,
                          'output_text_index': 0,
                          'pos_unk': False,
                          'normalize_probs': False,
                          'alpha_factor': 0.0,
                          'coverage_penalty': False,
                          'length_penalty': False,
                          'length_norm_factor': 0.0,
                          'coverage_norm_factor': 0.0,
                          'output_max_length_depending_on_x': False,
                          'output_max_length_depending_on_x_factor': 3,
                          'output_min_length_depending_on_x': False,
                          'output_min_length_depending_on_x_factor': 2
                          }
        params = self.checkParameters(self.params, default_params)

        scores_dict = dict()

        for s in params['predict_on_sets']:
            logging.info("<<< Scoring outputs of " + s + " set >>>")
            assert len(params['model_inputs']) > 0, 'We need at least one input!'
            if not params['optimized_search']:  # use optimized search model if available
                assert not params['pos_unk'], 'PosUnk is not supported with non-optimized beam search methods'
            params['pad_on_batch'] = self.dataset.pad_on_batch[params['dataset_inputs'][-1]]
            # Calculate how many interations are we going to perform
            n_samples = eval("self.dataset.len_" + s)
            num_iterations = int(math.ceil(float(n_samples)))  # / params['batch_size']))

            # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
            # TODO: We prepare data as model 0... Different data preparators for each model?
            data_gen = Data_Batch_Generator(s,
                                            self.models[0],
                                            self.dataset,
                                            num_iterations,
                                            shuffle=False,
                                            batch_size=1,
                                            normalization=params['normalize'],
                                            data_augmentation=False,
                                            mean_substraction=params['mean_substraction'],
                                            predict=False).generator()
            sources_sampling = []
            scores = []
            total_cost = 0
            sampled = 0
            start_time = time.time()
            eta = -1
            for j in range(num_iterations):
                data = data_gen.next()
                X = dict()
                s_dict = {}
                for input_id in params['model_inputs']:
                    X[input_id] = data[0][input_id]
                    s_dict[input_id] = X[input_id]
                sources_sampling.append(s_dict)

                Y = dict()
                for output_id in params['model_outputs']:
                    Y[output_id] = data[1][output_id]

                for i in range(len(X[params['model_inputs'][0]])):
                    sampled += 1
                    sys.stdout.write('\r')
                    sys.stdout.write("Scored %d/%d  -  ETA: %ds " % (sampled, n_samples, int(eta)))
                    sys.stdout.flush()
                    x = dict()

                    for input_id in params['model_inputs']:
                        x[input_id] = np.asarray([X[input_id][i]])
                    sample = one_hot_2_indices([Y[params['dataset_outputs'][params['output_text_index']]][i]],
                                               pad_sequences=True, verbose=0)[0]
                    score, alphas = self.score_cond_model(x, sample, params,
                                                          null_sym=self.dataset.extra_words['<null>'])

                    if params['length_penalty'] or params['coverage_penalty']:
                        if params['length_penalty']:
                            length_penalty = ((5 + len(sample)) ** params['length_norm_factor']
                                              / (5 + 1) ** params[
                                                  'length_norm_factor'])  # this 5 is a magic number by Google...
                        else:
                            length_penalty = 1.0

                        if params['coverage_penalty']:
                            coverage_penalties = []
                            # We assume that source sentences are at the first position of x
                            x_sentence = x[params['model_inputs'][0]][0]
                            alpha = np.asarray(alphas)
                            cp_penalty = 0.0
                            for cp_i in range(len(x_sentence)):
                                att_weight = 0.0
                                for cp_j in range(len(sample)):
                                    att_weight += alpha[cp_j, cp_i]
                                cp_penalty += np.log(min(att_weight, 1.0))
                            coverage_penalty = params['coverage_norm_factor'] * cp_penalty
                        else:
                            coverage_penalty = 0.0
                        score = score / length_penalty + coverage_penalty

                    elif params['normalize_probs']:
                        counts = float(len(sample) ** params['alpha_factor'])
                        score /= counts

                    scores.append(score)
                    total_cost += score
                    eta = (n_samples - sampled) * (time.time() - start_time) / sampled

            sys.stdout.write('Total cost of the translations: %f \t '
                             'Average cost of the translations: %f\n' % (total_cost, total_cost / n_samples))
            sys.stdout.write('The scoring took: %f secs (Speed: %f sec/sample)\n' %
                             ((time.time() - start_time), (time.time() - start_time) / n_samples))

            sys.stdout.flush()
            scores_dict[s] = scores
        return scores_dict

    def scoreSample(self, data):
        """
        Approximates by beam search the best predictions of the net on the dataset splits chosen.
        Params from config that affect the sarch process:
            * batch_size: size of the batch
            * n_parallel_loaders: number of parallel data batch loaders
            * normalization: apply data normalization on images/features or not (only if using images/features as input)
            * mean_substraction: apply mean data normalization on images or not (only if using images as input)
            * predict_on_sets: list of set splits for which we want to extract the predictions ['train', 'val', 'test']
            * optimized_search: boolean indicating if the used model has the optimized Beam Search implemented
             (separate self.model_init and self.model_next models for reusing the information from previous timesteps).

        The following attributes must be inserted to the model when building an optimized search model:

            * ids_inputs_init: list of input variables to model_init (must match inputs to conventional model)
            * ids_outputs_init: list of output variables of model_init (model probs must be the first output)
            * ids_inputs_next: list of input variables to model_next (previous word must be the first input)
            * ids_outputs_next: list of output variables of model_next (model probs must be the first output and
                                the number of out variables must match the number of in variables)
            * matchings_init_to_next: dictionary from 'ids_outputs_init' to 'ids_inputs_next'
            * matchings_next_to_next: dictionary from 'ids_outputs_next' to 'ids_inputs_next'

        :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        """

        # Check input parameters and recover default values if needed
        default_params = {'max_batch_size': 50,
                          'n_parallel_loaders': 8,
                          'beam_size': 5,
                          'normalize': False,
                          'mean_substraction': True,
                          'predict_on_sets': ['val'],
                          'maxlen': 20,
                          'n_samples': -1,
                          'pad_on_batch': True,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'state_below_index': -1,
                          'output_text_index': 0,
                          'pos_unk': False,
                          'mapping': None,
                          'normalize_probs': False,
                          'alpha_factor': 0.0,
                          'coverage_penalty': False,
                          'length_penalty': False,
                          'length_norm_factor': 0.0,
                          'coverage_norm_factor': 0.0,
                          'output_max_length_depending_on_x': False,
                          'output_max_length_depending_on_x_factor': 3,
                          'output_min_length_depending_on_x': False,
                          'output_min_length_depending_on_x_factor': 2
                          }
        params = self.checkParameters(self.params, default_params)

        scores = []
        total_cost = 0
        sampled = 0
        X = dict()
        for i, input_id in enumerate(params['model_inputs']):
            X[input_id] = data[0][i]
        Y = dict()
        for i, output_id in enumerate(params['model_outputs']):
            Y[output_id] = data[1][i]

        for i in range(len(X[params['model_inputs'][0]])):
            sampled += 1
            x = dict()

            for input_id in params['model_inputs']:
                x[input_id] = np.asarray([X[input_id][i]])
            sample = one_hot_2_indices([Y[params['dataset_outputs'][params['output_text_index']]][i]],
                                       pad_sequences=params['pad_on_batch'], verbose=0)[0]
            score, alphas = self.score_cond_model(x, sample,
                                                  params,
                                                  null_sym=self.dataset.extra_words['<null>'])

            if params['length_penalty'] or params['coverage_penalty']:
                if params['length_penalty']:
                    length_penalty = ((5 + len(sample)) ** params['length_norm_factor']
                                      / (5 + 1) ** params[
                                          'length_norm_factor'])  # this 5 is a magic number by Google...
                else:
                    length_penalty = 1.0

                if params['coverage_penalty']:
                    coverage_penalties = []
                    # We assume that source sentences are at the first position of x
                    x_sentence = x[params['model_inputs'][0]][0]
                    alpha = np.asarray(alphas)
                    cp_penalty = 0.0
                    for cp_i in range(len(x_sentence)):
                        att_weight = 0.0
                        for cp_j in range(len(sample)):
                            att_weight += alpha[cp_j, cp_i]
                        cp_penalty += np.log(min(att_weight, 1.0))
                    coverage_penalty = params['coverage_norm_factor'] * cp_penalty
                else:
                    coverage_penalty = 0.0
                score = score / length_penalty + coverage_penalty
            elif params['normalize_probs']:
                counts = float(len(sample) ** params['alpha_factor'])
                score /= counts

            scores.append(score)
            total_cost += score
        return scores

    def BeamSearchNet(self):
        """
        DEPRECATED, use predictBeamSearchNet() instead.
        """
        print "WARNING!: deprecated function, use predictBeamSearchNet() instead"
        return self.predictBeamSearchNet()

    @staticmethod
    def checkParameters(input_params, default_params):
        """
            Validates a set of input parameters and uses the default ones if not specified.
        """
        valid_params = [key for key in default_params]
        params = dict()

        # Check input parameters' validity
        for key, val in input_params.iteritems():
            if key in valid_params:
                params[key] = val

        # Use default parameters if not provided
        for key, default_val in default_params.iteritems():
            if key not in params:
                params[key] = default_val

        return params


class PredictEnsemble:
    def __init__(self, models, dataset, params_prediction, postprocess_fun=None, verbose=0):
        """

        :param models:
        :param dataset:
        :param params_prediction:
        """
        self.models = models
        self.postprocess_fun = postprocess_fun
        self.dataset = dataset
        self.params = params_prediction
        self.verbose = verbose

    # PREDICTION FUNCTIONS: Functions for making prediction on input samples

    def predict_generator(self, models, data_gen, val_samples=1, max_q_size=1):
        """
        Call the prediction functions of all models, according to their inputs
        The generator should return the same kind of data as accepted by
        `predict_on_batch`.

        # Arguments
            generator: generator yielding batches of input samples.
            val_samples: total number of samples to generate from `generator`
                before returning.
            max_q_size: maximum size for the generator queue
            nb_worker: maximum number of processes to spin up
            pickle_safe: if True, use process based threading. Note that because
                this implementation relies on multiprocessing, you should not pass non
                non picklable arguments to the generator as they can't be passed
                easily to children processes.

        # Returns
            A Numpy array of predictions.
        """

        outs_list = []
        for i, m in enumerate(models):
            outs_list.append(m.model.predict_on_batch(data_gen, val_samples, max_q_size))
        outs = sum(outs_list[i] for i in xrange(len(models))) / float(len(models))
        return outs

    def predict_on_batch(self, models, X, in_name=None, out_name=None, expand=False):
        """
        Applies a forward pass and returns the predicted values of all models.

        # Arguments
            generator: generator yielding batches of input samples.
            val_samples: total number of samples to generate from `generator`
                before returning.
            max_q_size: maximum size for the generator queue
            nb_worker: maximum number of processes to spin up
            pickle_safe: if True, use process based threading. Note that because
                this implementation relies on multiprocessing, you should not pass non
                non picklable arguments to the generator as they can't be passed
                easily to children processes.

        # Returns
            A Numpy array of predictions.
        """

        outs_list = []
        for i, m in enumerate(models):
            outs_list.append(m.model.predict_on_batch(X))
        outs = sum(outs_list[i] for i in xrange(len(models))) / float(len(models))
        return outs

    def predictNet(self):
        '''
            Returns the predictions of the net on the dataset splits chosen. The input 'parameters' is a dict()
            which may contain the following parameters:

            :param batch_size: size of the batch
            :param n_parallel_loaders: number of parallel data batch loaders
            :param normalize: apply data normalization on images/features or not (only if using images/features as input)
            :param mean_substraction: apply mean data normalization on images or not (only if using images as input)
            :param predict_on_sets: list of set splits for which we want to extract the predictions ['train', 'val', 'test']

            Additional parameters:

            :param postprocess_fun : post-processing function applied to all predictions before returning the result. The output of the function must be a list of results, one per sample. If postprocess_fun is a list, the second element will be used as an extra input to the function.

            :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        '''

        # Check input parameters and recover default values if needed
        default_params = {'batch_size': 50,
                          'n_parallel_loaders': 8,
                          'normalize': False,
                          'mean_substraction': True,
                          'model_inputs': ['input1'],
                          'model_outputs': ['output1'],
                          'dataset_inputs': ['input1'],
                          'dataset_outputs': ['output1'],
                          'n_samples': None,
                          'init_sample': -1,
                          'final_sample': -1,
                          'verbose': 1,
                          'predict_on_sets': ['val'],
                          'max_eval_samples': None
                          }

        params = self.checkParameters(self.params, default_params)
        predictions = dict()

        for s in params['predict_on_sets']:
            logging.info("\n <<< Predicting outputs of " + s + " set >>>")
            assert len(params['model_inputs']) > 0, 'We need at least one input!'

            # Calculate how many interations are we going to perform
            if params['n_samples'] is None:
                if params['init_sample'] > -1 and params['final_sample'] > -1:
                    n_samples = params['final_sample'] - params['init_sample']
                else:
                    n_samples = eval("self.dataset.len_" + s)
                num_iterations = int(math.ceil(float(n_samples) / params['batch_size']))
                n_samples = min(eval("self.dataset.len_" + s), num_iterations * params['batch_size'])

                # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                # TODO: We prepare data as model 0... Different data preparators for each model?
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                batch_size=params['batch_size'],
                                                normalization=params['normalize'],
                                                data_augmentation=False,
                                                mean_substraction=params['mean_substraction'],
                                                init_sample=params['init_sample'],
                                                final_sample=params['final_sample'],
                                                predict=True).generator()
            else:
                n_samples = params['n_samples']
                num_iterations = int(math.ceil(float(n_samples) / params['batch_size']))

                # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                batch_size=params['batch_size'],
                                                normalization=params['normalize'],
                                                data_augmentation=False,
                                                mean_substraction=params['mean_substraction'],
                                                predict=True,
                                                random_samples=n_samples).generator()
            # Predict on model
            """
            if self.postprocess_fun is None:

                out = self.predict_generator(self.models,
                                             data_gen,
                                             val_samples=n_samples,
                                             max_q_size=params['n_parallel_loaders'])
                predictions[s] = out
            """
            processed_samples = 0
            start_time = time.time()
            while processed_samples < n_samples:
                out = self.predict_on_batch(self.models, data_gen.next())
                # Apply post-processing function
                if self.postprocess_fun is not None:
                    if isinstance(self.postprocess_fun, list):
                        last_processed = min(processed_samples + params['batch_size'], n_samples)
                        out = self.postprocess_fun[0](out, self.postprocess_fun[1][processed_samples:last_processed])
                    else:
                        out = self.postprocess_fun(out)

                if predictions.get(s) is None:
                    predictions[s] = [out]
                else:
                    predictions[s].append(out)
                # Show progress
                processed_samples += params['batch_size']
                if processed_samples > n_samples:
                    processed_samples = n_samples
                eta = (n_samples - processed_samples) * (time.time() - start_time) / processed_samples
                sys.stdout.write('\r')
                sys.stdout.write("Predicting %d/%d  -  ETA: %ds " % (processed_samples, n_samples, int(eta)))
                sys.stdout.flush()
            predictions[s] = np.concatenate([pred for pred in predictions[s]])

        return predictions

    @staticmethod
    def checkParameters(input_params, default_params):
        """
            Validates a set of input parameters and uses the default ones if not specified.
        """
        valid_params = [key for key in default_params]
        params = dict()

        # Check input parameters' validity
        for key, val in input_params.iteritems():
            if key in valid_params:
                params[key] = val

        # Use default parameters if not provided
        for key, default_val in default_params.iteritems():
            if key not in params:
                params[key] = default_val

        return params
