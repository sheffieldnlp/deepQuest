import logging
import os

from keras.layers import *
from keras.models import model_from_json, Model
from keras.optimizers import Adam, RMSprop, Nadam, Adadelta, SGD, Adagrad, Adamax
from keras.regularizers import l2, AlphaRegularizer
from keras_wrapper.cnn_model import Model_Wrapper
from regularize import Regularize
from keras.initializers import Initializer


class TranslationModel(Model_Wrapper):
    """
    Translation model class. Instance of the Model_Wrapper class (see staged_keras_wrapper).

    :param params: all hyperparams of the model.
    :param model_type: network name type (corresponds to any method defined in the section 'MODELS' of this class).
                 Only valid if 'structure_path' == None.
    :param verbose: set to 0 if you don't want the model to output informative messages
    :param structure_path: path to a Keras' model json file.
                          If we speficy this parameter then 'type' will be only an informative parameter.
    :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
    :param model_name: optional name given to the network
                       (if None, then it will be assigned to current time as its name)
    :param vocabularies: vocabularies used for word embedding
    :param store_path: path to the folder where the temporal model packups will be stored
    :param set_optimizer: Compile optimizer or not.
    :param clear_dirs: Clean model directories or not.

    """

    def __init__(self, params, model_type='Translation_Model', verbose=1, structure_path=None, weights_path=None,
                 model_name=None, vocabularies=None, store_path=None, set_optimizer=True, clear_dirs=False, trainable_est=True, trainable_pred=True):
        """
        Translation_Model object constructor.

        :param params: all hyperparams of the model.
        :param model_type: network name type (corresponds to any method defined in the section 'MODELS' of this class).
                     Only valid if 'structure_path' == None.
        :param verbose: set to 0 if you don't want the model to output informative messages
        :param structure_path: path to a Keras' model json file.
                              If we speficy this parameter then 'type' will be only an informative parameter.
        :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
        :param model_name: optional name given to the network
                           (if None, then it will be assigned to current time as its name)
        :param vocabularies: vocabularies used for word embedding
        :param store_path: path to the folder where the temporal model packups will be stored
        :param set_optimizer: Compile optimizer or not.
        :param clear_dirs: Clean model directories or not.
        :param trainable: For stack multi-level training.
        """

        super(TranslationModel, self).__init__(type=model_type, model_name=model_name,
                                               silence=verbose == 0, models_path=store_path, inheritance=True)

        self.__toprint = ['_model_type', 'name', 'model_path', 'verbose']

        self.verbose = verbose
        self._model_type = model_type
        self.params = params
        self.vocabularies = vocabularies
        self.ids_inputs = params['INPUTS_IDS_MODEL']
        self.ids_outputs = params['OUTPUTS_IDS_MODEL']
        self.return_alphas = params['COVERAGE_PENALTY'] or params['POS_UNK']
        # Sets the model name and prepares the folders for storing the models
        self.setName(model_name, models_path=store_path, clear_dirs=clear_dirs)
        #self.trainable = trainable
        self.trainable_est = trainable_est
        self.trainable = trainable_pred

        # Prepare source word embedding
        if params['SRC_PRETRAINED_VECTORS'] is not None:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from:  " + params['SRC_PRETRAINED_VECTORS'] + " >>>")
            src_word_vectors = np.load(os.path.join(params['SRC_PRETRAINED_VECTORS'])).item()
            self.src_embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'],
                                                        params['SOURCE_TEXT_EMBEDDING_SIZE'])
            for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
                if src_word_vectors.get(word) is not None:
                    self.src_embedding_weights[index, :] = src_word_vectors[word]
            self.src_embedding_weights = [self.src_embedding_weights]
            self.src_embedding_weights_trainable = params['SRC_PRETRAINED_VECTORS_TRAINABLE']
            del src_word_vectors

        else:
            self.src_embedding_weights = None
            self.src_embedding_weights_trainable = True

        # Prepare target word embedding
        if params['TRG_PRETRAINED_VECTORS'] is not None:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from: " + params['TRG_PRETRAINED_VECTORS'] + " >>>")
            trg_word_vectors = np.load(os.path.join(params['TRG_PRETRAINED_VECTORS'])).item()
            self.trg_embedding_weights = np.random.rand(params['OUTPUT_VOCABULARY_SIZE'],
                                                        params['TARGET_TEXT_EMBEDDING_SIZE'])
            for word, index in self.vocabularies[self.ids_outputs[0]]['words2idx'].iteritems():
                if trg_word_vectors.get(word) is not None:
                    self.trg_embedding_weights[index, :] = trg_word_vectors[word]
            self.trg_embedding_weights = [self.trg_embedding_weights]
            self.trg_embedding_weights_trainable = params['TRG_PRETRAINED_VECTORS_TRAINABLE']
            del trg_word_vectors
        else:
            self.trg_embedding_weights = None
            self.trg_embedding_weights_trainable = True

        # Prepare model
        if structure_path:
            # Load a .json model
            if self.verbose > 0:
                logging.info("<<< Loading model structure from file " + structure_path + " >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, model_type):
                if self.verbose > 0:
                    logging.info("<<< Building " + model_type + " Translation_Model >>>")
                eval('self.' + model_type + '(params)')
            else:
                raise Exception('Translation_Model model_type "' + model_type + '" is not implemented.')

        # Load weights from file
        if weights_path:
            if self.verbose > 0:
                logging.info("<<< Loading weights from file " + weights_path + " >>>")
            self.model.load_weights(weights_path, by_name=True)

        # Print information of self
        if verbose > 0:
            print str(self)
            self.model.summary()
        if set_optimizer:
            self.setOptimizer()

    def setParams(self, params):
        self.params = params

    def setOptimizer(self, **kwargs):
        """
        Sets and compiles a new optimizer for the Translation_Model.
        :param kwargs:
        :return:
        """
        # compile differently depending if our model is 'Sequential' or 'Graph'
        if self.verbose > 0:
            logging.info("Preparing optimizer: %s [LR: %s - LOSS: %s] and compiling." %
                         (str(self.params['OPTIMIZER']), str(self.params.get('LR', 0.01)),
                          str(self.params.get('LOSS', 'categorical_crossentropy'))))

        if self.params['OPTIMIZER'].lower() == 'sgd':
            optimizer = SGD(lr=self.params.get('LR', 0.01),
                            momentum=self.params.get('MOMENTUM', 0.0),
                            decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                            nesterov=self.params.get('NESTEROV_MOMENTUM', False),
                            clipnorm=self.params.get('CLIP_C', 0.),
                            clipvalue=self.params.get('CLIP_V', 0.), )

        elif self.params['OPTIMIZER'].lower() == 'rsmprop':
            optimizer = RMSprop(lr=self.params.get('LR', 0.001),
                                rho=self.params.get('RHO', 0.9),
                                decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                clipnorm=self.params.get('CLIP_C', 0.),
                                clipvalue=self.params.get('CLIP_V', 0.))

        elif self.params['OPTIMIZER'].lower() == 'adagrad':
            optimizer = Adagrad(lr=self.params.get('LR', 0.01),
                                decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                clipnorm=self.params.get('CLIP_C', 0.),
                                clipvalue=self.params.get('CLIP_V', 0.))

        elif self.params['OPTIMIZER'].lower() == 'adadelta':
            optimizer = Adadelta(lr=self.params.get('LR', 1.0),
                                 rho=self.params.get('RHO', 0.9),
                                 decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                 clipnorm=self.params.get('CLIP_C', 0.),
                                 clipvalue=self.params.get('CLIP_V', 0.))

        elif self.params['OPTIMIZER'].lower() == 'adam':
            optimizer = Adam(lr=self.params.get('LR', 0.001),
                             beta_1=self.params.get('BETA_1', 0.9),
                             beta_2=self.params.get('BETA_2', 0.999),
                             decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                             clipnorm=self.params.get('CLIP_C', 0.),
                             clipvalue=self.params.get('CLIP_V', 0.))

        elif self.params['OPTIMIZER'].lower() == 'adamax':
            optimizer = Adamax(lr=self.params.get('LR', 0.002),
                               beta_1=self.params.get('BETA_1', 0.9),
                               beta_2=self.params.get('BETA_2', 0.999),
                               decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                               clipnorm=self.params.get('CLIP_C', 0.),
                               clipvalue=self.params.get('CLIP_V', 0.))

        elif self.params['OPTIMIZER'].lower() == 'nadam':
            optimizer = Nadam(lr=self.params.get('LR', 0.002),
                              beta_1=self.params.get('BETA_1', 0.9),
                              beta_2=self.params.get('BETA_2', 0.999),
                              schedule_decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                              clipnorm=self.params.get('CLIP_C', 0.),
                              clipvalue=self.params.get('CLIP_V', 0.))
        else:
            logging.info('\tWARNING: The modification of the LR is not implemented for the chosen optimizer.')
            optimizer = eval(self.params['OPTIMIZER'])

        sample_weight_mode = []
        sample_weight_dict = self.params['SAMPLE_WEIGHTS']

        for out_id in self.ids_outputs:

            if out_id in sample_weight_dict:
                sample_weight_mode.append('temporal')
            else:
                sample_weight_mode.append(None)

        self.model.compile(optimizer=optimizer, loss=self.params['LOSS'],
                           metrics=self.params.get('KERAS_METRICS', []),
                           sample_weight_mode=sample_weight_mode)


    def __str__(self):
        """
        Plots basic model information.

        :return: String containing model information.
        """
        obj_str = '-----------------------------------------------------------------------------------\n'
        class_name = self.__class__.__name__
        obj_str += '\t\t' + class_name + ' instance\n'
        obj_str += '-----------------------------------------------------------------------------------\n'

        # Print pickled attributes
        for att in self.__toprint:
            obj_str += att + ': ' + str(self.__dict__[att])
            obj_str += '\n'

        obj_str += '\n'
        obj_str += 'MODEL params:\n'
        obj_str += str(self.params)
        obj_str += '\n'
        obj_str += '-----------------------------------------------------------------------------------'

        return obj_str




    # ------------------------------------------------------- #
    #       PREDEFINED MODELS
    # ------------------------------------------------------- #


    #=============================
    # Word-level QE -- BiRNN model
    #=============================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated sentences (shape: (mini_batch_size, line_words))
    #
    ## Output:
    # 1. Word quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## Summary of the model:
    # The sentence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level, and used for making classification decisions.

    def EncWord(self, params):
        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True, trainable=self.trainable),
                                    name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                    merge_mode='concat')(trg_embedding)

        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        word_qe = TimeDistributed(Dense(params['WORD_QE_CLASSES'], activation=out_activation), name=self.ids_outputs[0])(annotations)

        self.model = Model(inputs=[src_words, trg_words],
                           outputs=[word_qe])



    #=================================
    # Sentence-level QE -- BiRNN model
    #=================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated documents (shape: (mini_batch_size, line_words))
    #
    ## Output:
    # 1. Sentence quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # The sententence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level, and the sentence representation is a weighted sum of its words.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j:
    #       alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting sentence vector is thus a weighted sum of word vectors:
    #       v = sum_j alpha_j*h_j
    # Sentence vectors are then directly used for making classification decisions.

    def EncSent(self, params):
        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True, trainable=self.trainable),
                                    name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                    merge_mode='concat')(trg_embedding)

        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        annotations = NonMasking()(annotations)
        # apply attention over words at the sentence-level
        annotations = attention_3d_block(annotations, params, 'sent')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_sent = Dense(1, activation=out_activation, name=self.ids_outputs[0])(annotations)

        self.model = Model(inputs=[src_words, trg_words],
                           outputs=[qe_sent])



    #========================================================================
    # Sentence-level QE with visual features -- BiRNN model + visual features
    #========================================================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated documents (shape: (mini_batch_size, line_words))
    #
    ## Output:
    # 1. Sentence quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # The sententence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level, and the sentence representation is a weighted sum of its words.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j:
    #       alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting sentence vector is thus a weighted sum of word vectors:
    #       v = sum_j alpha_j*h_j
    # Sentence vectors are then directly used for making classification decisions.

    def EncSentVis(self, params):
        # visual feature input
        visual_feature = Input(name=self.ids_inputs[2],
                            batch_shape=tuple([None, params['LEN_VISUAL_FEATURE']]), dtype='float32')
        rnd_seed = params.get('RND_SEED', 124)

        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        # visual feature added to the embedding
        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'embed':

            reduced_visual_feature_src = Dense(params['SOURCE_TEXT_EMBEDDING_SIZE'], activation='relu', name='reduced_visual_feature_src')(visual_feature)

            dropout_vis_src = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature_src)
            reshape_red_visual_feature_src = visual_feature_reshape(dropout_vis_src, params, 'src')

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'mult-mult':
                src_embedding = multiply([src_embedding, reshape_red_visual_feature_src], name='src_embedding_vis')
            else:
                src_embedding = concatenate([src_embedding, reshape_red_visual_feature_src], name='src_embedding_vis')
            print("src_embedding with visual feature")

        else:
            pass

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        # visual feature added to the embedding
        if not params.get('ONE_SIDE', False) and (params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'embed'):

            reduced_visual_feature_trg = Dense(params['TARGET_TEXT_EMBEDDING_SIZE'], activation='relu', name='reduced_visual_feature_trg')(visual_feature)

            dropout_vis_trg = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature_trg)
            reshape_red_visual_feature_trg = visual_feature_reshape(dropout_vis_trg, params, 'trg')

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'concat':
                trg_embedding = concatenate([trg_embedding, reshape_red_visual_feature_trg], name='trg_embedding_vis')
            else:
                trg_embedding = multiply([trg_embedding, reshape_red_visual_feature_trg], name='trg_embedding_vis') #concatenate
            print("trg_embedding with visual feature")

        else:
            pass

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True, trainable=self.trainable),
                                    name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                    merge_mode='concat')(trg_embedding)

        # visual feature concatenated in the time step
        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'annot':

            reduced_visual_feature = Dense((params['ENCODER_HIDDEN_SIZE'] * 2), activation='relu', name='reduced_visual_feature')(visual_feature)

            dropout_vis = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature)

            reshape_red_visual_feature = visual_feature_reshape(dropout_vis, params, 'src_trg')

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'concat':
                src_annotations_vis = concatenate([src_annotations, reshape_red_visual_feature], axis=1, name='src_annotations_vis')
                trg_annotations_vis = concatenate([trg_annotations, reshape_red_visual_feature], axis=1, name='trg_annotations_vis')
            else:
                src_annotations_vis = multiply([src_annotations, reshape_red_visual_feature], name='src_annotations_vis')
                trg_annotations_vis = multiply([trg_annotations, reshape_red_visual_feature], name='trg_annotations_vis')
            print("src_annotations shape:", K.int_shape(src_annotations_vis))
            annotations = concatenate([src_annotations_vis, trg_annotations_vis], name='anot_seq_concat')

        else:
            annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')

        #annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        annotations = NonMasking()(annotations)
        # apply attention over words at the sentence-level
        annotations = attention_3d_block(annotations, params, 'sent')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        # visual feature concatenated
        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'last':
            reduced_visual_feature = Dense(params['ENCODER_HIDDEN_SIZE'] * 4, activation='relu', name='reduced_visual_feature')(visual_feature)
            dropout_vis = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature)
            # concatenation of the visual features with the doc summary
            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'concat':
                sent_vis_concat = concatenate([annotations, dropout_vis], name='sent_vis_concat') #concatenate #multiply
            else:
                sent_vis_concat = multiply([annotations, dropout_vis], name='sent_vis_concat') #concatenate #multiply

            qe_sent = Dense(1, activation=out_activation, name=self.ids_outputs[0])(sent_vis_concat)

        else:
            qe_sent = Dense(1, activation=out_activation, name=self.ids_outputs[0])(annotations)


        self.model = Model(inputs=[src_words, trg_words, visual_feature],
                           outputs=[qe_sent])


    def EncBertSent(self, params):
        src_words_tokids = Input(name=self.ids_inputs[0],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        src_words_mask = Input(name=self.ids_inputs[1],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        src_words_segids = Input(name=self.ids_inputs[2],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # src_bert_inputs = dict(
            # input_ids=src_words_tokids, input_mask=src_words_mask, segment_ids=src_words_segids
        # )
        src_bert_input = [src_words_tokids, src_words_mask, src_words_segids]

        # src_embedding = BertLayer(
        #         max_seq_len=params['MAX_INPUT_TEXT_LEN'],
        #         trainable=False,
        #         name="src_word_embedding"
        #         )(src_bert_input)

        bert_layer = BertLayer(
                max_seq_len=params['MAX_INPUT_TEXT_LEN'],
                trainable=True,
                name="bert_embedding"
                )
        src_embedding = bert_layer(src_bert_input)

        # src_embedding =  Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
        #                           name='src_word_embedding',
        #                           embeddings_regularizer=l2(params['WEIGHT_DECAY']),
        #                           embeddings_initializer=params['INIT_FUNCTION'],
        #                           trainable=self.trainable,
        #                           mask_zero=True)(src_words_tokids)
                                  # mask_zero=True)(src_words)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(
            params['ENCODER_HIDDEN_SIZE'],
            kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            dropout=params['RECURRENT_INPUT_DROPOUT_P'],
            recurrent_dropout=params['RECURRENT_DROPOUT_P'],
            kernel_initializer=params['INIT_FUNCTION'],
            recurrent_initializer=params['INNER_INIT'],
            return_sequences=True,
            trainable=self.trainable),
            name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
            merge_mode='concat')(src_embedding)

        trg_words_tokids = Input(name=self.ids_inputs[3],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        trg_words_mask = Input(name=self.ids_inputs[4],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        trg_words_segids = Input(name=self.ids_inputs[5],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        trg_bert_input = [trg_words_tokids, trg_words_mask, trg_words_segids]

        # trg_embedding = BertLayer(
        #         max_seq_len=params['MAX_INPUT_TEXT_LEN'],
        #         trainable=True,
        #         name="trg_word_embedding"
        #         )(trg_bert_input)

        trg_embedding = bert_layer(trg_bert_input)

        # trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
        #                           name='target_word_embedding',
        #                           embeddings_regularizer=l2(params['WEIGHT_DECAY']),
        #                           embeddings_initializer=params['INIT_FUNCTION'],
        #                           trainable=self.trainable,
        #                           mask_zero=True)(trg_words_tokids)
                                  # mask_zero=True)(trg_words)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='target_state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(
            params['ENCODER_HIDDEN_SIZE'],
            kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            dropout=params['RECURRENT_INPUT_DROPOUT_P'],
            recurrent_dropout=params['RECURRENT_DROPOUT_P'],
            kernel_initializer=params['INIT_FUNCTION'],
            recurrent_initializer=params['INNER_INIT'],
            return_sequences=True, trainable=self.trainable),
            name='target_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
            merge_mode='concat')(trg_embedding)

        # annotations = concatenate([src_embedding, trg_embedding], name='anot_seq_concat')
        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        # import ipdb; ipdb.set_trace()
        annotations = NonMasking()(annotations)
        # apply attention over words at the sentence-level
        annotations = attention_3d_block(annotations, params, 'sent')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_sent = Dense(1, activation=out_activation, name=self.ids_outputs[0])(annotations)

        self.model = Model(inputs=[src_words_tokids, src_words_mask, src_words_segids, trg_words_tokids, trg_words_mask, trg_words_segids],
                           outputs=[qe_sent])


    def EncBertSentVis(self, params):
        # visual feature input
        visual_feature = Input(name=self.ids_inputs[6],
                            batch_shape=tuple([None, params['LEN_VISUAL_FEATURE']]), dtype='float32')
        rnd_seed = params.get('RND_SEED', 124)

        src_words_tokids = Input(name=self.ids_inputs[0],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        src_words_mask = Input(name=self.ids_inputs[1],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        src_words_segids = Input(name=self.ids_inputs[2],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # src_bert_inputs = dict(
            # input_ids=src_words_tokids, input_mask=src_words_mask, segment_ids=src_words_segids
        # )
        src_bert_input = [src_words_tokids, src_words_mask, src_words_segids]

        # src_embedding = BertLayer(
        #         max_seq_len=params['MAX_INPUT_TEXT_LEN'],
        #         trainable=False,
        #         name="src_word_embedding"
        #         )(src_bert_input)

        bert_layer = BertLayer(
                max_seq_len=params['MAX_INPUT_TEXT_LEN'],
                trainable=True,
                name="bert_embedding"
                )
        src_embedding = bert_layer(src_bert_input)

        # src_embedding =  Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
        #                           name='src_word_embedding',
        #                           embeddings_regularizer=l2(params['WEIGHT_DECAY']),
        #                           embeddings_initializer=params['INIT_FUNCTION'],
        #                           trainable=self.trainable,
        #                           mask_zero=True)(src_words_tokids)
                                  # mask_zero=True)(src_words)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        # visual feature added to the embedding
        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'embed':

            reduced_visual_feature_src = Dense(768, activation='relu', name='reduced_visual_feature_src')(visual_feature)
            #params['INPUT_VOCABULARY_SIZE']
            dropout_vis_src = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature_src)
            reshape_red_visual_feature_src = visual_feature_reshape(dropout_vis_src, params, 'src')

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'mult-mult':
                src_embedding = multiply([src_embedding, reshape_red_visual_feature_src], name='src_embedding_vis')
            else:
                src_embedding = concatenate([src_embedding, reshape_red_visual_feature_src], name='src_embedding_vis')
            print("src_embedding with visual feature")

        else:
            pass

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(
            params['ENCODER_HIDDEN_SIZE'],
            kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            dropout=params['RECURRENT_INPUT_DROPOUT_P'],
            recurrent_dropout=params['RECURRENT_DROPOUT_P'],
            kernel_initializer=params['INIT_FUNCTION'],
            recurrent_initializer=params['INNER_INIT'],
            return_sequences=True,
            trainable=self.trainable),
            name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
            merge_mode='concat')(src_embedding)

        trg_words_tokids = Input(name=self.ids_inputs[3],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        trg_words_mask = Input(name=self.ids_inputs[4],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        trg_words_segids = Input(name=self.ids_inputs[5],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        trg_bert_input = [trg_words_tokids, trg_words_mask, trg_words_segids]

        # trg_embedding = BertLayer(
        #         max_seq_len=params['MAX_INPUT_TEXT_LEN'],
        #         trainable=True,
        #         name="trg_word_embedding"
        #         )(trg_bert_input)

        trg_embedding = bert_layer(trg_bert_input)

        # trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
        #                           name='target_word_embedding',
        #                           embeddings_regularizer=l2(params['WEIGHT_DECAY']),
        #                           embeddings_initializer=params['INIT_FUNCTION'],
        #                           trainable=self.trainable,
        #                           mask_zero=True)(trg_words_tokids)
                                  # mask_zero=True)(trg_words)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='target_state')

        # visual feature added to the embedding
        if not params.get('ONE_SIDE', False) and (params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'embed'):

            reduced_visual_feature_trg = Dense(768, activation='relu', name='reduced_visual_feature_trg')(visual_feature)
            #params['INPUT_VOCABULARY_SIZE']
            dropout_vis_trg = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature_trg)
            reshape_red_visual_feature_trg = visual_feature_reshape(dropout_vis_trg, params, 'trg')

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'concat':
                trg_embedding = concatenate([trg_embedding, reshape_red_visual_feature_trg], name='trg_embedding_vis')
            else:
                trg_embedding = multiply([trg_embedding, reshape_red_visual_feature_trg], name='trg_embedding_vis') #concatenate
            print("trg_embedding with visual feature")

        else:
            pass

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(
            params['ENCODER_HIDDEN_SIZE'],
            kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            dropout=params['RECURRENT_INPUT_DROPOUT_P'],
            recurrent_dropout=params['RECURRENT_DROPOUT_P'],
            kernel_initializer=params['INIT_FUNCTION'],
            recurrent_initializer=params['INNER_INIT'],
            return_sequences=True, trainable=self.trainable),
            name='target_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
            merge_mode='concat')(trg_embedding)

        # visual feature concatenated in the time step
        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'annot':

            reduced_visual_feature = Dense((params['ENCODER_HIDDEN_SIZE'] * 2), activation='relu', name='reduced_visual_feature')(visual_feature)

            dropout_vis = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature)

            reshape_red_visual_feature = visual_feature_reshape(dropout_vis, params, 'src_trg')

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'concat':
                src_annotations_vis = concatenate([src_annotations, reshape_red_visual_feature], axis=1, name='src_annotations_vis')
                trg_annotations_vis = concatenate([trg_annotations, reshape_red_visual_feature], axis=1, name='trg_annotations_vis')
            else:
                src_annotations_vis = multiply([src_annotations, reshape_red_visual_feature], name='src_annotations_vis')
                trg_annotations_vis = multiply([trg_annotations, reshape_red_visual_feature], name='trg_annotations_vis')
            print("src_annotations shape:", K.int_shape(src_annotations_vis))
            annotations = concatenate([src_annotations_vis, trg_annotations_vis], name='anot_seq_concat')

        else:
            annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')

        # import ipdb; ipdb.set_trace()
        annotations = NonMasking()(annotations)
        # apply attention over words at the sentence-level
        annotations = attention_3d_block(annotations, params, 'sent')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_sent = Dense(1, activation=out_activation, name=self.ids_outputs[0])(annotations)

        # visual feature concatenated
        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'last':
            reduced_visual_feature = Dense(params['ENCODER_HIDDEN_SIZE'] * 4, activation='relu', name='reduced_visual_feature')(visual_feature)
            dropout_vis = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature)
            # concatenation of the visual features with the doc summary
            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'concat':
                sent_vis_last = concatenate([annotations, dropout_vis], name='sent_vis_last') #concatenate #multiply
            else:
                sent_vis_last = multiply([annotations, dropout_vis], name='sent_vis_last') #concatenate #multiply

            qe_sent = Dense(1, activation=out_activation, name=self.ids_outputs[0])(sent_vis_last)

        else:
            qe_sent = Dense(1, activation=out_activation, name=self.ids_outputs[0])(annotations)

        self.model = Model(inputs=[src_words_tokids, src_words_mask, src_words_segids, trg_words_tokids, trg_words_mask, trg_words_segids, visual_feature],
                           outputs=[qe_sent])




    #=================================
    # Document-level QE -- BiRNN model
    #=================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Parallel machine-translated documents (shape: (mini_batch_size, doc_lines, words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the document level.
    # The sentence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level.
    # A sentence representation is a weighted sum of its words.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j:
    #       alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting sentence vector is thus a weighted sum of word vectors:
    #       v = sum_j alpha_j*h_j
    # Sentence vectors are inputted in a doc-level bi-directional RNN.
    # The last hidden state of ths RNN is taken as the summary of an entire document.
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EncDoc(self, params):
        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        genreshape = GeneralReshape((None, params['MAX_INPUT_TEXT_LEN']), params)
        src_words_in = genreshape(src_words)

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        trg_words_in = genreshape(trg_words)

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words_in)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True, trainable=self.trainable),
                                    name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                    merge_mode='concat')(trg_embedding)

        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        annotations = NonMasking()(annotations)
        # apply sent-level attention over words
        annotations = attention_3d_block(annotations, params, 'sent')
        # reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['ENCODER_HIDDEN_SIZE'] * 4), params)
        annotations = genreshape_out(annotations)

        # bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         name='dec_doc_frw')(annotations)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='dec_doc_bkw')(annotations)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                            name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est, name='dec_doc_seq_concat')

        # we take the last bi-RNN state as doc summary
        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw], name='dec_doc_last_state_concat')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(dec_doc_last_state_concat)

        self.model = Model(inputs=[src_words, trg_words],
                           outputs=[qe_doc])


    #========================================================================
    # Document-level QE with visual features -- BiRNN model + visual features
    #========================================================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Parallel machine-translated documents (shape: (mini_batch_size, doc_lines, words))
    # 3. Visual features (shape: (mini_batch_size, number_of_features))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the document level.
    # The sentence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level.
    # A sentence representation is a weighted sum of its words.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j:
    #       alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting sentence vector is thus a weighted sum of word vectors:
    #       v = sum_j alpha_j*h_j
    # Sentence vectors are inputted in a doc-level bi-directional RNN.
    # The last hidden state of ths RNN is taken as the summary of an entire document.
    # The visual features are used as additional inputs.
    # Doc-level representations are then directly used for making classification decisions.
    #
    # Three merging strategies are tested for this model: last-layer merging, add-embedding and time-step.
    #
    ## References
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EncDocVis(self, params):
        # visual feature input
        visual_feature = Input(name=self.ids_inputs[2],
                            batch_shape=tuple([None, params['LEN_VISUAL_FEATURE']]), dtype='float32')
        rnd_seed = params.get('RND_SEED', 124)

        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        genreshape = GeneralReshape((None, params['MAX_INPUT_TEXT_LEN']), params)
        src_words_in = genreshape(src_words)

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                      name='src_word_embedding',
                                      embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                      embeddings_initializer=params['INIT_FUNCTION'],
                                      trainable=self.trainable,
                                      mask_zero=True)(src_words_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        # visual feature added to the source embedding
        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'embed':

            reduced_visual_feature_src = Dense(params['SOURCE_TEXT_EMBEDDING_SIZE'], activation='relu', name='reduced_visual_feature_src')(visual_feature)

            dropout_vis_src = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature_src)
            reshape_red_visual_feature_src = visual_feature_reshape(dropout_vis_src, params, 'src')

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'mult-mult':
                src_embedding = multiply([src_embedding, reshape_red_visual_feature_src], name='src_embedding_vis')
            else:
                src_embedding = concatenate([src_embedding, reshape_red_visual_feature_src], name='src_embedding_vis') #concatenate
            print("src_embedding with visual feature")

        else:
            pass


        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'state':

            reduced_visual_feature_src = Dense(params['ENCODER_HIDDEN_SIZE'], activation='relu', name='reduced_visual_feature_src')(visual_feature)
            print('type visual feature', type(reduced_visual_feature_src))
            print('visual feature shape', K.int_shape(reduced_visual_feature_src))

            if params['ENCODER_RNN_TYPE'] == 'GRU':
                reshape_red_visual_feature_src = visual_feature_reshape(reduced_visual_feature_src, params, 'src', dim=2)
                print('reshape visual feature shape', K.int_shape(reshape_red_visual_feature_src))

                initial_state_concat_src = concatenate([reshape_red_visual_feature_src, reshape_red_visual_feature_src], name='initial_state_concat_src')

                src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                 kernel_regularizer=l2(
                                                                                     params['RECURRENT_WEIGHT_DECAY']),
                                                                                 recurrent_regularizer=l2(
                                                                                     params['RECURRENT_WEIGHT_DECAY']),
                                                                                 bias_regularizer=l2(
                                                                                     params['RECURRENT_WEIGHT_DECAY']),
                                                                                 dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                                 recurrent_dropout=params[
                                                                                     'RECURRENT_DROPOUT_P'],
                                                                                 kernel_initializer=params['INIT_FUNCTION'],
                                                                                 recurrent_initializer=params['INNER_INIT'],
                                                                                 return_sequences=True,
                                                                                 trainable=self.trainable),
                                                name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                                merge_mode='concat')([src_embedding, initial_state_concat_src])

                '''

                src_annotations_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                 kernel_regularizer=l2(
                                                                                     params['RECURRENT_WEIGHT_DECAY']),
                                                                                 recurrent_regularizer=l2(
                                                                                     params['RECURRENT_WEIGHT_DECAY']),
                                                                                 bias_regularizer=l2(
                                                                                     params['RECURRENT_WEIGHT_DECAY']),
                                                                                 dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                                 recurrent_dropout=params[
                                                                                     'RECURRENT_DROPOUT_P'],
                                                                                 kernel_initializer=params['INIT_FUNCTION'],
                                                                                 recurrent_initializer=params['INNER_INIT'],
                                                                                 return_sequences=True,
                                                                                 trainable=self.trainable,
                                                                                 name='src_bidirectional_encoder_frw_' + params['ENCODER_RNN_TYPE']
                                                                                 )(src_embedding, initial_state=reshape_red_visual_feature_src)
                                                #name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                                #merge_mode='concat')([src_embedding, reshape_red_visual_feature_src])
                src_annotations_bkd = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                 kernel_regularizer=l2(
                                                                                     params['RECURRENT_WEIGHT_DECAY']),
                                                                                 recurrent_regularizer=l2(
                                                                                     params['RECURRENT_WEIGHT_DECAY']),
                                                                                 bias_regularizer=l2(
                                                                                     params['RECURRENT_WEIGHT_DECAY']),
                                                                                 dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                                 recurrent_dropout=params[
                                                                                     'RECURRENT_DROPOUT_P'],
                                                                                 kernel_initializer=params['INIT_FUNCTION'],
                                                                                 recurrent_initializer=params['INNER_INIT'],
                                                                                 return_sequences=True,
                                                                                 go_backwards=True,
                                                                                 trainable=self.trainable,
                                                                                 name='src_bidirectional_encoder_bkd_' + params['ENCODER_RNN_TYPE']
                                                                                 )(src_embedding, initial_state=reshape_red_visual_feature_src)
                src_annotations = concatenate([src_annotations_frw, src_annotations_bkd], name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'])
                '''

            else:
                raise NotImplementedError('RNN other than GRU currently not supported for this merging method')



        else:
            src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             return_sequences=True,
                                                                             trainable=self.trainable),
                                            name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                            merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        trg_words_in = genreshape(trg_words)

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words_in)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        # visual feature added to the target (MT) embedding
        if not params.get('ONE_SIDE', False) and (params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'embed'):

            reduced_visual_feature_trg = Dense(params['TARGET_TEXT_EMBEDDING_SIZE'], activation='relu', name='reduced_visual_feature_trg')(visual_feature)

            dropout_vis_trg = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature_trg)
            reshape_red_visual_feature_trg = visual_feature_reshape(dropout_vis_trg, params, 'trg')

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'concat':
                trg_embedding = concatenate([trg_embedding, reshape_red_visual_feature_trg], name='trg_embedding_vis')
            else:
                trg_embedding = multiply([trg_embedding, reshape_red_visual_feature_trg], name='trg_embedding_vis') #concatenate # multiply
            print("trg_embedding with visual feature")

        else:
            pass

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True, trainable=self.trainable),
                                    name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                    merge_mode='concat')(trg_embedding)


        # visual feature merged with the text features in the time step
        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'annot':

            reduced_visual_feature = Dense((params['ENCODER_HIDDEN_SIZE'] * 2), activation='relu', name='reduced_visual_feature')(visual_feature)

            dropout_vis = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature)

            reshape_red_visual_feature = visual_feature_reshape(dropout_vis, params, 'src_trg')

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'mult':
                src_annotations_vis = multiply([src_annotations, reshape_red_visual_feature], name='src_annotations_vis')#concatenate([src_annotations, reshape_red_visual_feature], axis=1, name='src_annotations_vis')
                trg_annotations_vis = multiply([trg_annotations, reshape_red_visual_feature], name='trg_annotations_vis')
            else:
                src_annotations_vis = concatenate([src_annotations, reshape_red_visual_feature], axis=1, name='src_annotations_vis')
                trg_annotations_vis = concatenate([trg_annotations, reshape_red_visual_feature], axis=1, name='trg_annotations_vis')
            print("src_annotations shape:", K.int_shape(src_annotations_vis))
            annotations = concatenate([src_annotations_vis, trg_annotations_vis], name='anot_seq_concat')

        else:
            annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')


        annotations = NonMasking()(annotations)
        # apply sent-level attention over words
        annotations = attention_3d_block(annotations, params, 'sent')
        # reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['ENCODER_HIDDEN_SIZE'] * 4), params)

        #annotations = genreshape_out(annotations)

        # visual feature concatenated in the time step
        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'annot-2':

            reduced_visual_feature = Dense(200, activation='relu', name='reduced_visual_feature')(visual_feature)

            #dropout_vis = Dropout(0.2)(reduced_visual_feature)
            # In the time step
            reshape_red_visual_feature = visual_feature_reshape(reduced_visual_feature, params, 'src_trg', dim=2)

            annotations = multiply([annotations, reshape_red_visual_feature], name='annot_vis')
            annotations = genreshape_out(annotations)

        else:
            annotations = genreshape_out(annotations)

        # bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         name='dec_doc_frw')(annotations)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='dec_doc_bkw')(annotations)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                            name='dec_reverse_doc_bkw')(dec_doc_bkw)

        #dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est, name='dec_doc_seq_concat')

        # we take the last bi-RNN state as doc summary
        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw], name='dec_doc_last_state_concat')

        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        # visual feature merged at the last layer
        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'last':

            reduced_visual_feature = Dense((params['DOC_DECODER_HIDDEN_SIZE'] * 2), activation='relu', name='reduced_visual_feature')(visual_feature)

            dropout_vis = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature)
            # concatenation of the visual features with the doc summary

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'mult':
                doc_vis_concat = multiply([dec_doc_last_state_concat, dropout_vis], name='doc_vis_concat') #concatenate
            else:
                doc_vis_concat = concatenate([dec_doc_last_state_concat, dropout_vis], name='doc_vis_concat')

            qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(doc_vis_concat)

        else:
            qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(dec_doc_last_state_concat)

        self.model = Model(inputs=[src_words, trg_words, visual_feature],
                           outputs=[qe_doc])





    #=============================================================================
    # Document-level QE with Attention mechanism -- BiRNN model Doc QE + Attention
    #=============================================================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Parallel machine-translated documents (shape: (mini_batch_size, doc_lines, words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the document level.
    # The sentence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level.
    # A sentence representation is a weighted sum of its words.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j:
    #       alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting sentence vector is thus a weighted sum of word vectors:
    #       v = sum_j alpha_j*h_j
    # Sentence vectors are inputted in a doc-level bi-directional RNN.
    # A document representation is a weighted sum of its sentences. We apply the attention function as described above.
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EncDocAtt(self, params):
        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        genreshape = GeneralReshape((None, params['MAX_INPUT_TEXT_LEN']), params)
        src_words_in = genreshape(src_words)

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        trg_words_in = genreshape(trg_words)

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words_in)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True, trainable=self.trainable),
                                    name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                    merge_mode='concat')(trg_embedding)

        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        annotations = NonMasking()(annotations)
        # apply sent-level attention over words
        annotations = attention_3d_block(annotations, params, 'sent')
        # reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['ENCODER_HIDDEN_SIZE'] * 4), params)
        annotations = genreshape_out(annotations)

        #bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         name='dec_doc_frw')(annotations)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='dec_doc_bkw')(annotations)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                            name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est, name='dec_doc_seq_concat')

        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw], name='dec_doc_last_state_concat')
        dec_doc_seq_concat = NonMasking()(dec_doc_seq_concat)

        # apply attention over doc sentences
        attention_mul = attention_3d_block(dec_doc_seq_concat, params, 'doc')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(attention_mul)

        self.model = Model(inputs=[src_words, trg_words],
                           outputs=[qe_doc])




    #===========================================
    # Document-level QE --POSTECH-inspired model
    #===========================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Machine-translated documents with one-position left-shifted sentences to represent the right context (shape: (mini_batch_size, doc_lines, words))
    # 3. Machine-translated documents with one-position rigth-shifted sentences to represent the left context (shape: (mini_batch_size, doc_lines, words))
    # 4. Machine-translated documents with unshifted sentences for evaluation (shape: (mini_batch_size, doc_lines,words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the      document level.
    # Sentence-level representations are created as by a POSTECH-inspired sentence-level QE model.
    # Those representations are inputted in a doc-level bi-directional RNN.
    # The last hidden state of ths RNN is taken as the summary of an entire document.
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EstimatorDoc(self, params):
        src_text = Input(name=self.ids_inputs[0],
                         batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        # Reshape input to 2d to produce sent-level vectors. Reshaping to (None, None) is necessary for compatibility with pre-trained Predictors.
        genreshape = GeneralReshape((None, None), params)
        src_text_in = genreshape(src_text)
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        next_words = Input(name=self.ids_inputs[1],
                           batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        next_words_in = genreshape(next_words)

        next_words_bkw = Input(name=self.ids_inputs[2],
                               batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]),
                               dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        next_words_bkw_in = genreshape(next_words_bkw)
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_in)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw_in)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)

        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []

        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        trg_words_in = genreshape(trg_words)

        next_words_one_hot = one_hot(trg_words_in, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         name='qe_frw', trainable=self.trainable_est)(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw', trainable=self.trainable_est)(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        enc_qe_concat = concatenate([enc_qe_frw, enc_qe_bkw], name='enc_qe_concat')
        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est,
                                        name='last_state_concat')

        #reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['QE_VECTOR_SIZE'] * 2), params)

        last_state_concat = genreshape_out(last_state_concat)

        #bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  name='dec_doc_frw')(last_state_concat)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  go_backwards=True, name='dec_doc_bkw')(last_state_concat)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                              name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est,
                                         name='dec_doc_seq_concat')

        #we take the last bi-RNN state as doc summary
        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw],
                                                name='dec_doc_last_state_concat')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(dec_doc_last_state_concat)
        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[qe_doc])





    #===========================================
    # Document-level QE --POSTECH-inspired model
    #===========================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Machine-translated documents with one-position left-shifted sentences to represent the right context (shape: (mini_batch_size, doc_lines, words))
    # 3. Machine-translated documents with one-position rigth-shifted sentences to represent the left context (shape: (mini_batch_size, doc_lines, words))
    # 4. Machine-translated documents with unshifted sentences for evaluation (shape: (mini_batch_size, doc_lines,words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the      document level.
    # Sentence-level representations are created as by a POSTECH-inspired sentence-level QE model.
    # Those representations are inputted in a doc-level bi-directional RNN.
    # The last hidden state of ths RNN is taken as the summary of an entire document.
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EstimatorDocVis(self, params):
        # visual feature part
        visual_feature = Input(name=self.ids_inputs[2],
                            batch_shape=tuple([None, params['LEN_VISUAL_FEATURE']]), dtype='float32')
        rnd_seed = params.get('RND_SEED', 124)

        src_text = Input(name=self.ids_inputs[0],
                         batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        # Reshape input to 2d to produce sent-level vectors. Reshaping to (None, None) is necessary for compatibility with pre-trained Predictors.
        genreshape = GeneralReshape((None, None), params)
        src_text_in = genreshape(src_text)
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        # visual feature added to the source embedding
        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'embed':

            reduced_visual_feature_src = Dense(params['SOURCE_TEXT_EMBEDDING_SIZE'], activation='relu', name='reduced_visual_feature_src')(visual_feature)

            dropout_vis_src = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature_src)
            reshape_red_visual_feature_src = visual_feature_reshape(dropout_vis_src, params, 'src')

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'mult-mult':
                src_embedding = multiply([src_embedding, reshape_red_visual_feature_src], name='src_embedding_vis') #concatenate
            else:
                src_embedding = concatenate([src_embedding, reshape_red_visual_feature_src], name='src_embedding_vis') #concatenate
            print("src_embedding with visual feature")

        else:
            pass

        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        next_words = Input(name=self.ids_inputs[1],
                           batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        next_words_in = genreshape(next_words)

        next_words_bkw = Input(name=self.ids_inputs[2],
                               batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]),
                               dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        next_words_bkw_in = genreshape(next_words_bkw)
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_in)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        # visual feature (below) added to the target (MT) embedding
        if (params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'embed'):

            reduced_visual_feature_below = Dense(params['TARGET_TEXT_EMBEDDING_SIZE'], activation='relu', name='reduced_visual_feature_below')(visual_feature)

            dropout_vis_below = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature_below)
            reshape_red_visual_feature_below = visual_feature_reshape(dropout_vis_below, params, 'below')

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'concat':
                state_below = concatenate([state_below, reshape_red_visual_feature_below], name='below_embedding_vis') #concatenate # multiply
            else:
                state_below = multiply([state_below, reshape_red_visual_feature_below], name='below_embedding_vis') #concatenate # multiply
            print("state_below with visual feature")

        else:
            pass

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw_in)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # visual feature (above) added to the target (MT) embedding
        if (params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'embed'):

            reduced_visual_feature_above = Dense(params['TARGET_TEXT_EMBEDDING_SIZE'], activation='relu', name='reduced_visual_feature_above')(visual_feature)

            dropout_vis_above = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature_above)
            reshape_red_visual_feature_above = visual_feature_reshape(dropout_vis_above, params, 'above')

            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'concat':
                state_above = concatenate([state_above, reshape_red_visual_feature_above], name='above_embedding_vis') #concatenate # multiply
            else:
                state_above = multiply([state_above, reshape_red_visual_feature_above], name='above_embedding_vis') #concatenate # multiply
            print("state_above with visual feature")

        else:
            pass

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)

        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []

        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        trg_words_in = genreshape(trg_words)

        next_words_one_hot = one_hot(trg_words_in, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         name='qe_frw', trainable=self.trainable_est)(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw', trainable=self.trainable_est)(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        enc_qe_concat = concatenate([enc_qe_frw, enc_qe_bkw], name='enc_qe_concat')
        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est,
                                        name='last_state_concat')

        #reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['QE_VECTOR_SIZE'] * 2), params)

        last_state_concat = genreshape_out(last_state_concat)

        #bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  name='dec_doc_frw')(last_state_concat)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  go_backwards=True, name='dec_doc_bkw')(last_state_concat)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                              name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est,
                                         name='dec_doc_seq_concat')

        #we take the last bi-RNN state as doc summary
        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw],
                                                name='dec_doc_last_state_concat')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        # visual feature concatenated
        if params.get('VISUAL_FEATURE_STRATEGY', 'none') == 'last':

            reduced_visual_feature = Dense((params['DOC_DECODER_HIDDEN_SIZE'] * 2), activation='relu', name='reduced_visual_feature')(visual_feature)

            dropout_vis = Dropout(0.5, seed=rnd_seed)(reduced_visual_feature)
            # concatenation of the visual features with the doc summary
            if params.get('VISUAL_FEATURE_METHOD', 'none') == 'concat':
                doc_vis_concat = concatenate([dec_doc_last_state_concat, dropout_vis], name='doc_vis_concat')
            else:
                doc_vis_concat = multiply([dec_doc_last_state_concat, dropout_vis], name='doc_vis_concat')

            qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(doc_vis_concat)

        else:
            qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(dec_doc_last_state_concat)


        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words, visual_feature],
                           outputs=[qe_doc])






    #=====================================================================
    # Document-level QE with Attention mechanism -- POSTECH-inspired model
    #=====================================================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Machine-translated documents with one-position left-shifted sentences to represent the right context (shape: (mini_batch_size, doc_lines, words))
    # 3. Machine-translated documents with one-position rigth-shifted sentences to represent the left context (shape: (mini_batch_size, doc_lines, words))
    # 4. Machine-translated documents with unshifted sentences for evaluation (shape: (mini_batch_size, doc_lines, doc_lines,words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the      document level.
    # Sentence-level representations are created as by a POSTECH-inspired sentence-level QE model.
    # Those representations are inputted in a doc-level bi-directional RNN.
    # A document representation is a weighted sum of its sentences.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j: alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting document vector is thus a weighted sum of sentence vectors:
    # v = sum_j alpha_j*h_j
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EstimatorDocAtt(self, params):
        src_text = Input(name=self.ids_inputs[0],
                         batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        # Reshape input to 2d to produce sent-level vectors. Reshaping to (None, None) is necessary for compatibility with pre-trained Predictors
        genreshape = GeneralReshape((None, None), params)
        src_text_in = genreshape(src_text)

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')
        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        # trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words = Input(name=self.ids_inputs[1],
                           batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        next_words_in = genreshape(next_words)

        next_words_bkw = Input(name=self.ids_inputs[2],
                               batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]),
                               dtype='int32')
        next_words_bkw_in = genreshape(next_words_bkw)

        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_in)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw_in)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)

        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []

        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        trg_words_in = genreshape(trg_words)

        next_words_one_hot = one_hot(trg_words_in, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         name='qe_frw', trainable=self.trainable_est)(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw', trainable=self.trainable_est)(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        enc_qe_concat = concatenate([enc_qe_frw, enc_qe_bkw], name='enc_qe_concat')
        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est,
                                        name='last_state_concat')

        # reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['QE_VECTOR_SIZE'] * 2), params)

        last_state_concat = genreshape_out(last_state_concat)

        #bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  name='dec_doc_frw')(last_state_concat)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  go_backwards=True, name='dec_doc_bkw')(last_state_concat)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                              name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est,
                                         name='dec_doc_seq_concat')

        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw],
                                                name='dec_doc_last_state_concat')

        dec_doc_seq_concat  = NonMasking()(dec_doc_seq_concat)

        # apply doc-level attention over sentences
        attention_mul = attention_3d_block(dec_doc_seq_concat, params, 'doc')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(attention_mul)

        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[qe_doc])




    #======================================================
    # Sentence-level QE -- POSTECH-inspired Estimator model
    #======================================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, words))
    # 2. One-position left-shifted machine-translated sentences to represent the right context (shape: (mini_batch_size, words))
    # 3. One-position rigth-shifted machine-translated sentences to represent the left context (shape: (mini_batch_size, words))
    # 4. Unshifted machine-translated sentences for evaluation (shape: (mini_batch_size, words))
    #
    ## Output:
    # 1. Sentence quality scores (shape: (mini_batch_size,))
    #
    ## References:
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.


    def EstimatorSent(self, params):
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable,name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True, trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable, name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        #trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words_bkw = Input(name=self.ids_inputs[2], batch_shape=tuple([None, None]), dtype='int32')

        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'],trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True,
                                                                     trainable=self.trainable,
                                                                     name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                         kernel_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         recurrent_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         bias_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                         recurrent_dropout=params[
                                                             'RECURRENT_DROPOUT_P'],
                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                         recurrent_initializer=params['INNER_INIT'],
                                                         return_sequences=True,
                                                         trainable=self.trainable,
                                                         go_backwards=True,
                                                         name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable, name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable, name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2),trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True, name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable, shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)

        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []

        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE'+params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE'+'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable = self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3], batch_shape=tuple([None, None]), dtype='int32')

        next_words_one_hot = one_hot(trg_words, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         name='qe_frw', trainable=self.trainable_est)(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw', trainable=self.trainable_est)(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est, name='last_state_concat')

        seq_concat = concatenate([enc_qe_frw, enc_qe_bkw], trainable=self.trainable_est, name='seq_concat')

        # uncomment for Post QE
        # fin_seq = concatenate([seq_concat, merged_states])
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_sent = Dense(1, activation=out_activation, trainable=self.trainable_est, name=self.ids_outputs[0])(last_state_concat)
        #word_qe = TimeDistributed(Dense(params['WORD_QE_CLASSES'], activation='sigmoid'), name=self.ids_outputs[2])(
        #    seq_concat)

        # self.model = Model(inputs=[src_text, next_words, next_words_bkw], outputs=[merged_states,softout, softoutQE])
        # if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
        #     self.model.add_loss(alpha_regularizer)
        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[qe_sent])


    #=============================
    # Phrase-level QE -- BiRNN model
    #=============================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated sentences (shape: (mini_batch_size, sentence_phrases, words))
    #
    ## Output:
    # 1. Phrase quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## Summary of the model:
    # The encoder encodes the source, the decoder at each timestep produces an output representation taking into account the previously produced representations, as well as the sum of source word representations weighted by the attention mechanism.
    # The resulting word-level representations are summarized (sum or average) into phrase-level representations used to make classification decisions.


    def EncPhraseAtt(self, params):
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        # trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, params['SECOND_DIM_SIZE'], None]), dtype='int32')
        trg_reshape = Reshape((-1,))
        # reshape MT input to 2d
        trg_words_reshaped = trg_reshape(trg_words)

        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(trg_words_reshaped)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_annotations = proj_h
        annotations = NonMasking()(trg_annotations)
        # reshape back to 3d
        annotations_reshape = Reshape((params['SECOND_DIM_SIZE'], -1, params['DECODER_HIDDEN_SIZE']))
        annotations_reshaped = annotations_reshape(annotations)
        #summarize phrase representations (average by default)
        merge_mode = params.get('WORD_MERGE_MODE', None)
        merge_words = Lambda(mask_aware_mean4d, mask_aware_merge_output_shape4d)
        if merge_mode == 'sum':
            merge_words = Lambda(sum4d, mask_aware_merge_output_shape4d)

        output_annotations = merge_words(annotations_reshaped)
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        phrase_qe = TimeDistributed(Dense(params['PHRASE_QE_CLASSES'], activation=out_activation), trainable=self.trainable,
                                  name=self.ids_outputs[0])(output_annotations)

        self.model = Model(inputs=[src_text, trg_words],
                           outputs=[phrase_qe])

    #======================================================
    # Phrase-level QE -- POSTECH-inspired Estimator model
    #======================================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated sentences (shape: (mini_batch_size, sentence_phrases, words))
    # 3. One-position rigth-shifted machine-translated sentences to represent the left context (shape: (mini_batch_size, sentence_phrases, words))
    # 4. Unshifted machine-translated sentences for evaluation (shape: (mini_batch_size, sentence_phrases, words))
    #
    ## Output:
    # 1. Phrase quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## References:
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.

    def EstimatorPhrase(self, params):

        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        #trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, params['SECOND_DIM_SIZE'], None]), dtype='int32')

        # Reshape phrase inputs to 2d
        trg_reshape = Reshape((-1, ))
        next_words_reshaped = trg_reshape(next_words)

        next_words_bkw = Input(name=self.ids_inputs[2], batch_shape=tuple([None, params['SECOND_DIM_SIZE'], None]), dtype='int32')
        next_words_bkw_reshaped = trg_reshape(next_words_bkw)
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_reshaped)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw_reshaped)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)
        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []
        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3], batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_TRG_INPUT_TEXT_LEN']]), dtype='int32')

        #reshape to 2d
        trg_words_reshaped = trg_reshape(trg_words)
        next_words_one_hot = one_hot(trg_words_reshaped, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], trainable=self.trainable_est, return_sequences=True, return_state=True,
                                         name='qe_frw')(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], trainable=self.trainable_est, return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw')(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est, name='last_state_concat')

        seq_concat = concatenate([enc_qe_frw, enc_qe_bkw], trainable=self.trainable_est, name='seq_concat')

        trg_annotations = seq_concat
        annotations = NonMasking()(trg_annotations)

        # reshape back to 3d
        annotations_reshape = Reshape((params['SECOND_DIM_SIZE'], -1, params['QE_VECTOR_SIZE']*2))
        annotations_reshaped = annotations_reshape(annotations)

        #summarize phrase representations (average by default)
        merge_mode = params.get('WORD_MERGE_MODE', None)
        merge_words = Lambda(mask_aware_mean4d, mask_aware_merge_output_shape4d)
        if merge_mode == 'sum':
            merge_words = Lambda(sum4d, mask_aware_merge_output_shape4d)

        output_annotations = merge_words(annotations_reshaped)
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        phrase_qe = TimeDistributed(Dense(params['PHRASE_QE_CLASSES'], activation=out_activation), trainable=self.trainable_est, name=self.ids_outputs[0])(output_annotations)

        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[phrase_qe])


    #==================================================
    # Word-level QE -- POSTECH-inspired Estimator model
    #==================================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, words))
    # 2. One-position left-shifted machine-translated sentences to represent the right context (shape: (mini_batch_size, words))
    # 3. One-position rigth-shifted machine-translated sentences to represent the left context (shape: (mini_batch_size, words))
    # 4. Unshifted machine-translated sentences for evaluation (shape: (mini_batch_size, words))
    #
    ## Output:
    # 1. Word quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## References:
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.

    def EstimatorWord(self, params):
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        #trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')

        next_words_bkw = Input(name=self.ids_inputs[2], batch_shape=tuple([None, None]), dtype='int32')
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)
        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []
        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3], batch_shape=tuple([None, None]), dtype='int32')
        next_words_one_hot = one_hot(trg_words, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], trainable=self.trainable_est, return_sequences=True, return_state=True,
                                         name='qe_frw')(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], trainable=self.trainable_est, return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw')(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est, name='last_state_concat')

        seq_concat = concatenate([enc_qe_frw, enc_qe_bkw], trainable=self.trainable_est, name='seq_concat')

        # uncomment for Post QE
        # fin_seq = concatenate([seq_concat, merged_states])
        #qe_sent = Dense(1, activation='sigmoid', name=self.ids_outputs[0])(last_state_concat)
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        word_qe = TimeDistributed(Dense(params['WORD_QE_CLASSES'], activation=out_activation), trainable=self.trainable_est, name=self.ids_outputs[0])(seq_concat)

        # self.model = Model(inputs=[src_text, next_words, next_words_bkw], outputs=[merged_states,softout, softoutQE])
        # if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
        #     self.model.add_loss(alpha_regularizer)
        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[word_qe])




    #===============================================================================
    # Word-level QE -- simplified RNN POSTECH model inspired by Jhaveri et al., 2018
    #===============================================================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, words))
    # 2. Machine-translated sentences (shape: (mini_batch_size, words))
    #
    ## Output:
    # 1. Word quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## References:
    # - Nisarg Jhaveri, Manish Gupta, and Vasudeva Varman. 2018. Translation quality estimation for indian languages. In Proceedings of th 21st International Conference of the European Association for Machine Transla- tion (EAMT).

    def EncWordAtt(self, params):
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        #trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')

        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(trg_words)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        word_qe = TimeDistributed(Dense(params['WORD_QE_CLASSES'], activation=out_activation), trainable=self.trainable, name=self.ids_outputs[0])(proj_h)

        # self.model = Model(inputs=[src_text, next_words, next_words_bkw], outputs=[merged_states,softout, softoutQE])
        # if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
        #     self.model.add_loss(alpha_regularizer)
        self.model = Model(inputs=[src_text, trg_words],
                           outputs=[word_qe])

    #================================
    # POSTECH-inspired Predictor model
    #================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, words))
    # 2. One-position left-shifted reference sentences to represent the right context (shape: (mini_batch_size, words))
    # 3. One-position rigth-shifted reference sentences to represent the left context (shape: (mini_batch_size, words))
    #
    ## Output:
    # 1. Machine-translated sentences (shape: (mini_batch_size, output_vocabulary_size))
    #
    ## References
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.

    def Predictor(self, params):
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable,name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True, trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')
        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable, name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')

        next_words_bkw = Input(name=self.ids_inputs[2], batch_shape=tuple([None, None]), dtype='int32')
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'],trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True,
                                                                     trainable=self.trainable,
                                                                     name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                         kernel_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         recurrent_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         bias_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                         recurrent_dropout=params[
                                                             'RECURRENT_DROPOUT_P'],
                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                         recurrent_initializer=params['INNER_INIT'],
                                                         return_sequences=True,
                                                         trainable=self.trainable,
                                                         go_backwards=True,
                                                         name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable, name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable, name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2),trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True, name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable, shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)
        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []

        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE'+params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE'+self.ids_outputs[0])

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable = self.trainable),
                                         name=self.ids_outputs[0])

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        self.model = Model(inputs=[src_text, next_words, next_words_bkw], outputs=[softout])
        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
            self.model.add_loss(alpha_regularizer)


    # ------------------------------------------------------- #
    #       END OF PREDEFINED MODELS
    # ------------------------------------------------------- #




def slice2d(x, dim, index):
    return x[:, index * dim: dim * (index + 1),:]


def get_slices2d(x, n):
    dim = 1
    return [Lambda(slice2d, arguments={'dim': dim, 'index': i}, output_shape=lambda s: (s[0], dim, s[2]))(x) for i in range(n)]


def slice3d(x, dim, index):
    return x[:, :, index * dim: dim * (index + 1)]

def get_slices3d(x, n):
    dim = int(K.int_shape(x)[2] / n)
    return [Lambda(slice3d, arguments={'dim': dim, 'index': i}, output_shape=lambda s: (s[0], s[1], dim))(x) for i in range(n)]

def merge(x, params, dim):
    return Lambda(lambda x: K.stack(x,axis=1), output_shape=(params['MAX_OUTPUT_TEXT_LEN'], dim * 2))(x)

def one_hot(x, params):
    return Lambda(lambda x: K.one_hot(x, params['OUTPUT_VOCABULARY_SIZE']), output_shape=(None, params['OUTPUT_VOCABULARY_SIZE']))(x)

def get_last_state(x, params):
    a = x[:, -1, :]
    return Lambda(lambda x: x[:, -1, :], output_shape=(1, params['QE_VECTOR_SIZE']*2))(x)

def max(x, params):
    return Lambda(lambda x: K.max(x, axis=2), output_shape=(None,params['MAX_OUTPUT_TEXT_LEN'], params['MAX_OUTPUT_TEXT_LEN']))(x)

def concat_time_distributed(input):
    a = input[0]
    b = input[1]
    return K.concatenate([a, b], axis=2)



class ShiftedConcat(Layer):

    def __init__(self, output_dim, params, reverse=False, **kwargs):
        self.output_dim = output_dim
        self.supports_masking = True
        self.params = params
        self.reverse = reverse
        super(ShiftedConcat, self).__init__(**kwargs)

    def call(self, x):
        seq1 = x[0]
        seq2 = x[1]

        if self.reverse:
            seq2= Reverse(seq2._keras_shape[2],axes=1)(seq2)

        # slice outputs per word
        sliced1 = get_slices2d(seq1, self.params['MAX_OUTPUT_TEXT_LEN'])
        sliced2 = get_slices2d(seq2, self.params['MAX_OUTPUT_TEXT_LEN'])

        states_merged = []

        #a loop over words
        for i in range(len(sliced1)):
            #for the first word and last words zeroes
            state_before = MyZeroesLayer()(sliced1[i])
            state_after = MyZeroesLayer()(sliced1[i])

            if i != 0:
                state_before = sliced1[i - 1]

            # for the last word EOS merge two identic states and embeddings
            if i < len(sliced1) - 1:
                state_after = sliced2[i + 1]

            # concatenate trg states, target embeddings
            state_merged = concatenate([state_before, state_after], axis=2)
            state_merged = K.batch_flatten(state_merged)
            states_merged.append(state_merged)

        return merge(states_merged, self.params, self.output_dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        else:
            mask= None
        return mask


class DenseTranspose(Layer):

  def __init__(self, output_dim, other_layer, other_layer_name, **kwargs):
      self.output_dim = output_dim
      self.other_layer=other_layer
      self.other_layer_name = other_layer_name
      super(DenseTranspose, self).__init__(**kwargs)

  def call(self, x):
      # w = self.other_layer.get_layer(self.other_layer_name).layer.kernel
      w = self.other_layer.layer.kernel
      w_trans = K.transpose(w)
      return K.dot(x, w_trans)

  def compute_output_shape(self, input_shape):
      return (input_shape[0], input_shape[1], self.output_dim)


class Reverse(Layer):

    def __init__(self, output_dim, axes, **kwargs):
        self.output_dim = output_dim
        self.axes = axes
        self.supports_masking = True
        super(Reverse, self).__init__(**kwargs)

    def call(self, x):
        return K.reverse(x, axes=self.axes)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        else:
            mask= None
        return mask


class MyZeroesLayer(Layer):

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MyZeroesLayer, self).__init__(**kwargs)

    def call(self, x):
        return K.zeros_like(x)

    def compute_output_shape(self, input_shape):
        return (input_shape)

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        else:
            mask= None
        return mask


class GeneralReshape(Layer):

    def __init__(self, output_dim, params, **kwargs):
        self.output_dim = output_dim
        self.params = params
        super(GeneralReshape, self).__init__(**kwargs)

    def call(self, x):
        if len(self.output_dim)==2:
            return K.reshape(x, (-1, self.params['MAX_INPUT_TEXT_LEN']))
        if len(self.output_dim)==3:
            return K.reshape(x, (-1, self.output_dim[1], self.output_dim[2]))
        if len(self.output_dim)==4:
            return K.reshape(x, (-1, self.output_dim[1], self.output_dim[2], self.output_dim[3]))

    def compute_output_shape(self, input_shape):
        return self.output_dim


def attention_3d_block(inputs, params, ext):
    '''
    simple attention: weights over time steps; as in https://github.com/philipperemy/keras-attention-mechanism
    '''
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]

    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax', name='soft_att' + ext)(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction' + ext, output_shape=(TIME_STEPS,))(a)
    a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec' + ext)(a)

    output_attention_mul = multiply([inputs, a_probs], name='attention_mul' + ext)
    sum = Lambda(reduce_sum, mask_aware_mean_output_shape)
    output = sum(output_attention_mul)

    return output


def reduce_max(x):
    return K.max(x, axis=1, keepdims=False)


def reduce_sum(x):
    return K.sum(x, axis=1, keepdims=False)


def visual_feature_reshape(inputs, params, ext, dim=3):
    '''
    Reshapes the visual features in order to be merged with text features
    '''
    # inputs.shape = (batch_size, 1, red_visual_feature)
    if params['MODEL_TYPE'] == 'EncDocVis':
        input_dim = params['SECOND_DIM_SIZE'] #K.int_shape(inputs)[2]
        if dim == 2:
            factor = 10
    else:
        pass


    vis_feature_shape = K.int_shape(inputs)[1]

    if (dim == 3) and (params['MODEL_TYPE'] == 'EncDocVis'):
        max_text_len_dim = params['MAX_INPUT_TEXT_LEN']

        a = Lambda(lambda x: K.reshape(x, (1, -1, vis_feature_shape)), output_shape=(-1, vis_feature_shape))(inputs)
        a = Permute((2, 1))(a)
        a = Lambda(lambda x: K.tile(x, n = (1, input_dim * max_text_len_dim, 1)),
                    output_shape=(vis_feature_shape * input_dim * max_text_len_dim, -1), name='tile' + ext)(a)

        a_perm = Permute((2, 1), name='visual_feature_reshape' + ext)(a)
        a_cut = Reshape((-1, vis_feature_shape * max_text_len_dim))(a_perm)
        output = Lambda(lambda x: K.reshape(x, (-1, max_text_len_dim, vis_feature_shape)), output_shape=(max_text_len_dim, vis_feature_shape))(a_cut)

        print("output:", K.int_shape(output))
    #print("type of output:", type(output)) #'theano.tensor.var.TensorVariable'

    elif (dim == 3) or (params['MODEL_TYPE'] == 'EncBertSentVis'): # For the EncSentVis model
        max_text_len_dim = params['MAX_INPUT_TEXT_LEN']

        a = Lambda(lambda x: K.reshape(x, (1, -1, vis_feature_shape)), output_shape=(-1, vis_feature_shape))(inputs)
        a = Permute((2, 1))(a)
        a = Lambda(lambda x: K.tile(x, n = (1, max_text_len_dim, 1)),
                    output_shape=(vis_feature_shape * max_text_len_dim, -1), name='tile' + ext)(a)

        a_perm = Permute((2, 1), name='visual_feature_reshape' + ext)(a)
        a_cut = Reshape((-1, vis_feature_shape * max_text_len_dim))(a_perm)
        output = Lambda(lambda x: K.reshape(x, (-1, max_text_len_dim, vis_feature_shape)), output_shape=(max_text_len_dim, vis_feature_shape))(a_cut)

        print("output:", K.int_shape(output))

    else:
        a = Lambda(lambda x: K.reshape(x, (1, -1, vis_feature_shape)), output_shape=(-1, vis_feature_shape))(inputs)
        a = Permute((2, 1))(a)
        a = Lambda(lambda x: K.tile(x, n = (1, factor, 1)),
                    output_shape=(vis_feature_shape * factor, -1), name='tile' + ext)(a)

        a_perm = Permute((2, 1), name='visual_feature_reshape_' + ext)(a)
        a_cut = Reshape((-1, vis_feature_shape))(a_perm)
        output = Lambda(lambda x: K.reshape(x, (-1, vis_feature_shape)), output_shape=(vis_feature_shape,))(a_cut)

        print("output:", K.int_shape(output))

    return output

class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def mask_aware_mean(x):
    '''
    see: https://github.com/keras-team/keras/issues/1579
    '''
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)
    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)

    x_mean = K.sum(x, axis=1, keepdims=False)
    x_mean = x_mean / n

    return x_mean


def mask_aware_mean_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    return (shape[0], shape[2])


def mask_aware_mean4d(x):
    '''
    see: https://github.com/keras-team/keras/issues/1579
    '''
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=3, keepdims=True), 0)
    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=2, keepdims=False)

    x_mean = K.sum(x, axis=2, keepdims=False)
    x_mean = x_mean / n

    return x_mean

def sum4d(x):

    return K.sum(x, axis=2, keepdims=False)


def mask_aware_merge_output_shape4d(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    return (shape[0], shape[1], shape[3])
