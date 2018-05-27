========
Tutorial
========

Training a QE model
*******************

One can use DeepQuest to train QE models at either word, sentence or document-level.
In its current version, DeepQuest provides the following, multi-level, QE models:

  - **POSTECH**: a two-stage end-to-end stacked neural architecture that combines a *Predictor* and an *Estimator*, designed by `Kim et al., 2017`_.
  - **BiRNN**: simple architecture relying on two bi-directional RNNs, designed by `Ive et al., 2018`_. 

.. _Kim et al., 2017: https://dl.acm.org/citation.cfm?id=3109480
.. _Ive et al., 2018: 

Depending on the desired level of prediction, the configuration will differ, and this section aims to give a detailed description of the customised parameters.

The first step is to create a configuration file (see `configs/example_config-WordQE.py`_ for an example), which defines the parameters of the model to train, starting with the definition of the task:

.. _configs/example_config-WordQE.py: https://github.com/sheffieldnlp/deepQuest/blob/master/configs/example_config-WordQE.py

  | ``TASK_NAME``: name given to the task; 
  | ``SRC_LAN``, ``TRG_LAN``: source and target language of the task;
  | ``DATA_ROOT_PATH``: directory where to find the data;
  | ``TEXT_FILES``: a (Python) dictionary that contains the names of the training, development and test sets (*without extension*).
  |
  | **Note**: for models other than Predictor takes a file with the extension '.mt' by default no matter what is set in ``TRG_LAN``, so that if Predictors and Estimators are trained consecutively (like in a Multi-Task Learning setting) both reference file with the extension .TRG_LAN and .mt file can be stored in a single folder.


0. ``INPUTS_IDS_DATASETS`` -- defines the datasets used to train the QE model

  | ``source_text`` -- source text 
  | ``state_below`` -- target text (reference for Predictor, MT for Estimator) one position right-shifted target text (for left POSTECH context, the same as previous word with NMT-Keras Teacher)
  | ``state_above`` -- target text (reference for Predictor, MT for Estimator) one position left-shifted target text (for right POSTECH context, the same as next word with NMT-Keras Teacher)
  | ``target`` -- MT text unshifted to obtain Predictor scores for it 
  |
  | **Note**: only ``source_text`` and ``target_text`` inputs are used for biRNN models.


1. For outputs of single-task models set an output in ``OUTPUTS_IDS_DATASET`` from the following (+ set ``MULTI_TASK=False``, keep pre-set task names):

  | ``target_text`` -- for Predictor, Predictor training can be stopped after 2-3 epochs as soon as the quality in BLEU will stop improving
  | ``word_qe`` -- for word-level quality Estimator
  | ``sent_qe`` -- for sentence-level quality Estimator
  | ``doc_qe`` -- for doc-level models


2. ``LOSS`` -- defines the loss function

  | ``categorical_crossentropy`` for Predictor (POSTECH architecture), or word-level QE
  | ``mse`` for sentence-level and document-level QE (regression optimization)


3. ``MODEL_TYPE`` -- defines the type of the model to train 

  | POSTECH: Predictor, Estimator{Word, Sent, Doc, DocAtt}
  | BiRNN: Enc{Word, Sent, Doc, DocAtt}
  | 
  | **Note**: document-level models take the last BiRNN states to produce the QE labels, while the document-level models with a Attention mechanism (DocAtt) take the sum of the BiRNN states, weighted by attention (see *model_zoo.py* for implementation details).


4. Parameters per model type:

  | ``WORD_QE_CLASSES`` -- constantly set to 5, except for OK and BAD labels , since we have a set of standard labels related to padding and other pre-processing
  | ``SAMPLE_WEIGHTS`` -- to specify a dictionary using task names above, labels and their weights (for non-regression tasks, like word-level QE)
  | ``PRED_SCORE`` -- set as the extension of the tag file, (*e.g. ``PRED_SCORE`` = 'bleu'*), for both sentence and document-level QE, while for word-level QE, sets as 'tags' extension
  | ``DOC_SIZE`` -- *(for document-level QE only)* to fix the size of a document (*e.g.* to the maximum length of the most frequent quartile)
  | ``DOC_ACTIVATION`` -- *(for document-level QE only)* set as 'relu' function if scores between (0, +infinity), as a 'sigmoid' function for scores similar to metrics such ash BLEU or HTER (*i.e.* between 0 and 1), or as a linear' function for scores between (-Infinity, +infinity).


5. ``MULTI_TASK`` -- Multi-Tasks Learning (MTL) (POSTECH model only)
  
  | ``MULTI_TASK`` = True / False, to activate / deactivate MTL
   
  | ``OUTPUTS_IDS_DATASET_FULL`` -- defines order for multiple outputs for Multi-Tasks Learning (MTL)
  | Standard order of tasks: ``target_text``, ``word_qe``, ``sent_qe`` (``LOSS`` and ``MODEL_TYPE`` will be ignored).
  | The MTL will first pre-train the word-level weigths (keeping Predictor weights unchanged), and the *Estimator* (sentence-level). 

  | ``EPOCH_PER_UPDATE`` = 1 -- times every task is consequently repeated (each of N epochs as specified by the parameters below)
  | ``EPOCH_PER_PRED`` = 5 -- Predictor epochs
  | ``EPOCH_PER_EST_SENT`` = 5 -- EstimatorSent epochs 
  | ``EPOCH_PER_EST_WORD`` = 5 -- EstimatorWord epochs


6. Neural network parameters (should be kept the same for the large Predictor training and then MTL learning). 

  | For a small **POSTECH-inspired** model the following parameters should be used:

  | ``IN{OUT}PUT_VOCABULARY_SIZE`` = 30000 
  | ``SOURCE{TARGET}_TEXT_EMBEDDING_SIZE`` = 300 
  | ``EN{DE}CODER_HIDDEN_SIZE`` = 500 
  | ``QE_VECTOR_SIZE`` = 75 

  | For a large **POSTECH-inspired** model:

  | ``IN{OUT}PUT_VOCABULARY_SIZE`` = 70000
  | ``SOURCE{TARGET}_TEXT_EMBEDDING_SIZE`` = 500
  | ``EN{DE}CODER_HIDDEN_SIZE`` = 700
  | ``QE_VECTOR_SIZE`` = 100

  | For document-level QE : ``DOC_DECODER_HIDDEN_SIZE`` = 50

  | For BiRNN models: ``ENCODER_HIDDEN_SIZE`` = 50

7. Other training-related parameters:

  | ``PRED_VOCAB`` -- set the dictionary pickle dumped by the pre-trained model (dumped to the datasets folder)
  | ``PRED_WEIGHTS`` -- set the pre-trained weights (as dumped to the trained_models/{model_name} folder)
  | ``BATCH_SIZE`` -- typically 50 or 70 for smaller models; set to 5 for doc QE
  | ``MAX_EPOCH`` -- max epochs the code will run (for MTL max quantity of iterations over all the three tasks)
  | ``MAX_IN(OUT)PUT_TEXT_LEN`` -- longer sequences are cut to the specified length
  | ``RELOAD`` = {epoch_number}, combined with ``RELOAD_EPOCH`` = True -- helpful when you want to continue training from a certain epoch, also a good idea to specify the vocabulary as previously pickeled (``PRED_VOCAB``)
  | ``OPTIMIZER`` = {optimizer}, also adjust the learning rate accordingly ``LR``
  | ``EARLY_STOP`` = True  -- activate early stopping with required ``PATIENCE`` = e.g. 5; set the right stop metric e.g. ``STOP_METRIC`` = e.g. 'pearson' (for regression QE tasks: alo 'mae', 'rmse'; for classification tasks: 'precision', 'recall', 'f1') 



Once all the training parameters are defined in the configuration file, one can run the training of the QE model as follows:

  .. code:: bash 

    THEANO_FLAGS=device=cuda{1,0} python main.py --config config.py |tee -a /tmp/deepQuest.log 2>&1 &

One can observe the progression of the training in the log file created in the temporary directory.


Scoring
*******

Test sets are scored after each epoch using the standard tests from the `WMT QE Shared task`_ metrics, with an inbuilt procedure.
The procedure to score new test sets with already trained models, is to be implemented. 

.. _`WMT QE Shared task`: http://www.statmt.org/wmt18/quality-estimation-task.html
