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
  | ``SRC_LAN``, ``TRG_LAN``: extensions of correspnding source language and MT files (target language file for Predictor);
  | ``DATA_ROOT_PATH``: directory where to find the data;
  | ``TEXT_FILES``: a (Python) dictionary that contains the names of the training, development and test sets (*without extension*).

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

    export KERAS_BACKEND=theano
    export MKL_THREADING_LAYER=GNU
    THEANO_FLAGS=device=cuda{1,0} python main.py --config config.py | tee -a /tmp/deepQuest.log 2>&1 &

One can observe the progression of the training in the log file created in the temporary directory.


Scoring
*******

Test sets are scored after each epoch using the standard tests from the `WMT QE Shared task`_ metrics, with an inbuilt procedure.
New test sets with already trained models can be scored by launching the same command as for training. Change the following parameters in your initial config (see `configs/config-sentQEbRNNEval.py`_ for an example, for now the scoring procedure is tested only for the sentence-level QE models):

  | ``EVAL_ON_SETS`` -- specify the set for scoring
  | ``PRED_VOCAB`` -- set the path to the vocabulary of the pre-trained model (as dumped to the datasets/Dataset_{task_name}_{src_extension}{trg_extension}.pkl folder)
  | ``PRED_WEIGHTS`` -- set the path to the pre-trained weights (as dumped to the trained_models/{model_name} folder) of the model that would be used for scoring
  | ``MODE`` -- set to 'sampling'
 
Note that the scoring procedure requires a file with gold-standard labels. Create a dummy file with, for example, random scores if you do not have gold-standard labels.

.. _`WMT QE Shared task`: http://www.statmt.org/wmt18/quality-estimation-task.html
.. _configs/config-sentQEbRNNEval.py: https://github.com/sheffieldnlp/deepQuest/blob/master/configs/config-sentQEbRNNEval.py

Examples
********

We also provide two scripts to train and test Sentence QE models for biRNN and POSTECH (`configs/train-test-sentQEbRNN.sh`_ and `configs/train-test-sentQEPostech.sh`_ respectively). Assuming that correct environment is already activated and all the environmental variables are set:

1. Copy the necessary BiRNN shell script to the 'quest' folder.
2. Sentence QE data in the format compatible for deepQuest could be downloaded, for example, from the `WMT QE Shared task 2016`_ page. 

Make sure to put train, dev and test files in one folder, e.g., 'quest/examples/qe-2016' ('examples' folder is used for data storage).

3. Launch the script from the 'quest' folder. Specify the name of the folder, extensions of the source and machine-translated files, as well the cuda device (specify 'cpu' to train on cpus):

.. code:: bash
 
  cd deepQuest/quest
  cp ../configs/train-test-sentQEbRNN.sh .  
  ./train-test-sentQEbRNN.sh --task qe-2016 --source src --target mt --score hter --device cuda0 &

The corresponding log is in quest/log-qe-2016_srcmt_EncSent.txt

The script will output the information on the number of the best epoch, e.g. 18.
The best model weights are in trained_models/qe-2016_srcmt_EncSent/epoch_18_weights.h5
The resulting test scores are in trained_models/qe-2016_srcmt_EncSent/test_epoch_18_output_0.pred

For POSTECH Predictor pre-training, parallel data containing human reference translations should be prepared. For example, the `Europarl`_ corpus can be used. The data can be pre-proccesed in a standard `Moses`_ pipeline (Corpus Preparation section). Typically, around 2M of parallel lines are used for training and 3K lines for testing (small Predictor model).

Assuming the Europarl training (train.{en,de}) and test data (test.{en,de}) are in 'quest/examples/europarl-en-de', launch the train-test-sentQEPostech.sh script:

.. code:: bash

  cd deepQuest/quest
  cp ../configs/train-test-sentQEPostech.sh .  
  ./train-test-sentQEPostech.sh --pred-task europarl-en-de --pred-source en --pred-target de --est-task qe-2016 --est-source src --est-target mt --score hter --device cuda0 &

The corresponding logs are in quest/log-europarl-en-de_ende_Predictor.txt and quest/log-qe-2016_srcmt_EstimatorSent.txt
The best model and test scores are stored as for BiRNN.

.. _`Europarl`: http://opus.nlpl.eu/Europarl.php
.. _`WMT QE Shared task 2016`: http://www.statmt.org/wmt16/quality-estimation-task.html
.. _`configs/train-test-sentQEbRNN.sh`: https://github.com/sheffieldnlp/deepQuest/blob/master/configs/train-test-sentQEbRNN.sh
.. _`configs/train-test-sentQEPostech.sh`: https://github.com/sheffieldnlp/deepQuest/blob/master/configs/train-test-sentQEPostech.sh
.. _`Moses`: http://www.statmt.org/moses/?n=Moses.Baseline

