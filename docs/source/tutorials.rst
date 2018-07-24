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
  | ``phrase_qe`` -- for phrase-level quality Estimator
  | ``sent_qe`` -- for sentence-level quality Estimator
  | ``doc_qe`` -- for doc-level models


2. ``LOSS`` -- defines the loss function

  | ``categorical_crossentropy`` for Predictor (POSTECH architecture)
  | ``mse`` for QE models


3. ``MODEL_TYPE`` -- defines the type of the model to train 

  | POSTECH: Predictor, Estimator{Word, Phrase, Sent, Doc, DocAtt}
  | BiRNN: Enc{Word, PhraseAtt, Sent, Doc, DocAtt}
  | 
  | **Note**: document-level models take the last BiRNN states to produce the QE labels, while the document-level models with an Attention mechanism (DocAtt) take the sum of the BiRNN states, weighted by attention (see *model_zoo.py* for implementation details). EncPhraseAtt takes into account attended parts of source while estimating MT phrase quality (useful in the absence of phrase alignments).

4. Parameters per model type:

  | ``WORD_QE_CLASSES``, ``PHRASE_QE_CLASSES`` -- constantly set to 5, except for OK and BAD labels , since we have a set of standard labels related to padding and other pre-processing
  | ``SAMPLE_WEIGHTS`` -- to specify a dictionary using task names above, labels and their weights (for non-regression tasks, like word-level QE)
  | ``PRED_SCORE`` -- set as the extension of the tag file, (*e.g. ``PRED_SCORE`` = 'bleu'*), for both sentence and document-level QE, while for word-level QE, sets as 'tags' extension
  | ``SECOND_DIM_SIZE`` -- *(for phrase- and document-level QE only)* to fix the size of a document (*e.g.* to the maximum length of the most frequent quartile)
  | ``OUT_ACTIVATION`` -- set as 'relu' function if predicted scores are in (0, +infinity), as a 'sigmoid' function for scores in (0,1) (for example, BLEU or HTER), or as a linear' function for scores in (-infinity, +infinity).


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
  | ``MAX_SRC(TRG)_INPUT_TEXT_LEN`` -- longer sequences are cut to the specified length; set this length separately if different for source and MT inputs (for example, for phrase-level QE, when source sentences and MT phrases are given as inputs)
  | ``RELOAD`` = {epoch_number}, combined with ``RELOAD_EPOCH`` = True -- helpful when you want to continue training from a certain epoch, also a good idea to specify the vocabulary as previously pickeled (``PRED_VOCAB``)
  | ``OPTIMIZER`` = {optimizer}, also adjust the learning rate accordingly ``LR``
  | ``EARLY_STOP`` = True  -- activate early stopping with required ``PATIENCE`` = e.g. 5; set the right stop metric e.g. ``STOP_METRIC`` = e.g. 'pearson' (for regression QE tasks: alo 'mae', 'rmse'; for classification tasks: 'precision', 'recall', 'f1') 



Once all the training parameters are defined in the configuration file quest/config.py, one can run the training of the QE model as follows:

  .. code:: bash 

    export KERAS_BACKEND=theano
    export MKL_THREADING_LAYER=GNU
    THEANO_FLAGS=device={device_name} python main.py | tee -a /tmp/deepQuest.log 2>&1 &

One can observe the progression of the training in the log file created in the temporary directory.


Scoring
*******

Test sets are scored after each epoch using the standard tests from the `WMT QE Shared task`_ metrics, with an inbuilt procedure.
New test sets with already trained models can be scored by launching the same command as for training. Change the following parameters in your initial config (see `configs/config-sentQEbRNNEval.py`_ for an example, for now the scoring procedure is tested only for the sentence-level QE models):

  | ``EVAL_ON_SETS`` -- specify the set for scoring
  | ``PRED_VOCAB`` -- set the path to the vocabulary of the pre-trained model (as dumped to the datasets/Dataset_{task_name}_{src_extension}{trg_extension}.pkl folder)
  | ``PRED_WEIGHTS`` -- set the path to the pre-trained weights (as dumped to the trained_models/{model_name} folder) of the model that would be used for scoring
  | ``MODE`` -- set to 'sampling'
 
**Note** that the scoring procedure requires a file with gold-standard labels. Create a dummy file with, for example, zero scores if you do not have gold-standard labels. Assuming your machine-translated file is test.mt and you want to generate dummy HTER scores:

 .. code:: bash 

   for i in `seq $(wc -l test.mt | cut -d ' ' -f 1)`; do echo "0.0000"; done > test.hter


.. _`WMT QE Shared task`: http://www.statmt.org/wmt18/quality-estimation-task.html
.. _configs/config-sentQEbRNNEval.py: https://github.com/sheffieldnlp/deepQuest/blob/master/configs/config-sentQEbRNNEval.py

Examples
********

We also provide two scripts to train and test Sentence QE models for biRNN and POSTECH (`configs/train-test-sentQEbRNN.sh`_ and `configs/train-test-sentQEPostech.sh`_ respectively). Assuming that correct environment is already activated and all the environmental variables are set:

1. Sentence QE data in the format compatible for deepQuest could be downloaded, for example, from the `WMT QE Shared task 2017`_ page. Download the `task1_en-de_training-dev.tar.gz`_, `task1_en-de_test.tar.gz`_ and `wmt17_en-de_gold.tar.gz`_ archives. Make sure to get original version of the data and not the latest version they were replaced with. Create the folder examples/qe-2017 in the quest directory and unarchive all the three archives into the folder. Execute the following commands to rename the 2017 test data:

 .. code:: bash
  
   cd examples/qe-2017
   rename 's/^test.2017/test/' *
   mv en-de_task1_test.2017.hter test.hter

2. Copy the necessary BiRNN shell script to the 'quest' folder. Launch the script from the 'quest' folder. Specify the name of the folder, extensions of the source and machine-translated files, as well the cuda device (specify 'cpu' to train on cpus):

 .. code:: bash
 
   cd deepQuest/quest
   cp ../configs/train-test-sentQEbRNN.sh .  
   ./train-test-sentQEbRNN.sh --task qe-2017 --source src --target mt --score hter --activation sigmoid --device cuda0 > log-sentQEbRNN-qe-2017.txt 2>&1 &

The complete log is in quest/log-qe-2016_srcmt_EncSent.txt.
The log log-sentQEbRNN-qe-2017.txt should show results comparable to the ones below:


 .. code:: bash
   
   cat log-sentQEbRNN-qe-2017.txt
   
   Analysing input parameters
   Traning the model qe-2017_srcmt_EncSent
   Best model weights are dumped into saved_models/qe-2017_srcmt_EncSent/epoch_12_weights.h5
   Scoring test.mt
   Model output in trained_models/qe-2017_srcmt_EncSent/test_epoch_12_output_0.pred
   Evaluations results
   [24/07/2018 12:08:33] **SentQE**
   [24/07/2018 12:08:33] Pearson 0.3871
   [24/07/2018 12:08:33] MAE 0.1380
   [24/07/2018 12:08:33] RMSE 0.1819

**Note** If you try to launch the scripts with your data and you do not have gold-standard labels for your test data cf. the respective note in the `Scoring`_ section.

For POSTECH Predictor pre-training, parallel data containing human reference translations should be prepared. For example, the `Europarl`_ corpus can be used. The data can be pre-proccesed in a standard `Moses`_ pipeline (Corpus Preparation section). Typically, around 2M of parallel lines are used for training and 3K lines for testing (small Predictor model).

We provide an example of the Postech architecture training using Europarl and WMT 2017 Sentence QE data:

1. Create a data directory and download the EN-DE Europarl data:

 .. code:: bash
  
   mkdir -p europarl/raw && cd "$_"
   wget http://opus.nlpl.eu/download.php?f=Europarl/de-en.txt.zip
   unzip download.php\?f=Europarl%2Fde-en.txt.zip

Create your copy of the Moses toolkit:
 
 .. code:: bash
   
   git clone https://github.com/moses-smt/mosesdecoder.git

Copy the preprocessing scripts provided with the deepQuest tool to your main data directory and launch the preprocessing scripts by specifying the data info and the Moses clone location. This step may take a while. 
  
 .. code:: bash

   cd /{your_path}/europarl
   cp deepQuest/configs/preprocess-data-predictor.sh ./
   cp deepQuest/configs/split.py ./
   ./preprocess-data-predictor.sh --name Europarl.de-en --source en --target de --dir /{your_path}/europarl --mosesdir /{your_path}/mosesdecoder

The final preprocessed data should look as follows:

 .. code:: bash

   wc -l /{your_path}/europarl/clean/en-de/*
  
   3000 clean/en-de/dev.de
   3000 clean/en-de/dev.en
   3000 clean/en-de/test.de
   3000 clean/en-de/test.en
   1862790 clean/en-de/train.de
   1862790 clean/en-de/train.en
   3737580 total


Copy the prepared data files into the quest data directory:

 .. code:: bash
  
   mkdir /{your_path}/quest/examples/europarl-en-de
   cp /{your_path}/europarl/clean/en-de/* /{your_path}/quest/examples/europarl-en-de

2. Launch the Postech script:

 .. code:: bash

   cd deepQuest/quest
   cp ../configs/train-test-sentQEPostech.sh .  
   ./train-test-sentQEPostech.sh --pred-task europarl-en-de --pred-source en --pred-target de --est-task qe-2017 --est-source src --est-target mt --score hter --activation sigmoid --device cuda0 > log-sentQEPostech-qe-2017.txt 2>&1 &

The complete logs are in quest/log-europarl-en-de_ende_Predictor.txt and quest/log-qe-2017_srcmt_EstimatorSent.txt
The log log-sentQEPostech-qe-2017.txt should show results comparable to the following ones:

 .. code:: bash
  
   cat log-sentQEPostech-qe-2017.txt
   
   Analysing input parameters
   Traning the model europarl-en-de_ende_Predictor
   Traning the model qe-2017_srcmt_EstimatorSent
   Best model weights are dumped into saved_models/qe-2017_srcmt_EstimatorSent/epoch_2_weights.h5
   Scoring test.mt
   Model output in trained_models/qe-2017_srcmt_EstimatorSent/test_epoch_2_output_0.pred
   Evaluations results
   [24/07/2018 10:52:50] Pearson 0.5102
   [24/07/2018 10:52:50] MAE 0.1261
   [24/07/2018 10:52:50] RMSE 0.1640
   [24/07/2018 10:52:50] Done evaluating on metric qe_metrics

.. _`Europarl`: http://opus.nlpl.eu/Europarl.php
.. _`WMT QE Shared task 2017`: http://www.statmt.org/wmt17/quality-estimation-task.html
.. _`configs/train-test-sentQEbRNN.sh`: https://github.com/sheffieldnlp/deepQuest/blob/master/configs/train-test-sentQEbRNN.sh
.. _`configs/train-test-sentQEPostech.sh`: https://github.com/sheffieldnlp/deepQuest/blob/master/configs/train-test-sentQEPostech.sh
.. _`Moses`: http://www.statmt.org/moses/?n=Moses.Baseline
.. _`task1_en-de_training-dev.tar.gz`: https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-1974
.. _`task1_en-de_test.tar.gz`: https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-2135
.. _`wmt17_en-de_gold.tar.gz`: http://www.quest.dcs.shef.ac.uk/wmt17_files_qe/wmt17_en-de_gold.tar.gz

