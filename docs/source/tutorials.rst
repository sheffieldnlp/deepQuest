=========
Tutorials
=========

Training
********

This section provides a detailed description of DeepQuest-related config parameters.
 
1. INPUTS_IDS_DATASETS:

    | ``source_text`` -- source text 
    | ``state_below`` -- target text (reference for Predictor, MT for Estimator) one position right-shifted target text (for left POSTECH context, the same as previous word with NMT-Keras Teacher)
    | ``state_above`` -- target text (reference for Predictor, MT for Estimator) one position left-shifted target text (for right POSTECH context, the same as next word with NMT-Keras Teacher)
    | ``target`` -- MT text unshifted to obtain Predictor scores for it 

  Only source_text and target_text inputs are used for biRNN models

2. For outputs of single-task models set an output in ``OUTPUTS_IDS_DATASET`` from the following (+ set ``MULTI_TASK=False``, keep pre-set task names):

| ``target_text`` -- for Predictor, Predictor training can be stopped after 2-3 epochs as soon as the quality in BLEU will stop improving
| ``word_qe`` -- for word-level quality Estimator
| ``sent_qe`` -- for sentence-level quality Estimator
| ``doc_qe`` -- for doc-level models


3. Set ``LOSS``:

``categorical_crossentropy`` -- for Predictor or Word QE

``mse`` - for sentence-level and document-level QE (regression optimization)

4. Set ``MODEL_TYPE``: 

Predictor, Estimator{Sent,Word,Doc,DocAtt}

for bi-RNN: Enc{Word,Sent,Doc,DocAtt}

Doc models take the last Bi-RNN states to produce QE labels;
DocAtt take the sum of Bi-RNN states weighted by attention

see model_zoo-{POSTECH,BiRRN}.py for more details

5. Parameters per model type:


- ``WORD_QE_CLASSES`` are constantly set to 5, since except for OK and BAD labels we have a set of standard labels related to padding and other pre-processing

- ``SAMPLE_WEIGHTS`` -- specify a dictionary using task names above, labels and their weights (for non-regression tasks, like Word QE)

- For Sent and Doc levels set ``PRED_SCORE`` for the extension of the tag file, e.g. ``PRED_SCORE`` = 'bleu'; for Word QE takes 'tags' extension 

- only for Doc level:

set ``DOC_SIZE`` -- fix the size of a document (eg. to the maximum length of the most frequent quartile)
set ``DOC_ACTIVATION`` depending on the type of score. 'relu' if the score in (0, +infinity), 'sigmoid'(for BLEU, HTER)  -- (0,1); 'linear' -- (-Infinity,+infinity)


6. For multiple outputs for MTL set their order in ``OUTPUTS_IDS_DATASET_FULL``
Standard order of tasks: ``target_text``, ``word_qe``, ``sent_qe`` (``LOSS`` and ``MODEL_TYPE`` will be ignored). The MTL will first pre-train Estimator sentence-level and word-level weights (keeping Predictor weights unchanged). 

For MTL also set:
   ``MULTI_TASK`` = True
   
   ``EPOCH_PER_UPDATE`` = 1 -- times every task is consequently repeated (each of N epochs as specified by the parameters below)
   
   ``EPOCH_PER_PRED`` = 5 -- Predictor epochs
   
   ``EPOCH_PER_EST_SENT`` = 5 -- EstimatorSent epochs 
   
   ``EPOCH_PER_EST_WORD`` = 5 -- EstimatorWord epochs


7. Neural network parameters (should be kept the same for the large Predictor training and then MTL learning). 

For a small **POSTECH-inspired** model the following parameters should be used:

 ``IN{OUT}PUT_VOCABULARY_SIZE`` = 30000 
 
 ``SOURCE{TARGET}_TEXT_EMBEDDING_SIZE`` = 300 
 
 ``EN{DE}CODER_HIDDEN_SIZE`` = 500 
 
 ``QE_VECTOR_SIZE`` = 75 
 
For a large **POSTECH-inspired** model:
 
 ``IN{OUT}PUT_VOCABULARY_SIZE`` = 70000
 
 ``SOURCE{TARGET}_TEXT_EMBEDDING_SIZE`` = 500
 
 ``EN{DE}CODER_HIDDEN_SIZE`` = 700
 
 ``QE_VECTOR_SIZE`` = 100

For doc level:

``DOC_DECODER_HIDDEN_SIZE`` = 50

For BiRNN models:

``ENCODER_HIDDEN_SIZE`` = 50

8. Other training-related parameters:


- ``PRED_VOCAB`` -- set the dictionary pickle dumped by the pre-trained model (dumped to the datasets folder)

- ``PRED_WEIGHTS`` -- set the pre-trained weights (as dumped to the trained_models/{model_name} folder)


- ``BATCH_SIZE`` -- typically 50 or 70 for smaller models; set to 5 for doc QE

- ``MAX_EPOCH`` -- max epochs the code will run (for MTL max quantity of iterations over all the three tasks)

- ``MAX_IN(OUT)PUT_TEXT_LEN`` -- longer sequences are cut to the specified length

- ``RELOAD`` = {epoch_number}, combined with ``RELOAD_EPOCH`` = True -- helpful when you want to continue training from a certain epoch, also a good idea to specify the vocabulary as previously pickeled (``PRED_VOCAB``)

- ``OPTIMIZER`` = {optimizer}, also adjust the learning rate accordingly ``LR``

- ``EARLY_STOP`` = True  -- activate early stopping with required ``PATIENCE`` = e.g. 5; set the right stop metric e.g. ``STOP_METRIC`` = e.g. 'pearson' (for regression QE tasks: alo 'mae', 'rmse'; for classification tasks: 'precision', 'recall', 'f1') 
