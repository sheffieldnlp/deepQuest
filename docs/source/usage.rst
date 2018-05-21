=====
Usage
=====

Training
********

  The training parameters are defined in a configuration files (by default, named `config.py`), and a training itself is ran as follows:

  .. code:: bash 

    THEANO_FLAGS=device=cuda{1,0} python main.py --config <myConfig.py | tee -a log.txt 2>&1 &

  The following pre-set DeepQuest configurations are provided: 

    - **POSTECH-inspired** models: config-{Predictor-small, Predictor-large, sentQE, wordQE} (the last two sentence-level and word-level Estimators, respectively);
    - **POSTECH-inspired** Multi-Task Learning models: config-MTL and config-MTL2.py for 2 and 3 tasks, respectively; 
    - **BiRNN** models: config-{wordQEEnc, sentQEEnc};
    - Document-level models: config-{docQE, docQEAtt, docEnc, docEncAtt} for **POSTECH-inspired** last-state and attention models, **BiRNN** last-state and attention models, respectively. 
    
    If you re-use configs set accordingly:
  
    | ``TASK_NAME``: name given to your task; 
    | ``SRC_LAN``, ``TRG_LAN``: source and target language for your task;
    | ``DATA_ROOT_PATH``: directory where to find your data;
    | ``TEXT_FILES``: a (Python) dictionary that contains the names of your training, development and test sets (without extension).
 
    **NB**: for models other than Predictor takes a file with the extension '.mt' by default no matter what is set in ``TRG_LAN``, so that if Predictors and Estimators are trained consecutively (like in MTL) both reference file with the extension .TRG_LAN and .mt file can be stored in one folder

 
*******
Scoring
*******
Test sets are scored after each epoch using the standard tests of WMT shared task metrics (http://www.statmt.org/wmt18/quality-estimation-task.html) with an inbuilt procedure. The procedure to score new test sets with already trained models is to be implemented. 
