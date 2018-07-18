cd ~/deepQuest/quest

device=cuda0
src=en
trg=es

# for COLING 4th quartile of the distribution of doc lengths; may be diffirent for \all and \filt configs
# by LP: EN-RU \all 28 \filt 28; DE-EN \all 34 \filt 34; EN-FR \all 36 \filt 34; EN-ES \all 43 \filt 45
# \all
doc_size=43
task_name=english-spanish-all
# activation depending on predicted score ReLU --(0, +infinity), Sigmoid -- (0,1); Linear -- (-Infinity,+infinity)
doc_activation=sigmoid
# config for doc-level POSTECH
conf=config-docQEPostech.py
# config for doc level biRNN
# conf=config-docbRNN.py
epochs_for_save=5

# POSTECH pre-trained Predictor + Vocab 
pred_vocab=vocabs-coling/Dataset_predictor-en-es-euro-newscom-small_enes.pkl
pred_weights=weights_en-es-coling/epoch_2_weights.h5

cp $conf ./config.py

#for encoder models loop over: models=(EncDoc EncDocAtt)
models=(EstimatorDoc EstimatorDocAtt)
scores=(bleu wbleu tfbleu)

for model_type in ${models[@]} ;do

for score in ${scores[@]} ;do

model_name=${task_name}_${model_type}_${score}_${doc_activation}
# for encoder models do not parametrize PRED_WEIGHTS
echo THEANO_FLAGS=device=$device python main.py TASK_NAME=$task_name DATASET_NAME=$task_name DATA_ROOT_PATH=examples/${task_name} SRC_LAN=${src} TRG_LAN=${trg} PRED_SCORE=$score MODEL_TYPE=$model_type MODEL_NAME=${model_name} STORE_PATH=doc_models/${model_name} SECOND_DIM_SIZE=$doc_size PRED_VOCAB=$pred_vocab PRED_WEIGHTS=$pred_weights OUT_ACTIVATION=${doc_activation} EPOCHS_FOR_SAVE=${epochs_for_save} SAVE_EACH_EVALUATION=False

THEANO_FLAGS=device=$device python main.py TASK_NAME=$task_name DATASET_NAME=$task_name DATA_ROOT_PATH=examples/${task_name} SRC_LAN=${src} TRG_LAN=${trg} PRED_SCORE=$score MODEL_TYPE=$model_type MODEL_NAME=${model_name} STORE_PATH=doc_models/${model_name} SECOND_DIM_SIZE=$doc_size PRED_VOCAB=$pred_vocab PRED_WEIGHTS=$pred_weights OUT_ACTIVATION=${doc_activation} EPOCHS_FOR_SAVE=${epochs_for_save} SAVE_EACH_EVALUATION=False > log-${model_name}.txt 2>&1
done
done
