#reading the parameters

echo "Analysing input parameters"

PARSED_OPTIONS=$(getopt -n "$0"  -o h --long "help,task:,target:,source:,score:,activation:,device:"  -- "$@")
 
if [ $# -eq 0 ];
then
  echo 'No argumants provided. Use --help option for more details.'
  exit 1
fi
 
eval set -- "$PARSED_OPTIONS"

while true;
do
  case "$1" in
 
    -h|--help)
     echo -e "usage $0 -h display help \n \
    --help display help \n \
    --task name of the folder containing the task \n \
    --source extension of the source language file \n \
    --target extension of the machine-translated file \n \
    --score extension of the file with predicted scores \n \
    --activation classification layer activation function set to ‘relu’ if predicted scores in (0, +infinity), ‘sigmoid’ -- (0, 1), or 'linear’ -- (-Infinity, +infinity) \n \
    --device cuda device "
      shift
      exit 0;;
   
    --task)
      if [ -n "$2" ];
      then
        task_name=$2
      fi
      shift 2;;

 
    --source)
      if [ -n "$2" ];
      then
        src=$2
      fi
      shift 2;;
   
   --target)
      if [ -n "$2" ];
      then
        trg=$2
      fi
      shift 2;;
    
   --score)
      if [ -n "$2" ];
      then
        score=$2
      fi
      shift 2;;
    
   --activation)
      if [ -n "$2" ];
      then
        out_activation=$2
      fi
      shift 2;;

  
   --device)
      if [ -n "$2" ];
      then
        device=$2
      fi
      shift 2;;

    --)
      shift;
      break;;
  esac
done


# we copy the base config
conf=config-sentQEbRNN.py
model_type=EncSent
model_name=${task_name}_${src}${trg}_${model_type}
patience=10

rm -rf config.*
ln -s ../configs/$conf ./config.py

echo "Traning the model "${model_name}
THEANO_FLAGS=device=$device python main.py TASK_NAME=$task_name DATASET_NAME=$task_name DATA_ROOT_PATH=examples/${task_name} SRC_LAN=${src} TRG_LAN=${trg} PRED_SCORE=$score OUT_ACTIVATION=$out_activation MODEL_TYPE=$model_type NEW_EVAL_ON_SETS=val PATIENCE=$patience SAVE_EACH_EVALUATION=True > log-${model_name}-prep.txt 2>&1

awk '/^$/ {nlstack=nlstack "\n";next;} {printf "%s",nlstack; nlstack=""; print;}' log-${model_name}-prep.txt > log-${model_name}.txt

best_epoch=$(tail -1 log-${model_name}.txt | tr ":" "\n" | tr ' ' '\n' | tail -3 | head -1)

echo 'Best model weights are dumped into 'trained_models/${model_name}/epoch_${best_epoch}_weights.h5

# pre-trained Weights + Vocab to use for scoring
pred_vocab=datasets/Dataset_${task_name}_${src}${trg}.pkl
pred_weights=trained_models/${model_name}/epoch_${best_epoch}_weights.h5

echo "Scoring test."${trg}

THEANO_FLAGS=device=$device python main.py TASK_NAME=$task_name DATASET_NAME=$task_name DATA_ROOT_PATH=examples/${task_name} SRC_LAN=${src} TRG_LAN=${trg} PRED_SCORE=$score OUT_ACTIVATION=$out_activation MODEL_TYPE=$model_type PRED_VOCAB=$pred_vocab PRED_WEIGHTS=$pred_weights MODE=sampling NEW_EVAL_ON_SETS=test PATIENCE=$patience SAVE_EACH_EVALUATION=True >> log-${model_name}.txt 2>&1

mv trained_models/${model_name}/test_epoch_0_output_0.pred trained_models/${model_name}/test_epoch_${best_epoch}_output_0.pred

echo "Model output in trained_models/"${model_name}"/test_epoch_"${best_epoch}"_output_0.pred"

