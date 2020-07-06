#reading the parameters

echo "Analysing input parameters"

PARSED_OPTIONS=$(getopt -n "$0"  -o h --long "help,task:,target:,source:,visual:,docsize:,score:,activation:,vis_strat:,vis_method:,attention:,seed:,device:"  -- "$@")

if [ $# -eq 0 ];
then
  echo 'No arguments provided. Use --help option for more details.'
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
    --docsize third quartile of the distribution of document length values (in sentences) for the training set\n \
    --score extension of the file with predicted scores \n \
    --activation classification layer activation function set to ‘relu’ if predicted scores in (0, +infinity), ‘sigmoid’ -- (0, 1), or 'linear’ -- (-Infinity, +infinity) \n \
    --attention true or false to use attention over document sentences or not \n \
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

   --visual)
       if [ -n "$2" ];
       then
         vis=$2
       fi
       shift 2;;

   --docsize)
      if [ -n "$2" ];
      then
        docsize=$2
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

   --vis_strat)
     if [ -n "$2" ];
     then
       vis_strat=$2
     fi
     shift 2;;

   --vis_method)
     if [ -n "$2" ];
     then
       vis_method=$2
     fi
     shift 2;;

   --attention)
      if [ -n "$2" ];
      then
        attention=$2
      fi
      shift 2;;

  --seed)
     if [ -n "$2" ];
     then
       seed=$2
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
if [ "${attention}" = true ]; then
	model_type=EncDocAtt
else
	model_type=EncDocVis
fi

conf=config-docBiRNN-vis.py
model_name=${task_name}_${src}${trg}_${model_type}
store_path=trained_models/${model_name}/
patience=10
batch_size=5 #10 #20
#random seed for reproducibility
rnd_seed=124
if [ $seed != 124 ]; then
	rnd_seed=$seed
fi

rm -rf config.*
ln -s ../configs/$conf ./config.py

echo "Traning the model "${model_name}
THEANO_FLAGS=device=$device PYTHONHASHSEED=0 python main.py TASK_NAME=$task_name DATASET_NAME=$task_name DATA_ROOT_PATH=examples/${task_name} SRC_LAN=${src} TRG_LAN=${trg} VISUAl_FEATURE=${vis} PRED_SCORE=$score OUT_ACTIVATION=$out_activation MODEL_TYPE=$model_type MODEL_NAME=$model_name STORE_PATH=$store_path SECOND_DIM_SIZE=$docsize VISUAL_FEATURE_STRATEGY=$vis_strat VISUAL_FEATURE_METHOD=$vis_method PATIENCE=$patience SAVE_EACH_EVALUATION=True RND_SEED=$rnd_seed BATCH_SIZE=$batch_size > log-${model_name}-prep.txt 2>&1

awk '/^$/ {nlstack=nlstack "\n";next;} {printf "%s",nlstack; nlstack=""; print;}' log-${model_name}-prep.txt > log-${model_name}.txt

best_epoch=$(tail -1 log-${model_name}.txt | tr ":" "\n" | tr ' ' '\n' | tail -3 | head -1)

# pre-trained Weights + Vocab to use for scoring
pred_vocab=saved_models/${model_name}/Dataset_${task_name}_${src}${trg}.pkl
pred_weights=saved_models/${model_name}/epoch_${best_epoch}_weights.h5

mkdir -p saved_models/${model_name}
cp datasets/Dataset_${task_name}_${src}${trg}.pkl saved_models/${model_name}
cp trained_models/${model_name}/epoch_${best_epoch}_weights.h5 saved_models/${model_name}

# Move new config file
rm -rf config.*
ln -s ../config-docBiRNN-vis-new.py ./config.py

echo 'Best model weights are dumped into 'saved_models/${model_name}/epoch_${best_epoch}_weights.h5

# Evaluation on the new test set
THEANO_FLAGS=device=$device PYTHONHASHSEED=0 python main.py TASK_NAME=$task_name DATASET_NAME=$task_name DATA_ROOT_PATH=examples/${task_name} SRC_LAN=${src} TRG_LAN=${trg} VISUAl_FEATURE=${vis} PRED_SCORE=$score OUT_ACTIVATION=$out_activation MODEL_TYPE=$model_type MODEL_NAME=$model_name STORE_PATH=$store_path SECOND_DIM_SIZE=$docsize PRED_VOCAB=$pred_vocab RELOAD=$best_epoch MODE=sampling NEW_EVAL_ON_SETS=test VISUAL_FEATURE_STRATEGY=$vis_strat VISUAL_FEATURE_METHOD=$vis_method PATIENCE=$patience SAVE_EACH_EVALUATION=True RND_SEED=$rnd_seed BATCH_SIZE=$batch_size > log-${model_name}.txt 2>&1


echo "Model output in trained_models/"${model_name}"/test_epoch_"${best_epoch}"_output_0.pred"
echo "Evaluations results"
tail -9 log-${model_name}.txt
