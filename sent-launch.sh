PARSED_OPTIONS=$(getopt -n "$0"  -o h --long "help,task:,data:,model:,vis:"  -- "$@")

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
     --data name of the data file (zip format; name without the .zip) \n \
     --vis use of visual features or not (bool)"
      shift
      exit 0;;

    --task)
      if [ -n "$2" ];
      then
        task_name=$2
      fi
      shift 2;;

    --data)
      if [ -n "$2" ];
      then
        data_name=$2
      fi
      shift 2;;

    --model)
      if [ -n "$2" ];
      then
        model_name=$2
      fi
      shift 2;;

    --vis)
      if [ -n "$2" ];
      then
        vis=$2
      fi
      shift 2;;

    --)
      shift;
      break;;
  esac
done

# Data name
datafile_name=${data_name}.zip

# Pre-trained model for POSTECH only

if [ "${model_name}" = 'POSTECH' ]; then
  mkdir -p quest/datasets
  mkdir -p quest/trained_models
  cp /data/jive/deepQuest/quest/fr-indomain/Dataset_wmt18-en-fr-predictor_enfr.pkl ./quest/datasets
  cp /data/jive/deepQuest/quest/fr-indomain/epoch_4_weights.h5 ./quest/trained_models
fi

### Move data files to the right folder in deepQuest ###
cd quest
mkdir -p examples/${task_name}
cd ..
cp ../${datafile_name} quest/examples/${task_name} # Change here for the data file path
#cp ../sentence_test.zip quest/examples/${task_name} # Example
cd quest/examples/${task_name}

# Unzip data files
unzip ${datafile_name}


# Move files out of the folder
mv ./${data_name}/* ./

# Return to deepQuest folder
cd ../../..


### Move shell and config files ###

# POSTECH models
if [ "${model_name}" = 'POSTECH' ]; then
  if [ "${vis}" = true ]; then # Multimodal document-level QE POSTECH models with visual features (EstimatorDocVis)
    cp train-test-sentQE${model_name}-vis.sh ./quest
    cp config-sentQE${model_name}-vis.py ./configs
  else # Baseline document-level POSTECH model (EstimatorDoc)
    cp train-test-sentQE${model_name}.sh ./quest
    cp config-sentQE${model_name}.py ./configs
  fi

# biRNN-BERT models
elif [ "${model_name}" = 'BERT' ]; then
  if [ "${vis}" = true ]; then
    cp config-sentQEBERTBiRNN-vis.py ./configs
    cp train-test-sentQEBERTBiRNN-vis.sh ./quest # Baseline EncBertSent (biRNN with BERT)
  else
    cp config-sentQEBERTBiRNN.py ./configs
    cp train-test-sentQEBERTBiRNN.sh ./quest # Baseline EncBertSent (biRNN with BERT)
  fi
  # BERT data processing
  cd quest/
  CUDA_VISIBLE_DEVICES=1 python data_engine/bert_processing.py ../config-sentQEBERTBiRNN.py

else

  # biRNN models
  if [ "${vis}" = true ]; then # Multimodal sentence-level QE biRNN models with visual features (EncSentVis)
    cp train-test-sentQEBiRNN-vis.sh ./quest
    cp config-sentQEBiRNN-vis.py ./configs
  else # Baseline sentence-level biRNN model (EncSent)
    cp config-sentQEBiRNN.py ./configs
    cp train-test-sentQEBiRNN.sh ./quest # Baseline EncSent
  fi
fi
