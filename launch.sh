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
    --model name of the model to use (biRNN by default) \n \
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
    mv train-test-docQE${model_name}-vis.sh ./quest
    mv config-docQE${model_name}-vis.py ./configs
  else # Baseline document-level POSTECH model (EstimatorDoc)
    mv train-test-docQE${model_name}.sh ./quest
    mv config-docQE${model_name}.py ./configs
  fi
else

  # biRNN models
  if [ "${vis}" = true ]; then # Multimodal document-level QE biRNN models with visual features (EncDocVis)
    mv train-test-docQEBiRNN-vis.sh ./quest
    mv config-docBiRNN-vis.py ./configs
  else # Baseline document-level biRNN model (EncDoc)
    mv train-test-docQEBiRNN.sh ./quest
    mv config-docBiRNN.py ./configs
  fi
fi
