#!/usr/bin/env bash


echo "Analysing input parameters"

PARSED_OPTIONS=$(getopt -n "$0"  -o h --long "help,task:,target:,source:,name:,dir:,mosesdir:"  -- "$@")
 
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
    --name name of a raw file without extension: e.g., if you download europarl.en-fr.en and europarl.en-fr.fr put here europarl.en-fr \n \
    --source extension of the source language file \n \
    --target extension of the reference file \n \
    --dir data directory containing the raw folder \n \
    --mosesdir cloned Moses directory "
      shift
      exit 0;;
   
    --name)
      if [ -n "$2" ];
      then
        name=$2
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
        tgt=$2
      fi
      shift 2;;
    
   --dir)
      if [ -n "$2" ];
      then
        WDIR=$2
      fi
      shift 2;;
    
   --mosesdir)
      if [ -n "$2" ];
      then
        MOSES_DIR=$2
      fi
      shift 2;;

    --)
      shift;
      break;;
  esac
done

DDIR=${WDIR}/raw
ODIR=${WDIR}/clean/${src}-${tgt}
if [ ! -d ${ODIR} ]; then
  mkdir -p ${ODIR}
fi

# TOOLS variables
MOSES_SCRIPTS_DIR=${MOSES_DIR}/scripts
TRUECASERTRAIN=${MOSES_SCRIPTS_DIR}/recaser/train-truecaser.perl
TRUECASER=${MOSES_SCRIPTS_DIR}/recaser/truecase.perl
CLEANER=${MOSES_SCRIPTS_DIR}/training/clean-corpus-n.perl
TRAINER=${MOSES_SCRIPTS_DIR}/training/train-model.perl
TOKENIZER=${MOSES_SCRIPTS_DIR}/tokenizer/tokenizer.perl
SPLITTER=./split-corpus.py
truemodel_src=${DDIR}/truecase-model.${name}.${src}
truemodel_tgt=${DDIR}/truecase-model.${name}.${tgt}


echo "STEP: tokenizing..."
perl -i.bckp -pe 's/\x80-\xff//g;' ${DDIR}/${name}.${src}
${TOKENIZER} -no-escape -threads 20 -l ${src} < ${DDIR}/${name}.${src} > ${DDIR}/${name}.tok.${src}
perl -i.bckp -pe 's/\x80-\xff//g;' ${DDIR}/${name}.${tgt}
${TOKENIZER} -no-escape -threads 20 -l ${tgt} < ${DDIR}/${name}.${tgt} > ${DDIR}/${name}.tok.${tgt}

echo "STEP: training truecasing models..."
${TRUECASERTRAIN} --model ${truemodel_src} --corpus ${DDIR}/${name}.tok.${src}
${TRUECASERTRAIN} --model ${truemodel_tgt} --corpus ${DDIR}/${name}.tok.${tgt}

echo "STEP: applying the truecasing models..."
${TRUECASER} --model ${truemodel_src} < ${DDIR}/${name}.tok.${src} > ${DDIR}/${name}.true.${src}
${TRUECASER} --model ${truemodel_tgt} < ${DDIR}/${name}.tok.${tgt} > ${DDIR}/${name}.true.${tgt}

cp ${DDIR}/${name}.true.${src} ${DDIR}/${name}.${src}
cp ${DDIR}/${name}.true.${tgt} ${DDIR}/${name}.${tgt}

echo "STEP: clean data..."
${CLEANER} ${DDIR}/${name}.true ${src} ${tgt} ${DDIR}/${name}.clean 1 70 >> ${DDIR}/clean.log
paste ${DDIR}/${name}.clean.${src} ${DDIR}/${name}.clean.${tgt} | sort -u > ${DDIR}/${name}.clean.merged

echo "STEP: splitting data..."
python ${SPLITTER} ${DDIR}/${name}.clean.merged
for data_set in train dev test; do
    cut -f1 -d$'\t' ${DDIR}/${data_set} > ${ODIR}/${data_set}.${src}
    cut -f2 -d$'\t' ${DDIR}/${data_set} > ${ODIR}/${data_set}.${tgt}
done
