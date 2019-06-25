GIGAPATH=sumdata

CODES=40000

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --gigapath)
	GIGAPATH="$2"; shift 2;;
  --reload_codes)
	RELOAD_CODES="$2"; shift 2;;
  --reload_vocab)
	RELOAD_VOCAB="$2"; shift 2;;
  --replace_ner)
	REPLACE_NER="$2"; shift 2;;
  --replace_unk)
	REPLACE_UNK="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

# Check parameters

if [ "$RELOAD_CODES" != "" ] && [ ! -f "$RELOAD_CODES" ]; then echo "cannot locate BPE codes"; exit; fi
if [ "$RELOAD_VOCAB" != "" ] && [ ! -f "$RELOAD_VOCAB" ]; then echo "cannot locate vocabulary"; exit; fi
if [ "$RELOAD_CODES" == "" -a "$RELOAD_VOCAB" != "" -o "$RELOAD_CODES" != "" -a "$RELOAD_VOCAB" == "" ]; then echo "BPE codes should be provided if and only if vocabulary is also provided"; exit; fi

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# main paths
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
PARA_PATH=$DATA_PATH/para/
PROC_PATH=$DATA_PATH/processed/giga/

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $PROC_PATH
mkdir -p $PARA_PATH

MOSES=$TOOLS_PATH/mosesdecoder

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

TRAIN_SRC_RAW=$GIGAPATH/train/train.article.txt
TRAIN_TGT_RAW=$GIGAPATH/train/train.title.txt
VALID_SRC_RAW=$GIGAPATH/train/valid.article.filter.txt
VALID_TGT_RAW=$GIGAPATH/train/valid.title.filter.txt
TEST_SRC_RAW=$GIGAPATH/Giga/input.txt
TEST_TGT_RAW=$GIGAPATH/Giga/task1_ref0.txt

TRAIN_SRC=$PARA_PATH/train.ar-ti.ar
TRAIN_TGT=$PARA_PATH/train.ar-ti.ti
VALID_SRC=$PARA_PATH/valid.ar-ti.ar
VALID_TGT=$PARA_PATH/valid.ar-ti.ti
TEST_SRC=$PARA_PATH/test.ar-ti.ar
TEST_TGT=$PARA_PATH/test.ar-ti.ar

TRAIN_SRC_BPE=$PROC_PATH/train.ar-ti.ar
TRAIN_TGT_BPE=$PROC_PATH/train.ar-ti.ti
VALID_SRC_BPE=$PROC_PATH/valid.ar-ti.ar
VALID_TGT_BPE=$PROC_PATH/valid.ar-ti.ti
TEST_SRC_BPE=$PROC_PATH/test.ar-ti.ar
TEST_TGT_BPE=$PROC_PATH/test.ar-ti.ar

BPE_CODES=$PROC_PATH/codes
FULL_VOCAB=$PROC_PATH/vocab.ar-ti

if [ ! -f $TRAIN_SRC_RAW ]; then
	gzip -d $TRAIN_SRC_RAW.gz
fi

if [ ! -f $TRAIN_TGT_RAW ]; then
	gzip -d $TRAIN_TGT_RAW.gz
fi

preprocess=""

if [ "$REPLACE_NER" == "true" ] && [ "$REPLACE_UNK" == "true" ]; then
	preprocess="sed 's/#/1/g' | sed 's/<unk>/unk/g' | sed 's/UNK/unk/g'"
else
	if [ "$REPLACE_UNK" == "true" ]; then
		preprocess="sed 's/<unk>/unk/g' | sed 's/UNK/unk/g'"
	fi
	if [ "$REPLACE_NER" == "true" ]; then
		preprocess="sed 's/#/1/g'"
	fi
fi

if ! [[ -f $TRAIN_SRC ]]; then
	eval "cat $TRAIN_SRC_RAW | $preprocess > $TRAIN_SRC"
fi

if ! [[ -f $TRAIN_TGT ]]; then
	eval "cat $TRAIN_TGT_RAW | $preprocess > $TRAIN_TGT"
fi

if [ ! -f "$BPE_CODES" ] && [ -f "$RELOAD_CODES" ]; then
  echo "Reloading BPE codes from $RELOAD_CODES ..."
  cp $RELOAD_CODES $BPE_CODES
fi

# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $TRAIN_SRC $TRAIN_TGT > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"

if [ ! -f "$FULL_VOCAB" ] && [ -f "$RELOAD_VOCAB" ]; then
  echo "Reloading vocabulary from $RELOAD_VOCAB ..."
  cp $RELOAD_VOCAB $FULL_VOCAB
fi

if [ ! -f "$TRAIN_SRC_BPE" ]; then
  echo "Applying article BPE codes..."
  $FASTBPE applybpe $TRAIN_SRC_BPE $TRAIN_SRC $BPE_CODES
fi

if [ ! -f "$TRAIN_TGT_BPE" ]; then
  echo "Applying title BPE codes..."
  $FASTBPE applybpe $TRAIN_TGT_BPE $TRAIN_TGT $BPE_CODES
fi

# extract full vocabulary
if ! [[ -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $TRAIN_SRC_BPE $TRAIN_TGT_BPE > $FULL_VOCAB
fi
echo "Full vocab in: $FULL_VOCAB"

eval "cat $VALID_SRC_RAW | $preprocess > $VALID_SRC"
eval "cat $VALID_TGT_RAW | $preprocess > $VALID_TGT"
eval "cat $TEST_SRC_RAW | $preprocess > $TEST_SRC"
eval "cat $TEST_TGT_RAW | $preprocess > $TEST_TGT"

$FASTBPE applybpe $VALID_SRC_BPE $VALID_SRC $BPE_CODES 
$FASTBPE applybpe $VALID_TGT_BPE $VALID_TGT $BPE_CODES 
$FASTBPE applybpe $TEST_SRC_BPE $TEST_SRC $BPE_CODES 
$FASTBPE applybpe $TEST_TGT_BPE $TEST_TGT $BPE_CODES 

$MAIN_PATH/preprocess.py $FULL_VOCAB $TRAIN_SRC_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $TRAIN_TGT_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $VALID_SRC_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $VALID_TGT_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $TEST_SRC_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $TEST_TGT_BPE
