MAIN_PATH=$(pwd)

N_MONO=10000000  # number of monolingual sentences for each language
CODES=60000     # number of BPE codes
N_THREADS=16

SRC=en
TGT=ro

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --reload_codes)
    RELOAD_CODES="$2"; shift 2;;
  --reload_vocab)
    RELOAD_VOCAB="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

# main paths
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para
PROC_PATH=$DATA_PATH/processed/$SRC-$TGT

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH
mkdir -p $PROC_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# Sennrich's WMT16 scripts for Romanian preprocessing
WMT16_SCRIPTS=$TOOLS_PATH/wmt16-scripts
NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/preprocess/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/preprocess/remove-diacritics.py

# raw and tokenized files
SRC_RAW=$MONO_PATH/$SRC/all.$SRC
TGT_RAW=$MONO_PATH/$TGT/all.$TGT
SRC_TOK=$SRC_RAW.tok
TGT_TOK=$TGT_RAW.tok

# BPE / vocab files
BPE_CODES=$PROC_PATH/codes
SRC_VOCAB=$PROC_PATH/vocab.$SRC
TGT_VOCAB=$PROC_PATH/vocab.$TGT
FULL_VOCAB=$PROC_PATH/vocab.$SRC-$TGT

# train / valid / test monolingual BPE data
SRC_TRAIN_BPE=$PROC_PATH/train.$SRC
TGT_TRAIN_BPE=$PROC_PATH/train.$TGT
SRC_VALID_BPE=$PROC_PATH/valid.$SRC
TGT_VALID_BPE=$PROC_PATH/valid.$TGT
SRC_TEST_BPE=$PROC_PATH/test.$SRC
TGT_TEST_BPE=$PROC_PATH/test.$TGT

# valid / test parallel BPE data
PARA_SRC_TRAIN_BPE=$PROC_PATH/train.$SRC-$TGT.$SRC
PARA_TGT_TRAIN_BPE=$PROC_PATH/train.$SRC-$TGT.$TGT
PARA_SRC_VALID_BPE=$PROC_PATH/valid.$SRC-$TGT.$SRC
PARA_TGT_VALID_BPE=$PROC_PATH/valid.$SRC-$TGT.$TGT
PARA_SRC_TEST_BPE=$PROC_PATH/test.$SRC-$TGT.$SRC
PARA_TGT_TEST_BPE=$PROC_PATH/test.$SRC-$TGT.$TGT

# en-ro valid / test data
PARA_SRC_TRAIN=$PARA_PATH/train.$SRC-$TGT.$SRC
PARA_TGT_TRAIN=$PARA_PATH/train.$SRC-$TGT.$TGT
PARA_SRC_VALID=$PARA_PATH/dev/newsdev2016-roen-ref.en
PARA_TGT_VALID=$PARA_PATH/dev/newsdev2016-enro-ref.ro
PARA_SRC_TEST=$PARA_PATH/dev/newstest2016-roen-ref.en
PARA_TGT_TEST=$PARA_PATH/dev/newstest2016-enro-ref.ro

# install tools
./install-tools.sh

#
# Download monolingual data
#

cd $MONO_PATH

if [ "$SRC" == "en" -o "$TGT" == "en" ]; then
  echo "Downloading English monolingual data ..."
  mkdir -p $MONO_PATH/en
  cd $MONO_PATH/en
  wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz
  # wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz
  # wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.en.shuffled.gz
  # wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz
  # wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.en.shuffled.gz
  # wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz
  # wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz
  # wget -c http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.en.shuffled.v2.gz
  # wget -c http://data.statmt.org/wmt16/translation-task/news.2015.en.shuffled.gz
  # wget -c http://data.statmt.org/wmt17/translation-task/news.2016.en.shuffled.gz
  # wget -c http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz
fi

if [ "$SRC" == "ro" -o "$TGT" == "ro" ]; then
  echo "Downloading Romanian monolingual data ..."
  mkdir -p $MONO_PATH/ro
  cd $MONO_PATH/ro
  wget -c http://data.statmt.org/wmt16/translation-task/news.2015.ro.shuffled.gz
fi

cd $MONO_PATH

# decompress monolingual data
for FILENAME in $SRC/news*gz $TGT/news*gz; do
  OUTPUT="${FILENAME::-3}"
  if [ ! -f "$OUTPUT" ]; then
    echo "Decompressing $FILENAME..."
    gunzip -k $FILENAME
  else
    echo "$OUTPUT already decompressed."
  fi
done

cd $PARA_PATH
#
# Download bilingual data
# 
echo "Downloading parallel data..."
if [[ ! -f "$PARA_SRC_TRAIN.raw" ]] && [[ ! -f "$PARA_TGT_TRAIN.raw" ]]; then
	wget -c http://data.statmt.org/wmt18/translation-task/dev.tgz
	wget http://www.statmt.org/europarl/v7/ro-en.tgz
	wget http://opus.lingfil.uu.se/download.php?f=SETIMES2/en-ro.txt.zip -O SETIMES2.ro-en.txt.zip
	wget -nc http://data.statmt.org/rsennrich/wmt16_backtranslations/ro-en/corpus.bt.ro-en.en.gz
	wget -nc http://data.statmt.org/rsennrich/wmt16_backtranslations/ro-en/corpus.bt.ro-en.ro.gz

	echo "Extracting parallel data..."
	tar -xzf dev.tgz
	tar -xf ro-en.tgz
	unzip SETIMES2.ro-en.txt.zip
	gzip -d corpus.bt.ro-en.en.gz corpus.bt.ro-en.ro.gz
fi


# concatenate bilingual data
if ! [[ -f "$PARA_SRC_TRAIN.raw" ]]; then
	echo "Concatenating $SRC bilingual data ..."
	cat corpus.bt.ro-en.$SRC europarl-v7.ro-en.$SRC SETIMES.en-ro.$SRC > $PARA_SRC_TRAIN.raw
fi

if ! [[ -f "$PARA_TGT_TRAIN.raw" ]]; then
	echo "Concatenating $TGT bilingual data ..."
	cat corpus.bt.ro-en.$TGT europarl-v7.ro-en.$TGT SETIMES.en-ro.$TGT > $PARA_TGT_TRAIN.raw
fi

SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR |                                            $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -l $TGT -no-escape -threads $N_THREADS"

# check valid and test files are here
if ! [[ -f "$PARA_SRC_VALID.sgm" ]]; then echo "$PARA_SRC_VALID.sgm is not found!"; exit; fi
if ! [[ -f "$PARA_TGT_VALID.sgm" ]]; then echo "$PARA_TGT_VALID.sgm is not found!"; exit; fi
if ! [[ -f "$PARA_SRC_TEST.sgm" ]];  then echo "$PARA_SRC_TEST.sgm is not found!";  exit; fi
if ! [[ -f "$PARA_TGT_TEST.sgm" ]];  then echo "$PARA_TGT_TEST.sgm is not found!";  exit; fi

echo "Tokenizing valid and test data..."
eval "$INPUT_FROM_SGM < $PARA_SRC_VALID.sgm | $SRC_PREPROCESSING > $PARA_SRC_VALID"
eval "$INPUT_FROM_SGM < $PARA_TGT_VALID.sgm | $TGT_PREPROCESSING > $PARA_TGT_VALID"
eval "$INPUT_FROM_SGM < $PARA_SRC_TEST.sgm  | $SRC_PREPROCESSING > $PARA_SRC_TEST"
eval "$INPUT_FROM_SGM < $PARA_TGT_TEST.sgm  | $TGT_PREPROCESSING > $PARA_TGT_TEST"

cat $PARA_SRC_TRAIN.raw | $SRC_PREPROCESSING > $PARA_SRC_TRAIN
cat $PARA_TGT_TRAIN.raw | $TGT_PREPROCESSING > $PARA_TGT_TRAIN

cd $MONO_PATH

if ! [[ -f "$SRC_RAW" ]]; then
	echo "Concatenating $SRC monolingual data..."
  	cat $(ls $SRC/news*$SRC* | grep -v gz) | head -n $N_MONO > $SRC_RAW
fi
if ! [[ -f "$TGT_RAW" ]]; then
  echo "Concatenating $TGT monolingual data..."
  cat $(ls $TGT/news*$TGT* | grep -v gz) | head -n $N_MONO > $TGT_RAW
  cat $PARA_PATH/SETIMES.en-ro.$TGT $PARA_PATH/europarl-v7.ro-en.$TGT >> $TGT_RAW
fi

# tokenize data
if ! [[ -f "$SRC_TOK" ]]; then
  echo "Tokenize $SRC monolingual data..."
  eval "cat $SRC_RAW | $SRC_PREPROCESSING > $SRC_TOK"
fi

if ! [[ -f "$TGT_TOK" ]]; then
  echo "Tokenize $TGT monolingual data..."
  eval "cat $TGT_RAW | $TGT_PREPROCESSING > $TGT_TOK"
fi
echo "$SRC monolingual data tokenized in: $SRC_TOK"
echo "$TGT monolingual data tokenized in: $TGT_TOK"

# reload BPE codes
cd $MAIN_PATH
if [ ! -f "$BPE_CODES" ] && [ -f "$RELOAD_CODES" ]; then
  echo "Reloading BPE codes from $RELOAD_CODES ..."
  cp $RELOAD_CODES $BPE_CODES
fi

# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $SRC_TOK $TGT_TOK > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"

# apply BPE codes
if ! [[ -f "$SRC_TRAIN_BPE" ]]; then
  echo "Applying $SRC BPE codes..."
  $FASTBPE applybpe $SRC_TRAIN_BPE $SRC_TOK $BPE_CODES
fi
if ! [[ -f "$TGT_TRAIN_BPE" ]]; then
  echo "Applying $TGT BPE codes..."
  $FASTBPE applybpe $TGT_TRAIN_BPE $TGT_TOK $BPE_CODES
fi
echo "BPE codes applied to $SRC in: $SRC_TRAIN_BPE"
echo "BPE codes applied to $TGT in: $TGT_TRAIN_BPE"

echo "Applying BPE to train, valid and test bilingual files..."
$FASTBPE applybpe $PARA_SRC_TRAIN_BPE $PARA_SRC_TRAIN $BPE_CODES
$FASTBPE applybpe $PARA_TGT_TRAIN_BPE $PARA_TGT_TRAIN $BPE_CODES
$FASTBPE applybpe $PARA_SRC_VALID_BPE $PARA_SRC_VALID $BPE_CODES
$FASTBPE applybpe $PARA_TGT_VALID_BPE $PARA_TGT_VALID $BPE_CODES
$FASTBPE applybpe $PARA_SRC_TEST_BPE  $PARA_SRC_TEST  $BPE_CODES
$FASTBPE applybpe $PARA_TGT_TEST_BPE  $PARA_TGT_TEST  $BPE_CODES

# extract source and target vocabulary
if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TRAIN_BPE > $SRC_VOCAB
  $FASTBPE getvocab $TGT_TRAIN_BPE > $TGT_VOCAB
fi
echo "$SRC vocab in: $SRC_VOCAB"
echo "$TGT vocab in: $TGT_VOCAB"

# reload full vocabulary
cd $MAIN_PATH
if [ ! -f "$FULL_VOCAB" ] && [ -f "$RELOAD_VOCAB" ]; then
  echo "Reloading vocabulary from $RELOAD_VOCAB ..."
  cp $RELOAD_VOCAB $FULL_VOCAB
fi

# extract full vocabulary
if ! [[ -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TRAIN_BPE $TGT_TRAIN_BPE > $FULL_VOCAB
fi
echo "Full vocab in: $FULL_VOCAB"

# binarize data
if ! [[ -f "$SRC_TRAIN_BPE.pth" ]]; then
  echo "Binarizing $SRC data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $SRC_TRAIN_BPE
fi
if ! [[ -f "$TGT_TRAIN_BPE.pth" ]]; then
  echo "Binarizing $TGT data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TGT_TRAIN_BPE
fi
echo "$SRC binarized data in: $SRC_TRAIN_BPE.pth"
echo "$TGT binarized data in: $TGT_TRAIN_BPE.pth"

echo "Binarizing data..."
rm -f $PARA_SRC_VALID_BPE.pth $PARA_TGT_VALID_BPE.pth $PARA_SRC_TEST_BPE.pth $PARA_TGT_TEST_BPE.pth
rm -f $PARA_SRC_TRAIN_BPE.pth $PARA_TGT_TRAIN_BPE.pth
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_TRAIN_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_TRAIN_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_VALID_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_VALID_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_TEST_BPE
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_TEST_BPE
#
# Link monolingual validation and test data to parallel data
#
ln -sf $PARA_SRC_VALID_BPE.pth $SRC_VALID_BPE.pth
ln -sf $PARA_TGT_VALID_BPE.pth $TGT_VALID_BPE.pth
ln -sf $PARA_SRC_TEST_BPE.pth  $SRC_TEST_BPE.pth
ln -sf $PARA_TGT_TEST_BPE.pth  $TGT_TEST_BPE.pth

cd $PARA_PATH
if [[ -f "dev.tgz" ]]; then
	rm dev.tgz ro-en.tgz SETIMES2.ro-en.txt.zip
	rm LICENSE README SETIMES.en-ro.* corpus.bt.ro-en.* europarl-v7.ro-en.*
fi
