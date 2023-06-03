\#!/usr/bin/env bash

data=/lusr/opt/kaldi/kaldi-master/egs/mini_librispeech/s5/corpus/
data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

. ./cmd.sh
. ./path.sh

stage=5
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
  local/download_lm.sh $lm_url $data data/local/lm
fi

if [ $stage -le 1 ]; then
  # format the data as Kaldi data directories
  for part in dev-clean-2 train-clean-5; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done

  local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
    data/local/lm data/local/lm data/local/dict_nosp

  utils/prepare_lang.sh data/local/dict_nosp \
    "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

  local/format_lms.sh --src-dir data/lang_nosp data/local/lm
fi

if [ $stage -le 2 ]; then
  mfccdir=mfcc
  train_data_dir=data/train_clean_5
  test_data_dir=data/dev_clean_2
  log_dir=exp/make_mfcc

  # TODO: extract MFCCs for train and test data
  steps/make_mfcc.sh $train_data_dir $log_dir $mfccdir
  steps/make_mfcc.sh $test_data_dir $log_dir $mfccdir

  steps/compute_cmvn_stats.sh $train_data_dir $log_dir $mfccdir
  steps/compute_cmvn_stats.sh $test_data_dir $log_dir $mfccdir
fi

# train a monophone system
if [ $stage -le 3 ]; then
  utils/subset_data_dir.sh --shortest data/train_clean_5 500 data/train_500short

  # TODO: train a monophone acoustic model
  data_dir=data/train_500short
  lang_dir=data/lang_nosp
  exp_dir=exp/mono

  steps/train_mono.sh --boost_silence 1.25 $data_dir $lang_dir $exp_dir
fi

# train a delta + delta-delta triphone system on all utterances
if [ $stage -le 4 ]; then
  data_dir=data/train_clean_5
  lang_dir=data/lang_nosp
  src_dir=exp/mono
  align_dir=exp/mono_ali_train_clean_5

  # TODO: 1) force-align the entire training set with the monophone model
  steps/align_si.sh --boost_silence 1.25 $data_dir $lang_dir $src_dir $align_dir 
  
  # TODO: 2) train the triphone model on the entire training set
  steps/train_deltas.sh 2000 10000 $data_dir $lang_dir $align_dir exp/tri1 
fi

if [ $stage -le 5 ]; then
  lang_dir=data/lang_nosp_test_tgsmall
  model_dir=exp/tri1
  graph_dir=exp/tri1/graph_tgsmall
  data_dir=data/dev_clean_2
  decode_dir=exp/tri1/decode_tgsmall_dev_clean_2

  # TODO: 1) build the decoding graph
  utils/mkgraph.sh $lang_dir $model_dir $graph_dir

  # TODO: 2) decode using the triphone model
  steps/decode.sh $graph_dir $data_dir $decode_dir

  # TODO: 3) rescore with the larger (tgmed) language model
  steps/lmrescore.sh $lang_dir data/lang_nosp_test_tgmed $data_dir $decode_dir exp/tri1/decode_tgmed_dev_clean_2 
fi
