#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



echo "Formating data..."

TASK=denoise
for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done


echo "Binarizing data for: $TASK ..."

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;


echo "Fine-Tuning Weights for: $TASK ..."

TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=1858      
LR=5e-06
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=/misc/vlgscratch4/HeGroup/jlm10003/bart.large/model.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train denoise-bin \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task denoising \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion masked_lm \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
