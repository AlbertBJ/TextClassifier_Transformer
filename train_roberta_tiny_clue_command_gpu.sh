#!/bin/bash
#description: BERT fine-tuning

export BERT_BASE_DIR=/export/huangdongxiao/huangdongxiao/AI_QA/models/RoBERTa-tiny-clue
export DATA_DIR=/export/huangdongxiao/huangdongxiao/AI_QA/ALBERT/1-data/command
export TRAINED_CLASSIFIER=./output
export MODEL_NAME=roberta_tiny_clue_command_gpu

export CUDA_VISIBLE_DEVICES=2
python run_classifier_serving_gpu.py \
  --task_name=command \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --do_export=false \
  --do_frozen=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --test_file=test  \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=1e-4 \
  --num_train_epochs=6.0 \
  --output_dir=$TRAINED_CLASSIFIER/$MODEL_NAME
