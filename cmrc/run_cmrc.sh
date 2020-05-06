export SQUAD_DIR=./CMRC_DIR
export BERT_BASE_DIR=/export/huangdongxiao/huangdongxiao/AI_QA/models/RoBERTa-tiny-clue  
python run_cmrc.py \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --do_train=False \
    --do_export=True \
    --do_predict=True \
    --train_file=$SQUAD_DIR/train.json \
    --predict_file=$SQUAD_DIR/dev.json \
    --train_batch_size=16 \
    --learning_rate=3e-5 \
    --num_train_epochs=8.0 \
    --max_seq_length=384 \
    --doc_stride=128 \
    --output_dir=./output/cmrc \
    --version_2_with_negative=False