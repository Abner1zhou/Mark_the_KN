# data processing
# python data_processor.py

python model/bert/run_classifiter.py \
  --task_name baidu_95 \
  --do_train true \
  --do_eval true \
  --do_predict true \
  --data_dir data/bert/all_labels \
  --vocab_file data/bert/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file data/bert/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint data/bert/chinese_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 6.0 \
  --output_dir data/bert/output/epochs6_baidu_95/

# evaluate test
# python model/bert/evaluate_test.py
