
比赛链接：https://www.datafountain.cn/competitions/424


。



这里用的是bert_wwm_wx_bert文件这里可以下载：https://github.com/ymcui/Chinese-BERT-wwm

训练执行：
python gf_run_squad.py \
 --model_type bert \
 --model_name_or_path pytorch_model.bin\
 --config_name bert_config.json\
  --tokenizer_name vocab.txt\
 --do_train \
 --do_eval \
 --do_lower_case \
 --train_file train.json\
  --predict_file valid.json\
  --per_gpu_train_batch_size 23 \
 --learning_rate 5e-5 \
     --num_train_epochs 4 \
     --max_seq_length 512 \
     --max_query_length 80 \
     --max_answer_length 400 \
     --warmup_steps 2500 \
     --doc_stride 128 \
     --output_dir  data/torch0428/home/40821/Desktop/knowledgegraph/epidemic-QA-assistant--2020-master/train_0.8.json \
    --predict_file /home40821/Desktop/knowledgegraph/epidemic-QA-assistant--2020-master/valid_0.2.json \
    --per_gpu_train_batch_size 6 \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --max_seq_length 512 \
    --max_query_length 80 \
    --max_answer_length 400 \
    --warmup_steps 2500 \
    --doc_stride 128 \
    --output_dir /home/40821/Desktop/knowledgegraph/epidemic-QA-assistant--2020-master/torch0413
 
 下面checkpoint-3是训练过程生成最优模型所在路径.
 预测执行：


python gf_run_squad.py \
    --model_type bert \
    --model_name_or_path "data/torch0428/checkpoint-4" \
    --do_eval \
    --do_test \
    --do_lower_case \
    --per_gpu_train_batch_size 22 \
    --predict_file test_torch_0428b_N3_json.json \
    --test_prefix "0428" \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --max_query_length 80 \
    --max_answer_length 400 \
    --doc_stride 128 \
    --output_dir "data/torch0428/checkpoint-4"