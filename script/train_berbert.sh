
DIR=$(pwd)
DATA=$(pwd)/data
MODEL=$(pwd)/model
RES=$(pwd)/res
NUM_GPU=$(nvidia-smi --list-gpus | wc -l)


encoder_name='sentence-transformers/distiluse-base-multilingual-cased' # 'sentence-transformers/stsb-xlm-r-multilingual' 
for model_name in 'xlm-roberta-base' 'roberta-base' 'microsoft/deberta-v3-base' 'hfl/chinese-bert-wwm-ext' 'hfl/chinese-roberta-wwm-ext' # 'hfl/chinese-roberta-wwm-ext-large' # 'WENGSYX/Deberta-Chinese-Large' 'IDEA-CCNL/Erlangshen-MegatronBert-1.3B' 
do
    for TASK in 'CSDS_berbert_token_classification' # 'MedQA_berbert_token_classification' # 
    do
        OUTPUT_DIR=${MODEL}/${TASK}/$(basename ${model_name})_$(basename ${encoder_name})
        
        python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=12345  src/run_dialog_encoding.py \
            --model_name_or_path ${model_name} \
            --utterance_encoder_name_or_path ${encoder_name} \
            --do_train \
            --do_predict \
            --do_eval \
            --train_file ${DATA}/${TASK}/train.json \
            --validation_file ${DATA}/${TASK}/dev.json \
            --test_file ${DATA}/${TASK}/test.json \
            --output_dir ${OUTPUT_DIR} \
            --overwrite_output_dir \
            --overwrite_cache \
            --max_steps 3000 \
            --per_device_train_batch_size 8 \
            --gradient_accumulation_steps 1 \
            --per_device_eval_batch_size 4 \
            --evaluation_strategy steps \
            --eval_steps 500 \
            --learning_rate 3e-5 \
            --save_strategy steps \
            --save_steps 500 \
            --load_best_model_at_end \
            --metric_for_best_model 'accuracy' \
            --max_seq_length 512 \
            --logging_steps 500 \
            --sharded_ddp simple 

        rm -rf ${OUTPUT_DIR}/checkpoint*
        python3 tools/evaluate.py --pred_file ${OUTPUT_DIR}/predictions.txt --ref_file ${DATA}/${TASK}/test.ref >> res/score.txt
    done
done



encoder_name='sentence-transformers/xlm-r-large-en-ko-nli-ststb' 
for model_name in 'xlm-roberta-large' 'roberta-large' 'microsoft/deberta-v3-large' 'hfl/chinese-roberta-wwm-ext-large' # 'hfl/chinese-bert-wwm-ext' 'hfl/chinese-roberta-wwm-ext' 'hfl/chinese-roberta-wwm-ext-large' # 'WENGSYX/Deberta-Chinese-Large' 'IDEA-CCNL/Erlangshen-MegatronBert-1.3B' 
do
    for TASK in 'CSDS_berbert_token_classification' #  'MedQA_berbert_token_classification' # 
    do
        OUTPUT_DIR=${MODEL}/${TASK}/$(basename ${model_name})_$(basename ${encoder_name})
        
        python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=12345  src/run_dialog_encoding.py \
            --model_name_or_path ${model_name} \
            --utterance_encoder_name_or_path ${encoder_name} \
            --do_train \
            --do_predict \
            --do_eval \
            --train_file ${DATA}/${TASK}/train.json \
            --validation_file ${DATA}/${TASK}/dev.json \
            --test_file ${DATA}/${TASK}/test.json \
            --output_dir ${OUTPUT_DIR} \
            --overwrite_output_dir \
            --overwrite_cache \
            --max_steps 3000 \
            --per_device_train_batch_size 8 \
            --gradient_accumulation_steps 1 \
            --per_device_eval_batch_size 4 \
            --evaluation_strategy steps \
            --eval_steps 500 \
            --learning_rate 3e-5 \
            --save_strategy steps \
            --save_steps 500 \
            --load_best_model_at_end \
            --metric_for_best_model 'accuracy' \
            --max_seq_length 512 \
            --logging_steps 500 \
            --sharded_ddp simple 

        rm -rf ${OUTPUT_DIR}/checkpoint*
        python3 tools/evaluate.py --pred_file ${OUTPUT_DIR}/predictions.txt --ref_file ${DATA}/${TASK}/test.ref >> res/score.txt
    done
done
