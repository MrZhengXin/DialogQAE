
DIR=$(pwd)
DATA=$(pwd)/data
MODEL=$(pwd)/model
RES=$(pwd)/res
NUM_GPU=$(nvidia-smi --list-gpus | wc -l)


for model_name in 'hfl/chinese-bert-wwm-ext' 'WENGSYX/Deberta-Chinese-Large' 'IDEA-CCNL/Erlangshen-MegatronBert-1.3B' 'hfl/chinese-roberta-wwm-ext' 'hfl/chinese-roberta-wwm-ext-large'
do
    for TASK in 'MedQA_mlm' # 'MedQA_mlm' # 9 epoch
    do
        OUTPUT_DIR=${MODEL}/${TASK}/$(basename ${model_name})
        python3 tools/evaluate.py --pred_file ${OUTPUT_DIR}/predict_results.txt --ref_file ${DATA}/${TASK}/test.ref >> res/score.txt
    done
done


for model_name in 'hfl/chinese-bert-wwm-ext' 'WENGSYX/Deberta-Chinese-Large' 'IDEA-CCNL/Erlangshen-MegatronBert-1.3B' 'hfl/chinese-roberta-wwm-ext' 'hfl/chinese-roberta-wwm-ext-large'
do
    for TASK in 'CSDS_mlm' # 'MedQA_mlm' 9 epoch
    do
        OUTPUT_DIR=${MODEL}/${TASK}/$(basename ${model_name})
        python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=12345 src/run_mlm.py \
            --model_name_or_path ${model_name} \
            --do_train \
            --do_predict \
            --do_eval \
            --train_file ${DATA}/${TASK}/train.json \
            --validation_file ${DATA}/${TASK}/dev.json \
            --test_file ${DATA}/${TASK}/test.json \
            --label2id_file ${DATA}/${TASK}/label2id.json \
            --output_dir ${OUTPUT_DIR} \
            --overwrite_output_dir \
            --num_train_epochs 9 \
            --overwrite_cache \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 2 \
            --per_device_eval_batch_size 4 \
            --evaluation_strategy epoch \
            --learning_rate 3e-5 \
            --save_strategy epoch \
            --load_best_model_at_end \
            --metric_for_best_model 'accuracy' \
            --pad_to_max_length \
            --max_seq_length 512 \
            --logging_steps 500 \
            --sharded_ddp simple \
            --fp16 \
            --fp16_full_eval
        rm -rf ${OUTPUT_DIR}/checkpoint*
        python3 tools/evaluate.py --pred_file ${OUTPUT_DIR}/predict_results.txt --ref_file ${DATA}/${TASK}/test.ref >> res/score.txt
    done
done

for model_name in 'xlm-roberta-base' 'xlm-roberta-large'
do
    for TASK in 'CSDS_xlm' # 'MedQA_xlm'
    do
        OUTPUT_DIR=${MODEL}/${TASK}/$(basename ${model_name})
        python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=11111 src/run_mlm.py \
            --model_name_or_path ${model_name} \
            --do_train \
            --do_predict \
            --do_eval \
            --train_file ${DATA}/${TASK}/train.json \
            --validation_file ${DATA}/${TASK}/dev.json \
            --test_file ${DATA}/${TASK}/test.json \
            --label2id_file ${DATA}/${TASK}/label2id.json \
            --output_dir ${OUTPUT_DIR} \
            --overwrite_output_dir \
            --num_train_epochs 9 \
            --overwrite_cache \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 2 \
            --per_device_eval_batch_size 4 \
            --evaluation_strategy epoch \
            --learning_rate 3e-5 \
            --save_strategy epoch \
            --load_best_model_at_end \
            --metric_for_best_model 'accuracy' \
            --pad_to_max_length \
            --max_seq_length 512 \
            --logging_steps 500 \
            --sharded_ddp simple \
            --fp16 \
            --fp16_full_eval
        rm -rf ${OUTPUT_DIR}/checkpoint*
        python3 tools/evaluate.py --pred_file ${OUTPUT_DIR}/predict_results.txt --ref_file ${DATA}/${TASK}/test.ref >> res/score.txt
    done
done
