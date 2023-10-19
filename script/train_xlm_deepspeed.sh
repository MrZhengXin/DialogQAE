
DIR=$(pwd)
DATA=$(pwd)/data
MODEL=$(pwd)/model
RES=$(pwd)/res
NUM_GPU=$(nvidia-smi --list-gpus | wc -l)


for model_name in 'facebook/xlm-roberta-xl' # 'facebook/xlm-roberta-xl'  # 'xlm-roberta-large' 'xlm-roberta-base' 
do
    for TASK in MedQA_xlm
    do
        OUTPUT_DIR=${MODEL}/${TASK}/$(basename ${model_name})
        python3 src/run_mlm.py \
            --model_name_or_path ${model_name} \
            --do_predict \
            --train_file ${DATA}/${TASK}/train.json \
            --validation_file ${DATA}/${TASK}/dev.json \
            --test_file ${DATA}/${TASK}/test.json \
            --label2id_file ${DATA}/${TASK}/label2id.json \
            --output_dir ${OUTPUT_DIR} \
            --overwrite_output_dir \
            --per_device_eval_batch_size 1 \
            --fp16_full_eval \
            --fp16_opt_level=O3

        # rm -rf ${OUTPUT_DIR}/checkpoint*
        python3 tools/evaluate.py --pred_file ${OUTPUT_DIR}/predict_results.txt --ref_file ${DATA}/${TASK}/test.ref >> res/score.txt
    done
done


        deepspeed src/run_mlm.py \
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
            --num_train_epochs 4 \
            --overwrite_cache \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 2 \
            --per_device_eval_batch_size 4 \
            --evaluation_strategy epoch \
            --learning_rate 3e-5 \
            --save_strategy no \
            --metric_for_best_model 'accuracy' \
            --pad_to_max_length \
            --max_seq_length 512 \
            --logging_steps 500 \
            --deepspeed src/ds_config_zero2.json