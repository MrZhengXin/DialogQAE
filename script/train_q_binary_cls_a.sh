
DIR=$(pwd)
DATA=$(pwd)/data
MODEL=$(pwd)/model
RES=$(pwd)/res
NUM_GPU=$(nvidia-smi --list-gpus | wc -l)
for TASK in 'CSDS' # 'MedQA'
do
    STAGE_1_TASK=${TASK}_seq2seq_Q_binary_cls
    STAGE_2_TASK=${TASK}_seq2seq_A
    for CLASSIFIER in 'hfl/chinese-roberta-wwm-ext-large'
    do
        CLASSIFIER_PATH=${MODEL}/${STAGE_1_TASK}/$(basename ${CLASSIFIER})

        python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} src/run_glue.py \
            --model_name_or_path ${CLASSIFIER} \
            --train_file ${DATA}/${STAGE_1_TASK}/train.json \
            --validation_file ${DATA}/${STAGE_1_TASK}/dev.json \
            --test_file ${DATA}/${STAGE_1_TASK}/test.json \
            --do_train \
            --do_eval \
            --do_predict \
            --max_seq_length 512 \
            --no_pad_to_max_length \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64 \
            --learning_rate 3e-5 \
            --num_train_epochs 8 \
            --output_dir ${CLASSIFIER_PATH} \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --load_best_model_at_end \
            --metric_for_best_model 'f1' \
            --logging_steps 500 \
            --overwrite_output_dir \
            --overwrite_cache \
            --fp16
        rm -rf ${CLASSIFIER_PATH}/checkpoint*

        STAGE_1_PRED_FILE=${CLASSIFIER_PATH}/predict_results.txt
        TEST_FILE=${DATA}/${STAGE_2_TASK}/test_${STAGE_1_TASK}_$(basename ${CLASSIFIER}).json
        python3 tools/construct_Q_binary_cls_second_stage_input.py --test_file ${DATA}/${STAGE_2_TASK}/test_original.json --pred_file ${STAGE_1_PRED_FILE} --output_file ${TEST_FILE}

        for model_name in google/mt5-base google/mt5-large google/mt5-xl 
        do
            OUTPUT_DIR=${MODEL}/${STAGE_2_TASK}/$(basename ${model_name})
            python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=12345  src/run_summarization.py \
                --model_name_or_path ${OUTPUT_DIR} \
                --do_predict \
                --train_file ${DATA}/${STAGE_2_TASK}/train.json \
                --validation_file ${DATA}/${STAGE_2_TASK}/dev.json \
                --test_file ${TEST_FILE} \
                --output_dir ${OUTPUT_DIR} \
                --overwrite_output_dir \
                --overwrite_cache \
                --predict_with_generate \
                --generation_num_beams 4 \
                --sharded_ddp simple 

            STAGE_2_PRED_FILE=${OUTPUT_DIR}/generated_predictions.txt
            MERGED_PRED_FILE=${OUTPUT_DIR}/generated_predictions_merged_${STAGE_1_TASK}.txt
            python3 tools/merge_two_stage_prediction.py --pred_file ${STAGE_2_PRED_FILE} --test_file ${TEST_FILE} --output_file ${MERGED_PRED_FILE}
            python3 tools/evaluate.py --pred_file ${MERGED_PRED_FILE} --ref_file ${DATA}/${STAGE_2_TASK}/test_original.ref >> res/score.txt

        done
    done
done
