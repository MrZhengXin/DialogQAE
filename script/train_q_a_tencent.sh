
DIR=$(pwd)
DATA=$(pwd)/data
MODEL=$(pwd)/model
RES=$(pwd)/res
NUM_GPU=$(nvidia-smi --list-gpus | wc -l)

for model_name in google/mt5-xl # google/mt5-base google/mt5-large 
do
    for TASK in 'carsales_Q' 'carsales_A' 'dhl_log_Q' 'dhl_log_A' 'ketang_Q' 'ketang_A'
    do
        OUTPUT_DIR=${MODEL}/${TASK}/$(basename ${model_name})
        echo ${OUTPUT_DIR}
        # cd ../transformers/examples/pytorch/summarization
        python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=12345  src/run_summarization.py \
            --model_name_or_path ${model_name} \
            --do_train \
            --do_predict \
            --do_eval \
            --train_file ${DATA}/${TASK}/train.json \
            --validation_file ${DATA}/${TASK}/test.json \
            --test_file ${DATA}/${TASK}/test.json \
            --output_dir ${OUTPUT_DIR} \
            --overwrite_output_dir \
            --num_train_epochs 6 \
            --overwrite_cache \
            --per_device_train_batch_size 3 \
            --gradient_accumulation_steps 2 \
            --per_device_eval_batch_size 4 \
            --evaluation_strategy epoch \
            --predict_with_generate \
            --generation_num_beams 4 \
            --learning_rate 3e-5 \
            --save_strategy no \
            --warmup_steps 50 \
            --logging_steps 500 \
            --sharded_ddp simple 
        # rm -rf ${OUTPUT_DIR}/checkpoint*
    done
done

for model_name in google/mt5-xl # google/mt5-base google/mt5-large 
do
    for TASK in 'carsales' 'ketang' 'dhl_log'
    do
    PRE_TASK=${TASK}_Q # MedQA_Q
    THIS_TASK=${TASK}_A # MedQA_A

    OUTPUT_DIR=${MODEL}/${PRE_TASK}/$(basename ${model_name})
    # cd ../transformers/examples/pytorch/summarization
    CUDA_VISIBLE_DEVICES=3 python3 src/run_summarization.py \
        --model_name_or_path ${OUTPUT_DIR} \
        --do_predict \
        --train_file ${DATA}/${PRE_TASK}/train.json \
        --validation_file ${DATA}/${PRE_TASK}/test.json \
        --test_file ${DATA}/${PRE_TASK}/test.json \
        --output_dir ${OUTPUT_DIR} \
        --overwrite_output_dir \
        --overwrite_cache \
        --predict_with_generate \
        --generation_num_beams 4 \

    STAGE_1_PRED_FILE=${MODEL}/${PRE_TASK}/$(basename ${model_name})/generated_predictions.txt
    TEST_FILE=${DATA}/${THIS_TASK}/test_${PRE_TASK}_$(basename ${model_name}).json
    echo $TEST_FILE
    python3 tools/construct_second_stage_input_tencent.py --test_file ${DATA}/${PRE_TASK}/test.json --pred_file ${STAGE_1_PRED_FILE} --output_file ${TEST_FILE}


    OUTPUT_DIR=${MODEL}/${THIS_TASK}/$(basename ${model_name})
    # cd ../transformers/examples/pytorch/summarization
    CUDA_VISIBLE_DEVICES=3 python3 src/run_summarization.py \
        --model_name_or_path ${OUTPUT_DIR} \
        --do_predict \
        --train_file ${DATA}/${THIS_TASK}/train.json \
        --validation_file ${DATA}/${THIS_TASK}/test.json \
        --test_file ${TEST_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --overwrite_output_dir \
        --overwrite_cache \
        --predict_with_generate \
        --generation_num_beams 4 \

    # cd ${DIR}
    STAGE_2_PRED_FILE=${OUTPUT_DIR}/generated_predictions.txt
    MERGED_PRED_FILE=${OUTPUT_DIR}/generated_predictions_merged.txt
    python3 tools/merge_two_stage_prediction.py --pred_file ${STAGE_2_PRED_FILE} --test_file ${TEST_FILE} --output_file ${MERGED_PRED_FILE}
    python3 tools/evaluate.py --pred_file ${MERGED_PRED_FILE} --ref_file ${DATA}/${TASK}/test.ref >> res/score.txt
    done
done
