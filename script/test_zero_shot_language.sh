
DIR=$(pwd)
DATA=$(pwd)/data
MODEL=$(pwd)/model
RES=$(pwd)/res
NUM_GPU=$(nvidia-smi --list-gpus | wc -l)
export CURL_CA_BUNDLE=""

PRED_TASK='zero_shot_MultiDoGO'
for model_name in google/mt5-xl #  google/mt5-base google/mt5-large
do
    for TRAINED_TASK in 'CSDS' 'MedQA'
    do
        for DOMAIN in 'airline' 'fastfood' 'finance' 'insurance' 'media' 'software'
        do
            OUTPUT_DIR=${MODEL}/${TRAINED_TASK}_seq2seq/$(basename ${model_name})
            echo ${OUTPUT_DIR}
            python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=12345  src/run_summarization.py \
                --model_name_or_path ${OUTPUT_DIR} \
                --do_predict \
                --validation_file ${DATA}/${PRED_TASK}/${DOMAIN}_${TRAINED_TASK}.json \
                --test_file ${DATA}/${PRED_TASK}/${DOMAIN}_${TRAINED_TASK}.json \
                --output_dir ${OUTPUT_DIR} \
                --overwrite_cache \
                --per_device_eval_batch_size 4 \
                --prediction_file ${PRED_TASK}_${DOMAIN}_${TRAINED_TASK} \
                --predict_with_generate \
                --generation_num_beams 4 \
                --sharded_ddp simple 
        done
    done
done