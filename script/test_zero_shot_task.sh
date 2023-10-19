
DIR=$(pwd)
DATA=$(pwd)/data
MODEL=$(pwd)/model
RES=$(pwd)/res
NUM_GPU=$(nvidia-smi --list-gpus | wc -l)
export CURL_CA_BUNDLE=""

for model_name in google/mt5-xl # google/mt5-base google/mt5-large 
do
    for TRAINED_TASK in 'carsales_dhllog_ketang' # 'carsales' 'dhl_log' 'ketang'
    do
        for TEST_TASK in 'carsales' 'dhl_log' 'ketang'
        do
            OUTPUT_DIR=${MODEL}/${TRAINED_TASK}/$(basename ${model_name})
            echo ${OUTPUT_DIR}
            python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=12345  src/run_summarization.py \
                --model_name_or_path ${OUTPUT_DIR} \
                --do_predict \
                --validation_file ${DATA}/${TEST_TASK}/test.json \
                --test_file ${DATA}/${TEST_TASK}/test.json \
                --output_dir ${OUTPUT_DIR} \
                --overwrite_cache \
                --per_device_eval_batch_size 4 \
                --prediction_file ${TEST_TASK}_trained_on_${TRAINED_TASK} \
                --predict_with_generate \
                --generation_num_beams 4 \
                --sharded_ddp simple 
            python3 tools/evaluate.py --pred_file ${OUTPUT_DIR}/${TEST_TASK}_trained_on_${TRAINED_TASK}.txt --ref_file ${DATA}/${TEST_TASK}/test.ref >> res/score.txt

        done
    done
done