
DIR=$(pwd)
DATA=$(pwd)/data
MODEL=$(pwd)/model
RES=$(pwd)/res
NUM_GPU=$(nvidia-smi --list-gpus | wc -l)
export CURL_CA_BUNDLE=""
export WANDB_DISABLED=true
for TASK in 'CSDS' 'MedQA'
do
    STAGE_1_TASK=${TASK}_seq2seq_Q_binary_cls
    STAGE_2_TASK=${TASK}_seq2seq_A_no_context


    

    for CLASSIFIER in 'hfl/chinese-roberta-wwm-ext-large'
    do
        CLASSIFIER_PATH=${MODEL}/${STAGE_2_TASK}/$(basename ${CLASSIFIER})

        # python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} 
        python3 src/run_glue.py \
            --model_name_or_path ${CLASSIFIER} \
            --train_file ${DATA}/${STAGE_2_TASK}/train.json \
            --validation_file ${DATA}/${STAGE_2_TASK}/dev.json \
            --test_file ${DATA}/${STAGE_2_TASK}/test.json \
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
    done
done
