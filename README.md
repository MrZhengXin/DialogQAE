# DialogQAE: N-to-N Question Answer Pair Extraction from Customer Service Chatlog

The fine-tuned models are available at Huggingface: [MedQA](https://huggingface.co/ExpectoZX/DialogQAE_CSDS) / [CSDS](https://huggingface.co/ExpectoZX/DialogQAE_CSDS)

Install huggingface transformers
```
pip install -r requirements.txt
SRC=$(pwd)/src
git clone https://github.com/huggingface/transformers.git
pip install -e .
wandb login
cp examples/pytorch/summarization/run_summarization.py ${SRC}
```


Put data at ```data/``` directory, with the format of 
```
{"text": "P: 今天上午做的( <extra_id_0> ); P: 各项数据都正常吗?有暗区问题大吗?( <extra_id_1> ); 
D: 依这个数据看,没有大问题的,放心哈( <extra_id_2> ); 
D: 怀孕几个月了( <extra_id_3> ); P: 一八周,请问有暗区是怎么回事儿。我们应该注意什么( <extra_id_4> );", 
"summary": "<extra_id_0> O <extra_id_1> Q1 <extra_id_2> A1 <extra_id_3> Q2 <extra_id_4> Q3"}
```


Train and evaluate by ```bash script/train.sh```

```

DIR=$(pwd)
DATA=$(pwd)/data
MODEL=$(pwd)/model
RES=$(pwd)/res
NUM_GPU=$(nvidia-smi --list-gpus | wc -l)

for model_name in google/mt5-xl 
do
    for TASK in 'MedQA_seq2seq'
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
            --validation_file ${DATA}/${TASK}/dev.json \
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
        rm -rf ${OUTPUT_DIR}/checkpoint*

        # cd ${DIR}
        python3 tools/evaluate.py --pred_file ${OUTPUT_DIR}/generated_predictions.txt --ref_file ${DATA}/${TASK}/test.ref >> res/score.txt

    done
done
```

Or use deepspeed training for large model:
```

DIR=$(pwd)
DATA=$(pwd)/data
MODEL=$(pwd)/model
RES=$(pwd)/res
NUM_GPU=$(nvidia-smi --list-gpus | wc -l)

for model_name in google/mt5-xl 
do
    for TASK in 'MedQA_seq2seq'
    do
        OUTPUT_DIR=${MODEL}/${TASK}/$(basename ${model_name})
        echo ${OUTPUT_DIR}
        # cd ../transformers/examples/pytorch/summarization
        deepspeed --num_gpus=${NUM_GPU} --master_port=12345  src/run_summarization.py \
            --model_name_or_path ${model_name} \
            --do_train \
            --do_eval \
            --train_file ${DATA}/${TASK}/train.json \
            --validation_file ${DATA}/${TASK}/dev.json \
            --test_file ${DATA}/${TASK}/test.json \
            --output_dir ${OUTPUT_DIR} \
            --overwrite_output_dir \
            --num_train_epochs 6 \
            --overwrite_cache \
            --per_device_train_batch_size 3 \
            --gradient_accumulation_steps 2 \
            --per_device_eval_batch_size 4 \
            --evaluation_strategy epoch \
            --learning_rate 3e-5 \
            --save_strategy no \
            --warmup_steps 50 \
            --logging_steps 500 \
            --deepspeed src/ds_config_zero3.json 
        rm -rf ${OUTPUT_DIR}/checkpoint*

        python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=12345  src/run_summarization.py \
            --model_name_or_path ${model_name} \
            --do_predict \
            --do_eval \
            --validation_file ${DATA}/${TASK}/dev.json \
            --test_file ${DATA}/${TASK}/test.json \
            --output_dir ${OUTPUT_DIR} \
            --overwrite_output_dir \
            --num_train_epochs 6 \
            --overwrite_cache \
            --per_device_eval_batch_size 4 \
            --evaluation_strategy epoch \
            --predict_with_generate \
            --generation_num_beams 4 \
            --logging_steps 500 \
            --sharded_ddp simple 


        # cd ${DIR}
        python3 tools/evaluate.py --pred_file ${OUTPUT_DIR}/generated_predictions.txt --ref_file ${DATA}/${TASK}/test.ref >> res/score.txt

    done
done

```



Citation
```
@article{zheng2022dialogqae,
  title={DialogQAE: N-to-N Question Answer Pair Extraction from Customer Service Chatlog},
  author={Zheng, Xin and Liu, Tianyu and Meng, Haoran and Wang, Xu and Jiang, Yufan and Rao, Mengliang and Lin, Binghuai and Sui, Zhifang and Cao, Yunbo},
  journal={arXiv preprint arXiv:2212.07112},
  year={2022}
}
```
