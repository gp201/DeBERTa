#!/bin/bash

set -exo pipefail

REPO_DIR=$(git rev-parse --show-toplevel)

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR
DATE=$(date '+%Y-%m-%d|%H:%M:%S')

cache_dir=~/.cache/deberta

max_seq_length=256
data_dir=$cache_dir/data

assets_dir=/home/deberta/.~DeBERTa/assets/latest/deberta-v3-base

mkdir -p $assets_dir $data_dir $cache_dir

cp $REPO_DIR/vocab.txt $assets_dir/vocab.txt

function setup_wiki_data(){
	task=$1
	mkdir -p $cache_dir
	if [[ ! -e  $cache_dir/spm.model ]]; then
		wget -q https://huggingface.co/microsoft/deberta-v3-base/resolve/main/spm.model -O $cache_dir/spm.model
	fi

	if [[ ! -e  $data_dir/test.txt ]]; then
		wget -q https://zenodo.org/record/8253367/files/train-test-eval_unpaired.tar.gz?download=1 -O $cache_dir/train-test-eval_unpaired.tar.gz
		wait
		tar -xzf $cache_dir/train-test-eval_unpaired.tar.gz -C $cache_dir
		wait
		python ./prepare_data.py -i $cache_dir/train-test-eval_unpaired/train.txt -o $data_dir/train.txt --max_seq_length $max_seq_length --vocab_path $assets_dir/vocab.txt
		python ./prepare_data.py -i $cache_dir/train-test-eval_unpaired/eval.txt -o $data_dir/valid.txt --max_seq_length $max_seq_length --vocab_path $assets_dir/vocab.txt
		python ./prepare_data.py -i $cache_dir/train-test-eval_unpaired/test.txt -o $data_dir/test.txt --max_seq_length $max_seq_length --vocab_path $assets_dir/vocab.txt
	fi
}

setup_wiki_data

Task=RTD

init=$1
tag=$init
case ${init,,} in
	deberta-v3-xsmall-continue)
	# wget https://huggingface.co/microsoft/deberta-v3-xsmall/resolve/main/pytorch_model.generator.bin
	# wget https://huggingface.co/microsoft/deberta-v3-xsmall/resolve/main/pytorch_model.bin
	parameters=" --num_train_epochs 1 \
	--model_config rtd_xsmall.json \
	--warmup 10000 \
	--num_training_steps 100000 \
	--learning_rate 5e-5 \
	--train_batch_size 256 \
	--init_generator <TODO: generator checkpoint> \
	--init_discriminator <TODO: discriminator checkpoint> \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-xsmall)
	parameters=" --num_train_epochs 20 \
	--model_config rtd_xsmall.json \
	--warmup 10000 \
	--learning_rate 3e-4 \
	--train_batch_size 64 \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-small-continue)
	# wget https://huggingface.co/microsoft/deberta-v3-small/resolve/main/pytorch_model.generator.bin
	# wget https://huggingface.co/microsoft/deberta-v3-small/resolve/main/pytorch_model.bin
	parameters=" --num_train_epochs 1 \
	--model_config rtd_small.json \
	--warmup 10000 \
	--num_training_steps 100000 \
	--learning_rate 5e-5 \
	--train_batch_size 256 \
	--init_generator <TODO: generator checkpoint> \
	--init_discriminator <TODO: discriminator checkpoint> \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-base)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_base.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-large)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_large.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--decoupled_training True \
	--fp16 True "
		;;
	*)
		echo "usage $0 <Pretrained model configuration>"
		echo "Supported configurations"
		echo "deberta-v3-xsmall - Pretrained DeBERTa v3 XSmall model with 9M backbone network parameters (12 layers, 256 hidden size) plus 32M embedding parameters(128k vocabulary size)"
		echo "deberta-v3-xsmall - Pretrained DeBERTa v3 Base model with 81M backbone network parameters (12 layers, 768 hidden size) plus 96M embedding parameters(128k vocabulary size)"
		echo "deberta-v3-xsmall - Pretrained DeBERTa v3 Large model with 288M backbone network parameters (24 layers, 1024 hidden size) plus 128M embedding parameters(128k vocabulary size)"
		exit 0
		;;
esac


# WANDB env variables
export WANDB_JOB_TYPE="Pretrain"
export WANDB_RUN_GROUP="Testing"
export WANDB_NAME="DeBERTa-$init-$Task-$DATE"
export WANDB_PROJECT="gp-deberta"
export WANDB_TAGS="testing"
export WANDB_DIR="~/models"

python -m DeBERTa.apps.run --model_config config.json  \
	--tag $tag \
	--do_train \
	--num_training_steps 100000 \
	--max_seq_len $max_seq_length \
	--dump 10000 \
	--task_name $Task \
	--data_dir $data_dir \
	--vocab_path $assets_dir/vocab.txt \
	--vocab_type custom \
	--output_dir ~/models \
	--wandb  $parameters
