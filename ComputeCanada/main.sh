#!/bin/bash
#SBATCH --exclusive
#SBATCH --account=def-lseoud #tranchon
#SBATCH --job-name=CenterNet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
##SBATCH --partition=multigpu
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=output/CenterNet.%j.out
#SBATCH --error=output/CenterNet.%j.err
eval "$(conda shell.bash hook)"
source ../env/bin/activate

# Define temporary variables
DF_SD=/home/tranchon/projects/def-lseoud/tranchon/code/

EPOCH=100
BATCH_SIZE=20
NUM_WORKERS=4
TRAIN_FILE=../dataset/train.csv
TRAIN_DIR=../dataset/train_images/
TEST_DIR=../dataset/test_images/
AUGMENT_DATA=False
LESLIE="true"
LR_MIN=0.0001
LR_MAX=1
DATASET=Kuzushiji
FILE_NAME_SAVE="run1"
SCHEDULER="None"

python main.py --epoch $EPOCH --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --dataset $DATASET --file_name_save $FILE_NAME_SAVE --train_file $TRAIN_FILE --train_dir $TRAIN_DIR --test_dir $TEST_DIR --augment_data $AUGMENT_DATA --leslie $LESLIE --lr_min $LR_MIN --lr_max $LR_MAX --scheduler $SCHEDULER
