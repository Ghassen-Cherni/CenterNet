#!/bin/bash
#SBATCH --exclusive
#SBATCH --account=def-lseoud #tranchon
#SBATCH --job-name=CenterNet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
##SBATCH --partition=multigpu
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=output/CenterNet.%j.out
#SBATCH --error=output/CenterNet.%j.err
eval "$(conda shell.bash hook)"
source ../env/bin/activate

# Define temporary variables
DF_SD=/home/tranchon/projects/def-lseoud/tranchon/INF8225/

EPOCH=20
BATCH_SIZE=20
NUM_WORKERS=4
TRAIN_FILE=train.csv
TRAIN_DIR=train_images/
TEST_DIR=test_images/

python main.py --epoch $EPOCH --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --train_file $TRAIN_FILE --train_dir $TRAIN_DIR --test_dir $TEST_DIR
