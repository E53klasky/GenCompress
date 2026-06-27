#!/bin/bash
#SBATCH --job-name=2D_train_caesar  # Job name
#SBATCH --output=logs/vae_model_dim4.out  # Std output log
#SBATCH --error=logs/vae_model_dim4.err   # Std error log
#SBATCH --mail-type=ALL                     # Email notifications for all job states
#SBATCH --mail-user=klaskyethan@gmail.com  # Email address for notifications
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks (processes)
#SBATCH --cpus-per-task=18                   # Number of CPU cores per task
#SBATCH --mem=650GB                       # Memory per node
#SBATCH --partition=hpg-b200                     # GPU partition
#SBATCH --gpus=b200                      # Number of GPUs (A100)
#SBATCH --time=72:00:00                     # Maximum job runtime

echo "Date       = $(date)"
echo "Host       = $(hostname -s)"
echo "Directory  = $(pwd)"

module purge
export OMP_NUM_THREADS=18
source ../data/test/set_env_ufl_caesar.sh
source ../caesar_venv/bin/activate

T1=$(date +%s)
# --train_set="S3D,JHTDB,Hurricane,ERA5,Sunquake,Blastnet" \
# Run the VAE3D training script

vae_path="./snapshots/vae/laten_dim/model_dim4"
train_set="E3SM,ERA5,HYCOM,JHTDB,S3D,Cavity2D,Hurricane,PDE,TUM,GX_42x83,GX_2x83,GX_2x42,GX_2x96"
test_set="E3SM_test"


torchrun --num_workers=3 python3 train_vae2d.py \
    --save_path=$vae_path \
    --batch_size=128 \
    --iterations=500 \
    --model_dim=4 \
    --lr=0.0005 \
    --beta_start=0.5 \
    --train_set=$train_set \
    --test_set=$test_set \
    --init_beta=0.00001 \
    --end_beta=0.00002\
    --num_workers=18
    --sr_dim=-1\
    # --pretrain="./snapshots/vae/e3sm/train_on_5Sets/model_bs32_ep200k.pt"

T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED seconds"
