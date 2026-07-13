#!/bin/bash
#SBATCH --job-name=2D_train_caesar
#SBATCH --output=logs/vae_model_dim4.out
#SBATCH --error=logs/vae_model_dim4.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=klaskyethan@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=720GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=4
#SBATCH --time=12:00:00

echo "Date       = $(date)"
echo "Host       = $(hostname -s)"
echo "Directory  = $(pwd)"

module purge
source ../data/test/set_env_ufl_caesar.sh
source ../caesar_venv/bin/activate

export OMP_NUM_THREADS=32

T1=$(date +%s)

vae_path="./snapshots/vae/laten_dim/model_dim4"
train_set="E3SM,ERA5,HYCOM,JHTDB,S3D,Cavity2D,Hurricane,PDE,TUM,GX_42x83,GX_2x83,GX_2x42,GX_2x96,Microscopy,Turb_Rot,S3D_step3,GX_ion_imag,GX_ion_real,GX_electron_real"
test_set="E3SM_test"

torchrun --standalone --nproc_per_node=4 train_vae2d.py \
    --save_path=$vae_path \
    --batch_size=128 \
    --iterations=2500 \
    --model_dim=4 \
    --lr=0.0005 \
    --beta_start=0.5 \
    --train_set=$train_set \
    --test_set=$test_set \
    --init_beta=0.00001 \
    --end_beta=0.00002 \
    --num_workers=8 \
    --sr_dim=-1

T2=$(date +%s)
ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED seconds"
