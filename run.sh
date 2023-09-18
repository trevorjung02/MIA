#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tjung2@uw.edu
#SBATCH --output=slurm_new/slurm-%j.out

# I use source to initialize conda into the right environment.
cat $0
echo "--------------------"

source ~/.bashrc
conda activate mia

srun python src/MIA/MIA_llama.py --target_path=checkpoints/gpt2/run_5/epoch-3_perplexity-20.9533 --ref_path=checkpoints/gpt2/run_2/epoch-4_perplexity-20.8704 --gamma=1 --num_z=6000
# srun python src/detectGPT.py
# srun python src/train.    py --train_path=data/newsSpace_other_ref_train.csv --val_path=data/newsSpace_oracle_val.csv

# srun conda env update -f environment.yml
# srun jupyter-lab