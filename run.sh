#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
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

# Target paths
# target_path=checkpoints/gpt2/run_5/epoch-0_perplexity-23.6118
# target_path=checkpoints/gpt2/run_5/epoch-3_perplexity-20.9533

# Ref models
# ref_model=EleutherAI/pythia-1b-deduped

srun python src/MIA/MIA_llama.py --target_path=checkpoints/gpt2/run_5/epoch-3_perplexity-20.9533 --gamma=-1 --num_z=100 --input_path=data/newsSpace_oracle_debug.jsonl -online --z_sampling=prefix --prefix_length=0.1 -contrastive_dec -ref_z
# srun python src/train.py --train_path=data/newsSpace_other_ref_train.csv --val_path=data/newsSpace_oracle_val.csv

# srun conda env update -f environment.yml
# srun jupyter-lab