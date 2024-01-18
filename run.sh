#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tjung2@uw.edu
#SBATCH --output=slurm_new/slurm-%j.out
# --array=1,2
#--dependency=afterok:15937969

# I use source to initialize conda into the right environment.
cat $0
echo "--------------------"

source ~/.bashrc
conda activate mia  

# Target paths
# target_path=checkpoints/gpt2/run_5/epoch-0_perplexity-23.6118
# target_path=checkpoints/gpt2/run_5/epoch-3_perplexity-20.9533

# target_path=checkpoints/gpt2/run_8/epoch-2_perplexity-68.3903

# models
# --generation_model=facebook/opt-350m
# --ref_model=EleutherAI/pythia-1b-deduped
# huggyllama/llama-7b

srun python src/MIA/MIA_llama.py --target_path=checkpoints/gpt2/run_5/epoch-3_perplexity-20.9533 --gamma=-1 --num_z=25 --input_path=data/newsSpace_oracle_debug.jsonl --z_sampling=perturb --idx_frac=0.1 -fixed 

# srun python src/MIA/MIA_llama.py --target_path=checkpoints/gpt2/run_8/epoch-2_perplexity-68.3903 --gamma=-1 --num_z=500 --input_path=data/sentiment140_oracle_debug.jsonl -online --z_sampling=prefix --prefix_length=0.1 -perturb --idx_frac=0.4 

# srun python src/MIA/MIA_llama.py --target_model=EleutherAI/pythia-160m-deduped --target_path=checkpoints/gpt2/run_4/epoch-2_perplexity-30.5387 --ref_model=EleutherAI/pythia-160m-deduped --gamma=-1 --num_z=500 --input_path=data/newsSpace_oracle_debug.jsonl -online --z_sampling=prefix --prefix_length=0.1 -perturb --idx_frac=0.4
# srun python src/train.py --train_path=data/newsSpace_oracle_target_train.csv --val_path=data/newsSpace_oracle_val.csv --model_name=EleutherAI/pythia-160m-deduped
# srun python src/train.py --train_path=sentiment140 --val_path=sentiment140 --model_name=EleutherAI/pythia-160m-deduped

# srun conda env update -f environment.yml
# srun jupyter-lab
