#!/bin/bash
#SBATCH --partition=gpu-l40
#SBAtCH --account=ark
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --output=slurm_new/slurm-%j.out

# I use source to initialize conda into the right environment.
cat $0
echo "--------------------"

source ~/.bashrc
conda activate mia  

# Target paths
# NewsSpace
# GPT2
# target_path=checkpoints/gpt2/run_5/epoch-0_perplexity-23.6118
# target_path=checkpoints/gpt2/run_5/epoch-3_perplexity-20.9533

# Sentiment140
# GPT2
# target_path=checkpoints/gpt2/run_8/epoch-2_perplexity-68.3903

# Wikitext
# GPT2 
# checkpoints/gpt2/run_10/epoch-2_perplexity-26.4751

# models
# --generation_model=facebook/opt-350m
# --ref_model=EleutherAI/pythia-1b-deduped
# huggyllama/llama-7b

# srun python analysis_ratios.py
# srun python analysis.py

# srun python src/MIA/MIA_llama.py --target_model=openai-community/gpt2-medium --ref_model=openai-community/gpt2 --gamma=-1 --input_path=data/wikimia.jsonl --z_sampling=perturb --idx_frac=0.1 -fixed --num_z=100

# srun python src/MIA/MIA_llama.py --target_model=swj0419/7b_finetuned_llama2_3epoch --ref_model=meta-llama/Llama-2-7b-hf --gamma=-1 --input_path=data/bookmia_debug.jsonl --z_sampling=perturb --idx_frac=0.1 -fixed --num_z=100

# srun python src/MIA/MIA_llama.py --target_model=EleutherAI/pythia-160m-deduped --target_path=checkpoints/gpt2/run_12/epoch-3_perplexity-111.0108 --ref_model=EleutherAI/pythia-160m-deduped --gamma=-1 --input_path=data/sentiment140_oracle_debug.jsonl --z_sampling=perturb --idx_frac=0.1 -fixed --num_z=250 -no_perturbation

# srun python src/MIA/MIA_llama.py --target_path=checkpoints/gpt2/run_10/epoch-2_perplexity-26.4751 --gamma=-1 --input_path=data/wikitext_oracle_debug.jsonl --z_sampling=perturb --idx_frac=0.1 -fixed --num_z=250 -no_perturbation

# srun python src/MIA/MIA_llama.py --target_path=checkpoints/gpt2/run_5/epoch-3_perplexity-20.9533 --gamma=-1 --input_path=data/newsSpace_oracle_debug.jsonl --z_sampling=perturb --idx_frac=0.1 -fixed --num_z=50 

# srun python src/MIA/MIA_llama.py --target_path=checkpoints/gpt2/run_5/epoch-3_perplexity-20.9533 --gamma=-1 --num_z=250 --input_path=data/newsSpace_oracle_tiny.jsonl --z_sampling=perturb --idx_frac=0.1 -fixed

# srun python src/MIA/MIA_llama.py --target_path=checkpoints/gpt2/run_8/epoch-2_perplexity-68.3903 --gamma=-1 --input_path=data/sentiment140_oracle_debug.jsonl --z_sampling=perturb --idx_frac=0.7 -fixed -no_perturbation

# srun python src/MIA/MIA_llama.py --target_model=EleutherAI/pythia-160m-deduped --target_path=checkpoints/gpt2/run_4/epoch-2_perplexity-30.5387 --ref_model=EleutherAI/pythia-160m-deduped --gamma=-1 --num_z=250 --input_path=data/newsSpace_oracle_debug.jsonl --z_sampling=perturb --idx_frac=0.1 -fixed -no_perturbation

# srun python src/train.py --train_path=data/newsSpace_oracle_target_train.csv --val_path=data/newsSpace_oracle_val.csv --model_name=EleutherAI/pythia-160m-deduped
# srun python src/train.py --train_path=sentiment140 --val_path=sentiment140 --model_name=EleutherAI/pythia-160m-deduped

# srun conda env update -f environment.yml
# srun jupyter-lab
