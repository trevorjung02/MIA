# Basic Run with Prefix-generated z's
python src/MIA/MIA_llama.py --target_path=\<target model checkpoint\> --gamma=-1 --num_z=100 --input_path=\<dataset path\> --z_sampling=prefix --idx_frac=0.1

# Run with Perturb-generated z's
python src/MIA/MIA_llama.py --target_path=\<target model checkpoint\> --gamma=-1 --num_z=100 --input_path=\<dataset path\> --z_sampling=perturb --idx_frac=0.1

# Command Line Arguments 
--z_sampling: set to 'prefix' or 'perturb' for diffferent z sampling methods \
-perturb_generation: Use this flag if z's should be generated with chain rule \
-perturb_span: Use this flag if the perturbed indices should be selected as spans instead of uniformally randomly \
--idx_frac: The maximum fraction of indices to perturb \
-fixed: Use this flag to perturb a fixed percentage of indices (equal to idx_frac). Without this flag, each z will randomly select a fraction from 0 to idx_frac of the indices to perturb. \
--num_z: The number of neighbors to generate per sentence x

# Using different Models
--target_model: Name of the target model on huggingface \
--target_path: path to a checkpoint for the target model \
--ref_model: Name of the reference model on huggingface \
--ref_path: path to a checkpoint for the reference model

# Using different datasets
--input_path: path to the dataset. The dataset should be a jsonl file, with each row representing a different piece of data. The text of the sentence within each row should be stored with the key 'input'
