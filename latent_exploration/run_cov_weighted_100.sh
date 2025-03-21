#!/bin/bash
#SBATCH --job-name=segger_experiments      # Job name
#SBATCH --output=segger_experiments_w100.out    # Output file (%j will be replaced with the job ID)
#SBATCH --time=48:00:00
#SBATCH --partition=gpuqueue
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:3g.39gb:1

python train_segger_experiments.py \
  --cov_data_dir "/data/peer/riffled/segger/development_tools/data/xenium_human_colon/datasets_with_cov_with_genes_cleaned_for_nuclear" \
  --no_cov_data_dir /data/peer/riffled/segger/development_tools/data/xenium_human_colon/datasets_with_cov_with_genes_cleaned_for_nuclear \
  --epochs 50 \
  --batch_size_train 3 \
  --num_tx_tokens 500 \
  --model_dir /data/peer/riffled/segger/development_tools/data/xenium_human_colon/models/models_k=8_with_cov_notebook_no_cov \
  --transcripts_parquet /data/peer/riffled/segger/development_tools/data/xenium_human_colon/xenium/transcripts.parquet \
  --bce_loss=True \
  --triplet_loss=False \
  --enrich_global_edges=False \
  --global_gene_cov_weight 100.0 \
  --output_dir data/peer/riffled/segger/development_tools/segger_experiments/bce_true_triplet_false_enrich_global_edges_false_global_gene_cov_weight_100.0

python train_segger_experiments.py \
  --cov_data_dir "/data/peer/riffled/segger/development_tools/data/xenium_human_colon/datasets_with_cov_with_genes_cleaned_for_nuclear" \
  --no_cov_data_dir /data/peer/riffled/segger/development_tools/data/xenium_human_colon/datasets_with_cov_with_genes_cleaned_for_nuclear \
  --epochs 50 \
  --batch_size_train 3 \
  --num_tx_tokens 500 \
  --model_dir /data/peer/riffled/segger/development_tools/data/xenium_human_colon/models/models_k=8_with_cov_notebook_no_cov \
  --transcripts_parquet /data/peer/riffled/segger/development_tools/data/xenium_human_colon/xenium/transcripts.parquet \
  --bce_loss=True \
  --triplet_loss=False \
  --enrich_global_edges=True \
  --global_gene_cov_weight 100.0 \
  --output_dir data/peer/riffled/segger/development_tools/segger_experiments/bce_true_triplet_false_enrich_global_edges_true_global_gene_cov_weight_100.0

python train_segger_experiments.py \
  --cov_data_dir "/data/peer/riffled/segger/development_tools/data/xenium_human_colon/datasets_with_cov_with_genes_cleaned_for_nuclear" \
  --no_cov_data_dir /data/peer/riffled/segger/development_tools/data/xenium_human_colon/datasets_with_cov_with_genes_cleaned_for_nuclear \
  --epochs 50 \
  --batch_size_train 3 \
  --num_tx_tokens 500 \
  --model_dir /data/peer/riffled/segger/development_tools/data/xenium_human_colon/models/models_k=8_with_cov_notebook_no_cov \
  --transcripts_parquet /data/peer/riffled/segger/development_tools/data/xenium_human_colon/xenium/transcripts.parquet \
  --bce_loss=Fals  \
  --triplet_loss=False \
  --enrich_global_edges=True \
  --global_gene_cov_weight 100.0 \
  --output_dir data/peer/riffled/segger/development_tools/segger_experiments/bce_false_triplet_false_enrich_global_edges_true_global_gene_cov_weight_100.0

python train_segger_experiments.py \
  --cov_data_dir "/data/peer/riffled/segger/development_tools/data/xenium_human_colon/datasets_with_cov_with_genes_cleaned_for_nuclear" \
  --no_cov_data_dir /data/peer/riffled/segger/development_tools/data/xenium_human_colon/datasets_with_cov_with_genes_cleaned_for_nuclear \
  --epochs 50 \
  --batch_size_train 3 \
  --num_tx_tokens 500 \
  --model_dir /data/peer/riffled/segger/development_tools/data/xenium_human_colon/models/models_k=8_with_cov_notebook_no_cov \
  --transcripts_parquet /data/peer/riffled/segger/development_tools/data/xenium_human_colon/xenium/transcripts.parquet \
  --bce_loss=Fals  \
  --triplet_loss=False \
  --enrich_global_edges=False \
  --global_gene_cov_weight 100.0 \
  --output_dir data/peer/riffled/segger/development_tools/segger_experiments/bce_false_triplet_false_enrich_global_edges_false_global_gene_cov_weight_100.0