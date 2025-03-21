#!/bin/bash
#SBATCH --job-name=latent_gifs      # Job name
#SBATCH --output=latent_gifs.out    # Output file (%j will be replaced with the job ID)
#SBATCH --time=36:00:00
#SBATCH --partition=gpuqueue
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:3g.39gb:1

python latent_gif_maker.py