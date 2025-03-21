import glob
import os
import imageio
import re
from natsort import natsorted

def create_gif(image_files, output_path, duration=0.5):
    """Create a GIF from a list of image files."""
    images = [imageio.imread(img) for img in image_files]
    imageio.mimsave(output_path, images, duration=duration)
    print(f"Saved GIF: {output_path}")

def process_experiment_directory(root_dir):
    """Process each experiment directory and generate GIFs."""
    for experiment in os.listdir(root_dir):
        experiment_path = os.path.join(root_dir, experiment)
        
        if os.path.isdir(experiment_path) and experiment.startswith("bce"):
            print(f"Processing experiment: {experiment}")
            
            for trial in os.listdir(experiment_path):
                trial_path = os.path.join(experiment_path, trial)
                
                if os.path.isdir(trial_path):
                    print(f"  Processing trial: {trial}")
    
                    umap_transcripts_files = natsorted(glob.glob(os.path.join(trial_path, "umap_epoch_*_transcripts_by_celltype.png")))
                    
                    if umap_transcripts_files:
                        gif_transcripts_path = os.path.join(trial_path, "umap_epochs_transcripts_by_celltype.gif")
                        create_gif(umap_transcripts_files, gif_transcripts_path)

if __name__ == "__main__":
    root_directory = "data/peer/riffled/segger/development_tools/segger_experiments"
    process_experiment_directory(root_directory)
