"""
Training script for DTM with LunarLander environment using iterator-based data loading.

This script demonstrates how to train a DTM on data generated on-the-fly from a gym environment,
without needing to load a full dataset into memory. The model is trained in an unconditioned manner
to generate random LunarLander frames.

Key features:
- Iterator-based data loading (no full dataset in memory)
- Unconditioned generation (no class labels)
- RGB images (32x32 with 3 color channels = 3072 pixels)
- On-the-fly data generation from LunarLander-v3 environment
"""

import sys
sys.path.insert(0, '/home/ubuntu/Extropic_Hackaton')

from thrmlDenoising.DTM import DTM
from thrmlDenoising.utils import make_cfg
from thrmlDenoising.iterator_dataloader import IteratorDataLoader
from src.dataloader import GymDataGenerator

# ---------- Data parameters ----------
# For iterator-based training with unconditioned generation
data_params = {
    "dataset_name": "iterator_lunarlander",  # Custom name for logging
    "target_classes": (0,),  # Single dummy class for unconditioned generation
    "pixel_threshold_for_single_trials": 0.5,  # Threshold for binarization
}

# ---------- Graph parameters ----------
# Configured for 32x32 RGB images (3072 pixels total)
# IMPORTANT: With PoissonBinomialIsingGraphManager, each pixel needs n_trials=grayscale_levels spins
# So total nodes = n_pixels × grayscale_levels = 3072 × grayscale_levels
# Must satisfy: n_pixels × grayscale_levels < grid_size/2
graph_params = {
    "graph_preset_architecture": 80_12,  # Grid: side=80, degree=12 (size=6400, capacity=3200)
    "num_label_spots": 1,  # Minimal labels for unconditioned generation
    "grayscale_levels": 1,  # Binary images (1 bit per channel: 0 or 1)
                            # Using 1 means: 3072 pixels × 1 = 3072 nodes (fits in 3200 capacity)
                            # Using 4 would mean: 3072 × 4 = 12,288 nodes (too large!)
    "torus": True,
    "base_graph_manager": "poisson_binomial_ising_graph_manager",
}

# ---------- Sampling (Gibbs/CD) schedule parameters ----------
sampling_params = {
    "batch_size": 50,  # Smaller batch size for faster iteration
    "n_samples": 30,  # Fewer samples for faster training
    "steps_per_sample": 6,
    "steps_warmup": 300,  # Reduced warmup
    "training_beta": 1.0,
}

# ---------- Image Generation parameters ----------
generation_params = {
    "generation_beta_start": 0.8,
    "generation_beta_end": 1.2,
    "fid_images_per_digit": 100,  # Fewer images for evaluation
    "steps_warmup": 400,
}

# ---------- Diffusion schedule (time grid) parameters ----------
diffusion_schedule_params = {
    "num_diffusion_steps": 1,  # Start with single step for simplicity
    "kind": "log",
    "diffusion_offset": 0.1,
}

# ---------- Diffusion rates (forward/noising) parameters ----------
diffusion_rates_params = {
    "image_rate": 0.8,
    "label_rate": 0.2,  # Not used in unconditioned training
}

# ---------- Optim / LR decay parameters ----------
optim_params = {
    "momentum": 0.9,
    "b2_adam": 0.999,
    "step_learning_rates": (0.03,),  # Lower learning rate for stability
    "alpha_cosine_decay": 0.2,
    "n_epochs_for_lrd": 20,  # Cosine decay over 20 epochs
}

# ---------- Regularization: Correlation Penalty parameters ----------
cp_params = {
    "correlation_penalty": (0.0,),
    "adaptive_cp": False,  # Disabled for iterator training
    "cp_min": 0.001,
    "adaptive_threshold": 0.016,
}

# ---------- Regularization: Weight Decay parameters ----------
wd_params = {
    "weight_decay": (0.0,),
    "adaptive_wd": False,  # Disabled for iterator training
    "wd_min": 0.001,
}

# ---------- Meta / run parameters ----------
exp_params = {
    "seed": 42,
    "graph_seeds": (),
    "descriptor": "lunarlander_rgb",
    "n_cores": 1,
    "compute_autocorr": False,  # Disabled for iterator training (requires test dataset)
    "generate_gif": True,  # Now supports RGB GIF generation!
    "drawn_images_per_digit": 4,
    "animated_images_per_digit": 2,
    "steps_per_sample_in_gif": 10,
}

# ---------- Training parameters ----------
n_epochs = 20  # Number of epochs to train
batches_per_epoch = 100  # Number of batches per epoch (controls epoch length)
evaluate_every = 1  # Evaluate and save every N epochs (0 to disable)

# ---------- LunarLander Data Generator parameters ----------
# IMPORTANT: Now using ALL frames from each episode (not just the last one!)
# Each episode with state_size=32 yields 32 individual training frames
gym_params = {
    "state_size": 32,  # Number of frames to collect per episode
    "environment_name": "LunarLander-v3",
    "training_examples": 160,  # Episodes per epoch (each yields 32 frames = ~5120 training samples)
    "autoencoder_time_compression": 4,
    "return_anyways": True,  # Return frames even if lander leaves the frame
    "resolution": 32,  # 32x32 resolution (matches graph parameters)
}

# Data efficiency calculation:
# - batches_per_epoch=100 × batch_size=50 = 5000 samples needed per epoch
# - training_examples=160 episodes × state_size=32 frames = 5120 frames available
# - Perfect! Just enough data to fill all batches without repetition

# Build the config
cfg = make_cfg(
    exp=exp_params,
    data=data_params,
    graph=graph_params,
    sampling=sampling_params,
    generation=generation_params,
    diffusion_schedule=diffusion_schedule_params,
    diffusion_rates=diffusion_rates_params,
    optim=optim_params,
    cp=cp_params,
    wd=wd_params,
)

# Calculate n_image_pixels for RGB 32x32 images
n_image_pixels = 32 * 32 * 3  # 3072 pixels

print(f"Initializing DTM for iterator-based training...")
print(f"Image dimensions: {gym_params['resolution']}x{gym_params['resolution']} RGB = {n_image_pixels} pixels")
print(f"Grayscale levels: {graph_params['grayscale_levels']} (binary: each RGB channel is 0 or 1)")
print(f"Graph capacity: 80x80 grid = {80*80} nodes, capacity = {80*80//2} visible nodes")
print(f"Required nodes: {n_image_pixels} pixels × {graph_params['grayscale_levels']} levels = {n_image_pixels * graph_params['grayscale_levels']} nodes")
print(f"Batch size: {sampling_params['batch_size']}")
print(f"Batches per epoch: {batches_per_epoch}")

# Create DTM with dummy dataset for initialization
dtm = DTM(cfg, use_dummy_dataset=True, n_image_pixels=n_image_pixels)

print(f"DTM initialized successfully!")
print(f"Model has {len(dtm.steps)} diffusion step(s)")

# Define the data iterator factory
def create_data_iterator():
    """Factory function that creates a new data iterator for each epoch."""
    # Create the gym environment data generator
    gym_generator = GymDataGenerator(**gym_params)
    
    # Wrap it with our IteratorDataLoader
    dataloader = IteratorDataLoader(
        data_iterator=iter(gym_generator),
        batch_size=sampling_params['batch_size'],
        n_grayscale_levels=graph_params['grayscale_levels'],  # Will be 1 (binary)
        max_batches_per_epoch=batches_per_epoch,
    )
    
    return iter(dataloader)

print(f"\nStarting training for {n_epochs} epochs...")
print(f"This will process {batches_per_epoch} batches per epoch")
print(f"Total training steps: {n_epochs * batches_per_epoch}")
print("-" * 60)

# Train the DTM using the iterator
dtm.train_from_iterator(
    n_epochs=n_epochs,
    data_iterator_factory=create_data_iterator,
    batches_per_epoch=batches_per_epoch,
    evaluate_every=evaluate_every,
)

print("\nTraining complete!")
print(f"Model saved to: {dtm.logging_and_saving_dir}")

