"""
Training script for CONDITIONAL DTM with LunarLander environment using iterator-based data loading.

This script demonstrates how to train a conditional DTM on data generated on-the-fly from a gym environment.
The model learns to predict the next frame given the previous 2 frames and actions.

Key features:
- Iterator-based data loading (no full dataset in memory)
- CONDITIONAL generation (conditioned on previous frames + actions)
- 3-class images (32x32 single channel with 3 classes = 1024 pixels per frame)
  - Class 0: Black background
  - Class 1: Terrain and other elements
  - Class 2: Purple lunar lander
- Triplet-based training: predict frame[2] + action[2] given frames[0:2] + actions[0:2]
- On-the-fly data generation from LunarLander-v3 environment
"""

import sys
sys.path.insert(0, '/home/ubuntu/Extropic_Hackaton')

from thrmlDenoising.DTM import DTM
from thrmlDenoising.utils import make_cfg
from thrmlDenoising.conditional_iterator_dataloader import ConditionalIteratorDataLoader
from src.dataloader import GymDataGenerator

# ---------- Data parameters ----------
# For iterator-based training with CONDITIONAL generation
data_params = {
    "dataset_name": "conditional_lunarlander",  # Custom name for logging
    "target_classes": (0,),  # Single dummy class (not used in conditional training)
    "pixel_threshold_for_single_trials": 0.5,  # Threshold for binarization
}

# ---------- Graph parameters ----------
# Configured for CONDITIONAL training with triplets using 3-class representation:
# - Each pixel with 3 classes requires 2 binary nodes in the graph
# - Target: 1 frame (1024 pixels × 2 = 2048 bits) + 1 action (4 bits) = 2052 nodes
# - Condition: 2 frames STACKED on channel axis for local attention
#   * Frame format: (H, W, 4) where first 2 channels = frame0, last 2 = frame1
#   * Flattened: (1024 pixels × 4 = 4096 bits) + 2 actions (8 bits) = 4104 nodes
# Total visible nodes ≈ 6156 nodes (framework reports needing 8208 with overhead)
# We need grid capacity > 8208, so use 140×140 grid (capacity = 9800)
graph_params = {
    "graph_preset_architecture": 140_12,  # Grid: side=140, degree=12 (size=19600, capacity=9800)
    "num_label_spots": 1,  # Labels will store conditioning data (2 frames + 2 actions)
    "grayscale_levels": 2,  # 3 classes (0, 1, 2): black, terrain, lander (requires 2 bits per pixel)
    "torus": True,
    "base_graph_manager": "poisson_binomial_ising_graph_manager",
}

# ---------- Sampling (Gibbs/CD) schedule parameters ----------
sampling_params = {
    "batch_size": 128,  # Smaller batch size for faster iteration
    "n_samples": 50,  # Fewer samples for faster training
    "steps_per_sample": 8,
    "steps_warmup": 400,  # Reduced warmup
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
    "num_diffusion_steps": 3,  # Start with single step for simplicity
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
    "descriptor": "lunarlander_conditional",
    "n_cores": 1,
    "compute_autocorr": False,  # Disabled for iterator training (requires test dataset)
    "generate_gif": True,  # Now supports RGB GIF generation!
    "drawn_images_per_digit": 4,
    "animated_images_per_digit": 2,
    "steps_per_sample_in_gif": 10,
}

# ---------- Training parameters ----------
n_epochs = 50 # Number of epochs to train
batches_per_epoch = 5  # Number of batches per epoch (controls epoch length)
evaluate_every = 1  # Evaluate and save every N epochs (0 to disable)

# ---------- LunarLander Data Generator parameters ----------
# IMPORTANT: Each episode with state_size=32 yields 10 triplets (30 frames used, 2 discarded)
gym_params = {
    "state_size": 32,  # Number of frames to collect per episode
    "environment_name": "LunarLander-v3",
    "training_examples": 500,  # Episodes per epoch (each yields 10 triplets = 5000 training samples)
    "autoencoder_time_compression": 4,
    "return_anyways": True,  # Return frames even if lander leaves the frame
    "resolution": 32,  # 32x32 resolution (matches graph parameters)
}

# Data efficiency calculation:
# - batches_per_epoch=100 × batch_size=50 = 5000 samples needed per epoch
# - training_examples=500 episodes × 10 triplets = 5000 triplets available
# - Perfect! Exactly enough triplets to fill all batches without repetition

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

# Calculate dimensions for conditional training with 3-class representation
# With grayscale_levels=2, each pixel is encoded as 2 bits
n_pixels_per_frame = 32 * 32  # 1024 pixels per frame
n_bits_per_pixel = 2  # 3 classes require 2 bits per pixel
n_image_pixels = n_pixels_per_frame * n_bits_per_pixel  # 2048 bits per frame
n_action_classes = 4  # LunarLander has 4 actions
n_target_nodes = n_image_pixels + n_action_classes  # 2052 nodes (1 frame + 1 action)
n_condition_nodes = 2 * n_image_pixels + 2 * n_action_classes  # 4104 nodes (2 frames + 2 actions)
total_visible_nodes = n_target_nodes + n_condition_nodes  # 6156 nodes

print(f"Initializing DTM for CONDITIONAL iterator-based training...")
print(f"Image dimensions: {gym_params['resolution']}x{gym_params['resolution']} with 3 classes = {n_pixels_per_frame} pixels × {n_bits_per_pixel} bits = {n_image_pixels} bits per frame")
print(f"  Class 0: Black background")
print(f"  Class 1: Terrain and other elements")
print(f"  Class 2: Purple lunar lander")
print(f"Action classes: {n_action_classes}")
print(f"Target nodes: {n_target_nodes} (1 frame + 1 action)")
print(f"Condition nodes: {n_condition_nodes} (2 frames + 2 actions)")
print(f"Total visible nodes (calculated): {total_visible_nodes}")
print(f"Grayscale levels: {graph_params['grayscale_levels']} (3 classes: 0, 1, 2)")
print(f"Note: Each pixel requires 2 binary nodes for 3-class representation")
print(f"Graph capacity: 140×140 grid = {140*140} nodes, capacity = {140*140//2} visible nodes")
print(f"Batch size: {sampling_params['batch_size']}")
print(f"Batches per epoch: {batches_per_epoch}")

# Create DTM with dummy dataset for initialization
# For conditional training:
# - n_image_pixels = target data size (frame 2 + action 2)
# - n_label_nodes = conditioning data size (frames 0,1 + actions 0,1)
dtm = DTM(cfg, use_dummy_dataset=True, n_image_pixels=n_target_nodes, n_label_nodes=n_condition_nodes)

print(f"DTM initialized successfully!")
print(f"Model has {len(dtm.steps)} diffusion step(s)")

# Define the data iterator factory
def create_data_iterator():
    """Factory function that creates a new data iterator for each epoch."""
    # Create the gym environment data generator
    gym_generator = GymDataGenerator(**gym_params)
    
    # Wrap it with our ConditionalIteratorDataLoader
    dataloader = ConditionalIteratorDataLoader(
        data_iterator=iter(gym_generator),
        batch_size=sampling_params['batch_size'],
        n_grayscale_levels=graph_params['grayscale_levels'],  # Will be 1 (binary)
        n_action_classes=n_action_classes,  # 4 actions for LunarLander
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

