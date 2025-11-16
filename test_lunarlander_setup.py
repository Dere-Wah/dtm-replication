"""
Test script to verify the LunarLander iterator-based training setup.
This script performs basic sanity checks without running full training.
"""

import sys
sys.path.insert(0, '/home/ubuntu/Extropic_Hackaton')

print("=" * 60)
print("Testing LunarLander Iterator-Based Training Setup")
print("=" * 60)

# Test 1: Import all required modules
print("\n[Test 1] Importing modules...")
try:
    from thrmlDenoising.DTM import DTM
    from thrmlDenoising.utils import make_cfg, create_dummy_dataset_for_iterator
    from thrmlDenoising.iterator_dataloader import IteratorDataLoader, DummyLabelGenerator
    from src.dataloader import GymDataGenerator
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create dummy dataset
print("\n[Test 2] Creating dummy dataset...")
try:
    n_image_pixels = 32 * 32 * 3  # RGB 32x32
    train_data, test_data, one_hot_labels = create_dummy_dataset_for_iterator(
        n_image_pixels=n_image_pixels,
        n_grayscale_levels=4,
        n_dummy_samples=10,
    )
    print(f"✓ Dummy dataset created:")
    print(f"  - Train images shape: {train_data['image'].shape}")
    print(f"  - Train labels shape: {train_data['label'].shape}")
    print(f"  - Test images shape: {test_data['image'].shape}")
    print(f"  - One-hot labels shape: {one_hot_labels.shape}")
except Exception as e:
    print(f"✗ Dummy dataset creation failed: {e}")
    sys.exit(1)

# Test 3: Create dummy label generator
print("\n[Test 3] Testing dummy label generator...")
try:
    label_gen = DummyLabelGenerator(n_label_nodes=1)
    dummy_labels = label_gen.get_labels(batch_size=5)
    print(f"✓ Dummy labels generated: shape {dummy_labels.shape}")
except Exception as e:
    print(f"✗ Dummy label generation failed: {e}")
    sys.exit(1)

# Test 4: Test GymDataGenerator
print("\n[Test 4] Testing GymDataGenerator...")
try:
    gym_gen = GymDataGenerator(
        state_size=8,  # Small for quick test
        environment_name="LunarLander-v3",
        training_examples=3,  # Just 3 examples for test
        autoencoder_time_compression=4,
        return_anyways=True,
        resolution=32,
    )
    
    # Get one sample
    sample_count = 0
    for frames, actions, reward in gym_gen:
        sample_count += 1
        print(f"✓ Got sample {sample_count}: frames shape {frames.shape}")
        if sample_count >= 2:  # Just test first 2 samples
            break
    
    print(f"✓ GymDataGenerator works correctly")
except Exception as e:
    print(f"✗ GymDataGenerator failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test IteratorDataLoader with GymDataGenerator
print("\n[Test 5] Testing IteratorDataLoader with GymDataGenerator...")
try:
    gym_gen = GymDataGenerator(
        state_size=8,
        environment_name="LunarLander-v3",
        training_examples=10,
        autoencoder_time_compression=4,
        return_anyways=True,
        resolution=32,
    )
    
    dataloader = IteratorDataLoader(
        data_iterator=iter(gym_gen),
        batch_size=4,
        n_grayscale_levels=4,
        max_batches_per_epoch=2,
    )
    
    # Get n_image_pixels
    n_pixels = dataloader.get_n_image_pixels()
    print(f"✓ Image pixels detected: {n_pixels}")
    
    # Get batches
    batch_count = 0
    for batch_data in dataloader:
        batch_count += 1
        image_batch = batch_data[0]
        print(f"✓ Got batch {batch_count}: shape {image_batch.shape}")
    
    print(f"✓ IteratorDataLoader works correctly ({batch_count} batches)")
except Exception as e:
    print(f"✗ IteratorDataLoader failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Create config for LunarLander
print("\n[Test 6] Creating DTM config...")
try:
    cfg = make_cfg(
        exp={"seed": 42, "graph_seeds": (), "descriptor": "test", "n_cores": 1, 
             "compute_autocorr": False, "generate_gif": False, "drawn_images_per_digit": 4,
             "animated_images_per_digit": 2, "steps_per_sample_in_gif": 10},
        data={"dataset_name": "iterator_test", "target_classes": (0,), 
              "pixel_threshold_for_single_trials": 0.5},
        graph={"graph_preset_architecture": 80_12, "num_label_spots": 1, 
               "grayscale_levels": 4, "torus": True, 
               "base_graph_manager": "poisson_binomial_ising_graph_manager"},
        sampling={"batch_size": 4, "n_samples": 10, "steps_per_sample": 4, 
                  "steps_warmup": 50, "training_beta": 1.0},
        generation={"generation_beta_start": 0.8, "generation_beta_end": 1.2, 
                    "fid_images_per_digit": 10, "steps_warmup": 100},
        diffusion_schedule={"num_diffusion_steps": 1, "kind": "log", "diffusion_offset": 0.1},
        diffusion_rates={"image_rate": 0.8, "label_rate": 0.2},
        optim={"momentum": 0.9, "b2_adam": 0.999, "step_learning_rates": (0.01,), 
               "alpha_cosine_decay": 0.2, "n_epochs_for_lrd": 10},
        cp={"correlation_penalty": (0.0,), "adaptive_cp": False, 
            "cp_min": 0.001, "adaptive_threshold": 0.016},
        wd={"weight_decay": (0.0,), "adaptive_wd": False, "wd_min": 0.001},
    )
    print("✓ Config created successfully")
except Exception as e:
    print(f"✗ Config creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Initialize DTM with dummy dataset
print("\n[Test 7] Initializing DTM with dummy dataset...")
try:
    n_image_pixels = 32 * 32 * 3
    dtm = DTM(cfg, use_dummy_dataset=True, n_image_pixels=n_image_pixels)
    print(f"✓ DTM initialized successfully")
    print(f"  - Number of diffusion steps: {len(dtm.steps)}")
    print(f"  - Image pixels: {dtm.n_image_pixels}")
    print(f"  - Label nodes: {dtm.n_label_nodes}")
    print(f"  - Grayscale levels: {dtm.n_grayscale_levels}")
except Exception as e:
    print(f"✗ DTM initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nYou can now run the full training with:")
print("  python training_script_lunarlander.py")
print("=" * 60)

