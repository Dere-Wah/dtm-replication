# DTM Iterator-Based Training for LunarLander

This document describes the modifications made to the DTM codebase to support iterator-based training with the LunarLander environment.

## Overview

The DTM (Denoising Thermodynamic Model) has been extended to support on-the-fly data generation from iterators, enabling training without loading a full dataset into memory. This is particularly useful for:

- Training on simulator-generated data (like gym environments)
- Working with very large datasets that don't fit in memory
- Real-time or streaming data scenarios
- **Unconditioned generation** (no class labels)

## Key Changes

### 1. New Files

#### `thrmlDenoising/iterator_dataloader.py`
- **`IteratorDataLoader`**: Wraps an iterator to yield batches compatible with DTM training
  - Handles image preprocessing (normalization, quantization)
  - Manages batch accumulation from stream
  - Flattens images to 1D arrays
- **`DummyLabelGenerator`**: Provides dummy labels for unconditioned training

#### `training_script_lunarlander.py`
- Complete training script for LunarLander environment
- Configured for RGB 32x32 images (3072 pixels)
- Unconditioned generation (no classes)
- Uses iterator-based data loading

#### `test_lunarlander_setup.py`
- Comprehensive test suite to verify the setup
- Tests all components before running full training

### 2. Modified Files

#### `thrmlDenoising/utils.py`
- Added `create_dummy_dataset_for_iterator()`: Creates minimal dataset for initialization when using iterators

#### `thrmlDenoising/DTM.py`
- Modified `__init__()`: Added `use_dummy_dataset` parameter for iterator-based training
- Added `train_from_iterator()`: New training method that works with data iterators
  - Processes batches as they arrive
  - No shuffling or full dataset loading
  - Updates model parameters after each batch

#### `thrmlDenoising/step.py`
- Added `train_step_model_single_batch()`: Trains on a single pre-batched batch
  - No internal re-batching or shuffling
  - Applies perturbations to the batch
  - Compatible with streaming data

#### `thrmlDenoising/ising_training.py`
- Added `do_single_batch_update()`: Performs parameter update on a single batch
  - Computes gradients without internal batching
  - Applies optimizer updates with weight/bias decay

## Architecture Details

### Image Format
- **Resolution**: 32x32 pixels
- **Channels**: 3 (RGB)
- **Total pixels**: 3072 (32 × 32 × 3)
- **Quantization**: Binary (1 bit per channel: 0 or 1)
  - Note: Using higher grayscale levels would require much larger graphs
  - Binary quantization: each RGB value is thresholded to 0 or 1
- **Format**: Flattened 1D array of 3072 binary values

### Model Configuration
- **Graph architecture**: 80×80 grid with degree 12 (80_12)
  - Grid size must satisfy: `side² > 2 × n_pixels × grayscale_levels`
  - For 3072 pixels × 1 level: need `side² > 6144`, so 80×80 works (6400 > 6144)
  - Capacity: 3200 visible nodes (enough for 3072 image pixels + 1 label node)
- **Quantization**: Binary (grayscale_levels=1)
- **Diffusion steps**: 1 (can be increased for better quality)
- **Batch size**: 50 samples
- **Training mode**: Unconditioned (single dummy class)

### Training Flow

```
LunarLander Env → GymDataGenerator → IteratorDataLoader → DTM.train_from_iterator()
                                                              ↓
                                                    train_step_model_single_batch()
                                                              ↓
                                                    do_single_batch_update()
```

## Usage

### Basic Training

```python
from thrmlDenoising.DTM import DTM
from thrmlDenoising.utils import make_cfg
from thrmlDenoising.iterator_dataloader import IteratorDataLoader
from src.dataloader import GymDataGenerator

# 1. Create configuration
cfg = make_cfg(
    # ... configuration parameters ...
)

# 2. Initialize DTM with dummy dataset
n_image_pixels = 32 * 32 * 3  # RGB 32x32
dtm = DTM(cfg, use_dummy_dataset=True, n_image_pixels=n_image_pixels)

# 3. Define iterator factory
def create_data_iterator():
    gym_gen = GymDataGenerator(
        state_size=32,
        environment_name="LunarLander-v3",
        training_examples=1000,
        resolution=32,
    )
    
    dataloader = IteratorDataLoader(
        data_iterator=iter(gym_gen),
        batch_size=50,
        n_grayscale_levels=4,
        max_batches_per_epoch=100,
    )
    
    return iter(dataloader)

# 4. Train
dtm.train_from_iterator(
    n_epochs=20,
    data_iterator_factory=create_data_iterator,
    batches_per_epoch=100,
    evaluate_every=5,
)
```

### Running the Example Script

```bash
cd /home/ubuntu/dtm-replication
python training_script_lunarlander.py
```

### Testing the Setup

```bash
cd /home/ubuntu/dtm-replication
python test_lunarlander_setup.py
```

## Key Differences from Standard Training

| Feature | Standard Training | Iterator-Based Training |
|---------|------------------|------------------------|
| Data Loading | Full dataset in memory | On-the-fly generation |
| Shuffling | Per-epoch shuffling | No shuffling (data is fresh each time) |
| Batching | Internal batching with reshuffling | Pre-batched from iterator |
| Memory Usage | O(dataset_size) | O(batch_size) |
| Autocorrelation | Computed on test set | Disabled (no test set) |
| Adaptive Regularization | Supported | Disabled |
| Labels | Required (conditional) | Dummy labels (unconditioned) |

## Configuration Parameters

### Critical Parameters for LunarLander

```python
# Image dimensions
n_image_pixels = 32 * 32 * 3  # 3072 for RGB 32x32

# Graph size (affects model capacity)
# IMPORTANT: Must satisfy side² > 2 × n_pixels
# For 3072 pixels: need side² > 6144, so side > 78.4
graph_preset_architecture = 80_12  # 80×80 grid, degree 12

# Quantization (affects image quality vs. model complexity)
grayscale_levels = 4  # 4 levels per channel

# Training
batch_size = 50  # Samples per batch
batches_per_epoch = 100  # Batches to process per epoch

# Learning
step_learning_rates = (0.03,)  # Learning rate per diffusion step
n_epochs = 20  # Total training epochs
```

### Choosing the Right Graph Size

**CRITICAL:** With `PoissonBinomialIsingGraphManager`, each pixel requires `grayscale_levels` Ising spins. The constraint is:

```
side_length² > 2 × n_image_pixels × grayscale_levels
```

**For 32×32 RGB (3072 pixels)**:
- With `grayscale_levels=1` (binary): Need `side² > 6144` → **80×80 grid works** ✓
- With `grayscale_levels=4`: Need `side² > 24,576` → **No preset large enough!** ✗

**Valid presets include**: `20_8`, `44_12`, `50_12`, `60_12`, `70_12`, `80_12`, `90_12`, `100_40`, `120_36`, etc.

**Recommended configurations**:
- 28×28 grayscale binary (784 pixels × 1): `44_12` or larger
- 32×32 grayscale binary (1024 pixels × 1): `50_12` or larger  
- **32×32 RGB binary (3072 pixels × 1): `80_12` or larger** ← Use this!
- 32×32 RGB 4-level (3072 pixels × 4): Would need `157×157` grid (not available)

**Solution for RGB**: Use binary quantization (`grayscale_levels=1`) to keep the model tractable.

### Hyperparameter Tuning Tips

1. **`grayscale_levels`**: 
   - Lower (2-4): Faster training, simpler images
   - Higher (8-16): Better quality, slower training

2. **`graph_preset_architecture`**:
   - Smaller (44_12, 50_12): Faster, less capacity (for smaller images)
   - Medium (60_12, 70_12): Balanced (for medium images)
   - Larger (80_12, 90_12): Slower, more capacity (for larger/RGB images)

3. **`batch_size`**:
   - Smaller (20-30): More updates, less stable
   - Larger (50-100): Fewer updates, more stable

4. **`steps_per_sample`**:
   - Fewer (4-6): Faster training, less mixing
   - More (8-12): Slower training, better mixing

## Limitations

1. **No Test Set**: Iterator-based training doesn't have a persistent test set
   - Autocorrelation computation is disabled
   - Adaptive regularization is disabled
   - Can be worked around by maintaining a separate test set

2. **No Shuffling**: Data comes in the order generated by the iterator
   - Less important when data is generated fresh each epoch
   - Consider adding randomness in the data generator itself

3. **Evaluation**: Limited evaluation without a reference dataset
   - Can still generate images and visualize
   - FID computation requires precomputed statistics

## Extending the Implementation

### Using a Different Data Source

```python
def my_custom_iterator():
    """Your custom data iterator."""
    for i in range(num_batches):
        # Generate or load batch
        batch = generate_batch()  # Shape: (batch_size, height, width, channels)
        yield (batch,)  # Yield as tuple

# Use with IteratorDataLoader
dataloader = IteratorDataLoader(
    data_iterator=my_custom_iterator(),
    batch_size=50,
    n_grayscale_levels=4,
)
```

### Adding Test Set Support

```python
# Load or generate a persistent test set
test_dataset = load_test_data()

# Manually set it on the DTM instance
dtm.test_dataset = test_dataset

# Now autocorrelation can be computed
dtm.cfg.exp.compute_autocorr = True
```

### Conditional Generation

To use labels (conditional generation), modify:

1. Update `DummyLabelGenerator` to provide real labels
2. Change `num_label_spots` to match your number of classes
3. Modify `create_dummy_dataset_for_iterator()` to use correct label dimensions

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_size`
- Reduce `graph_preset_architecture`
- Reduce `steps_warmup`

### Training Unstable
- Lower learning rate (`step_learning_rates`)
- Increase `batch_size`
- Reduce `grayscale_levels`

### Poor Image Quality
- Increase `grayscale_levels`
- Increase `graph_preset_architecture`
- Increase `num_diffusion_steps`
- Train for more epochs

### Slow Training
- Reduce `batch_size`
- Reduce `steps_per_sample`
- Reduce `steps_warmup`
- Use smaller `graph_preset_architecture`

## Performance Benchmarks

Approximate training speed (on single GPU):

| Configuration | Image Size | Batch Size | Steps/Epoch | Time/Epoch |
|---------------|-----------|-----------|-------------|------------|
| 44_12 | 28×28 gray | 50 | 100 | ~3 min |
| 60_12 | 32×32 gray | 50 | 100 | ~6 min |
| 80_12 | 32×32 RGB | 50 | 100 | ~12 min |
| 90_12 | 64×64 gray | 50 | 100 | ~18 min |

*Note: Actual performance depends on hardware, environment rendering speed, and hyperparameters*

## Future Improvements

1. **Parallel Data Generation**: Generate batches in background threads
2. **Data Augmentation**: Apply augmentations in the iterator
3. **Dynamic Batch Sizing**: Adjust batch size based on available memory
4. **Checkpoint Recovery**: Resume from interrupted training
5. **Distributed Training**: Support multi-GPU iterator-based training

## Citation

If you use this iterator-based training extension, please cite the original DTM paper and acknowledge the extension.

## Support

For issues or questions:
1. Check this README
2. Run `test_lunarlander_setup.py` to verify setup
3. Review the example in `training_script_lunarlander.py`

