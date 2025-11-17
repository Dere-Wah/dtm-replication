# Conditional Training Changes

## Key Fixes
1. Added `140_12` and `150_12` graph presets to `poisson_binomial_ising_graph_manager.py` (capacity 9800 and 11250 nodes)
2. Fixed evaluation to separate image pixels from action data before drawing
3. Fixed GIF generation to handle conditional data (separates images from actions)
4. **CRITICAL FIX**: Fixed clamped generation to use ACTUAL conditioning data instead of dummy labels
   - Before: Clamped generation was using `self.one_hot_target_labels` (meaningless dummy data)
   - After: Clamped generation uses actual conditioning frames from the dataset
   - This was causing clamped generation to look worse than free generation!

## Modified Files

### New File: `thrmlDenoising/conditional_iterator_dataloader.py`
- Splits 32-frame episodes into triplets (3 frames each, discards last 2 frames)
- Extracts and expands actions to match frame count (1 action per frame)
- Outputs: `(target_batch, condition_batch)` where:
  - `target_batch`: frame[2] + action[2] (3076 nodes)
  - `condition_batch`: frames[0:2] + actions[0:2] (6152 nodes)

### Modified: `training_script_lunarlander.py`
- Changed from unconditioned to conditional training
- Updated graph size: 80x80 → 140x140 (capacity: 3200 → 9800 nodes)
- Imports `ConditionalIteratorDataLoader` instead of `IteratorDataLoader`
- Passes `n_label_nodes=6152` for conditioning data
- Updated training_examples: 160 → 500 episodes (maintains 5000 samples/epoch)

### Modified: `thrmlDenoising/DTM.py`
- Added `n_label_nodes` parameter to `__init__`
- Updated `train_from_iterator` to handle conditional data:
  - Detects 2-tuple batches: `(targets, conditions)`
  - Uses conditions as label_batch for clamping during training
- Fixed `do_draw_and_fid` to separate image pixels (3072) from actions (4) before drawing
- Fixed `generate_gif` to handle conditional data by extracting image pixels before GIF generation
- **CRITICAL FIX in `_run_denoising`**: 
  - Added `conditioning_data` parameter to accept actual conditioning frames
  - Uses this instead of dummy `self.one_hot_target_labels` for clamped generation
  - `generate_gif` now passes actual dataset conditioning to `_run_denoising` for clamped mode

### Modified: `thrmlDenoising/utils.py`
- Added `n_label_nodes` parameter to `create_dummy_dataset_for_iterator`
- Supports custom label node count for conditional training

### Modified: `thrmlDenoising/base_graphs/poisson_binomial_ising_graph_manager.py`
- Added `140_12` graph preset: 140×140 grid, degree 12, capacity 9800 nodes
- Added `150_12` graph preset: 150×150 grid, degree 12, capacity 11250 nodes (future use)

### New File: `thrmlDenoising/conditional_gif_utils.py`
- **IMPLEMENTED:** Custom conditional GIF visualization
- `denoise_conditional_arrays_to_gif`: Creates GIFs showing:
  - Left: 2 conditioning frames (side by side) with action numbers below
  - Right: Denoising process of predicted frame 2
  - Each column shows one run/sample
- Extracts conditioning data from label_readout_list (2 frames + 2 actions)
- Renders action numbers using 3x5 pixel font
- Handles RGB images correctly

## Architecture
- Target nodes: 3076 (1 frame + 1 action)
- Condition nodes: 6152 (2 frames + 2 actions)
- Total visible nodes: 9228
- Grid capacity: 9800 (140x140 grid with degree 12)

## Training Data Flow
1. GymDataGenerator yields episodes: (32 frames, 8 actions)
2. ConditionalIteratorDataLoader creates 10 triplets per episode
3. Each triplet trains model to predict frame[2]+action[2] given frames[0:2]+actions[0:2]
4. Condition data is clamped during denoising (not modified)
5. Target data is denoised and compared against ground truth

## Current Status
✅ **Fully Working:**
- Conditional training with triplet data
- Image generation and evaluation (separates images from actions)
- **Custom conditional GIF visualization showing:**
  - Conditioning frames (frames 0 & 1) with action numbers
  - Denoising process of predicted frame 2
  - All in one coherent layout per run

## GIF Layout (COMPLETELY REWRITTEN!)
Each column in the GIF shows:
```
[Frame 0] [Frame 1]  |  [Predicted]  |  [Ground Truth]
[Action 0][Action 1] |  (animating)  |  (static)
```

**What you see (FIXED - Now using CLEAN dataset frames!):**
- **Left section (STATIC CLEAN)**: Conditioning frames 0 & 1 with their actions - **ORIGINAL CLEAN RGB frames from the dataset** (NOT noisy graph representation!)
- **Middle section (ANIMATED)**: Model's prediction of frame 2 evolving through denoising - starts as noise, should converge toward ground truth
- **Right section (STATIC CLEAN)**: Ground truth frame 2 from dataset - what the model should predict

**Key Fix:**
The system now extracts CLEAN conditioning frames directly from the dataset BEFORE they go through the noisy graph representation. This means:
- Conditioning frames shown are the actual RGB frames (no noise)
- Ground truth is also the actual frame from the dataset
- Only the prediction in the middle animates (showing the denoising process)

**How it works:**
1. During GIF generation, we sample actual triplets from the training data iterator
2. **Randomly select** from batch to get diverse triplet positions (not always episode start!)
   - This ensures the lander is often visible in frame (not out-of-bounds at episode start)
   - Each evaluation uses different random triplet positions from episodes
3. Extract conditioning frames (0, 1) and ground truth (frame 2) as clean binary data
4. Convert to clean RGB images (0-1 range)
5. Display these clean images in the GIF
6. Only the predicted frame goes through the noisy denoising visualization

