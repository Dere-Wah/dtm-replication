"""
Conditional iterator-based dataloader for DTM training.
Supports guided diffusion by conditioning on previous frames and actions.
"""

import jax.numpy as jnp
import numpy as np
from typing import Iterator, Tuple, Optional


def process_frame(frame):
    """
    Converts the frame to a single channel with 3 classes:
    - Class 0: Black pixels (background) - all channels == 0
    - Class 1: Other pixels (terrain, etc.) - everything else
    - Class 2: Purple pixels (lunar lander) - #8066E6 = RGB(128, 102, 230)
    
    Args:
        frame: Array of shape (H, W, 3) with RGB values in [0, 255] or [0, 1]
        
    Returns:
        Single-channel array of shape (H, W) with integer class labels 0, 1, or 2
    """
    # Normalize to [0, 255] if needed
    if jnp.max(frame) <= 1.0:
        frame = frame * 255.0
    
    # If input frame has shape (H, W, 3)
    if frame.shape[-1] == 3:
        # Black pixels: all channels == 0
        black_mask = jnp.all(frame == 0, axis=-1)
        
        # Purple detection for lunar lander color #8066E6 = RGB(128, 102, 230)
        # Allow some tolerance for color matching
        r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
        target_r, target_g, target_b = 128, 102, 230
        tolerance = 30
        
        purple_mask = (
            (jnp.abs(r - target_r) < tolerance) &
            (jnp.abs(g - target_g) < tolerance) &
            (jnp.abs(b - target_b) < tolerance)
        )
        
        # Assign classes: 0 for black, 2 for purple, 1 for everything else
        processed = jnp.ones(frame.shape[:2], dtype=jnp.int32)
        processed = jnp.where(purple_mask, 2, processed)
        processed = jnp.where(black_mask, 0, processed)
    else:
        # If already single channel, just apply 0: stays 0; any other-->1
        processed = jnp.where(frame == 0, 0, 1)
    
    return processed


class ConditionalIteratorDataLoader:
    """Iterator dataloader for conditional DTM training on frame prediction.
    
    This dataloader processes episodes from a gym environment and creates triplets of frames
    where the model learns to predict the 3rd frame given the first 2 frames and all actions.
    
    Episode structure:
    - Each episode has 32 frames and 8 actions (each action applies to 4 consecutive frames)
    - We split into groups of 3 frames (30 frames total, discarding last 2)
    - This gives us 10 triplets per episode
    
    Triplet structure:
    - Conditioning: frames[0:2] (flattened) + actions[0:2] (one-hot encoded)
    - Target: frame[2] (flattened) + action[2] (one-hot encoded)
    
    Args:
        data_iterator: Iterator that yields (frames, actions, rewards) tuples from gym
        batch_size: Number of triplets per batch
        image_preprocessing: Optional function to preprocess images
        n_grayscale_levels: Number of grayscale levels for quantization (1 for binary)
        n_action_classes: Number of possible actions (4 for LunarLander)
        max_batches_per_epoch: Maximum number of batches to yield per epoch (None for unlimited)
    """
    
    def __init__(
        self,
        data_iterator: Iterator,
        batch_size: int,
        image_preprocessing: Optional[callable] = None,
        n_grayscale_levels: int = 1,
        n_action_classes: int = 4,
        max_batches_per_epoch: Optional[int] = None,
    ):
        self.data_iterator = data_iterator
        self.batch_size = batch_size
        self.image_preprocessing = image_preprocessing
        self.n_grayscale_levels = n_grayscale_levels
        self.n_action_classes = n_action_classes
        self.max_batches_per_epoch = max_batches_per_epoch
        
        # Dimensions will be calculated from first sample
        self._first_triplets = None
        self._n_image_pixels = None
        self._n_condition_nodes = None
        self._n_target_nodes = None
        
    def _preprocess_image(self, frame: jnp.ndarray, keep_spatial: bool = False) -> jnp.ndarray:
        """Preprocess a single frame using 3-class color space.
        
        Args:
            frame: Array of shape (H, W, C) with RGB values
            keep_spatial: If True, return (H, W, channels) instead of flattened
            
        Returns:
            Processed frame - format depends on n_grayscale_levels and keep_spatial:
            - n_grayscale_levels=1: boolean array
            - n_grayscale_levels=2: boolean array with 2 bits per pixel (for 3 classes)
            Shape: (H, W, bits) if keep_spatial=True, else (H*W*bits,)
        """
        # Apply custom preprocessing if provided
        if self.image_preprocessing is not None:
            frame = self.image_preprocessing(frame)
        
        # Convert RGB to 3-class representation (0=black, 1=other, 2=purple/lander)
        frame = process_frame(frame)
        
        # Quantize based on grayscale levels setting
        if self.n_grayscale_levels == 1:
            # Binary: convert to bool (any non-zero becomes True)
            frame = jnp.asarray(frame > 0, dtype=jnp.bool_)
            if keep_spatial:
                return frame[..., jnp.newaxis]  # Add channel dimension
            return frame.reshape(-1)
        elif self.n_grayscale_levels == 2:
            # 3 classes require 2 bits per pixel
            # Convert each class to 2-bit binary representation:
            # Class 0 → [0, 0] = False, False
            # Class 1 → [0, 1] = False, True  
            # Class 2 → [1, 0] = True, False
            H, W = frame.shape
            # Create output with 2 bits per pixel
            frame_2bit = jnp.zeros((H, W, 2), dtype=jnp.bool_)
            
            # Set bits based on class
            frame_2bit = frame_2bit.at[:, :, 0].set(frame >= 2)  # High bit: True for class 2
            frame_2bit = frame_2bit.at[:, :, 1].set((frame == 1) | (frame == 3))  # Low bit: True for class 1 (and 3 if exists)
            
            # Return with or without spatial structure
            if keep_spatial:
                return frame_2bit  # (H, W, 2)
            else:
                # Flatten to 1D: [H*W*2] where every 2 elements represent one pixel
                return frame_2bit.reshape(-1)
        else:
            # For other grayscale levels, keep the 3-class structure
            frame = jnp.asarray(frame, dtype=jnp.int32)
            if keep_spatial:
                return frame[..., jnp.newaxis]
            return frame.reshape(-1)
    
    def _action_to_onehot(self, action: int) -> jnp.ndarray:
        """Convert action integer to one-hot encoding.
        
        Args:
            action: Integer action in range [0, n_action_classes)
            
        Returns:
            One-hot encoded action as bool array of shape (n_action_classes,)
        """
        onehot = jnp.zeros(self.n_action_classes, dtype=jnp.bool_)
        onehot = onehot.at[int(action)].set(True)
        return onehot
    
    def _expand_actions_to_frames(self, actions: jnp.ndarray, n_frames: int) -> jnp.ndarray:
        """Expand actions to match frame count.
        
        In the gym environment, actions are sampled every 4 frames (with autoencoder_time_compression=4
        and frame_collection_interval=2). We need to expand them so each action corresponds to a frame.
        
        Args:
            actions: Array of shape (n_actions,) - typically 8 actions for 32 frames
            n_frames: Number of frames - typically 32
            
        Returns:
            Expanded actions of shape (n_frames,) where each action is repeated 4 times
        """
        frames_per_action = n_frames // len(actions)  # Should be 4 for standard config
        expanded = jnp.repeat(actions, frames_per_action)
        return expanded[:n_frames]  # Ensure exact length
    
    def _create_triplets_from_episode(self, frames: jnp.ndarray, actions: jnp.ndarray) -> list:
        """Create triplets from an episode.
        
        Args:
            frames: Array of shape (32, H, W, C)
            actions: Array of shape (8,) - action indices
            
        Returns:
            List of (condition_data, target_data) tuples
        """
        # Expand actions to match frames (32 actions for 32 frames)
        expanded_actions = self._expand_actions_to_frames(actions, frames.shape[0])
        
        # Use first 30 frames (discard last 2) to get 10 triplets
        n_triplets = 30 // 3  # = 10
        triplets = []
        
        for i in range(n_triplets):
            start_idx = i * 3
            # Get 3 consecutive frames and their actions
            triplet_frames = frames[start_idx:start_idx + 3]  # (3, H, W, C)
            triplet_actions = expanded_actions[start_idx:start_idx + 3]  # (3,)
            
            # Preprocess frames - keep spatial structure for conditioning frames
            # This allows stacking on channel axis for better local attention
            processed_frame0_spatial = self._preprocess_image(triplet_frames[0], keep_spatial=True)  # (H, W, 2)
            processed_frame1_spatial = self._preprocess_image(triplet_frames[1], keep_spatial=True)  # (H, W, 2)
            processed_frame2_flat = self._preprocess_image(triplet_frames[2], keep_spatial=False)    # (H*W*2,)
            
            # Convert actions to one-hot
            onehot_actions = [self._action_to_onehot(triplet_actions[j]) for j in range(3)]
            
            # Condition: Stack frames 0,1 on channel axis for local attention
            # This keeps corresponding spatial positions close in the flattened array
            # Shape: (H, W, 2) + (H, W, 2) → (H, W, 4) → flatten to (H*W*4,)
            stacked_conditioning_frames = jnp.concatenate([
                processed_frame0_spatial,
                processed_frame1_spatial
            ], axis=-1)  # Stack on channel axis: (H, W, 4)
            
            condition_frames_flat = stacked_conditioning_frames.reshape(-1)  # Flatten: (H*W*4,)
            
            # Append actions after the stacked frames
            condition_data = jnp.concatenate([
                condition_frames_flat,  # Stacked frames: (H*W*4,)
                onehot_actions[0],      # Action 0: (4,)
                onehot_actions[1],      # Action 1: (4,)
            ])
            
            # Target: frame 2 + action 2 (what we want to predict)
            target_data = jnp.concatenate([
                processed_frame2_flat,  # Frame 2: (H*W*2,)
                onehot_actions[2],      # Action 2: (4,)
            ])
            
            triplets.append((condition_data, target_data))
        
        return triplets
    
    def get_n_image_pixels(self) -> int:
        """Get the number of pixels in a single frame."""
        if self._n_image_pixels is None:
            self._ensure_dimensions()
        return self._n_image_pixels
    
    def get_n_condition_nodes(self) -> int:
        """Get the number of nodes in conditioning data."""
        if self._n_condition_nodes is None:
            self._ensure_dimensions()
        return self._n_condition_nodes
    
    def get_n_target_nodes(self) -> int:
        """Get the number of nodes in target data."""
        if self._n_target_nodes is None:
            self._ensure_dimensions()
        return self._n_target_nodes
    
    def _ensure_dimensions(self):
        """Peek at first episode to determine dimensions."""
        if self._first_triplets is None:
            first_episode = next(self.data_iterator)
            frames, actions, _ = first_episode
            self._first_triplets = self._create_triplets_from_episode(frames, actions)
        
        # Get dimensions from first triplet
        condition_data, target_data = self._first_triplets[0]
        self._n_condition_nodes = condition_data.shape[0]
        self._n_target_nodes = target_data.shape[0]
        
        # Calculate image pixels (target has 1 frame + 1 action)
        self._n_image_pixels = self._n_target_nodes - self.n_action_classes
    
    def __iter__(self):
        """Iterate over batches of triplet data.
        
        Yields:
            Tuple of (batch_targets, batch_conditions) where:
            - batch_targets: shape (batch_size, n_pixels_per_frame + n_action_classes)
            - batch_conditions: shape (batch_size, 2*n_pixels_per_frame + 2*n_action_classes)
        """
        batch_conditions = []
        batch_targets = []
        batch_count = 0
        
        # Use first triplets if we peeked at them
        if self._first_triplets is not None:
            for condition_data, target_data in self._first_triplets:
                batch_conditions.append(np.array(condition_data))
                batch_targets.append(np.array(target_data))
            self._first_triplets = None
        
        # Process episodes and create triplets
        for frames, actions, _ in self.data_iterator:
            triplets = self._create_triplets_from_episode(frames, actions)
            
            for condition_data, target_data in triplets:
                batch_conditions.append(np.array(condition_data))
                batch_targets.append(np.array(target_data))
                
                # When we have a full batch, yield it
                if len(batch_targets) >= self.batch_size:
                    batch_t = jnp.array(np.stack(batch_targets[:self.batch_size]))
                    batch_c = jnp.array(np.stack(batch_conditions[:self.batch_size]))
                    batch_targets = batch_targets[self.batch_size:]
                    batch_conditions = batch_conditions[self.batch_size:]
                    
                    yield (batch_t, batch_c)
                    batch_count += 1
                    
                    # Stop if we've reached max batches per epoch
                    if self.max_batches_per_epoch is not None and batch_count >= self.max_batches_per_epoch:
                        return
        
        # Yield remaining partial batch if any (pad to batch_size)
        if batch_targets and (self.max_batches_per_epoch is None or batch_count < self.max_batches_per_epoch):
            # Pad the last batch to batch_size by repeating samples
            while len(batch_targets) < self.batch_size:
                batch_targets.append(batch_targets[-1])
                batch_conditions.append(batch_conditions[-1])
            batch_t = jnp.array(np.stack(batch_targets))
            batch_c = jnp.array(np.stack(batch_conditions))
            yield (batch_t, batch_c)

