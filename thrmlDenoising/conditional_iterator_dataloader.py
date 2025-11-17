"""
Conditional iterator-based dataloader for DTM training.
Supports guided diffusion by conditioning on previous frames and actions.
"""

import jax.numpy as jnp
import numpy as np
from typing import Iterator, Tuple, Optional


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
        
    def _preprocess_image(self, frame: jnp.ndarray) -> jnp.ndarray:
        """Preprocess a single frame.
        
        Args:
            frame: Array of shape (H, W, C)
            
        Returns:
            Processed frame as a 1D array
        """
        # Apply custom preprocessing if provided
        if self.image_preprocessing is not None:
            frame = self.image_preprocessing(frame)
        
        # Normalize to [0, 1] if needed
        if jnp.max(frame) > 1.0:
            frame = frame / 255.0
        
        # Quantize to grayscale levels
        if self.n_grayscale_levels == 1:
            # Binary: threshold at 0.5
            frame = jnp.asarray(frame > 0.5, dtype=jnp.bool_)
        else:
            # Multi-level quantization
            frame = jnp.asarray(
                jnp.rint(frame * self.n_grayscale_levels),
                dtype=jnp.int32
            )
        
        # Flatten to 1D
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
            
            # Preprocess frames
            processed_frames = [self._preprocess_image(triplet_frames[j]) for j in range(3)]
            
            # Convert actions to one-hot
            onehot_actions = [self._action_to_onehot(triplet_actions[j]) for j in range(3)]
            
            # Condition: frames 0,1 + actions 0,1 (what we know)
            condition_data = jnp.concatenate([
                processed_frames[0],
                onehot_actions[0],
                processed_frames[1],
                onehot_actions[1],
            ])
            
            # Target: frame 2 + action 2 (what we want to predict)
            target_data = jnp.concatenate([
                processed_frames[2],
                onehot_actions[2],
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

