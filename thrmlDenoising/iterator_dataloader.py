"""
Iterator-based dataloader for DTM training.
Supports on-the-fly data generation without loading the full dataset into memory.
"""

import jax.numpy as jnp
import numpy as np
from typing import Iterator, Tuple, Optional


class IteratorDataLoader:
    """Wrapper for iterator-based data sources that yields batches for DTM training.
    
    This class enables training on datasets that are generated on-the-fly, such as
    from simulators or large datasets that don't fit in memory. It wraps an iterator
    that yields (frames, actions, rewards) and converts them to the format expected
    by DTM training.
    
    Args:
        data_iterator: Iterator that yields (frames, actions, rewards) tuples
        batch_size: Number of examples per batch
        image_preprocessing: Optional function to preprocess images (e.g., normalize, resize)
        n_grayscale_levels: Number of grayscale levels for quantization (1 for binary)
        max_batches_per_epoch: Maximum number of batches to yield per epoch (None for unlimited)
    """
    
    def __init__(
        self,
        data_iterator: Iterator,
        batch_size: int,
        image_preprocessing: Optional[callable] = None,
        n_grayscale_levels: int = 1,
        max_batches_per_epoch: Optional[int] = None,
    ):
        self.data_iterator = data_iterator
        self.batch_size = batch_size
        self.image_preprocessing = image_preprocessing
        self.n_grayscale_levels = n_grayscale_levels
        self.max_batches_per_epoch = max_batches_per_epoch
        
        # Get first sample to determine dimensions
        self._first_sample = None
        self._n_image_pixels = None
        
    def _preprocess_image(self, frames: jnp.ndarray) -> jnp.ndarray:
        """Preprocess a single sample of frames.
        
        Args:
            frames: Array of shape (n_frames, H, W, C) or (1, H, W, C) for single frame
            
        Returns:
            Processed frame as a 1D array
        """
        # Apply custom preprocessing if provided
        if self.image_preprocessing is not None:
            frames = self.image_preprocessing(frames)
        
        # Take the first frame (or only frame if single frame passed) - shape (H, W, C)
        frame = frames[0] if frames.shape[0] > 0 else frames
        
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
    
    def get_n_image_pixels(self) -> int:
        """Get the number of pixels in processed images."""
        if self._n_image_pixels is None:
            # Peek at first sample
            if self._first_sample is None:
                self._first_sample = next(self.data_iterator)
            frames, _, _ = self._first_sample
            processed = self._preprocess_image(frames)
            self._n_image_pixels = processed.shape[0]
        return self._n_image_pixels
    
    def __iter__(self):
        """Iterate over batches of data.
        
        Yields:
            Tuple of (batch_images,) where batch_images has shape (batch_size, n_pixels)
            Note: Returns a tuple for compatibility with existing code structure
        """
        batch_images = []
        batch_count = 0
        
        # Use first sample if we peeked at it
        if self._first_sample is not None:
            frames, _, _ = self._first_sample
            processed = self._preprocess_image(frames)
            batch_images.append(np.array(processed))
            self._first_sample = None
        
        # Collect samples into batches
        # NEW: Process ALL frames from each episode, not just the last one
        for frames, actions, rewards in self.data_iterator:
            # frames has shape (n_frames, H, W, C)
            # Process each frame individually
            for i in range(frames.shape[0]):
                single_frame = frames[i:i+1]  # Keep dims: (1, H, W, C)
                processed = self._preprocess_image(single_frame)
                batch_images.append(np.array(processed))
                
                # When we have a full batch, yield it
                if len(batch_images) >= self.batch_size:
                    batch = jnp.array(np.stack(batch_images[:self.batch_size]))
                    batch_images = batch_images[self.batch_size:]
                    
                    yield (batch,)
                    batch_count += 1
                    
                    # Stop if we've reached max batches per epoch
                    if self.max_batches_per_epoch is not None and batch_count >= self.max_batches_per_epoch:
                        return  # Exit early if max batches reached
        
        # Yield remaining partial batch if any (pad to batch_size)
        if batch_images and (self.max_batches_per_epoch is None or batch_count < self.max_batches_per_epoch):
            # Pad the last batch to batch_size by repeating samples
            while len(batch_images) < self.batch_size:
                batch_images.append(batch_images[-1])
            batch = jnp.array(np.stack(batch_images))
            yield (batch,)


class DummyLabelGenerator:
    """Generates dummy labels for unconditioned training.
    
    Since we're doing unconditioned generation, we don't need real labels.
    This class provides dummy labels that are ignored during training.
    """
    
    def __init__(self, n_label_nodes: int = 1):
        """Initialize with a minimal label size."""
        self.n_label_nodes = n_label_nodes
    
    def get_labels(self, batch_size: int) -> jnp.ndarray:
        """Get dummy labels for a batch.
        
        Returns zeros since labels are not used in unconditioned training.
        """
        return jnp.zeros((batch_size, self.n_label_nodes), dtype=jnp.bool_)

