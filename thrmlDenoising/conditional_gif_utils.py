"""
Utility functions for creating conditional GIFs that show:
- Conditioning frames (first 2 frames of triplet)
- Denoising process (frame 2 being generated)
- Action visualizations as numbers
"""

import numpy as np
import imageio.v2 as imageio
from typing import Optional
import os


def denoise_conditional_arrays_to_gif(
    image_readout_list: list[np.ndarray],
    label_readout_list: list[np.ndarray],
    out_path: str,
    *,
    n_grayscale_levels: int,
    runs_per_label: int,
    frame_stride: int,
    fps: int,
    image_side_len: int,
    pad: int = 2,
    steps_per_sample: int = 1,
    is_rgb: bool = False,
    is_3class: bool = False,  # NEW: 3-class representation (0=black, 1=terrain, 2=lander)
    clean_conditioning_frames: Optional[list] = None,  # List of [frame0, frame1] for each run
    clean_conditioning_actions: Optional[list] = None,  # List of [action0, action1] for each run  
    clean_ground_truth_frames: Optional[list] = None,  # List of frame2 for each run
    clean_ground_truth_actions: Optional[list] = None,  # List of action2 for each run
):
    """
    Creates a conditional GIF showing:
    - Left column: 2 conditioning frames (frames 0, 1) with action numbers - CLEAN from dataset
    - Middle: Denoising GIF of frame 2 (predicted)
    - Right: Ground truth frame 2 (optional)
    
    Args:
        image_readout_list: list over steps; each array (n_labels, runs, n_samples, image_pixels)
        label_readout_list: list over steps; each (n_labels, runs, n_samples, condition_data)
        out_path: Path to save GIF
        n_grayscale_levels: Grayscale levels for normalization
        runs_per_label: Number of runs per label
        frame_stride: Keep 1 of every N samples
        fps: Frames per second
        image_side_len: Side length of square images (32)
        pad: Padding between images
        steps_per_sample: Gibbs steps between samples
        is_rgb: If True, treats as RGB images
        is_3class: If True, treats as 3-class images (0=black, 1=terrain, 2=lander)
        clean_conditioning_frames: CLEAN conditioning frames from dataset (not noisy graph representation)
        clean_conditioning_actions: Actions for conditioning frames
        clean_ground_truth_frames: Ground truth target frames
        clean_ground_truth_actions: Ground truth target actions
    """
    assert len(image_readout_list) > 0, "empty image_readout_list"
    n_steps = len(image_readout_list)
    n_labels, rp_avail, n_samples, image_size = image_readout_list[0].shape
    assert rp_avail >= runs_per_label, "runs_per_label exceeds available runs"
    
    # Extract conditioning data dimensions
    condition_size = label_readout_list[0].shape[-1] if label_readout_list else 0
    
    # Calculate bits per frame based on representation
    if is_3class:
        # 3-class uses 2 bits per pixel
        n_bits_per_frame = image_side_len * image_side_len * 2
    elif is_rgb:
        n_bits_per_frame = image_side_len * image_side_len * 3
    else:
        n_bits_per_frame = image_side_len * image_side_len
    
    n_action_classes = 4  # LunarLander has 4 actions
    
    # Conditioning data: 2 frames stacked on channel axis + 2 actions (one-hot encoded)
    # For 3-class: (H*W*4) stacked frames + 4 + 4 actions = 4096 + 8 = 4104
    expected_condition_size = 2 * n_bits_per_frame + 2 * n_action_classes
    
    # Simple 3x5 font for action numbers
    FONT_3x5 = {
        "0": ["111", "101", "101", "101", "111"],
        "1": ["010", "110", "010", "010", "111"],
        "2": ["111", "001", "111", "100", "111"],
        "3": ["111", "001", "111", "001", "111"],
        "4": ["101", "101", "111", "001", "001"],
        " ": ["000", "000", "000", "000", "000"],
        "A": ["010", "101", "111", "101", "101"],  # For "Action"
        "c": ["011", "100", "100", "100", "011"],
        "t": ["010", "111", "010", "010", "001"],
        "i": ["010", "000", "010", "010", "010"],
        "o": ["010", "101", "101", "101", "010"],
        "n": ["000", "110", "101", "101", "101"],
    }
    
    def render_number(num: int, H: int, W: int) -> np.ndarray:
        """Render a number centered in HxW canvas."""
        text = str(num)
        base_h, base_w = 5, 3
        s = max(1, min(H // base_h, W // (base_w * len(text))))
        glyph_h = base_h * s
        glyph_w = base_w * s
        gap = max(1, s // 2)
        
        # Compose bitmap
        bitmaps = []
        for ch in text:
            patt = FONT_3x5.get(ch, FONT_3x5[" "])
            g = np.array([[1.0 if c == "1" else 0.0 for c in row] for row in patt], dtype=np.float32)
            g_up = np.kron(g, np.ones((s, s), dtype=np.float32))
            bitmaps.append(g_up)
        
        if len(bitmaps) == 0:
            return np.zeros((H, W), dtype=np.float32)
        
        # Place with gaps
        total_w = len(bitmaps) * glyph_w + (len(bitmaps) - 1) * gap
        canvas = np.zeros((H, W), dtype=np.float32)
        y0 = (H - glyph_h) // 2
        x0 = (W - total_w) // 2
        x = x0
        for i, g in enumerate(bitmaps):
            h, w = g.shape
            if x + w <= W and y0 + h <= H:
                canvas[y0:y0 + h, x:x + w] = np.maximum(canvas[y0:y0 + h, x:x + w], g)
            x += w + gap
        return canvas
    
    def class_to_rgb(class_img: np.ndarray) -> np.ndarray:
        """Convert 3-class image to RGB for visualization.
        
        Args:
            class_img: Array with integer class labels 0, 1, or 2
            
        Returns:
            RGB image array with shape (..., 3)
        """
        # Create RGB output
        rgb = np.zeros((*class_img.shape, 3), dtype=np.float32)
        
        # Class 0: Black (0, 0, 0)
        mask0 = (class_img == 0)
        rgb[mask0] = [0.0, 0.0, 0.0]
        
        # Class 1: White/terrain (255, 255, 255) -> normalized to (1.0, 1.0, 1.0)
        mask1 = (class_img == 1)
        rgb[mask1] = [1.0, 1.0, 1.0]
        
        # Class 2: Purple lander (128, 102, 230) -> normalized to (0.502, 0.400, 0.902)
        mask2 = (class_img == 2)
        rgb[mask2] = [128.0/255.0, 102.0/255.0, 230.0/255.0]
        
        return rgb
    
    def norm_img(x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=np.float32) / float(n_grayscale_levels), 0.0, 1.0)
    
    def extract_conditioning_frames_and_actions(condition_data: np.ndarray):
        """Extract 2 frames and 2 actions from conditioning data.
        
        Note: Frames are stacked on channel axis for local attention.
        For 3-class: condition_data has shape (4104,) = 4096 stacked frames + 8 actions
        """
        # Stacked frames: first (2 * n_bits_per_frame) elements
        stacked_frames_size = 2 * n_bits_per_frame
        stacked_frames_flat = condition_data[:stacked_frames_size]
        
        # Reshape stacked frames: (H*W*4,) → (H, W, 4)
        if is_3class:
            stacked_frames = stacked_frames_flat.reshape(image_side_len, image_side_len, 4)
            # Split into frame0 (channels 0-1) and frame1 (channels 2-3)
            frame0 = stacked_frames[:, :, 0:2].reshape(-1)  # (H*W*2,)
            frame1 = stacked_frames[:, :, 2:4].reshape(-1)  # (H*W*2,)
        elif is_rgb:
            stacked_frames = stacked_frames_flat.reshape(image_side_len, image_side_len, 6)
            frame0 = stacked_frames[:, :, 0:3].reshape(-1)  # (H*W*3,)
            frame1 = stacked_frames[:, :, 3:6].reshape(-1)  # (H*W*3,)
        else:
            # Grayscale stacked
            stacked_frames = stacked_frames_flat.reshape(image_side_len, image_side_len, 2)
            frame0 = stacked_frames[:, :, 0].reshape(-1)
            frame1 = stacked_frames[:, :, 1].reshape(-1)
        
        # Actions come after stacked frames
        action0_onehot = condition_data[stacked_frames_size:stacked_frames_size + n_action_classes]
        action1_onehot = condition_data[stacked_frames_size + n_action_classes:stacked_frames_size + 2*n_action_classes]
        
        # Convert one-hot to integer
        action0 = int(np.argmax(action0_onehot)) if action0_onehot.sum() > 0 else 0
        action1 = int(np.argmax(action1_onehot)) if action1_onehot.sum() > 0 else 0
        
        return frame0, frame1, action0, action1
    
    def frame_to_image(frame_data: np.ndarray) -> np.ndarray:
        """Convert flat frame data to HxWxC image."""
        if is_3class:
            # For 3-class data with n_grayscale_levels=2, data is 2-bit encoded
            # Each pixel uses 2 bits: [high_bit, low_bit]
            # Decode: class = high_bit*2 + low_bit
            # Class 0 → [0, 0], Class 1 → [0, 1], Class 2 → [1, 0]
            
            # Reshape from flat array to 2-bit pairs
            frame_2bit = frame_data.reshape(image_side_len, image_side_len, 2)
            
            # Convert 2-bit to class labels (0, 1, 2)
            class_img = frame_2bit[..., 0].astype(np.int32) * 2 + frame_2bit[..., 1].astype(np.int32)
            
            # Convert class labels to RGB for visualization
            img = class_to_rgb(class_img)
        elif is_rgb:
            normalized = norm_img(frame_data)
            img = normalized.reshape(image_side_len, image_side_len, 3)
        else:
            normalized = norm_img(frame_data)
            img = normalized.reshape(image_side_len, image_side_len)
        return img
    
    # Prepare frames
    total_samples = n_samples
    frames_to_keep = list(range(0, total_samples, frame_stride))
    if frames_to_keep[-1] != total_samples - 1:
        frames_to_keep.append(total_samples - 1)
    
    # Pre-extract STATIC conditioning data for each run
    # Use CLEAN conditioning frames from dataset if provided, otherwise extract from noisy graph
    conditioning_cache = []
    ground_truth_cache = []
    
    if clean_conditioning_frames is not None and clean_conditioning_actions is not None:
        # Use CLEAN frames from dataset (no noise!)
        for run_idx in range(min(runs_per_label, len(clean_conditioning_frames))):
            frame0, frame1 = clean_conditioning_frames[run_idx]
            action0, action1 = clean_conditioning_actions[run_idx]
            conditioning_cache.append((frame0, frame1, action0, action1))
            
            # Also cache ground truth if available
            if clean_ground_truth_frames is not None and clean_ground_truth_actions is not None:
                if run_idx < len(clean_ground_truth_frames):
                    gt_frame = clean_ground_truth_frames[run_idx]
                    gt_action = clean_ground_truth_actions[run_idx]
                    ground_truth_cache.append((gt_frame, gt_action))
    
    elif label_readout_list and len(label_readout_list) > 0:
        # Fallback: extract from noisy graph representation (for free generation or when no clean frames)
        for run_idx in range(runs_per_label):
            condition_data = label_readout_list[-1][0, run_idx, 0, :]
            frame0, frame1, action0, action1 = extract_conditioning_frames_and_actions(condition_data)
            img0 = frame_to_image(frame0)
            img1 = frame_to_image(frame1)
            conditioning_cache.append((img0, img1, action0, action1))
    
    # Layout: each column shows one run
    # Left: STATIC conditioning frames | Right: ANIMATED denoising frame
    # We'll show runs_per_label columns, each with the same layout
    
    all_frames = []
    
    for frame_idx in frames_to_keep:
        # Create canvas for this frame
        # Layout per column: [cond_frame_0] [cond_frame_1] | [denoising] | [ground_truth]
        #                    [action_0   ] [action_1   ]
        
        action_text_height = 10  # Height for action number text
        col_height = image_side_len + action_text_height + pad
        col_width_left = 2 * image_side_len + pad  # 2 conditioning frames side by side
        col_width_middle = image_side_len  # 1 denoising frame
        col_width_right = image_side_len if len(ground_truth_cache) > 0 else 0  # Ground truth (if available)
        col_width_total = col_width_left + pad + col_width_middle + (pad + col_width_right if col_width_right > 0 else 0)
        
        canvas_height = col_height
        canvas_width = runs_per_label * col_width_total + (runs_per_label - 1) * pad
        
        # For 3-class representation, we always use RGB canvas for visualization
        if is_rgb or is_3class:
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
        else:
            canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        
        # Fill each column (one per run)
        for run_idx in range(runs_per_label):
            x_offset = run_idx * (col_width_total + pad)
            
            # Get STATIC conditioning data from cache (these frames DON'T CHANGE - they're the CLEAN input!)
            if run_idx < len(conditioning_cache):
                img0, img1, action0, action1 = conditioning_cache[run_idx]
                
                # Place conditioning frame 0 (top left) - STATIC CLEAN
                y0, x0 = 0, x_offset
                if is_rgb or is_3class:
                    canvas[y0:y0+image_side_len, x0:x0+image_side_len, :] = np.clip(img0, 0, 1)
                else:
                    canvas[y0:y0+image_side_len, x0:x0+image_side_len] = np.clip(img0.squeeze(), 0, 1)
                
                # Place conditioning frame 1 (top right of left section) - STATIC CLEAN
                x1 = x_offset + image_side_len + pad // 2
                if is_rgb or is_3class:
                    canvas[y0:y0+image_side_len, x1:x1+image_side_len, :] = np.clip(img1, 0, 1)
                else:
                    canvas[y0:y0+image_side_len, x1:x1+image_side_len] = np.clip(img1.squeeze(), 0, 1)
                
                # Add action numbers below conditioning frames - STATIC
                action_y = image_side_len
                # Action 0 below frame 0
                action0_text = render_number(action0, action_text_height, image_side_len)
                if is_rgb or is_3class:
                    for c in range(3):
                        canvas[action_y:action_y+action_text_height, x0:x0+image_side_len, c] = action0_text
                else:
                    canvas[action_y:action_y+action_text_height, x0:x0+image_side_len] = action0_text
                
                # Action 1 below frame 1 - STATIC
                if is_rgb or is_3class:
                    for c in range(3):
                        canvas[action_y:action_y+action_text_height, x1:x1+image_side_len, c] = render_number(action1, action_text_height, image_side_len)
                else:
                    canvas[action_y:action_y+action_text_height, x1:x1+image_side_len] = render_number(action1, action_text_height, image_side_len)
            
            # Get ANIMATED denoising frame (this changes over time - it's the prediction evolving!)
            # This shows how the model's prediction of frame 2 evolves during denoising
            denoising_frame = image_readout_list[-1][0, run_idx, frame_idx, :]
            denoising_img = frame_to_image(denoising_frame)
            
            # Place denoising frame (middle section) - ANIMATED PREDICTION
            x_denoise = x_offset + col_width_left + pad
            y_denoise = 0
            if is_rgb or is_3class:
                canvas[y_denoise:y_denoise+image_side_len, x_denoise:x_denoise+image_side_len, :] = denoising_img
            else:
                canvas[y_denoise:y_denoise+image_side_len, x_denoise:x_denoise+image_side_len] = denoising_img
            
            # Place ground truth frame if available (right section) - STATIC GROUND TRUTH
            if run_idx < len(ground_truth_cache):
                gt_frame, gt_action = ground_truth_cache[run_idx]
                x_gt = x_denoise + col_width_middle + pad
                if is_rgb or is_3class:
                    canvas[y_denoise:y_denoise+image_side_len, x_gt:x_gt+image_side_len, :] = np.clip(gt_frame, 0, 1)
                else:
                    canvas[y_denoise:y_denoise+image_side_len, x_gt:x_gt+image_side_len] = np.clip(gt_frame.squeeze(), 0, 1)
        
        # Convert to uint8 for GIF
        canvas_uint8 = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
        all_frames.append(canvas_uint8)
    
    # Save GIF
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, all_frames, fps=fps, loop=0)
    print(f"Saved conditional GIF to {out_path}")

