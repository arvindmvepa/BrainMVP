import torch
import torch.nn.functional as F
from typing import Callable, List, Sequence, Tuple, Union, Any
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple, look_up_option
from models.Uniformer import SSLEncoder
import monai.transforms as transforms

def sliding_window_embedding_inference(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    return_patch_locations: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple]]]:
    """
    Sliding window inference for embedding extraction.
    
    Args:
        inputs: input image to be processed (NCHW[D])
        roi_size: spatial window size for inference patches
        sw_batch_size: batch size for processing patches
        predictor: embedding model that takes patch and returns features
        overlap: overlap between patches (0-1)
        mode: blending mode for overlapping regions
        sigma_scale: Gaussian weighting parameter
        padding_mode: padding mode when roi_size > input size
        cval: constant value for padding
        sw_device: device for patch processing
        device: device for output
        return_patch_locations: if True, also return patch coordinates
        
    Returns:
        If return_patch_locations=False: aggregated embedding tensor
        If return_patch_locations=True: (embeddings, patch_locations)
    """
    
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # Determine image spatial size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    
    # Pad input if needed
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    
    num_win = len(slices)
    total_slices = num_win * batch_size

    # Create importance map for weighting
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Storage for embeddings and weights
    output_embeddings = None
    count_map = None
    patch_locations = [] if return_patch_locations else None
    _initialized = False

    # Process patches in batches
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]

        # Extract patch data
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        
        # Get embeddings from predictor
        patch_embeddings = predictor(window_data, *args, **kwargs).to(device)
        
        # Initialize output tensors on first iteration
        if not _initialized:
            embedding_dim = patch_embeddings.shape[1]  # Assumes shape: (batch, embedding_dim, ...)
            
            # For spatial embeddings (feature maps)
            if len(patch_embeddings.shape) > 2:
                spatial_shape = patch_embeddings.shape[2:]
                output_shape = [batch_size, embedding_dim] + list(image_size)
                # Scale spatial dimensions if embedding has different spatial size than input
                if spatial_shape != roi_size:
                    scale_factor = [image_size[i] / spatial_shape[i] for i in range(num_spatial_dims)]
                    scaled_image_size = [int(image_size[i] / (roi_size[i] / spatial_shape[i])) 
                                       for i in range(num_spatial_dims)]
                    output_shape = [batch_size, embedding_dim] + scaled_image_size
            else:
                # For global embeddings (vectors)
                output_shape = [batch_size, embedding_dim] + list(image_size)
                
            output_embeddings = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # Store embeddings with importance weighting
        for idx, original_idx in zip(slice_range, unravel_slice):
            embedding = patch_embeddings[idx - slice_g]
            
            # Handle different embedding shapes
            if len(embedding.shape) == 1:  # Global embedding vector
                # Broadcast to spatial dimensions
                spatial_importance = importance_map[0, 0]  # Get spatial weight
                weighted_embedding = embedding.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * spatial_importance
                output_embeddings[original_idx] += weighted_embedding
                count_map[original_idx] += importance_map[0:1]  # Single channel for counting
            else:  # Spatial embedding
                # Resize if needed
                if embedding.shape[1:] != importance_map.shape[1:]:
                    embedding = F.interpolate(
                        embedding.unsqueeze(0), 
                        size=importance_map.shape[1:], 
                        mode='trilinear', 
                        align_corners=False
                    ).squeeze(0)
                
                output_embeddings[original_idx] += importance_map * embedding
                count_map[original_idx] += importance_map
            
            # Store patch location if requested
            if return_patch_locations:
                patch_coords = tuple(s.start for s in original_idx[2:])  # Skip batch and channel dims
                patch_locations.append(patch_coords)

    # Average overlapping regions
    output_embeddings = output_embeddings / count_map

    # Remove padding
    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_embeddings.shape):
        final_slicing.insert(0, slice(None))
    
    final_embeddings = output_embeddings[final_slicing]
    
    if return_patch_locations:
        return final_embeddings, patch_locations
    return final_embeddings


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """Compute scan interval according to image size, roi size and overlap."""
    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def extract_volume_embeddings(volume, encoder, patch_size=96, overlap=0.5):
    """
    Extract embeddings from full volume using sliding window approach
    
    Args:
        volume: Input volume tensor (1, 4, H, W, D)
        encoder: Pretrained encoder model
        patch_size: Size of patches for processing
        overlap: Overlap between patches
        
    Returns:
        Aggregated embeddings for the full volume
    """
    
    def embedding_predictor(patches):
        """Wrapper to extract embeddings from patches"""
        with torch.no_grad():
            features = encoder(patches)
            # Return final layer features or global pooled features
            if isinstance(features, list):
                return features[-1]  # Last layer features
            return features
    
    # Extract embeddings using sliding window
    embeddings = sliding_window_embedding_inference(
        inputs=volume,
        roi_size=(patch_size, patch_size, patch_size),
        sw_batch_size=1,  # Process one patch at a time
        predictor=embedding_predictor,
        overlap=overlap,
        mode='gaussian',  # Use gaussian blending
    )
    
    return embeddings

# Example usage
def main():
    # Load your volume and encoder
    file = "/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2/BraTS-GLI-03064-100/BraTS-GLI-03064-100-t1c.nii.gz"
    checkpoint_pth = "BrainMVP_uniformer.pt"
    val_transform = transforms.Compose([
        transforms.LoadImaged(keys=['image']), 
        transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=5, upper=95, b_min=0.0, b_max=1.0, channel_wise=True), 
        transforms.Orientationd(keys=['image'], axcodes='RAS'), 
        transforms.Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),  # Changed: mode='bilinear' instead of mode=['bilinear', 'nearest']
        transforms.CropForegroundd(keys=['image'], source_key='image', margin=1)
    ])
    volume = val_transform({'image': file})['image'].unsqueeze(0)  # Add batch dim
    print("test volume loaded and transformed")
    print(volume.shape)
    encoder = SSLEncoder(num_phase=1, initial_checkpoint=checkpoint_pth)
    encoder.load_state_dict({k[8:]:v for k,v in state_dict.items() if 'encoder' in k})
    print("model loaded!")
    encoder.eval()
    
    # Extract embeddings
    embeddings = extract_volume_embeddings(volume, encoder, patch_size=96, overlap=0.5)
    print(f"Volume embeddings shape: {embeddings.shape}")
    
    # For global volume representation, you can pool the spatial dimensions
    global_embedding = torch.mean(embeddings, dim=[2, 3, 4])  # Average pool spatial dims
    print(f"Global volume embedding shape: {global_embedding.shape}")

if __name__ == "__main__":
    main()
