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

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    
    num_win = len(slices)
    total_slices = num_win * batch_size

    # Create importance map for weighting
    importance_map = None

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
        patch_embeddings = predictor(window_data, *args, **kwargs)
        
        # Handle tuple output from UniFormer
        if isinstance(patch_embeddings, tuple):
            patch_embeddings = patch_embeddings[-1].to(device)  # Use the last/deepest features
        else:
            patch_embeddings = patch_embeddings.to(device)

        # Initialize output tensors on first iteration
        if not _initialized:
            embedding_dim = patch_embeddings.shape[1]  # Assumes shape: (batch, embedding_dim, ...)
            embedding_spatial_shape = patch_embeddings.shape[2:]  # Actual embedding spatial size
            print(f"Embedding spatial shape: {embedding_spatial_shape}")
            print(f"Input ROI size: {roi_size}")

            scale_factor = [roi_size[i] / embedding_spatial_shape[i] for i in range(num_spatial_dims)]
            scaled_image_size = [int(image_size[i] / scale_factor[i]) for i in range(num_spatial_dims)]
            output_shape = [batch_size, embedding_dim] + scaled_image_size
                
            output_embeddings = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)

            importance_map = compute_importance_map(
                get_valid_patch_size(scaled_image_size, embedding_spatial_shape),
                mode=mode,
                sigma_scale=sigma_scale,
                device=device
            )

            print(f"Importance map shape: {importance_map.shape}")
            _initialized = True

        # Store embeddings with importance weighting
        for idx, original_idx in zip(slice_range, unravel_slice):
            embedding = patch_embeddings[idx - slice_g]
            print(embedding.shape)
            print(importance_map.shape)
            
            # Scale the slice indices from input space to embedding space
            scaled_idx = [original_idx[0], original_idx[1]]  # Keep batch and channel dims
            for dim_idx in range(2, len(original_idx)):  # Scale spatial dimensions
                s = original_idx[dim_idx]
                if isinstance(s, slice):
                    start_scaled = int(s.start / scale_factor[dim_idx - 2]) if s.start is not None else None
                    stop_scaled = int(s.stop / scale_factor[dim_idx - 2]) if s.stop is not None else None
                    scaled_idx.append(slice(start_scaled, stop_scaled, s.step))
                else:
                    scaled_idx.append(s)
            print(scaled_idx)
            
            weighted_embedding = importance_map.unsqueeze(0) * embedding
            output_embeddings[scaled_idx] += weighted_embedding
            count_map[scaled_idx] += importance_map.unsqueeze(0)

            # Store patch location if requested
            if return_patch_locations:
                patch_coords = tuple(s.start for s in original_idx[2:])
                patch_locations.append(patch_coords)

    print(count_map)
    print(torch.max(count_map))
    print(torch.min(count_map))
    # account for any overlapping sections
    output_embeddings = output_embeddings / count_map

    if return_patch_locations:
        return output_embeddings, patch_locations
    return output_embeddings


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
        mode='constant',  # Use constant blending
    )
    
    return embeddings

# Example usage
def main():
    # Load your volume and encoder
    file = "/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2/BraTS-GLI-03064-100/BraTS-GLI-03064-100-t1c.nii.gz"
    checkpoint_pth = "BrainMVP_uniformer.pt"
    val_transform = transforms.Compose([
        transforms.LoadImaged(keys=['image']),
        transforms.EnsureChannelFirstd(keys=['image']),
        transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=5, upper=95, b_min=0.0, b_max=1.0, channel_wise=True), 
        transforms.Orientationd(keys=['image'], axcodes='RAS'), 
        transforms.Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),  # Changed: mode='bilinear' instead of mode=['bilinear', 'nearest']
        transforms.CropForegroundd(keys=['image'], source_key='image', margin=1)
    ])
    volume = val_transform({'image': file})['image'].unsqueeze(0)  # Add batch dim
    print("test volume loaded and transformed")
    print(volume.shape)
    encoder = SSLEncoder(num_phase=1, initial_checkpoint=checkpoint_pth)
    checkpoint = torch.load(checkpoint_pth, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    encoder_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('module.', '')
        if 'encoder.' in clean_key:
            encoder_key = clean_key.replace('encoder.', '')
            encoder_state_dict[encoder_key] = v
    encoder.load_state_dict(encoder_state_dict)
    print("model loaded!")
    encoder.eval()
    
    # Set device to first GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Move data and model to GPU
    volume = volume.to(device)
    encoder = encoder.to(device)
    
    # Extract embeddings
    embeddings = extract_volume_embeddings(volume, encoder, patch_size=96, overlap=0.5)
    print(f"Volume embeddings shape: {embeddings.shape}")
    
    # For global volume representation, you can pool the spatial dimensions
    global_embedding = torch.mean(embeddings, dim=[2, 3, 4])  # Average pool spatial dims
    print(f"Global volume embedding shape: {global_embedding.shape}")

if __name__ == "__main__":
    main()
