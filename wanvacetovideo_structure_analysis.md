# WanVaceToVideo Node Structure Analysis

## Overview
Based on the code analysis of `comfy_extras/nodes_wan.py` and `workflow_api_2.py`, here's the detailed structure analysis of the WanVaceToVideo node and latent generation step.

## WanVaceToVideo Node Analysis

### Input Parameters (from workflow_api_2.py):
- **width**: 480
- **height**: 832  
- **length**: 37 (video frames)
- **batch_size**: 1
- **strength**: 1.0
- **positive**: Text conditioning from previous step
- **negative**: Text conditioning from previous step
- **vae**: VAE model from vaeloader
- **control_video**: Video data from vhs_loadvideo_1
- **reference_image**: Image from loadimage_4

### WanVaceToVideo Processing Steps:

#### 1. Control Video Processing:
```python
# Line 309-314 in nodes_wan.py
if control_video is not None:
    control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    if control_video.shape[0] < length:
        control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5)
else:
    control_video = torch.ones((length, height, width, 3)) * 0.5
```
**Result**: `control_video` shape = `[37, 832, 480, 3]`

#### 2. Reference Image Processing:
```python
# Line 316-319 in nodes_wan.py
if reference_image is not None:
    reference_image = comfy.utils.common_upscale(reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    reference_image = vae.encode(reference_image[:, :, :, :3])
    reference_image = torch.cat([reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))], dim=1)
```
**Result**: `reference_image` shape = `[1, 16, height//8, width//8]` = `[1, 16, 104, 60]`

#### 3. Mask Processing:
```python
# Line 321-329 in nodes_wan.py
if control_masks is None:
    mask = torch.ones((length, height, width, 1))
else:
    mask = control_masks
    # ... processing ...
```
**Result**: `mask` shape = `[37, 832, 480, 1]`

#### 4. VAE Encoding of Control Video:
```python
# Line 335-337 in nodes_wan.py
inactive = vae.encode(inactive[:, :, :, :3])
reactive = vae.encode(reactive[:, :, :, :3])
control_video_latent = torch.cat((inactive, reactive), dim=1)
```
**Result**: `control_video_latent` shape = `[37, 32, 104, 60]` (16 channels each for inactive + reactive)

#### 5. Conditioning Enhancement:
```python
# Line 358-359 in nodes_wan.py
positive = node_helpers.conditioning_set_values(positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
negative = node_helpers.conditioning_set_values(negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
```

#### 6. Latent Generation:
```python
# Line 361-364 in nodes_wan.py
latent_length = ((length - 1) // 4) + 1  # = ((37 - 1) // 4) + 1 = 10
latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8], device=comfy.model_management.intermediate_device())
out_latent = {}
out_latent["samples"] = latent
return io.NodeOutput(positive, negative, out_latent, trim_latent)
```

## Expected Output Structure

### WanVaceToVideo NodeOutput:
```python
io.NodeOutput(
    positive,      # Enhanced positive conditioning
    negative,      # Enhanced negative conditioning  
    out_latent,    # Dictionary with "samples" key
    trim_latent    # Integer (0 in this case)
)
```

### Positive/Negative Conditioning Structure:
```python
# Original conditioning + VACE data
{
    # Original text conditioning tensor
    [tensor([1, 512, 4096])],  # Main text conditioning
    
    # VACE enhancement data
    {
        'pooled_output': None,
        'vace_frames': [control_video_latent],  # Shape: [37, 32, 104, 60]
        'vace_mask': [mask],                   # Shape: [1, 64, 10, 104, 60] (processed)
        'vace_strength': [1.0]                # Float value
    }
}
```

### Latent Output Structure:
```python
out_latent = {
    "samples": tensor([1, 16, 10, 104, 60])  # [batch, channels, frames, height, width]
}
```

## Key Insights:

1. **Latent Length Calculation**: `latent_length = ((37 - 1) // 4) + 1 = 10` frames
2. **Spatial Compression**: Video dimensions `832x480` → Latent dimensions `104x60` (8x compression)
3. **Temporal Compression**: Video frames `37` → Latent frames `10` (3.7x compression)
4. **VACE Enhancement**: Adds control video latents and masks to conditioning
5. **Reference Image**: If provided, adds additional latent channels

## Expected K-Sampler Input Analysis Output:

```
📋 Positive Conditioning (K-Sampler input):
   Type: list
   List length: 1
     Tensor 1:
       Shape: torch.Size([1, 512, 4096])  # Original text conditioning
       Dtype: torch.float32
       Device: cpu
       Memory: 8.00 MB
     Key 'pooled_output': NoneType
       -> None
     Key 'vace_frames': list
       -> Contains 1 items
       Tensor 2:
         Shape: torch.Size([37, 32, 104, 60])  # Control video latent
         Dtype: torch.float16
         Device: cuda:0
         Memory: ~45 MB
     Key 'vace_mask': list  
       -> Contains 1 items
       Tensor 3:
         Shape: torch.Size([1, 64, 10, 104, 60])  # Processed mask
         Dtype: torch.float32
         Device: cuda:0
         Memory: ~91 MB
     Key 'vace_strength': list
       -> Contains 1 items
       -> Value: [1.0]
   Total tensors found: 3
   Total memory: ~144 MB
   ⚠️  WARNING: Found 3 tensors (expected 1)

📋 Latent Image (K-Sampler input):
   Type: dict
   Dict keys: ['samples']
   Samples type: Tensor
   Shape: torch.Size([1, 16, 10, 104, 60])  # Video latent
   Dtype: torch.float32
   Device: cpu
   Memory: 4.19 MB
   Value Range: [0.0000, 0.0000], Mean: 0.0000  # All zeros (initial noise)
```

This analysis explains why you're seeing multiple tensors in the conditioning - the VACE enhancement adds control video latents and masks to guide the generation process.
