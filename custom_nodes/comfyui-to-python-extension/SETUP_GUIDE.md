# ComfyUI to Python Extension - Setup Guide

## Input File Structure

Your workflow expects the following input files:

### Required Files:
1. **Video File**: `IMG_4179.mp4`
   - Used by: VHS_LoadVideo node (Node 1)
   - Purpose: Source video for processing

2. **Image File**: `machan_2.JPG`
   - Used by: LoadImage node (Node 4)
   - Purpose: Reference image for the video generation

### Directory Structure:
```
comfyui-to-python-extension/
├── input_files/
│   ├── IMG_4179.mp4          # Your input video
│   └── machan_2.JPG          # Your reference image
├── workflow_api.json          # Your workflow configuration
├── comfyui_to_python.py      # The conversion script
└── workflow_api.py           # Generated Python code (output)
```

## Setup Instructions:

### 1. Prepare Your Input Files
Place your video and image files in the `input_files/` directory:

```bash
# Create the input directory if it doesn't exist
mkdir -p input_files

# Copy your files to the input directory
cp /path/to/your/video.mp4 input_files/IMG_4179.mp4
cp /path/to/your/image.jpg input_files/machan_2.JPG
```

### 2. Update Workflow JSON (if needed)
If your files have different names, update the `workflow_api.json` file:

```json
{
  "1": {
    "inputs": {
      "video": "your_video_name.mp4",  // Change this to your video filename
      ...
    }
  },
  "4": {
    "inputs": {
      "image": "your_image_name.jpg",   // Change this to your image filename
      ...
    }
  }
}
```

### 3. Run the Conversion Script
```bash
# Basic usage (uses default workflow_api.json)
python comfyui_to_python.py

# Specify custom input/output files
python comfyui_to_python.py -f your_workflow.json -o output.py

# Specify queue size (number of times to run)
python comfyui_to_python.py -q 5
```

### 4. Run the Generated Python Code
```bash
# The script will generate workflow_api.py
python workflow_api.py
```

## Workflow Analysis

Your current workflow performs the following operations:

1. **Load Video** (Node 1): Loads `IMG_4179.mp4` with specific settings
2. **Load Image** (Node 4): Loads `machan_2.JPG` as reference
3. **Load Models**: Loads UNet, VAE, CLIP, and LoRA models
4. **Text Encoding**: Processes positive and negative prompts
5. **Video Processing**: Uses WanVaceToVideo for video generation
6. **Sampling**: Uses KSampler for image generation
7. **Video Output**: Combines frames into final video

## Model Requirements

Your workflow requires these models (make sure they're in your ComfyUI models directory):
- `Wan2.1_14B_VACE-Q6_K.gguf` (UNet)
- `wan_2.1_vae.safetensors` (VAE)
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (CLIP)
- `Wan21_CausVid_14B_T2V_lora_rank32.safetensors` (LoRA)

## Troubleshooting

### Common Issues:
1. **File not found**: Make sure your input files are in the correct location
2. **Model not found**: Ensure all required models are in your ComfyUI models directory
3. **Import errors**: The script should now handle ComfyUI imports correctly

### File Formats Supported:
- **Video**: MP4, AVI, MOV, etc. (depends on your video processing nodes)
- **Image**: JPG, PNG, WEBP, etc.

## Example Usage

```bash
# 1. Prepare your files
cp my_video.mp4 input_files/IMG_4179.mp4
cp my_image.jpg input_files/machan_2.JPG

# 2. Convert workflow to Python
python comfyui_to_python.py

# 3. Run the generated code
python workflow_api.py
```

The generated Python code will be saved as `workflow_api.py` and can be executed independently. 