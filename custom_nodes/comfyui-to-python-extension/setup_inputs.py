#!/usr/bin/env python3
"""
Helper script to set up input files for the ComfyUI to Python extension.
This script helps you organize your input files and validates the workflow structure.
"""

import os
import json
import shutil
from pathlib import Path

def create_input_directory():
    """Create the input_files directory if it doesn't exist."""
    input_dir = Path("input_files")
    input_dir.mkdir(exist_ok=True)
    print(f"✅ Created input directory: {input_dir}")
    return input_dir

def analyze_workflow():
    """Analyze the workflow JSON to find required input files."""
    workflow_file = "workflow_api.json"
    
    if not os.path.exists(workflow_file):
        print(f"❌ Workflow file not found: {workflow_file}")
        return None
    
    try:
        with open(workflow_file, 'r') as f:
            workflow = json.load(f)
        
        required_files = {}
        
        for node_id, node_data in workflow.items():
            if "inputs" in node_data:
                inputs = node_data["inputs"]
                
                # Check for video files
                if "video" in inputs:
                    video_file = inputs["video"]
                    required_files[video_file] = {
                        "type": "video",
                        "node": node_id,
                        "class": node_data.get("class_type", "Unknown")
                    }
                
                # Check for image files
                if "image" in inputs:
                    image_file = inputs["image"]
                    required_files[image_file] = {
                        "type": "image", 
                        "node": node_id,
                        "class": node_data.get("class_type", "Unknown")
                    }
        
        return required_files
    
    except Exception as e:
        print(f"❌ Error reading workflow file: {e}")
        return None

def check_input_files(required_files):
    """Check which input files exist and which are missing."""
    input_dir = Path("input_files")
    
    existing_files = []
    missing_files = []
    
    for filename, info in required_files.items():
        file_path = input_dir / filename
        
        if file_path.exists():
            existing_files.append((filename, info))
            print(f"✅ Found: {filename} ({info['type']}) - Used by {info['class']}")
        else:
            missing_files.append((filename, info))
            print(f"❌ Missing: {filename} ({info['type']}) - Used by {info['class']}")
    
    return existing_files, missing_files

def copy_files_to_input_directory():
    """Interactive function to copy files to input directory."""
    input_dir = Path("input_files")
    
    print("\n📁 Setting up input files...")
    print("Please provide the paths to your input files:")
    
    # Ask for video file
    video_path = input("Enter path to your video file (or press Enter to skip): ").strip()
    if video_path and os.path.exists(video_path):
        video_name = "IMG_4179.mp4"  # Default name from workflow
        dest_path = input_dir / video_name
        
        try:
            shutil.copy2(video_path, dest_path)
            print(f"✅ Copied video to: {dest_path}")
        except Exception as e:
            print(f"❌ Error copying video: {e}")
    elif video_path:
        print(f"❌ Video file not found: {video_path}")
    
    # Ask for image file
    image_path = input("Enter path to your image file (or press Enter to skip): ").strip()
    if image_path and os.path.exists(image_path):
        image_name = "machan_2.JPG"  # Default name from workflow
        dest_path = input_dir / image_name
        
        try:
            shutil.copy2(image_path, dest_path)
            print(f"✅ Copied image to: {dest_path}")
        except Exception as e:
            print(f"❌ Error copying image: {e}")
    elif image_path:
        print(f"❌ Image file not found: {image_path}")

def main():
    """Main function to set up input files."""
    print("🎬 ComfyUI to Python Extension - Input Setup")
    print("=" * 50)
    
    # Create input directory
    input_dir = create_input_directory()
    
    # Analyze workflow
    print("\n📋 Analyzing workflow...")
    required_files = analyze_workflow()
    
    if not required_files:
        print("❌ Could not analyze workflow. Please check your workflow_api.json file.")
        return
    
    print(f"\n📄 Found {len(required_files)} required input files:")
    for filename, info in required_files.items():
        print(f"  - {filename} ({info['type']}) - Used by {info['class']}")
    
    # Check existing files
    print("\n🔍 Checking existing input files...")
    existing_files, missing_files = check_input_files(required_files)
    
    if not missing_files:
        print("\n✅ All required input files are present!")
        print("You can now run: python comfyui_to_python.py")
    else:
        print(f"\n⚠️  Missing {len(missing_files)} input files.")
        
        # Offer to copy files
        if input("\nWould you like to copy your files to the input directory? (y/n): ").lower() == 'y':
            copy_files_to_input_directory()
            
            # Check again after copying
            print("\n🔍 Re-checking input files...")
            existing_files, missing_files = check_input_files(required_files)
            
            if not missing_files:
                print("\n✅ All required input files are now present!")
                print("You can now run: python comfyui_to_python.py")
            else:
                print(f"\n❌ Still missing {len(missing_files)} files. Please add them manually.")
        else:
            print("\n📝 Please manually copy your files to the input_files/ directory:")
            for filename, info in missing_files:
                print(f"  - Copy your {info['type']} file to: input_files/{filename}")

if __name__ == "__main__":
    main() 