import os
import random
import sys
import time
import psutil
from typing import Sequence, Mapping, Any, Union
import torch

# Direct imports for VHS functionality
try:
    # Add custom_nodes to path for direct imports
    custom_nodes_path = os.path.join(os.path.dirname(__file__), 'custom_nodes', 'comfyui-videohelpersuite')
    if custom_nodes_path not in sys.path:
        sys.path.insert(0, custom_nodes_path)
    
    # Import the specific classes we need
    from videohelpersuite.load_video_nodes import LoadVideoUpload, LoadVideoPath
    from videohelpersuite.utils import video_extensions
    
    print("✅ Successfully imported VHS video loading classes directly")
    VHS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import VHS classes directly: {e}")
    print("   Will use fallback video loading approach")
    VHS_AVAILABLE = False

# Add monitoring and debugging utilities
class ModelLoadingMonitor:
    """Comprehensive monitoring for model loading steps"""
    
    def __init__(self):
        self.baseline_ram = None
        self.baseline_gpu = None
        self.monitoring_data = {}
        self.step_start_time = None
    
    def start_monitoring(self, step_name):
        """Start monitoring a specific step"""
        self.step_start_time = time.time()
        
        # Record baseline memory state
        self.baseline_ram = psutil.virtual_memory()
        if torch.cuda.is_available():
            self.baseline_gpu = {
                'allocated': torch.cuda.memory_allocated() / 1024**2,
                'reserved': torch.cuda.memory_reserved() / 1024**2
            }
        
        print(f"\n🔍 STARTING MONITORING FOR: {step_name.upper()}")
        print(f"   Baseline RAM: {self.baseline_ram.used / 1024**3:.1f} GB used, {self.baseline_ram.available / 1024**3:.1f} GB available")
        if self.baseline_gpu:
            print(f"   Baseline GPU: {self.baseline_gpu['allocated']:.1f} MB allocated, {self.baseline_gpu['reserved']:.1f} MB reserved")
    
    def end_monitoring(self, step_name, loader_result, model_type):
        """End monitoring and display comprehensive debugging info"""
        elapsed_time = time.time() - self.step_start_time
        
        # Get current memory state
        current_ram = psutil.virtual_memory()
        current_gpu = None
        if torch.cuda.is_available():
            current_gpu = {
                'allocated': torch.cuda.memory_allocated() / 1024**2,
                'reserved': torch.cuda.memory_reserved() / 1024**2
            }
        
        # Calculate memory changes
        ram_change = (current_ram.used - self.baseline_ram.used) / 1024**2  # MB
        gpu_change = None
        if self.baseline_gpu and current_gpu:
            gpu_change = {
                'allocated': current_gpu['allocated'] - self.baseline_gpu['allocated'],
                'reserved': current_gpu['reserved'] - self.baseline_gpu['reserved']
            }
        
        # Store monitoring data
        self.monitoring_data[step_name] = {
            'model_type': model_type,
            'elapsed_time': elapsed_time,
            'ram_change_mb': ram_change,
            'gpu_change_mb': gpu_change,
            'loader_result': loader_result
        }
        
        # Display comprehensive debugging info
        print(f"\n🔍 {step_name.upper()} DEBUGGING COMPLETE")
        print("=" * 60)
        
        # Performance metrics
        print(f"⏱️  PERFORMANCE:")
        print(f"   Loading Time: {elapsed_time:.3f} seconds")
        
        # Memory analysis
        print(f"💾 MEMORY ANALYSIS:")
        print(f"   RAM Change: {ram_change:+.1f} MB")
        print(f"   Current RAM: {current_ram.used / 1024**3:.1f} GB used, {current_ram.available / 1024**3:.1f} GB available")
        
        if gpu_change:
            print(f"   GPU Change: {gpu_change['allocated']:+.1f} MB allocated, {gpu_change['reserved']:+.1f} MB reserved")
            print(f"   Current GPU: {current_gpu['allocated']:.1f} MB allocated, {current_gpu['reserved']:.1f} MB reserved")
        
        # Model information extraction
        print(f"🔧 MODEL INFORMATION:")
        self._extract_model_info(loader_result, model_type)
        
        print("=" * 60)
    
    def _extract_model_info(self, loader_result, model_type):
        """Extract detailed information from the loader result"""
        try:
            # Get the actual model from the loader result
            if isinstance(loader_result, (list, tuple)) and len(loader_result) > 0:
                model = loader_result[0]
            else:
                model = loader_result
            
            print(f"   Model Type: {model_type}")
            print(f"   Result Type: {type(loader_result).__name__}")
            print(f"   Model Class: {type(model).__name__}")
            
            # Extract device information
            if hasattr(model, 'device'):
                print(f"   Device: {model.device}")
            elif hasattr(model, 'model') and hasattr(model.model, 'device'):
                print(f"   Device: {model.model.device}")
            
            # Extract model size information
            if hasattr(model, 'model'):
                if hasattr(model.model, 'parameters'):
                    param_count = sum(p.numel() for p in model.model.parameters())
                    print(f"   Parameters: {param_count:,}")
                if hasattr(model.model, 'state_dict'):
                    state_dict = model.model.state_dict()
                    print(f"   State Dict Keys: {len(state_dict)}")
            
            # Extract specific model attributes
            if model_type == "VAE":
                if hasattr(model, 'first_stage_model'):
                    print(f"   Has First Stage Model: ✅")
                if hasattr(model, 'scale_factor'):
                    print(f"   Scale Factor: {model.scale_factor}")
            
            elif model_type == "UNET":
                if hasattr(model, 'model'):
                    if hasattr(model.model, 'num_blocks'):
                        print(f"   Number of Blocks: {model.model.num_blocks}")
                    if hasattr(model.model, 'in_channels'):
                        print(f"   Input Channels: {model.model.in_channels}")
            
            elif model_type == "CLIP":
                if hasattr(model, 'patcher'):
                    print(f"   Has ModelPatcher: ✅")
                if hasattr(model, 'max_length'):
                    print(f"   Max Length: {model.max_length}")
            
        except Exception as e:
            print(f"   ⚠️  Error extracting model info: {e}")
    
    def print_summary(self):
        """Print summary of all monitored steps"""
        print(f"\n📊 STEP 1 MODEL LOADING SUMMARY")
        print("=" * 80)
        
        total_time = sum(data['elapsed_time'] for data in self.monitoring_data.values())
        total_ram_change = sum(data['ram_change_mb'] for data in self.monitoring_data.values())
        
        print(f"⏱️  Total Loading Time: {total_time:.3f} seconds")
        print(f"💾 Total RAM Change: {total_ram_change:+.1f} MB")
        
        for step_name, data in self.monitoring_data.items():
            print(f"\n🔍 {step_name.upper()}:")
            print(f"   Model: {data['model_type']}")
            print(f"   Time: {data['elapsed_time']:.3f}s")
            print(f"   RAM: {data['ram_change_mb']:+.1f} MB")
            if data['gpu_change_mb']:
                print(f"   GPU: {data['gpu_change_mb']['allocated']:+.1f} MB allocated")
        
        print("=" * 80)

# Initialize the monitor
model_monitor = ModelLoadingMonitor()


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path and change to the ComfyUI directory
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        # Add ComfyUI directory to Python path
        sys.path.insert(0, comfyui_path)
        # Change to the ComfyUI directory to ensure relative imports work correctly
        original_cwd = os.getcwd()
        os.chdir(comfyui_path)


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> dict:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    # Ensure we're in the ComfyUI directory for imports to work correctly
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        original_cwd = os.getcwd()
        original_sys_path = sys.path.copy()

        # Add ComfyUI directory to Python path and change directory
        sys.path.insert(0, comfyui_path)
        os.chdir(comfyui_path)

        try:
            import asyncio
            import execution
            from nodes import init_extra_nodes
            import server

            # Creating a new event loop and setting it as the default loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Creating an instance of PromptServer with the loop
            server_instance = server.PromptServer(loop)
            execution.PromptQueue(server_instance)

            # Initializing custom nodes
            loop.run_until_complete(init_extra_nodes())
            
        except Exception as e:
            pass
        
        # Manual fallback: try to import and merge custom nodes directly
        try:
            # Add custom_nodes to path
            custom_nodes_path = os.path.join(comfyui_path, 'custom_nodes')
            sys.path.insert(0, custom_nodes_path)
            
            # Try importing videohelpersuite
            try:
                import comfyui_videohelpersuite.videohelpersuite.nodes as vhs_nodes
                return vhs_nodes.NODE_CLASS_MAPPINGS
            except ImportError:
                pass
                
            # Try the full path approach
            try:
                sys.path.insert(0, os.path.join(custom_nodes_path, 'comfyui-videohelpersuite'))
                import videohelpersuite.nodes as vhs_nodes
                return vhs_nodes.NODE_CLASS_MAPPINGS
            except ImportError:
                pass
                
        except Exception as e:
            pass
        
        finally:
            # Restore original working directory and sys.path
            os.chdir(original_cwd)
            sys.path = original_sys_path
    
    return {}


def main():
    # Load custom nodes FIRST, before importing NODE_CLASS_MAPPINGS
    custom_node_mappings = import_custom_nodes()
    
    # Now import NODE_CLASS_MAPPINGS after custom nodes are loaded
    from nodes import (
        NODE_CLASS_MAPPINGS,
        CLIPTextEncode,
        UNETLoader,
        CLIPLoader,
        LoraLoader,
        KSampler,
        VAEDecode,
        VAELoader,
        LoadImage,
    )
    
    # Manually merge custom nodes if they weren't merged automatically
    if custom_node_mappings:
        NODE_CLASS_MAPPINGS.update(custom_node_mappings)
    
    # Load comfy_extras nodes that contain the missing workflow nodes
    try:
        # Import the specific modules we need
        import comfy_extras.nodes_wan
        import comfy_extras.nodes_model_advanced
        
        # Get the node mappings - handle both old and new style
        wan_nodes = {}
        model_nodes = {}
        
        # Check if nodes_wan has NODE_CLASS_MAPPINGS (old style)
        if hasattr(comfy_extras.nodes_wan, 'NODE_CLASS_MAPPINGS'):
            wan_nodes = comfy_extras.nodes_wan.NODE_CLASS_MAPPINGS
        else:
            # New style - try to get nodes from ComfyExtension
            try:
                if hasattr(comfy_extras.nodes_wan, 'WanExtension'):
                    # Create a temporary instance to get the node list
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    extension = loop.run_until_complete(comfy_extras.nodes_wan.comfy_entrypoint())
                    node_list = loop.run_until_complete(extension.get_node_list())
                    
                    # Convert to NODE_CLASS_MAPPINGS format
                    for node_class in node_list:
                        if hasattr(node_class, 'define_schema'):
                            schema = node_class.define_schema()
                            if hasattr(schema, 'node_id'):
                                wan_nodes[schema.node_id] = node_class
                    
            except Exception as e:
                pass
        
        # Check if nodes_model_advanced has NODE_CLASS_MAPPINGS
        if hasattr(comfy_extras.nodes_model_advanced, 'NODE_CLASS_MAPPINGS'):
            model_nodes = comfy_extras.nodes_model_advanced.NODE_CLASS_MAPPINGS
        
        # Merge them into the main NODE_CLASS_MAPPINGS
        if wan_nodes:
            NODE_CLASS_MAPPINGS.update(wan_nodes)
        if model_nodes:
            NODE_CLASS_MAPPINGS.update(model_nodes)
                
    except Exception as e:
        pass
    
    # Check if custom nodes are available, but continue for debugging purposes
    if 'VHS_LoadVideo' not in NODE_CLASS_MAPPINGS:
        print("⚠️  WARNING: VHS_LoadVideo not found in NODE_CLASS_MAPPINGS!")
        print("🔍 This is expected if custom nodes aren't loaded. Continuing with model loading debugging...")
        print("📋 Available nodes: " + ", ".join(list(NODE_CLASS_MAPPINGS.keys())[:10]) + "...")
    else:
        print("✅ VHS_LoadVideo found in NODE_CLASS_MAPPINGS")
    
    with torch.inference_mode():
        # === STEP 1 START: MODEL LOADING ===
        print("1. Loading diffusion model components...")
        
        # Load video and reference image using direct imports
        if VHS_AVAILABLE:
            print("🎥 Loading video using direct VHS imports...")
            try:
                # Use LoadVideoPath for direct file loading
                video_loader = LoadVideoPath()
                vhs_loadvideo_1 = video_loader.load_video(
                    video="safu.mp4",
                    force_rate=0,
                    custom_width=0,
                    custom_height=0,
                    frame_load_cap=0,
                    skip_first_frames=0,
                    select_every_nth=1
                )
                print("✅ Video loaded successfully using direct VHS import")
            except Exception as e:
                print(f"❌ Error loading video with direct import: {e}")
                vhs_loadvideo_1 = None
        else:
            print("⏭️  Skipping video loading (VHS not available)")
            vhs_loadvideo_1 = None
        
        # Load reference image
        try:
            loadimage = LoadImage()
            loadimage_4 = loadimage.load_image(image="safu.jpg")
            print("✅ Reference image loaded successfully")
            
            # Debug: Show what was loaded
            if loadimage_4:
                print(f"   Image type: {type(loadimage_4)}")
                if isinstance(loadimage_4, (list, tuple)) and len(loadimage_4) > 0:
                    print(f"   Image data type: {type(loadimage_4[0])}")
                    if hasattr(loadimage_4[0], 'shape'):
                        print(f"   Image shape: {loadimage_4[0].shape}")
        except Exception as e:
            print(f"❌ Error loading reference image: {e}")
            loadimage_4 = None
        
        # Debug: Show video loading results
        if vhs_loadvideo_1:
            print(f"   Video type: {type(vhs_loadvideo_1)}")
            if isinstance(vhs_loadvideo_1, (list, tuple)) and len(vhs_loadvideo_1) > 0:
                print(f"   Video data type: {type(vhs_loadvideo_1[0])}")
                if hasattr(vhs_loadvideo_1[0], 'shape'):
                    print(f"   Video shape: {vhs_loadvideo_1[0].shape}")
                if len(vhs_loadvideo_1) > 1:
                    print(f"   Frame count: {vhs_loadvideo_1[1]}")
        else:
            print("   Video: Not loaded")

        # === ENHANCED MODEL LOADING WITH COMPREHENSIVE DEBUGGING ===
        
        # Check if model files exist
        print("\n🔍 CHECKING MODEL FILES:")
        model_files = {
            "VAE": "vae.safetensors",
            "UNET": "model.safetensors", 
            "CLIP": "clip.safetensors"
        }
        
        for model_name, filename in model_files.items():
            if os.path.exists(filename):
                print(f"   ✅ {model_name}: {filename} - EXISTS")
            else:
                print(f"   ❌ {model_name}: {filename} - NOT FOUND")
                print(f"      Current working directory: {os.getcwd()}")
                print(f"      Looking for: {os.path.abspath(filename)}")
        
        print()
        
        # Load VAE with monitoring
        try:
            model_monitor.start_monitoring("vae_loading")
            vaeloader = VAELoader()
            vaeloader_7 = vaeloader.load_vae(vae_name="vae.safetensors")
            model_monitor.end_monitoring("vae_loading", vaeloader_7, "VAE")
        except Exception as e:
            print(f"❌ ERROR loading VAE: {e}")
            vaeloader_7 = None

        # Load UNET with monitoring
        try:
            model_monitor.start_monitoring("unet_loading")
            unetloader = UNETLoader()
            unetloader_27 = unetloader.load_unet(
                unet_name="model.safetensors", weight_dtype="default"
            )
            model_monitor.end_monitoring("unet_loading", unetloader_27, "UNET")
        except Exception as e:
            print(f"❌ ERROR loading UNET: {e}")
            unetloader_27 = None

        # Load CLIP with monitoring
        try:
            model_monitor.start_monitoring("clip_loading")
            cliploader = CLIPLoader()
            cliploader_23 = cliploader.load_clip(
                clip_name="clip.safetensors", type="wan", device="default"
            )
            model_monitor.end_monitoring("clip_loading", cliploader_23, "CLIP")
        except Exception as e:
            print(f"❌ ERROR loading CLIP: {e}")
            cliploader_23 = None
        
        # Print comprehensive summary of all model loading steps
        model_monitor.print_summary()
        
        print("✅ Step 1 completed: Model Loading")
        # === STEP 1 END: MODEL LOADING ===
        
        # Stop execution after step 1 for debugging purposes
        print("\n🛑 STOPPING EXECUTION AFTER STEP 1 (MODEL LOADING)")
        print("🔍 All model loading debugging information has been displayed above.")
        print("📊 Check the monitoring data above to analyze model loading performance.")
        return

        # === STEP 2 START: LORA APPLICATION ===
        print("2. Applying LoRA...")
        
        loraloader = LoraLoader()
        loraloader_24 = loraloader.load_lora(
            lora_name="lora.safetensors",
            strength_model=0.5000000000000001,
            strength_clip=1,
            model=get_value_at_index(unetloader_27, 0),
            clip=get_value_at_index(cliploader_23, 0),
        )
        
        print("✅ Step 2 completed: LoRA Application")
        # === STEP 2 END: LORA APPLICATION ===

        # === STEP 3 START: TEXT ENCODING ===
        print("3. Encoding text prompts...")
        
        cliptextencode = CLIPTextEncode()
        cliptextencode_10 = cliptextencode.encode(
            text="a cinematic video.", clip=get_value_at_index(loraloader_24, 1)
        )

        cliptextencode_11 = cliptextencode.encode(
            text="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 , extra hands, extra arms, extra legs",
            clip=get_value_at_index(loraloader_24, 1),
        )
        
        print("✅ Step 3 completed: Text Encoding")
        # === STEP 3 END: TEXT ENCODING ===

        # === STEP 4 START: MODEL SAMPLING ===
        print("4. Applying ModelSamplingSD3...")
        
        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        modelsamplingsd3_15 = modelsamplingsd3.patch(
            shift=8.000000000000002, model=get_value_at_index(loraloader_24, 0)
        )
        
        print("✅ Step 4 completed: Model Sampling")
        # === STEP 4 END: MODEL SAMPLING ===

        # === STEP 5 START: INITIAL LATENT GENERATION ===
        print("5. Generating initial latents...")
        
        wanvacetovideo = NODE_CLASS_MAPPINGS["WanVaceToVideo"]()
        wanvacetovideo_13 = wanvacetovideo.EXECUTE_NORMALIZED(
            width=480,
            height=832,
            length=37,
            batch_size=1,
            strength=1,
            positive=get_value_at_index(cliptextencode_10, 0),
            negative=get_value_at_index(cliptextencode_11, 0),
            vae=get_value_at_index(vaeloader_7, 0),
            control_video=get_value_at_index(vhs_loadvideo_1, 0),
            reference_image=get_value_at_index(loadimage_4, 0),
        )
        
        print("✅ Step 5 completed: Initial Latent Generation")
        # === STEP 5 END: INITIAL LATENT GENERATION ===

        # === STEP 6 START: UNET SAMPLING ===
        print("6. Running KSampler...")
        
        ksampler = KSampler()
        ksampler_14 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=4,
            cfg=1,
            sampler_name="ddim",
            scheduler="normal",
            denoise=1,
            model=get_value_at_index(modelsamplingsd3_15, 0),
            positive=get_value_at_index(wanvacetovideo_13, 0),
            negative=get_value_at_index(wanvacetovideo_13, 1),
            latent_image=get_value_at_index(wanvacetovideo_13, 2),
        )
        
        print("✅ Step 6 completed: UNET Sampling")
        # === STEP 6 END: UNET SAMPLING ===

        # === STEP 7 START: VIDEO TRIMMING ===
        print("7. Trimming video latent...")
        
        trimvideolatent = NODE_CLASS_MAPPINGS["TrimVideoLatent"]()
        trimvideolatent_16 = trimvideolatent.EXECUTE_NORMALIZED(
            trim_amount=get_value_at_index(wanvacetovideo_13, 3),
            samples=get_value_at_index(ksampler_14, 0),
        )
        
        print("✅ Step 7 completed: Video Trimming")
        # === STEP 7 END: VIDEO TRIMMING ===

        # === STEP 8 START: VAE DECODING ===
        print("8. Decoding latents to frames...")
        
        vaedecode = VAEDecode()
        vaedecode_18 = vaedecode.decode(
            samples=get_value_at_index(trimvideolatent_16, 0),
            vae=get_value_at_index(vaeloader_7, 0),
        )
        
        print("✅ Step 8 completed: VAE Decoding")
        # === STEP 8 END: VAE DECODING ===

        # === STEP 9 START: VIDEO EXPORT ===
        print("9. Exporting video...")
        
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        vhs_videocombine_19 = vhs_videocombine.combine_video(
            frame_rate=19,
            loop_count=0,
            filename_prefix="AnimateDiff",
            format="video/h264-mp4",
            pix_fmt="yuv420p",
            crf=19,
            save_metadata=True,
            trim_to_audio=False,
            pingpong=False,
            save_output=True,
            images=get_value_at_index(vaedecode_18, 0),
        )
        
        print("✅ Step 9 completed: Video Export")
        print("🎉 All workflow steps completed successfully!")
        
        # === FINAL MONITORING SUMMARY ===
        print("\n" + "="*80)
        print("🔍 FINAL WORKFLOW MONITORING SUMMARY")
        print("="*80)
        model_monitor.print_summary()
        print("="*80)
        
        # === STEP 9 END: VIDEO EXPORT ===


if __name__ == "__main__":
    main()
