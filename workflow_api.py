import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


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
        print(f"{name} found: {path_name}")
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
        print(f"'{comfyui_path}' added to sys.path")

        # Change to the ComfyUI directory to ensure relative imports work correctly
        original_cwd = os.getcwd()
        os.chdir(comfyui_path)
        print(f"Changed working directory from '{original_cwd}' to '{comfyui_path}'")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> dict:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current sys.path: {sys.path[:3]}...")  # Show first 3 paths
    
    # Ensure we're in the ComfyUI directory for imports to work correctly
    comfyui_path = find_path("ComfyUI")
    print(f"find_path('ComfyUI') returned: {comfyui_path}")
    
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        print(f"ComfyUI path found and is directory: {comfyui_path}")
        original_cwd = os.getcwd()
        original_sys_path = sys.path.copy()

        # Add ComfyUI directory to Python path and change directory
        sys.path.insert(0, comfyui_path)
        os.chdir(comfyui_path)
        print(f"Changed to ComfyUI directory: {comfyui_path}")

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
            print("Running init_extra_nodes...")
            loop.run_until_complete(init_extra_nodes())
            
        except Exception as e:
            print(f"init_extra_nodes failed: {e}")
        
        # Manual fallback: try to import and merge custom nodes directly
        print("Attempting manual custom node import...")
        try:
            # Add custom_nodes to path
            custom_nodes_path = os.path.join(comfyui_path, 'custom_nodes')
            sys.path.insert(0, custom_nodes_path)
            print(f"Added {custom_nodes_path} to Python path")
            
            # List what's in the custom_nodes directory
            print(f"Contents of {custom_nodes_path}: {os.listdir(custom_nodes_path)}")
            
            # Try different import approaches
            try:
                # Approach 1: Direct import
                import comfyui_videohelpersuite.videohelpersuite.nodes as vhs_nodes
                print(f"✓ Approach 1 succeeded: {len(vhs_nodes.NODE_CLASS_MAPPINGS)} nodes")
                return vhs_nodes.NODE_CLASS_MAPPINGS
            except ImportError as e1:
                print(f"Approach 1 failed: {e1}")
                
                # Approach 2: Try with underscore
                try:
                    import comfyui_videohelpersuite.videohelpersuite.nodes as vhs_nodes
                    print(f"✓ Approach 2 succeeded: {len(vhs_nodes.NODE_CLASS_MAPPINGS)} nodes")
                    return vhs_nodes.NODE_CLASS_MAPPINGS
                except ImportError as e2:
                    print(f"Approach 2 failed: {e2}")
                    
                    # Approach 3: Try importing the package first
                    try:
                        import comfyui_videohelpersuite
                        print("✓ Imported comfyui_videohelpersuite package")
                        from comfyui_videohelpersuite.videohelpersuite.nodes import NODE_CLASS_MAPPINGS
                        print(f"✓ Approach 3 succeeded: {len(NODE_CLASS_MAPPINGS)} nodes")
                        return NODE_CLASS_MAPPINGS
                    except ImportError as e3:
                        print(f"Approach 3 failed: {e3}")
                        
                        # Approach 4: Try importing from the full path
                        try:
                            sys.path.insert(0, os.path.join(custom_nodes_path, 'comfyui-videohelpersuite'))
                            import videohelpersuite.nodes as vhs_nodes
                            print(f"✓ Approach 4 succeeded: {len(vhs_nodes.NODE_CLASS_MAPPINGS)} nodes")
                            return vhs_nodes.NODE_CLASS_MAPPINGS
                        except ImportError as e4:
                            print(f"Approach 4 failed: {e4}")
                            raise ImportError(f"All import approaches failed. Last error: {e4}")
            
        except Exception as e:
            print(f"Manual import failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Restore original working directory and sys.path
            os.chdir(original_cwd)
            sys.path = original_sys_path
            print(f"Restored working directory: {original_cwd}")
    else:
        print(f"ComfyUI path not found or not a directory. comfyui_path: {comfyui_path}")
        print("Trying alternative approach...")
        
        # Alternative: try to work from current directory
        current_dir = os.getcwd()
        print(f"Working from current directory: {current_dir}")
        
        # Check if we're already in a ComfyUI-like structure
        if os.path.exists('nodes.py') and os.path.exists('custom_nodes'):
            print("✓ Found nodes.py and custom_nodes in current directory")
            
            # Initialize ComfyUI environment first
            try:
                print("Initializing ComfyUI environment...")
                import asyncio
                import execution
                import server
                
                # Create event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Initialize PromptServer properly
                server_instance = server.PromptServer(loop)
                execution.PromptQueue(server_instance)
                
                print("✓ ComfyUI environment initialized")
                
                # Now try to import custom nodes
                try:
                    custom_nodes_path = os.path.join(current_dir, 'custom_nodes')
                    sys.path.insert(0, custom_nodes_path)
                    print(f"Added {custom_nodes_path} to Python path")
                    
                    # List what's in the custom_nodes directory
                    print(f"Contents of {custom_nodes_path}: {os.listdir(custom_nodes_path)}")
                    
                    # Try importing videohelpersuite with proper server context
                    try:
                        # Approach 1: Direct import
                        import comfyui_videohelpersuite.videohelpersuite.nodes as vhs_nodes
                        print(f"✓ Alternative import succeeded: {len(vhs_nodes.NODE_CLASS_MAPPINGS)} nodes")
                        return vhs_nodes.NODE_CLASS_MAPPINGS
                    except ImportError as e:
                        print(f"Direct import failed: {e}")
                        
                        # Approach 2: Try the full path approach
                        try:
                            sys.path.insert(0, os.path.join(custom_nodes_path, 'comfyui-videohelpersuite'))
                            import videohelpersuite.nodes as vhs_nodes
                            print(f"✓ Full path import succeeded: {len(vhs_nodes.NODE_CLASS_MAPPINGS)} nodes")
                            return vhs_nodes.NODE_CLASS_MAPPINGS
                        except ImportError as e2:
                            print(f"Full path import failed: {e2}")
                            
                            # Approach 3: Try using init_extra_nodes
                            try:
                                print("Trying init_extra_nodes from current directory...")
                                from nodes import init_extra_nodes
                                loop.run_until_complete(init_extra_nodes())
                                print("✓ init_extra_nodes completed")
                                
                                # Check if nodes are now available
                                from nodes import NODE_CLASS_MAPPINGS
                                if 'VHS_LoadVideo' in NODE_CLASS_MAPPINGS:
                                    print("✓ VHS_LoadVideo found after init_extra_nodes")
                                    return {}  # Return empty since nodes are already in main mappings
                                else:
                                    print("✗ VHS_LoadVideo still not found after init_extra_nodes")
                                    raise ImportError("VHS_LoadVideo not found after init_extra_nodes")
                                    
                            except Exception as e3:
                                print(f"init_extra_nodes approach failed: {e3}")
                                raise ImportError(f"All alternative approaches failed. Last error: {e3}")
                                
                except Exception as e:
                    print(f"Custom node import failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as e:
                print(f"ComfyUI environment initialization failed: {e}")
                import traceback
                traceback.print_exc()
    
    print("No custom nodes could be loaded")
    return {}


def main():
    # Load custom nodes FIRST, before importing NODE_CLASS_MAPPINGS
    print("Loading custom nodes...")
    custom_node_mappings = import_custom_nodes()
    
    # Now import NODE_CLASS_MAPPINGS after custom nodes are loaded
    print("Importing node mappings...")
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
        print(f"Merging {len(custom_node_mappings)} custom nodes into main mappings...")
        NODE_CLASS_MAPPINGS.update(custom_node_mappings)
        print(f"Total nodes after merge: {len(NODE_CLASS_MAPPINGS)}")
        
        # Show what custom nodes we have
        print(f"Custom nodes available: {list(custom_node_mappings.keys())}")
        
        # Check for specific nodes the workflow needs
        required_custom_nodes = ['WanVaceToVideo', 'ModelSamplingSD3', 'TrimVideoLatent', 'VHS_VideoCombine']
        missing_nodes = []
        for node in required_custom_nodes:
            if node in NODE_CLASS_MAPPINGS:
                print(f"✓ {node} found")
            else:
                print(f"✗ {node} missing")
                missing_nodes.append(node)
        
        if missing_nodes:
            print(f"\n⚠️  Missing required custom nodes: {missing_nodes}")
            print("This workflow requires additional custom nodes that aren't available.")
            return
    
    # Verify VHS_LoadVideo is available
    if 'VHS_LoadVideo' not in NODE_CLASS_MAPPINGS:
        print("ERROR: VHS_LoadVideo not found in NODE_CLASS_MAPPINGS!")
        print("Available nodes:", list(NODE_CLASS_MAPPINGS.keys())[:20])
        
        # Show what custom nodes we have
        if custom_node_mappings:
            print(f"\nCustom nodes available: {list(custom_node_mappings.keys())}")
        
        return
    
    print("✓ VHS_LoadVideo found, proceeding with workflow...")
    
    with torch.inference_mode():
        vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
        vhs_loadvideo_1 = vhs_loadvideo.load_video(
            video="safu.mp4",
            force_rate=0,
            custom_width=0,
            custom_height=0,
            frame_load_cap=0,
            skip_first_frames=0,
            select_every_nth=1,
            format="Wan",
        )

        loadimage = LoadImage()
        loadimage_4 = loadimage.load_image(image="safu.jpg")

        vaeloader = VAELoader()
        vaeloader_7 = vaeloader.load_vae(vae_name="vae.safetensors")

        unetloader = UNETLoader()
        unetloader_27 = unetloader.load_unet(
            unet_name="model.safetensors", weight_dtype="default"
        )

        cliploader = CLIPLoader()
        cliploader_23 = cliploader.load_clip(
            clip_name="clip.safetensors", type="wan", device="default"
        )

        loraloader = LoraLoader()
        loraloader_24 = loraloader.load_lora(
            lora_name="lora.safetensors",
            strength_model=0.5000000000000001,
            strength_clip=1,
            model=get_value_at_index(unetloader_27, 0),
            clip=get_value_at_index(cliploader_23, 0),
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_10 = cliptextencode.encode(
            text="a cinematic video.", clip=get_value_at_index(loraloader_24, 1)
        )

        cliptextencode_11 = cliptextencode.encode(
            text="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 , extra hands, extra arms, extra legs",
            clip=get_value_at_index(loraloader_24, 1),
        )

        wanvacetovideo = NODE_CLASS_MAPPINGS["WanVaceToVideo"]()
        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        ksampler = KSampler()
        trimvideolatent = NODE_CLASS_MAPPINGS["TrimVideoLatent"]()
        vaedecode = VAEDecode()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        for q in range(10):
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

            modelsamplingsd3_15 = modelsamplingsd3.patch(
                shift=8.000000000000002, model=get_value_at_index(loraloader_24, 0)
            )

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

            trimvideolatent_16 = trimvideolatent.EXECUTE_NORMALIZED(
                trim_amount=get_value_at_index(wanvacetovideo_13, 3),
                samples=get_value_at_index(ksampler_14, 0),
            )

            vaedecode_18 = vaedecode.decode(
                samples=get_value_at_index(trimvideolatent_16, 0),
                vae=get_value_at_index(vaeloader_7, 0),
            )

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


if __name__ == "__main__":
    main()
