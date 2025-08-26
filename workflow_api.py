import os
import random
import sys
import time
import psutil
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np

# Try to import video loading libraries
try:
    import torchvision.io as tvio
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Direct imports for VHS functionality
VHS_AVAILABLE = False
VHS_LoadVideoUpload = None
VHS_LoadVideoPath = None

def attempt_vhs_import():
    """Attempt to import VHS classes directly from custom_nodes"""
    global VHS_AVAILABLE, VHS_LoadVideoUpload, VHS_LoadVideoPath
    
    print("🔍 ATTEMPTING VHS IMPORT...")
    
    # Strategy 1: Direct import from custom_nodes directory
    comfyui_path = find_path("ComfyUI")
    if comfyui_path and os.path.isdir(comfyui_path):
        print(f"   🔍 Found ComfyUI path: {comfyui_path}")
        custom_nodes_path = os.path.join(comfyui_path, 'custom_nodes')
        print(f"   🔍 Custom nodes path: {custom_nodes_path}")
        
        if os.path.exists(custom_nodes_path):
            print(f"   ✅ Custom nodes directory exists")
            # List contents of custom_nodes directory
            try:
                custom_node_contents = os.listdir(custom_nodes_path)
                print(f"   📂 Custom nodes contents: {custom_node_contents}")
            except Exception as e:
                print(f"   ⚠️  Could not list custom nodes contents: {e}")
        else:
            print(f"   ❌ Custom nodes directory does not exist")
        
        # Try to find VHS in custom_nodes
        vhs_paths = [
            os.path.join(custom_nodes_path, 'comfyui-videohelpersuite'),
            os.path.join(custom_nodes_path, 'VideoHelperSuite'),
            os.path.join(custom_nodes_path, 'videohelpersuite'),
            os.path.join(custom_nodes_path, 'ComfyUI-VideoHelperSuite'),
            os.path.join(custom_nodes_path, 'ComfyUI-VideoHelperSuite-main')
        ]
        
        for vhs_path in vhs_paths:
            if os.path.exists(vhs_path):
                print(f"   🔍 Found VHS directory: {vhs_path}")
                try:
                    # Add to Python path
                    if vhs_path not in sys.path:
                        sys.path.insert(0, vhs_path)
                        print(f"      ✅ Added {vhs_path} to Python path")
                    
                    # List contents of VHS directory
                    try:
                        vhs_contents = os.listdir(vhs_path)
                        print(f"      📂 VHS directory contents: {vhs_contents}")
                    except Exception as e:
                        print(f"      ⚠️  Could not list VHS contents: {e}")
                    
                    # Try multiple import strategies for this path
                    import_strategies = [
                        # Strategy 1: From videohelpersuite submodule (correct path based on user's directory structure)
                        ("from videohelpersuite.load_video_nodes import LoadVideoUpload, LoadVideoPath", "videohelpersuite"),
                        # Strategy 2: From videohelpersuite.nodes (alternative path)
                        ("from videohelpersuite.nodes import LoadVideoUpload, LoadVideoPath", "videohelpersuite.nodes"),
                        # Strategy 3: Direct import from videohelpersuite directory
                        ("from load_video_nodes import LoadVideoUpload, LoadVideoPath", "direct"),
                        # Strategy 4: From nodes submodule
                        ("from nodes.load_video_nodes import LoadVideoUpload, LoadVideoPath", "nodes"),
                        # Strategy 5: From comfyui_videohelpersuite
                        ("from comfyui_videohelpersuite.videohelpersuite.nodes import LoadVideoUpload, LoadVideoPath", "comfyui_videohelpersuite"),
                        # Strategy 6: From comfyui_videohelpersuite.nodes
                        ("from comfyui_videohelpersuite.nodes import LoadVideoUpload, LoadVideoPath", "comfyui_videohelpersuite.nodes")
                    ]
                    
                    # Also try to add the videohelpersuite subdirectory to the path for direct import
                    videohelpersuite_subdir = os.path.join(vhs_path, 'videohelpersuite')
                    if os.path.exists(videohelpersuite_subdir):
                        if videohelpersuite_subdir not in sys.path:
                            sys.path.insert(0, videohelpersuite_subdir)
                            print(f"      ✅ Added videohelpersuite subdirectory to Python path: {videohelpersuite_subdir}")
                        
                        # Try direct import from the subdirectory
                        try:
                            print(f"      🔍 Trying direct import from videohelpersuite subdirectory...")
                            from load_video_nodes import LoadVideoUpload, LoadVideoPath
                            VHS_LoadVideoUpload = LoadVideoUpload
                            VHS_LoadVideoPath = LoadVideoPath
                            VHS_AVAILABLE = True
                            print(f"✅ Successfully imported VHS classes directly from videohelpersuite subdirectory: {videohelpersuite_subdir}")
                            print(f"   LoadVideoUpload: {VHS_LoadVideoUpload}")
                            print(f"   LoadVideoPath: {VHS_LoadVideoPath}")
                            return True
                        except ImportError as e:
                            print(f"         ⚠️  Direct import from videohelpersuite subdirectory failed: {e}")
                        except Exception as e:
                            print(f"         ⚠️  Direct import from videohelpersuite subdirectory error: {e}")
                    
                    # Try importing from the main package level (this should work with relative imports)
                    try:
                        print(f"      🔍 Trying package-level import from {vhs_path}...")
                        # Add the main VHS directory to the path
                        if vhs_path not in sys.path:
                            sys.path.insert(0, vhs_path)
                        
                        # Import from the main package level
                        from comfyui_videohelpersuite.videohelpersuite.load_video_nodes import LoadVideoUpload, LoadVideoPath
                        VHS_LoadVideoUpload = LoadVideoUpload
                        VHS_LoadVideoPath = LoadVideoPath
                        VHS_AVAILABLE = True
                        print(f"✅ Successfully imported VHS classes using package-level import from: {vhs_path}")
                        print(f"   LoadVideoUpload: {VHS_LoadVideoUpload}")
                        print(f"   LoadVideoPath: {VHS_LoadVideoPath}")
                        return True
                    except ImportError as e:
                        print(f"         ⚠️  Package-level import failed: {e}")
                    except Exception as e:
                        print(f"         ⚠️  Package-level import error: {e}")
                    
                    # Try importing as a module from the current working directory
                    try:
                        print(f"      🔍 Trying module import from current working directory...")
                        # Change to the VHS directory temporarily
                        original_cwd = os.getcwd()
                        os.chdir(vhs_path)
                        
                        # Try to import the module
                        # Note: sys is already imported at module level, no need to import again
                        sys.path.insert(0, vhs_path)
                        
                        # Import using the module name
                        import videohelpersuite.load_video_nodes
                        LoadVideoUpload = videohelpersuite.load_video_nodes.LoadVideoUpload
                        LoadVideoPath = videohelpersuite.load_video_nodes.LoadVideoPath
                        
                        VHS_LoadVideoUpload = LoadVideoUpload
                        VHS_LoadVideoPath = LoadVideoPath
                        VHS_AVAILABLE = True
                        
                        # Change back to original directory
                        os.chdir(original_cwd)
                        
                        print(f"✅ Successfully imported VHS classes using module import from: {vhs_path}")
                        print(f"   LoadVideoUpload: {VHS_LoadVideoUpload}")
                        print(f"   LoadVideoPath: {VHS_LoadVideoPath}")
                        return True
                    except ImportError as e:
                        print(f"         ⚠️  Module import failed: {e}")
                        # Change back to original directory if there was an error
                        if 'original_cwd' in locals():
                            os.chdir(original_cwd)
                    except Exception as e:
                        print(f"         ⚠️  Module import error: {e}")
                        # Change back to original directory if there was an error
                        if 'original_cwd' in locals():
                            os.chdir(original_cwd)
                    
                    # Try importing as a ComfyUI custom node package (this should handle relative imports correctly)
                    try:
                        print(f"      🔍 Trying ComfyUI custom node package import...")
                        # Add the custom_nodes directory to the path
                        custom_nodes_dir = os.path.dirname(vhs_path)
                        if custom_nodes_dir not in sys.path:
                            sys.path.insert(0, custom_nodes_dir)
                            print(f"         ✅ Added custom_nodes directory to Python path: {custom_nodes_dir}")
                        
                        # Try to import the package as a ComfyUI custom node
                        import comfyui_videohelpersuite
                        print(f"         ✅ Successfully imported comfyui_videohelpersuite package")
                        
                        # Now try to access the classes through the package
                        if hasattr(comfyui_videohelpersuite, 'videohelpersuite'):
                            videohelpersuite_module = comfyui_videohelpersuite.videohelpersuite
                            if hasattr(videohelpersuite_module, 'load_video_nodes'):
                                load_video_nodes_module = videohelpersuite_module.load_video_nodes
                                if hasattr(load_video_nodes_module, 'LoadVideoUpload') and hasattr(load_video_nodes_module, 'LoadVideoPath'):
                                    LoadVideoUpload = load_video_nodes_module.LoadVideoUpload
                                    LoadVideoPath = load_video_nodes_module.LoadVideoPath
                                    
                                    VHS_LoadVideoUpload = LoadVideoUpload
                                    VHS_LoadVideoPath = LoadVideoPath
                                    VHS_AVAILABLE = True
                                    
                                    print(f"✅ Successfully imported VHS classes using ComfyUI custom node package import from: {vhs_path}")
                                    print(f"   LoadVideoUpload: {VHS_LoadVideoUpload}")
                                    print(f"   LoadVideoPath: {VHS_LoadVideoPath}")
                                    return True
                                else:
                                    print(f"         ⚠️  LoadVideoUpload or LoadVideoPath not found in load_video_nodes module")
                            else:
                                print(f"         ⚠️  load_video_nodes module not found in videohelpersuite")
                        else:
                            print(f"         ⚠️  videohelpersuite module not found in comfyui_videohelpersuite package")
                            
                    except ImportError as e:
                        print(f"         ⚠️  ComfyUI custom node package import failed: {e}")
                    except Exception as e:
                        print(f"         ⚠️  ComfyUI custom node package import error: {e}")
                    
                    for import_statement, strategy_name in import_strategies:
                        try:
                            print(f"      🔍 Trying strategy: {strategy_name}")
                            # Execute the import statement
                            exec(import_statement)
                            VHS_LoadVideoUpload = LoadVideoUpload
                            VHS_LoadVideoPath = LoadVideoPath
                            VHS_AVAILABLE = True
                            print(f"✅ Successfully imported VHS classes using strategy '{strategy_name}' from: {vhs_path}")
                            print(f"   LoadVideoUpload: {VHS_LoadVideoUpload}")
                            print(f"   LoadVideoPath: {VHS_LoadVideoPath}")
                            return True
                        except ImportError as import_error:
                            print(f"         ⚠️  Strategy '{strategy_name}' failed: {import_error}")
                            continue
                        except Exception as e:
                            print(f"         ⚠️  Strategy '{strategy_name}' error: {e}")
                            continue
                        
                except Exception as e:
                    print(f"   ⚠️  Error processing {vhs_path}: {e}")
                    continue
            else:
                print(f"   ❌ VHS directory not found: {vhs_path}")
    else:
        print(f"   ❌ ComfyUI path not found")
    
    # Strategy 2: Try pip-installed package
    print("   🔍 Trying pip-installed package...")
    try:
        from videohelpersuite.load_video_nodes import LoadVideoUpload, LoadVideoPath
        VHS_LoadVideoUpload = LoadVideoUpload
        VHS_LoadVideoPath = LoadVideoPath
        VHS_AVAILABLE = True
        print("✅ Successfully imported VHS from pip-installed package")
        return True
    except ImportError as e:
        print(f"      ❌ Pip package import failed: {e}")
    
    # Strategy 3: Try to find VHS in current working directory or subdirectories
    print("   🔍 Searching current directory for VHS...")
    try:
        current_dir = os.getcwd()
        print(f"      🔍 Current directory: {current_dir}")
        for root, dirs, files in os.walk(current_dir):
            if 'load_video_nodes.py' in files or 'videohelpersuite' in dirs:
                try:
                    vhs_root = root
                    print(f"      🔍 Found potential VHS location: {vhs_root}")
                    if vhs_root not in sys.path:
                        sys.path.insert(0, vhs_root)
                        print(f"         ✅ Added {vhs_root} to Python path")
                    
                    # Try to import from this location
                    try:
                        from load_video_nodes import LoadVideoUpload, LoadVideoPath
                        VHS_LoadVideoUpload = LoadVideoUpload
                        VHS_LoadVideoPath = LoadVideoPath
                        VHS_AVAILABLE = True
                        print(f"✅ Successfully imported VHS from discovered location: {vhs_root}")
                        return True
                    except ImportError as e:
                        print(f"         ❌ Import failed from {vhs_root}: {e}")
                        continue
                        
                except Exception as e:
                    print(f"         ⚠️  Error processing {vhs_root}: {e}")
                    continue
    except Exception as e:
        print(f"      ⚠️  Directory search failed: {e}")
    
    print("⚠️  Warning: Could not import VHS classes directly")
    print("   Will use fallback video loading approach")
    VHS_AVAILABLE = False
    return False

# VHS import will be attempted after find_path function is defined

# Add monitoring and debugging utilities
class ModelLoadingMonitor:
    """Comprehensive monitoring for model loading steps"""
    
    def __init__(self):
        self.baseline_ram = None
        self.baseline_gpu = None
        self.monitoring_data = {}
        self.step_start_time = None
        
        # Create output directories for tensor dumps
        self._create_output_directories()
        
        # Test tensor creation to verify directory works
        self._test_tensor_creation()
    
    def _create_output_directories(self):
        """Create output directories for tensor dumps and analysis"""
        output_dirs = [
            "./W_out",
            "./W_out/step3"
        ]
        
        for output_dir in output_dirs:
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"📁 Created output directory: {output_dir}")
            except Exception as e:
                print(f"⚠️  Warning: Could not create directory {output_dir}: {e}")
    
    def _test_tensor_creation(self):
        """Test tensor creation to verify directory and file creation works"""
        try:
            output_dir = "./W_out/step3"
            test_filename = "test_tensor.npy"
            test_filepath = os.path.join(output_dir, test_filename)
            
            # Create a simple test tensor
            test_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            np.save(test_filepath, test_tensor)
            
            if os.path.exists(test_filepath):
                size_mb = os.path.getsize(test_filepath) / (1024**2)
                print(f"📁 Test tensor created successfully: {test_filepath} ({size_mb:.4f} MB)")
                # Clean up test file
                os.remove(test_filepath)
                print(f"🧹 Test tensor cleaned up")
            else:
                print(f"⚠️  Warning: Test tensor creation failed")
        except Exception as e:
            print(f"⚠️  Warning: Test tensor creation failed: {e}")
    
    def _debug_conditioning_structure(self, conditioning, tensor_type):
        """Debug helper to understand conditioning structure"""
        print(f"\n🔍 DEBUGGING {tensor_type.upper()} CONDITIONING STRUCTURE:")
        print(f"   Type: {type(conditioning).__name__}")
        print(f"   Repr: {repr(conditioning)[:200]}...")
        
        if hasattr(conditioning, '__len__'):
            print(f"   Length: {len(conditioning)}")
            if len(conditioning) > 0:
                print(f"   First item type: {type(conditioning[0]).__name__}")
                if hasattr(conditioning[0], 'shape'):
                    print(f"   First item shape: {conditioning[0].shape}")
                else:
                    print(f"   First item attributes: {[attr for attr in dir(conditioning[0]) if not attr.startswith('_')][:10]}")
        
        if hasattr(conditioning, '__dir__'):
            print(f"   Available attributes: {[attr for attr in dir(conditioning) if not attr.startswith('_')][:10]}")
        
        print("=" * 60)
    
    def _extract_tensor_recursively(self, obj, max_depth=5, current_depth=0):
        """Recursively search for tensor objects in nested structures"""
        if current_depth >= max_depth:
            return None, f"Max depth {max_depth} reached"
        
        # Direct tensor check
        if hasattr(obj, 'shape'):
            return obj, f"Found tensor at depth {current_depth}"
        
        # List/tuple check
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            for i, item in enumerate(obj):
                result, reason = self._extract_tensor_recursively(item, max_depth, current_depth + 1)
                if result is not None:
                    return result, f"Found in {type(obj).__name__}[{i}] at depth {current_depth}: {reason}"
        
        # Dictionary check
        elif isinstance(obj, dict):
            for key, value in obj.items():
                result, reason = self._extract_tensor_recursively(value, max_depth, current_depth + 1)
                if result is not None:
                    return result, f"Found in dict['{key}'] at depth {current_depth}: {reason}"
        
        return None, f"No tensor found at depth {current_depth}"
    
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
        
        # Initialize peak memory tracking
        self.peak_memory = {
            'ram_peak_mb': self.baseline_ram.used / (1024**2),
            'gpu_allocated_peak_mb': self.baseline_gpu['allocated'] if self.baseline_gpu else 0,
            'gpu_reserved_peak_mb': self.baseline_gpu['reserved'] if self.baseline_gpu else 0,
            'peak_timestamps': []
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
        
        # Enhanced model information extraction
        print(f"🔧 ENHANCED MODEL INFORMATION:")
        self._extract_enhanced_model_info(loader_result, model_type)
        
        print("=" * 60)
        
        return elapsed_time
    
    def update_peak_memory(self):
        """Update peak memory values - call this during monitoring to track peaks"""
        try:
            # Update RAM peak
            current_ram = psutil.virtual_memory()
            current_ram_mb = current_ram.used / (1024**2)
            if current_ram_mb > self.peak_memory['ram_peak_mb']:
                self.peak_memory['ram_peak_mb'] = current_ram_mb
                self.peak_memory['peak_timestamps'].append({
                    'type': 'ram',
                    'value_mb': current_ram_mb,
                    'timestamp': time.time() - self.step_start_time
                })
            
            # Update GPU peak
            if torch.cuda.is_available():
                current_gpu_allocated = torch.cuda.memory_allocated() / (1024**2)
                current_gpu_reserved = torch.cuda.memory_reserved() / (1024**2)
                
                if current_gpu_allocated > self.peak_memory['gpu_allocated_peak_mb']:
                    self.peak_memory['gpu_allocated_peak_mb'] = current_gpu_allocated
                    self.peak_memory['peak_timestamps'].append({
                        'type': 'gpu_allocated',
                        'value_mb': current_gpu_allocated,
                        'timestamp': time.time() - self.step_start_time
                    })
                
                if current_gpu_reserved > self.peak_memory['gpu_reserved_peak_mb']:
                    self.peak_memory['gpu_reserved_peak_mb'] = current_gpu_reserved
                    self.peak_memory['peak_timestamps'].append({
                        'type': 'gpu_reserved',
                        'value_mb': current_gpu_reserved,
                        'timestamp': time.time() - self.step_start_time
                    })
        except Exception as e:
            print(f"⚠️  Warning: Peak memory update failed: {e}")
    
    def get_peak_memory_summary(self):
        """Get summary of peak memory usage during monitoring"""
        return {
            'ram_peak_mb': self.peak_memory['ram_peak_mb'],
            'gpu_allocated_peak_mb': self.peak_memory['gpu_allocated_peak_mb'],
            'gpu_reserved_peak_mb': self.peak_memory['gpu_reserved_peak_mb'],
            'peak_timestamps': self.peak_memory['peak_timestamps']
        }
    
    def _extract_enhanced_model_info(self, loader_result, model_type):
        """Extract comprehensive model information including architecture, dimensions, and memory analysis"""
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
            device_info = self._get_device_info(model)
            print(f"   Device: {device_info}")
            
            # Extract model size information
            param_info = self._get_parameter_info(model)
            print(f"   Parameters: {param_info['count']:,}")
            print(f"   Model Size: {param_info['size_mb']:.1f} MB")
            
            # Extract state dict information
            state_dict_info = self._get_state_dict_info(model)
            print(f"   State Dict Keys: {state_dict_info['key_count']}")
            
            # Extract model-specific detailed information
            if model_type == "VAE":
                self._extract_vae_details(model)
            elif model_type == "UNET":
                self._extract_unet_details(model)
            elif model_type == "CLIP":
                self._extract_clip_details(model)
            
            # Memory efficiency analysis
            self._analyze_memory_efficiency(model, model_type, param_info)
            
        except Exception as e:
            print(f"   ⚠️  Error extracting enhanced model info: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_device_info(self, model):
        """Get comprehensive device information"""
        try:
            if hasattr(model, 'device'):
                return str(model.device)
            elif hasattr(model, 'model') and hasattr(model.model, 'device'):
                return str(model.model.device)
            elif hasattr(model, 'patcher') and hasattr(model.patcher, 'model'):
                if hasattr(model.patcher.model, 'device'):
                    return str(model.patcher.model.device)
            return "unknown"
        except:
            return "unknown"
    
    def _get_parameter_info(self, model):
        """Get detailed parameter information"""
        try:
            param_count = 0
            if hasattr(model, 'parameters'):
                param_count = sum(p.numel() for p in model.parameters())
            elif hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                param_count = sum(p.numel() for p in model.model.parameters())
            elif hasattr(model, 'patcher') and hasattr(model.patcher, 'model'):
                if hasattr(model.patcher.model, 'parameters'):
                    param_count = sum(p.numel() for p in model.patcher.model.parameters())
            
            # Estimate model size (assuming float32)
            size_mb = (param_count * 4) / (1024 * 1024)
            return {'count': param_count, 'size_mb': size_mb}
        except:
            return {'count': 0, 'size_mb': 0.0}
    
    def _get_state_dict_info(self, model):
        """Get state dictionary information"""
        try:
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
                return {'key_count': len(state_dict), 'keys': list(state_dict.keys())}
            elif hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
                state_dict = model.model.state_dict()
                return {'key_count': len(state_dict), 'keys': list(state_dict.keys())}
            return {'key_count': 0, 'keys': []}
        except:
            return {'key_count': 0, 'keys': []}
    
    def _extract_vae_details(self, model):
        """Extract detailed VAE information"""
        print(f"   🎨 VAE SPECIFIC DETAILS:")
        
        # Latent channels
        if hasattr(model, 'latent_channels'):
            print(f"     Latent Channels: {model.latent_channels}")
        
        # Downscale/upscale ratios
        if hasattr(model, 'downscale_ratio'):
            print(f"     Downscale Ratio: {model.downscale_ratio}")
        if hasattr(model, 'upscale_ratio'):
            print(f"     Upscale Ratio: {model.upscale_ratio}")
        
        # First stage model
        if hasattr(model, 'first_stage_model'):
            print(f"     Has First Stage Model: ✅")
            first_stage = model.first_stage_model
            if hasattr(first_stage, 'latent_channels'):
                print(f"     First Stage Latent Channels: {first_stage.latent_channels}")
        
        # Scale factor
        if hasattr(model, 'scale_factor'):
            print(f"     Scale Factor: {model.scale_factor}")
        
        # Model type detection
        self._detect_vae_type(model)
    
    def _extract_unet_details(self, model):
        """Extract detailed UNET information"""
        print(f"   🧠 UNET SPECIFIC DETAILS:")
        
        # Get the actual UNET model
        unet_model = model
        if hasattr(model, 'model'):
            unet_model = model.model
        
        # Basic architecture info
        if hasattr(unet_model, 'in_channels'):
            print(f"     Input Channels: {unet_model.in_channels}")
        if hasattr(unet_model, 'out_channels'):
            print(f"     Output Channels: {unet_model.out_channels}")
        if hasattr(unet_model, 'model_channels'):
            print(f"     Model Channels: {unet_model.model_channels}")
        
        # Channel multipliers
        if hasattr(unet_model, 'channel_mult'):
            print(f"     Channel Multipliers: {unet_model.channel_mult}")
        
        # Residual blocks
        if hasattr(unet_model, 'num_res_blocks'):
            print(f"     Residual Blocks: {unet_model.num_res_blocks}")
        
        # Attention info
        if hasattr(unet_model, 'num_heads'):
            print(f"     Attention Heads: {unet_model.num_heads}")
        if hasattr(unet_model, 'transformer_depth'):
            print(f"     Transformer Depth: {unet_model.transformer_depth}")
        
        # Context dimension
        if hasattr(unet_model, 'context_dim'):
            print(f"     Context Dimension: {unet_model.context_dim}")
        
        # Model type detection
        self._detect_unet_type(unet_model)
    
    def _extract_clip_details(self, model):
        """Extract detailed CLIP information"""
        print(f"   📝 CLIP SPECIFIC DETAILS:")
        
        # ModelPatcher info
        if hasattr(model, 'patcher'):
            print(f"     Has ModelPatcher: ✅")
            patcher_model = model.patcher.model
            
            # Text model configuration
            if hasattr(patcher_model, 'config'):
                config = patcher_model.config
                if hasattr(config, 'hidden_size'):
                    print(f"     Hidden Size: {config.hidden_size}")
                if hasattr(config, 'num_hidden_layers'):
                    print(f"     Num Layers: {config.num_hidden_layers}")
                if hasattr(config, 'num_attention_heads'):
                    print(f"     Attention Heads: {config.num_attention_heads}")
                if hasattr(config, 'vocab_size'):
                    print(f"     Vocab Size: {config.vocab_size}")
                if hasattr(config, 'max_position_embeddings'):
                    print(f"     Max Position Embeddings: {config.max_position_embeddings}")
                if hasattr(config, 'intermediate_size'):
                    print(f"     Intermediate Size: {config.intermediate_size}")
        
        # Max length
        if hasattr(model, 'max_length'):
            print(f"     Max Length: {model.max_length}")
        
        # Model type detection
        self._detect_clip_type(model)
    
    def _detect_vae_type(self, model):
        """Detect VAE model type based on architecture"""
        print(f"     🔍 VAE TYPE DETECTION:")
        
        # Check for SD vs SDXL vs SD3 characteristics
        if hasattr(model, 'latent_channels'):
            if model.latent_channels == 4:
                print(f"       Detected: Standard Diffusion (SD/SDXL)")
            elif model.latent_channels == 16:
                print(f"       Detected: Stable Cascade (Stage C)")
            else:
                print(f"       Detected: Custom VAE ({model.latent_channels} channels)")
        
        # Check for video capabilities
        if hasattr(model, 'first_stage_model'):
            first_stage = model.first_stage_model
            if hasattr(first_stage, 'video_kernel_size'):
                print(f"       Capabilities: Video Support ✅")
            else:
                print(f"       Capabilities: Image Only")
    
    def _detect_unet_type(self, model):
        """Detect UNET model type based on architecture"""
        print(f"     🔍 UNET TYPE DETECTION:")
        
        # Check for SD vs SDXL vs SD3 vs WAN
        if hasattr(model, 'context_dim'):
            if model.context_dim == 768:
                print(f"       Detected: SD 1.5")
            elif model.context_dim == 1024:
                print(f"       Detected: SD 2.1")
            elif model.context_dim == 1280:
                print(f"       Detected: SDXL")
            elif model.context_dim == 2048:
                print(f"       Detected: SD 3 / WAN")
            else:
                print(f"       Detected: Custom UNET (context_dim: {model.context_dim})")
        
        # Check for temporal capabilities
        if hasattr(model, 'use_temporal_resblocks'):
            if model.use_temporal_resblocks:
                print(f"       Capabilities: Video/Temporal Support ✅")
            else:
                print(f"       Capabilities: Image Only")
    
    def _detect_clip_type(self, model):
        """Detect CLIP model type based on architecture"""
        print(f"     🔍 CLIP TYPE DETECTION:")
        
        # Check for different CLIP variants
        if hasattr(model, 'patcher'):
            patcher_model = model.patcher.model
            if hasattr(patcher_model, 'config'):
                config = patcher_model.config
                if hasattr(config, 'hidden_size'):
                    if config.hidden_size == 768:
                        print(f"       Detected: SD 1.5 CLIP")
                    elif config.hidden_size == 1024:
                        print(f"       Detected: SD 2.1 CLIP")
                    elif config.hidden_size == 1280:
                        print(f"       Detected: SDXL CLIP")
                    elif config.hidden_size == 2048:
                        print(f"       Detected: SD 3 / WAN T5")
                    else:
                        print(f"       Detected: Custom CLIP (hidden_size: {config.hidden_size})")
    
    def _analyze_memory_efficiency(self, model, model_type, param_info):
        """Analyze memory efficiency and provide recommendations"""
        print(f"   💡 MEMORY EFFICIENCY ANALYSIS:")
        
        # Parameter efficiency
        if param_info['count'] > 0:
            if param_info['count'] > 1000000000:  # 1B+ parameters
                print(f"     Model Size: Large ({param_info['size_mb']:.1f} MB)")
                print(f"     Recommendation: Consider GPU offloading for memory efficiency")
            elif param_info['count'] > 500000000:  # 500M+ parameters
                print(f"     Model Size: Medium ({param_info['size_mb']:.1f} MB)")
                print(f"     Recommendation: Monitor memory usage during operations")
            else:
                print(f"     Model Size: Small ({param_info['size_mb']:.1f} MB)")
                print(f"     Recommendation: Should fit comfortably in memory")
        
        # Device placement efficiency
        device_info = self._get_device_info(model)
        if device_info == "cpu":
            print(f"     Device Placement: CPU (memory efficient, slower inference)")
        elif device_info == "cuda:0":
            print(f"     Device Placement: GPU (faster inference, higher memory usage)")
        else:
            print(f"     Device Placement: {device_info}")
    
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
    
    def print_comprehensive_summary(self, lora_analysis=None, text_encoding_analysis=None, model_sampling_analysis=None, latent_generation_analysis=None):
        """Print comprehensive summary including model loading, LoRA application, text encoding, model sampling, and latent generation"""
        print(f"\n📊 COMPREHENSIVE WORKFLOW MONITORING SUMMARY")
        print("=" * 80)
        
        # Step 1: Model Loading Summary
        print(f"🔍 STEP 1: MODEL LOADING")
        print("   ✅ Models loaded successfully (monitoring disabled)")
        
        # Step 2: LoRA Application Summary
        print(f"\n🔍 STEP 2: LORA APPLICATION")
        print("   ✅ LoRA applied successfully (monitoring disabled)")
        
        # Step 3: Text Encoding Summary
        print(f"\n🔍 STEP 3: TEXT ENCODING")
        print("   ✅ Text encoding completed (monitoring disabled)")
        
        # Step 4: Model Sampling Summary
        print(f"\n🔍 STEP 4: MODEL SAMPLING")
        print("   ✅ Model sampling completed (monitoring disabled)")
        
        # Step 5: Input Preparation Summary
        if latent_generation_analysis:
            print(f"\n🔍 STEP 5: INPUT PREPARATION FOR WANVACETOVIDEO NODE")
            self.print_latent_generation_analysis_summary(latent_generation_analysis)
        else:
            print(f"\n🔍 STEP 5: INPUT PREPARATION FOR WANVACETOVIDEO NODE")
            print("   ❌ No analysis available")
        
        print("=" * 80)
    
    # === LORA APPLICATION MONITORING METHODS ===
    
    def capture_lora_baseline(self, unet_model, clip_model):
        """Capture baseline state before LoRA application"""
        # Capture RAM baseline
        ram_baseline = psutil.virtual_memory()
        
        # Capture GPU baseline
        gpu_baseline = None
        if torch.cuda.is_available():
            gpu_baseline = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'total': torch.cuda.get_device_properties(0).total_memory,
                'device_name': torch.cuda.get_device_name(0)
            }
        
        baseline = {
            'timestamp': time.time(),
            'ram': {
                'used_mb': ram_baseline.used / (1024**2),
                'available_mb': ram_baseline.available / (1024**2),
                'total_mb': ram_baseline.total / (1024**2),
                'percent_used': ram_baseline.percent
            },
            'gpu': gpu_baseline,
            'unet': {
                'model_id': id(unet_model),
                'class': type(unet_model).__name__,
                'device': getattr(unet_model, 'device', None),
                'patches_count': len(getattr(unet_model, 'patches', {})),
                'patches_uuid': getattr(unet_model, 'patches_uuid', None)
            },
            'clip': {
                'model_id': id(clip_model),
                'class': type(clip_model).__name__,
                'device': getattr(clip_model, 'device', None),
                'patcher_patches_count': len(getattr(clip_model.patcher, 'patches', {})) if hasattr(clip_model, 'patcher') else 0,
                'patcher_patches_uuid': getattr(clip_model.patcher, 'patches_uuid', None) if hasattr(clip_model, 'patcher') else None
            }
        }
        return baseline
    
    def track_model_identity_changes(self, original_model, modified_model, model_type):
        """Track changes in model identity and structure"""
        
        # 1. Model Instance Changes
        model_cloned = original_model is not modified_model
        model_class_changed = type(original_model) != type(modified_model)
        
        # 2. ModelPatcher Changes - handle different model structures
        original_patch_count = 0
        modified_patch_count = 0
        patches_added = 0
        original_uuid = None
        modified_uuid = None
        uuid_changed = False
        
        # Get original patch count
        if hasattr(original_model, 'patches'):
            original_patch_count = len(original_model.patches)
        elif hasattr(original_model, 'patcher') and hasattr(original_model.patcher, 'patches'):
            original_patch_count = len(original_model.patcher.patches)
        
        # Get modified patch count
        if hasattr(modified_model, 'patches'):
            modified_patch_count = len(modified_model.patches)
        elif hasattr(modified_model, 'patcher') and hasattr(modified_model.patcher, 'patches'):
            modified_patch_count = len(modified_model.patcher.patches)
        
        patches_added = modified_patch_count - original_patch_count
        
        # 3. Patch UUID Changes
        if hasattr(original_model, 'patches_uuid'):
            original_uuid = original_model.patches_uuid
        elif hasattr(original_model, 'patcher') and hasattr(original_model.patcher, 'patches_uuid'):
            original_uuid = original_model.patcher.patches_uuid
            
        if hasattr(modified_model, 'patches_uuid'):
            modified_uuid = modified_model.patches_uuid
        elif hasattr(modified_model, 'patcher') and hasattr(modified_model.patcher, 'patches_uuid'):
            modified_uuid = modified_model.patcher.patches_uuid
        
        uuid_changed = original_uuid != modified_uuid
        
        return {
            'model_cloned': model_cloned,
            'class_changed': model_class_changed,
            'patches_added': patches_added,
            'uuid_changed': uuid_changed,
            'original_patch_count': original_patch_count,
            'modified_patch_count': modified_patch_count,
            'original_model_type': type(original_model).__name__,
            'modified_model_type': type(modified_model).__name__
        }
    
    def track_weight_modifications(self, original_model, modified_model, model_type):
        """Track how LoRA modifies model weights"""
        
        # 1. State Dict Changes
        original_state = {}
        modified_state = {}
        
        try:
            if hasattr(original_model, 'state_dict'):
                original_state = original_model.state_dict()
            if hasattr(modified_model, 'state_dict'):
                modified_state = modified_model.state_dict()
        except Exception as e:
            print(f"⚠️  Warning: Could not access state_dict for {model_type}: {e}")
        
        # 2. Key Differences
        original_keys = set(original_state.keys())
        modified_keys = set(modified_state.keys())
        keys_added = modified_keys - original_keys
        keys_removed = original_keys - modified_keys
        keys_modified = original_keys & modified_keys
        
        # 3. Weight Value Changes (for accessible weights)
        weight_changes = {}
        for key in keys_modified:
            if key in original_state and key in modified_state:
                orig_weight = original_state[key]
                mod_weight = modified_state[key]
                
                if hasattr(orig_weight, 'shape') and hasattr(mod_weight, 'shape'):
                    shape_changed = orig_weight.shape != mod_weight.shape
                    dtype_changed = orig_weight.dtype != mod_weight.dtype
                    device_changed = orig_weight.device != mod_weight.device
                    
                    weight_changes[key] = {
                        'shape_changed': shape_changed,
                        'dtype_changed': dtype_changed,
                        'device_changed': device_changed,
                        'original_shape': str(orig_weight.shape),
                        'modified_shape': str(mod_weight.shape)
                    }
        
        return {
            'keys_added': list(keys_added),
            'keys_removed': list(keys_removed),
            'keys_modified': list(keys_modified),
            'weight_changes': weight_changes,
            'total_keys_original': len(original_keys),
            'total_keys_modified': len(modified_keys)
        }
    
    def analyze_lora_patches(self, modified_model, model_type):
        """Analyze the specific LoRA patches applied to the model"""
        
        # Handle different model structures
        patches = {}
        model_structure = "unknown"
        
        if hasattr(modified_model, 'patches'):
            patches = modified_model.patches
            model_structure = "direct_patches"
        elif hasattr(modified_model, 'patcher') and hasattr(modified_model.patcher, 'patches'):
            patches = modified_model.patcher.patches
            model_structure = "patcher_patches"
        else:
            return {
                'error': f'Model has no accessible patches attribute. Model type: {type(modified_model).__name__}',
                'model_structure': model_structure,
                'available_attrs': [attr for attr in dir(modified_model) if not attr.startswith('_')][:10]
            }
        
        lora_patches = {}
        total_patches = 0
        
        for key, patch_list in patches.items():
            if patch_list:  # If patches exist for this key
                total_patches += len(patch_list)
                # Each patch is a tuple: (strength_patch, patch_data, strength_model, offset, function)
                for patch in patch_list:
                    if len(patch) >= 2:
                        strength_patch, patch_data = patch[0], patch[1]
                        
                        # Determine patch type
                        if isinstance(patch_data, dict) and 'lora_up.weight' in str(patch_data):
                            patch_type = 'lora_up'
                        elif isinstance(patch_data, dict) and 'lora_down.weight' in str(patch_data):
                            patch_type = 'lora_down'
                        elif isinstance(patch_data, dict) and 'diff' in str(patch_data):
                            patch_type = 'diff'
                        else:
                            patch_type = 'unknown'
                        
                        lora_patches[key] = {
                            'strength_patch': strength_patch,
                            'patch_type': patch_type,
                            'patch_data_shape': str(type(patch_data)),
                            'patch_count': len(patch_list)
                        }
        
        return {
            'total_patched_keys': len(lora_patches),
            'total_patches': total_patches,
            'patch_details': lora_patches,
            'model_type': model_type,
            'model_structure': model_structure,
            'raw_patches_count': len(patches)
        }
    
    def track_model_placement_changes(self, original_model, modified_model, model_type):
        """Track changes in model device placement"""
        
        # 1. Device Changes
        original_device = getattr(original_model, 'device', None)
        modified_device = getattr(modified_model, 'device', None)
        
        # 2. ModelPatcher Device Changes
        original_load_device = getattr(original_model, 'load_device', None)
        modified_load_device = getattr(modified_model, 'load_device', None)
        
        original_offload_device = getattr(original_model, 'offload_device', None)
        modified_offload_device = getattr(modified_model, 'offload_device', None)
        
        # 3. CLIP-specific device tracking
        clip_model_device = None
        clip_patcher_load_device = None
        clip_patcher_offload_device = None
        
        if model_type == 'CLIP' and hasattr(modified_model, 'patcher'):
            try:
                clip_model_device = getattr(modified_model.patcher.model, 'device', None)
                clip_patcher_load_device = getattr(modified_model.patcher, 'load_device', None)
                clip_patcher_offload_device = getattr(modified_model.patcher, 'offload_device', None)
            except Exception as e:
                print(f"⚠️  Warning: Could not access CLIP patcher device info: {e}")
        
        return {
            'model_device_changed': original_device != modified_device,
            'load_device_changed': original_load_device != modified_load_device,
            'offload_device_changed': original_offload_device != modified_offload_device,
            'original_device': str(original_device),
            'modified_device': str(modified_device),
            'clip_model_device': str(clip_model_device) if clip_model_device else None,
            'clip_patcher_devices': {
                'load': str(clip_patcher_load_device),
                'offload': str(clip_patcher_offload_device)
            } if clip_patcher_load_device else None
        }
    
    def calculate_memory_change(self, baseline_info, current_model):
        """Calculate memory usage changes for a model"""
        try:
            # Get current RAM state
            current_ram = psutil.virtual_memory()
            
            # Get current GPU state
            current_gpu = None
            if torch.cuda.is_available():
                current_gpu = {
                    'allocated': torch.cuda.memory_allocated(),
                    'reserved': torch.cuda.memory_reserved(),
                    'total': torch.cuda.get_device_properties(0).total_memory
                }
            
            # Calculate RAM changes
            ram_changes = {
                'used_change_mb': (current_ram.used - baseline_info['ram']['used_mb'] * (1024**2)) / (1024**2),
                'available_change_mb': (current_ram.available - baseline_info['ram']['available_mb'] * (1024**2)) / (1024**2),
                'current_used_mb': current_ram.used / (1024**2),
                'current_available_mb': current_ram.available / (1024**2),
                'current_total_mb': current_ram.total / (1024**2),
                'current_percent_used': current_ram.percent,
                'baseline_used_mb': baseline_info['ram']['used_mb'],
                'baseline_available_mb': baseline_info['ram']['available_mb'],
                'baseline_percent_used': baseline_info['ram']['percent_used']
            }
            
            # Calculate GPU changes
            gpu_changes = None
            if current_gpu and baseline_info['gpu']:
                allocated_change = current_gpu['allocated'] - baseline_info['gpu']['allocated']
                reserved_change = current_gpu['reserved'] - baseline_info['gpu']['reserved']
                
                gpu_changes = {
                    'allocated_change_mb': allocated_change / (1024**2),
                    'reserved_change_mb': reserved_change / (1024**2),
                    'current_allocated_mb': current_gpu['allocated'] / (1024**2),
                    'current_reserved_mb': current_gpu['reserved'] / (1024**2),
                    'current_total_mb': current_gpu['total'] / (1024**2),
                    'baseline_allocated_mb': baseline_info['gpu']['allocated'] / (1024**2),
                    'baseline_reserved_mb': baseline_info['gpu']['reserved'] / (1024**2),
                    'baseline_total_mb': baseline_info['gpu']['total'] / (1024**2),
                    'allocated_change_pct': (allocated_change / baseline_info['gpu']['allocated'] * 100) if baseline_info['gpu']['allocated'] > 0 else 0,
                    'reserved_change_pct': (reserved_change / baseline_info['gpu']['reserved'] * 100) if baseline_info['gpu']['reserved'] > 0 else 0
                }
            
            return {
                'ram': ram_changes,
                'gpu': gpu_changes
            }
        except Exception as e:
            return {'error': f'Memory calculation failed: {e}'}
    
    def analyze_lora_application_results(self, baseline, modified_unet, modified_clip, lora_result):
        """Analyze the results of LoRA application"""
        
        analysis = {
            'lora_application_success': lora_result is not None,
            'models_returned': len(lora_result) if lora_result else 0,
            'unet_changes': self.track_model_identity_changes(
                baseline['unet'], modified_unet, 'UNET'
            ),
            'clip_changes': self.track_model_identity_changes(
                baseline['clip'], modified_clip, 'CLIP'
            ),
            'unet_weight_changes': self.track_weight_modifications(
                baseline['unet'], modified_clip, 'UNET'
            ),
            'clip_weight_changes': self.track_weight_modifications(
                baseline['clip'], modified_clip, 'CLIP'
            ),
            'unet_lora_patches': self.analyze_lora_patches(modified_unet, 'UNET'),
            'clip_lora_patches': self.analyze_lora_patches(modified_clip, 'CLIP'),
            'placement_changes': {
                'unet': self.track_model_placement_changes(
                    baseline['unet'], modified_unet, 'UNET'
                ),
                'clip': self.track_model_placement_changes(
                    baseline['clip'], modified_clip, 'CLIP'
                )
            },
            'memory_impact': self.calculate_memory_change(baseline, modified_unet),
            'peak_memory': self.get_peak_memory_summary()
        }
        
        return analysis
    
    # === TEXT ENCODING MONITORING METHODS ===
    
    def capture_text_encoding_baseline(self, unet_model, clip_model):
        """Capture baseline state before text encoding"""
        # Capture RAM baseline
        ram_baseline = psutil.virtual_memory()
        
        # Capture GPU baseline
        gpu_baseline = None
        if torch.cuda.is_available():
            gpu_baseline = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'total': torch.cuda.get_device_properties(0).total_memory,
                'device_name': torch.cuda.get_device_name(0)
            }
        
        baseline = {
            'timestamp': time.time(),
            'ram': {
                'used_mb': ram_baseline.used / (1024**2),
                'available_mb': ram_baseline.available / (1024**2),
                'total_mb': ram_baseline.total / (1024**2),
                'percent_used': ram_baseline.percent
            },
            'gpu': gpu_baseline,
            'unet': {
                'model_id': id(unet_model),
                'class': type(unet_model).__name__,
                'device': getattr(unet_model, 'device', None),
                'patches_count': len(getattr(unet_model, 'patches', {})),
                'patches_uuid': getattr(unet_model, 'patches_uuid', None)
            },
            'clip': {
                'model_id': id(clip_model),
                'class': type(clip_model).__name__,
                'device': getattr(clip_model, 'device', None),
                'patcher_patches_count': len(getattr(clip_model.patcher, 'patches', {})) if hasattr(clip_model, 'patcher') else 0,
                'patcher_patches_uuid': getattr(clip_model.patcher, 'patches_uuid', None) if hasattr(clip_model, 'patcher') else None
            }
        }
        return baseline
    
    def analyze_text_encoding_results(self, baseline, positive_cond, negative_cond, positive_prompt, negative_prompt, elapsed_time):
        """Analyze the results of text encoding"""
        
        # Check if baseline is available
        if baseline is None:
            print("⚠️  WARNING: Baseline not available - using limited analysis")
        
        # Get current memory state
        current_ram = psutil.virtual_memory()
        current_gpu = None
        if torch.cuda.is_available():
            current_gpu = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'total': torch.cuda.get_device_properties(0).total_memory
            }
        
        # Analyze conditioning tensors
        positive_analysis = self._analyze_conditioning_tensor(positive_cond, "Positive")
        negative_analysis = self._analyze_conditioning_tensor(negative_cond, "Negative")
        
        # Calculate memory changes
        memory_impact = self._calculate_text_encoding_memory_change(baseline, current_ram, current_gpu)
        
        # Analyze text processing
        text_analysis = self._analyze_text_processing(positive_prompt, negative_prompt)
        
        analysis = {
            'encoding_success': positive_cond is not None and negative_cond is not None,
            'elapsed_time': elapsed_time,
            'positive_conditioning': positive_analysis,
            'negative_conditioning': negative_analysis,
            'text_processing': text_analysis,
            'memory_impact': memory_impact,
            'peak_memory': self.get_peak_memory_summary(),
            'baseline': baseline,
            'current_memory': {
                'ram': current_ram,
                'gpu': current_gpu
            }
        }
        
        return analysis
    
    def _analyze_conditioning_tensor(self, conditioning, tensor_type):
        """Analyze a conditioning tensor in detail"""
        if conditioning is None:
            print(f"   ❌ {tensor_type} conditioning is None")
            return {
                'status': 'failed',
                'error': 'Conditioning tensor is None'
            }
        
        # First, debug the conditioning structure to understand it
        self._debug_conditioning_structure(conditioning, tensor_type)
        
        try:
            # Handle different conditioning structures from ComfyUI
            tensor_data = None
            metadata = {}
            
            print(f"   🔍 Attempting to extract tensor from {tensor_type} conditioning...")
            
            # Strategy 1: Direct tensor
            if hasattr(conditioning, 'shape'):
                tensor_data = conditioning
                metadata = {}
                print(f"   ✅ Strategy 1 SUCCESS: Found direct tensor with shape: {tensor_data.shape}")
            
            # Strategy 2: List/tuple format [tensor, metadata]
            elif isinstance(conditioning, (list, tuple)) and len(conditioning) > 0:
                print(f"   🔍 Strategy 2: Checking list/tuple with {len(conditioning)} items")
                if hasattr(conditioning[0], 'shape'):
                    tensor_data = conditioning[0]
                    metadata = conditioning[1] if len(conditioning) > 1 else {}
                    print(f"   ✅ Strategy 2 SUCCESS: Found tensor in list[0] with shape: {tensor_data.shape}")
                else:
                    # Try to find tensor in the list
                    print(f"   🔍 Strategy 2: Item[0] has no shape, searching through list...")
                    for i, item in enumerate(conditioning):
                        print(f"      Checking item[{i}]: type={type(item).__name__}, has_shape={hasattr(item, 'shape')}")
                        if hasattr(item, 'shape'):
                            tensor_data = item
                            metadata = {f'list_index_{i}': item for j, item in enumerate(conditioning) if j != i}
                            print(f"   ✅ Strategy 2 SUCCESS: Found tensor in list[{i}] with shape: {tensor_data.shape}")
                            break
            
            # Strategy 3: Dictionary format
            elif isinstance(conditioning, dict):
                print(f"   🔍 Strategy 3: Checking dictionary with {len(conditioning)} keys")
                # Look for tensor-like objects in the dictionary
                for key, value in conditioning.items():
                    print(f"      Checking key '{key}': type={type(value).__name__}, has_shape={hasattr(value, 'shape')}")
                    if hasattr(value, 'shape'):
                        tensor_data = value
                        metadata = {k: v for k, v in conditioning.items() if k != key}
                        print(f"   ✅ Strategy 3 SUCCESS: Found tensor in dict['{key}'] with shape: {tensor_data.shape}")
                        break
            
            # Strategy 4: Recursive deep search for nested structures
            if tensor_data is None:
                print(f"   🔍 Strategy 4: Attempting recursive deep search...")
                tensor_data, reason = self._extract_tensor_recursively(conditioning)
                if tensor_data is not None:
                    print(f"   ✅ Strategy 4 SUCCESS: {reason}")
                    # Create metadata from the original structure
                    metadata = {'extraction_method': 'recursive', 'reason': reason}
                else:
                    print(f"   ❌ Strategy 4 FAILED: {reason}")
            
            # If we found tensor data, analyze it
            if tensor_data is not None and hasattr(tensor_data, 'shape'):
                print(f"   🎯 Tensor extraction SUCCESS for {tensor_type}")
                shape = tensor_data.shape
                dtype = str(tensor_data.dtype)
                device = str(tensor_data.device)
                
                # Calculate tensor size in MB
                num_elements = tensor_data.numel()
                size_mb = (num_elements * tensor_data.element_size()) / (1024**2)
                
                # Determine CLIP variant based on dimensions
                clip_variant = self._detect_clip_variant_from_tensor(shape)
                
                # Create output directory for tensor dumps
                output_dir = "./W_out/step3"
                os.makedirs(output_dir, exist_ok=True)
                print(f"   📁 Output directory ensured: {output_dir}")
                
                # Generate filename for tensor dump
                timestamp = int(time.time())
                tensor_filename = f"{tensor_type.lower()}_conditioning_{timestamp}.npy"
                tensor_filepath = os.path.join(output_dir, tensor_filename)
                print(f"   📄 Will save to: {tensor_filepath}")
                
                # Store tensor for later comparison (dump to file)
                try:
                    print(f"   🔄 Converting tensor to numpy...")
                    # Convert to numpy and save
                    if hasattr(tensor_data, 'detach'):
                        print(f"      Using detach().cpu().numpy() method")
                        numpy_tensor = tensor_data.detach().cpu().numpy()
                    elif hasattr(tensor_data, 'cpu'):
                        print(f"      Using cpu().numpy() method")
                        numpy_tensor = tensor_data.cpu().numpy()
                    else:
                        print(f"      Using direct .numpy() method")
                        numpy_tensor = tensor_data.numpy()
                    
                    print(f"   💾 Saving numpy tensor to file...")
                    # Save tensor to file
                    np.save(tensor_filepath, numpy_tensor)
                    
                    # Verify file was created
                    if os.path.exists(tensor_filepath):
                        actual_size = os.path.getsize(tensor_filepath) / (1024**2)
                        print(f"   ✅ File saved successfully! Size: {actual_size:.2f} MB")
                    else:
                        print(f"   ❌ ERROR: File was not created!")
                    
                    tensor_dump = {
                        'shape': shape,
                        'dtype': dtype,
                        'device': device,
                        'size_mb': size_mb,
                        'num_elements': num_elements,
                        'clip_variant': clip_variant,
                        'filepath': tensor_filepath,
                        'filename': tensor_filename,
                        'numpy_shape': numpy_tensor.shape,
                        'numpy_dtype': str(numpy_tensor.dtype)
                    }
                    
                    print(f"   💾 Tensor dumped to: {tensor_filepath}")
                    print(f"   📊 Tensor info: {shape}, {dtype}, {size_mb:.2f} MB")
                    
                except Exception as dump_error:
                    print(f"   ⚠️  Warning: Could not dump tensor: {dump_error}")
                    import traceback
                    print(f"   🔍 Dump error traceback:")
                    traceback.print_exc()
                    tensor_dump = {
                        'shape': shape,
                        'dtype': dtype,
                        'device': device,
                        'size_mb': size_mb,
                        'num_elements': num_elements,
                        'clip_variant': clip_variant,
                        'filepath': 'FAILED',
                        'error': str(dump_error)
                    }
                
                return {
                    'status': 'success',
                    'shape': shape,
                    'dtype': dtype,
                    'device': device,
                    'size_mb': size_mb,
                    'num_elements': num_elements,
                    'clip_variant': clip_variant,
                    'metadata': metadata,
                    'tensor_dump': tensor_dump
                }
            else:
                print(f"   ❌ All strategies failed - no tensor data found")
                # Debug information about the conditioning structure
                debug_info = {
                    'type': type(conditioning).__name__,
                    'length': len(conditioning) if hasattr(conditioning, '__len__') else 'N/A',
                    'attributes': [attr for attr in dir(conditioning) if not attr.startswith('_')][:10] if hasattr(conditioning, '__dir__') else 'N/A'
                }
                
                return {
                    'status': 'failed',
                    'error': f'Could not extract tensor data from conditioning structure',
                    'debug_info': debug_info,
                    'conditioning_type': str(type(conditioning))
                }
                
        except Exception as e:
            print(f"   ❌ Analysis failed with exception: {str(e)}")
            import traceback
            print(f"   🔍 Exception traceback:")
            traceback.print_exc()
            return {
                'status': 'failed',
                'error': f'Analysis failed: {str(e)}',
                'traceback': str(e)
            }
    
    def _detect_clip_variant_from_tensor(self, shape):
        """Detect CLIP variant based on tensor dimensions"""
        if len(shape) >= 2:
            last_dim = shape[-1]
            if last_dim == 768:
                return "SD 1.5 CLIP"
            elif last_dim == 1024:
                return "SD 2.1 CLIP"
            elif last_dim == 1280:
                return "SDXL CLIP"
            elif last_dim == 2048:
                return "SD 3 / WAN T5"
            else:
                return f"Unknown CLIP (dim: {last_dim})"
        return "Unknown CLIP (invalid shape)"
    
    def _analyze_text_processing(self, positive_prompt, negative_prompt):
        """Analyze text processing characteristics"""
        return {
            'positive_prompt': {
                'text': positive_prompt,
                'length_chars': len(positive_prompt),
                'length_words': len(positive_prompt.split()),
                'special_chars': sum(1 for c in positive_prompt if not c.isalnum() and not c.isspace())
            },
            'negative_prompt': {
                'text': negative_prompt,
                'length_chars': len(negative_prompt),
                'length_words': len(negative_prompt.split()),
                'special_chars': sum(1 for c in negative_prompt if not c.isalnum() and not c.isspace())
            }
        }
    
    def _calculate_text_encoding_memory_change(self, baseline, current_ram, current_gpu):
        """Calculate memory usage changes during text encoding"""
        try:
            # Check if baseline is available
            if baseline is None:
                return {'error': 'Baseline not available for memory calculation'}
            
            # Calculate RAM changes
            ram_changes = {
                'used_change_mb': (current_ram.used - baseline.get('ram', {}).get('used_mb', 0) * (1024**2)) / (1024**2),
                'available_change_mb': (current_ram.available - baseline.get('ram', {}).get('available_mb', 0) * (1024**2)) / (1024**2),
                'current_used_mb': current_ram.used / (1024**2),
                'current_available_mb': current_ram.available / (1024**2),
                'current_total_mb': current_ram.total / (1024**2),
                'current_percent_used': current_ram.percent,
                'baseline_used_mb': baseline.get('ram', {}).get('used_mb', 0),
                'baseline_available_mb': baseline.get('ram', {}).get('available_mb', 0),
                'baseline_percent_used': baseline.get('ram', {}).get('percent_used', 0)
            }
            
            # Calculate GPU changes
            gpu_changes = None
            if current_gpu and baseline.get('gpu'):
                baseline_gpu = baseline['gpu']
                allocated_change = current_gpu['allocated'] - baseline_gpu.get('allocated', 0)
                reserved_change = current_gpu['reserved'] - baseline_gpu.get('reserved', 0)
                
                gpu_changes = {
                    'allocated_change_mb': allocated_change / (1024**2),
                    'reserved_change_mb': reserved_change / (1024**2),
                    'current_allocated_mb': current_gpu['allocated'] / (1024**2),
                    'current_reserved_mb': current_gpu['reserved'] / (1024**2),
                    'current_total_mb': current_gpu['total'] / (1024**2),
                    'baseline_allocated_mb': baseline_gpu.get('allocated', 0) / (1024**2),
                    'baseline_reserved_mb': baseline_gpu.get('reserved', 0) / (1024**2),
                    'baseline_total_mb': baseline_gpu.get('total', 0) / (1024**2),
                    'allocated_change_pct': (allocated_change / baseline_gpu.get('allocated', 1) * 100) if baseline_gpu.get('allocated', 0) > 0 else 0,
                    'reserved_change_pct': (reserved_change / baseline_gpu.get('reserved', 1) * 100) if baseline_gpu.get('reserved', 0) > 0 else 0
                }
            
            return {
                'ram': ram_changes,
                'gpu': gpu_changes
            }
        except Exception as e:
            return {'error': f'Memory calculation failed: {e}'}
    
    def print_text_encoding_analysis_summary(self, analysis):
        """Print comprehensive text encoding analysis"""
        print(f"\n🔍 TEXT ENCODING ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Check if analysis is None or invalid
        if analysis is None:
            print("❌ ERROR: Text encoding analysis is None")
            print("   This indicates the text encoding step failed or analysis was not generated")
            print("=" * 80)
            return
        
        # Basic success info
        print(f"✅ Text Encoding Success: {'YES' if analysis.get('encoding_success', False) else 'NO'}")
        print(f"⏱️  Total Execution Time: {analysis.get('elapsed_time', 0):.3f} seconds")
        
        # Text Processing Analysis
        print(f"\n📝 TEXT PROCESSING ANALYSIS:")
        text_processing = analysis.get('text_processing')
        if text_processing is None:
            print("   ❌ ERROR: Text processing analysis not available")
        else:
            positive = text_processing.get('positive_prompt', {})
            print(f"   🔤 POSITIVE PROMPT:")
            print(f"      Text: '{positive.get('text', 'N/A')}'")
            print(f"      Length: {positive.get('length_chars', 0)} characters, {positive.get('length_words', 0)} words")
            print(f"      Special Characters: {positive.get('special_chars', 0)}")
            
            negative = text_processing.get('negative_prompt', {})
            print(f"   🔤 NEGATIVE PROMPT:")
            print(f"      Text: '{negative.get('text', 'N/A')}'")
            print(f"      Length: {negative.get('length_chars', 0)} characters, {negative.get('length_words', 0)} words")
            print(f"      Special Characters: {negative.get('special_chars', 0)}")
        
        # Positive Conditioning Analysis
        print(f"\n🔧 POSITIVE CONDITIONING ANALYSIS:")
        positive_cond = analysis.get('positive_conditioning')
        if positive_cond is None:
            print("   ❌ ERROR: Positive conditioning analysis not available")
        elif positive_cond.get('status') == 'success':
            print(f"   ✅ Status: SUCCESS")
            print(f"   📐 Shape: {positive_cond.get('shape', 'N/A')}")
            print(f"   🏷️  Data Type: {positive_cond.get('dtype', 'N/A')}")
            print(f"   📱 Device: {positive_cond.get('device', 'N/A')}")
            print(f"   💾 Size: {positive_cond.get('size_mb', 0):.2f} MB")
            print(f"   🔢 Elements: {positive_cond.get('num_elements', 0):,}")
            print(f"   🎯 CLIP Variant: {positive_cond.get('clip_variant', 'N/A')}")
            
            # Store tensor dump for later comparison
            if positive_cond.get('tensor_dump'):
                print(f"   💾 Tensor Dump: Stored for comparison")
        else:
            print(f"   ❌ Status: FAILED")
            print(f"   Error: {positive_cond.get('error', 'Unknown error')}")
        
        # Negative Conditioning Analysis
        print(f"\n🔧 NEGATIVE CONDITIONING ANALYSIS:")
        negative_cond = analysis.get('negative_conditioning')
        if negative_cond is None:
            print("   ❌ ERROR: Negative conditioning analysis not available")
        elif negative_cond.get('status') == 'success':
            print(f"   ✅ Status: SUCCESS")
            print(f"   📐 Shape: {negative_cond.get('shape', 'N/A')}")
            print(f"   🏷️  Data Type: {negative_cond.get('dtype', 'N/A')}")
            print(f"   📱 Device: {negative_cond.get('device', 'N/A')}")
            print(f"   💾 Size: {negative_cond.get('size_mb', 0):.2f} MB")
            print(f"   🔢 Elements: {negative_cond.get('num_elements', 0):,}")
            print(f"   🎯 CLIP Variant: {negative_cond.get('clip_variant', 'N/A')}")
            
            # Store tensor dump for later comparison
            if negative_cond.get('tensor_dump'):
                print(f"   💾 Tensor Dump: Stored for comparison")
        else:
            print(f"   ❌ Status: FAILED")
            print(f"   Error: {negative_cond.get('error', 'Unknown error')}")
        
        # Tensor Dump Summary
        print(f"\n💾 TENSOR DUMP SUMMARY:")
        print(f"   📁 Output Directory: ./W_out/step3/")
        
        # Check what tensors were successfully dumped
        dumped_tensors = []
        if positive_cond and positive_cond.get('status') == 'success' and positive_cond.get('tensor_dump'):
            tensor_dump = positive_cond['tensor_dump']
            if tensor_dump.get('filepath') and tensor_dump.get('filepath') != 'FAILED':
                dumped_tensors.append({
                    'type': 'Positive',
                    'filepath': tensor_dump['filepath'],
                    'filename': tensor_dump['filename'],
                    'shape': positive_cond.get('shape'),
                    'size_mb': positive_cond.get('size_mb', 0)
                })
        
        if negative_cond and negative_cond.get('status') == 'success' and negative_cond.get('tensor_dump'):
            tensor_dump = negative_cond['tensor_dump']
            if tensor_dump.get('filepath') and tensor_dump.get('filepath') != 'FAILED':
                dumped_tensors.append({
                    'type': 'Negative',
                    'filepath': tensor_dump['filepath'],
                    'filename': tensor_dump['filename'],
                    'shape': negative_cond.get('shape'),
                    'size_mb': negative_cond.get('size_mb', 0)
                })
        
        if dumped_tensors:
            print(f"   ✅ Successfully dumped {len(dumped_tensors)} tensors:")
            for tensor in dumped_tensors:
                print(f"      {tensor['type']}: {tensor['filename']}")
                print(f"         Shape: {tensor['shape']}, Size: {tensor['size_mb']:.2f} MB")
                print(f"         Path: {tensor['filepath']}")
        else:
            print(f"   ❌ No tensors were successfully dumped")
            print(f"   💡 Check the error messages above for details")
        
        # Show output directory contents
        output_dir = "./W_out/step3"
        if os.path.exists(output_dir):
            try:
                files = os.listdir(output_dir)
                if files:
                    print(f"\n   📂 Output directory contents:")
                    for file in sorted(files):
                        filepath = os.path.join(output_dir, file)
                        if os.path.isfile(filepath):
                            size_mb = os.path.getsize(filepath) / (1024**2)
                            print(f"      📄 {file} ({size_mb:.2f} MB)")
                else:
                    print(f"\n   📂 Output directory is empty")
            except Exception as e:
                print(f"\n   ⚠️  Could not read output directory: {e}")
        else:
            print(f"\n   📂 Output directory does not exist")
        
        # Memory Impact
        print(f"\n💾 MEMORY IMPACT:")
        memory_impact = analysis.get('memory_impact')
        
        if memory_impact is None:
            print("   ❌ ERROR: Memory impact analysis not available")
        elif 'error' not in memory_impact:
            # RAM Changes
            if 'ram' in memory_impact:
                ram = memory_impact['ram']
                print(f"\n   🖥️  RAM CHANGES:")
                print(f"      Used: {ram.get('used_change_mb', 0):+.1f} MB ({ram.get('baseline_used_mb', 0):.1f} → {ram.get('current_used_mb', 0):.1f} MB)")
                print(f"      Available: {ram.get('available_change_mb', 0):+.1f} MB ({ram.get('baseline_available_mb', 0):.1f} → {ram.get('current_available_mb', 0):.1f} MB)")
                print(f"      Usage: {ram.get('baseline_percent_used', 0):.1f}% → {ram.get('current_percent_used', 0):.1f}%")
            
            # GPU Changes
            if 'gpu' in memory_impact and memory_impact['gpu']:
                gpu = memory_impact['gpu']
                print(f"\n   🎮 GPU CHANGES:")
                print(f"      Allocated: {gpu.get('allocated_change_mb', 0):+.1f} MB ({gpu.get('baseline_allocated_mb', 0):.1f} → {gpu.get('current_allocated_mb', 0):.1f} MB)")
                print(f"      Reserved: {gpu.get('reserved_change_mb', 0):+.1f} MB ({gpu.get('baseline_reserved_mb', 0):.1f} → {gpu.get('current_reserved_mb', 0):.1f} MB)")
                print(f"      Total VRAM: {gpu.get('current_total_mb', 0):.1f} MB")
        else:
            print(f"   ❌ Memory calculation failed: {memory_impact.get('error', 'Unknown error')}")
        
        # Peak Memory Information
        peak_memory = analysis.get('peak_memory')
        if peak_memory is None:
            print(f"\n📊 PEAK MEMORY DURING TEXT ENCODING:")
            print("   ❌ ERROR: Peak memory information not available")
        else:
            print(f"\n📊 PEAK MEMORY DURING TEXT ENCODING:")
            print(f"   🖥️  RAM Peak: {peak_memory.get('ram_peak_mb', 0):.1f} MB")
            print(f"   🎮 GPU Allocated Peak: {peak_memory.get('gpu_allocated_peak_mb', 0):.1f} MB")
            print(f"   🎮 GPU Reserved Peak: {peak_memory.get('gpu_reserved_peak_mb', 0):.1f} MB")
            
            # Show peak timestamps if available
            peak_timestamps = peak_memory.get('peak_timestamps')
            if peak_timestamps:
                print(f"   ⏱️  Peak Timestamps:")
                for peak in peak_memory['peak_timestamps'][:5]:  # Show first 5 peaks
                    print(f"      {peak.get('type', 'unknown')}: {peak.get('value_mb', 0):.1f} MB at {peak.get('timestamp', 0):.2f}s")
        
        print("=" * 80)
    
    def _calculate_model_sampling_memory_change(self, unet_baseline, gpu_baseline, current_ram, current_gpu):
        """Calculate memory usage changes during model sampling"""
        try:
            # Check if baselines are available
            if unet_baseline is None:
                return {'error': 'Baselines not available for memory calculation'}
            
            # Calculate RAM changes
            ram_changes = {
                'used_change_mb': (current_ram.used - unet_baseline.get('ram', {}).get('used_mb', 0) * (1024**2)) / (1024**2),
                'available_change_mb': (current_ram.available - unet_baseline.get('ram', {}).get('available_mb', 0) * (1024**2)) / (1024**2),
                'current_used_mb': current_ram.used / (1024**2),
                'current_available_mb': current_ram.available / (1024**2),
                'current_total_mb': current_ram.total / (1024**2),
                'current_percent_used': current_ram.percent,
                'baseline_used_mb': unet_baseline.get('ram', {}).get('used_mb', 0),
                'baseline_available_mb': unet_baseline.get('ram', {}).get('available_mb', 0),
                'baseline_percent_used': unet_baseline.get('ram', {}).get('percent_used', 0)
            }
            
            # Calculate GPU changes
            gpu_changes = None
            if current_gpu and unet_baseline.get('gpu'):
                baseline_gpu = unet_baseline['gpu']
                allocated_change = current_gpu['allocated'] - baseline_gpu.get('allocated', 0)
                reserved_change = current_gpu['reserved'] - baseline_gpu.get('reserved', 0)
                
                gpu_changes = {
                    'allocated_change_mb': allocated_change / (1024**2),
                    'reserved_change_mb': reserved_change / (1024**2),
                    'current_allocated_mb': current_gpu['allocated'] / (1024**2),
                    'current_reserved_mb': current_gpu['reserved'] / (1024**2),
                    'current_total_mb': current_gpu['total'] / (1024**2),
                    'baseline_allocated_mb': baseline_gpu.get('allocated', 0) / (1024**2),
                    'baseline_reserved_mb': baseline_gpu.get('reserved', 0) / (1024**2),
                    'baseline_total_mb': baseline_gpu.get('total', 0) / (1024**2),
                    'allocated_change_pct': (allocated_change / baseline_gpu.get('allocated', 1) * 100) if baseline_gpu.get('allocated', 0) > 0 else 0,
                    'reserved_change_pct': (reserved_change / baseline_gpu.get('reserved', 1) * 100) if baseline_gpu.get('reserved', 0) > 0 else 0
                }
            
            return {
                'ram': ram_changes,
                'gpu': gpu_changes
            }
        except Exception as e:
            return {'error': f'Memory calculation failed: {e}'}
    
    def print_model_sampling_analysis_summary(self, analysis):
        """Print comprehensive model sampling analysis"""
        print(f"\n🔍 MODEL SAMPLING ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Check if analysis is None or invalid
        if analysis is None:
            print("❌ ERROR: Model sampling analysis is None")
            print("   This indicates the model sampling step failed or analysis was not generated")
            print("=" * 80)
            return
        
        # Basic success info
        print(f"✅ ModelSamplingSD3 Success: {'YES' if analysis.get('sampling_success', False) else 'NO'}")
        print(f"⏱️  Total Execution Time: {analysis.get('elapsed_time', 0):.3f} seconds")
        
        # UNET Changes Analysis
        print(f"\n🔧 UNET MODEL CHANGES:")
        unet_changes = analysis.get('unet_changes')
        if unet_changes is None:
            print("   ❌ ERROR: UNET changes analysis not available")
        else:
            print(f"   Model Cloned: {'✅ YES' if unet_changes.get('model_cloned', False) else '❌ NO'}")
            print(f"   Class Changed: {'✅ YES' if unet_changes.get('class_changed', False) else '❌ NO'}")
            print(f"   Patches Added: {unet_changes.get('patches_added', 0)}")
            print(f"   UUID Changed: {'✅ YES' if unet_changes.get('uuid_changed', False) else '❌ NO'}")
            print(f"   Original Patches: {unet_changes.get('original_patch_count', 0)}")
            print(f"   Modified Patches: {unet_changes.get('modified_patch_count', 0)}")
        
        # UNET Patches Analysis
        print(f"\n🔧 UNET PATCHES ANALYSIS:")
        unet_patches = analysis.get('unet_patches')
        if unet_patches is None:
            print("   ❌ ERROR: UNET patches analysis not available")
        elif 'error' not in unet_patches:
            print(f"   Total Patched Keys: {unet_patches.get('total_patched_keys', 0)}")
            print(f"   Total Patches: {unet_patches.get('total_patches', 0)}")
            print(f"   Model Structure: {unet_patches.get('model_structure', 'N/A')}")
        else:
            print(f"   UNET Patches: {unet_patches.get('error', 'Unknown error')}")
        
        # Immediate GPU Changes
        print(f"\n💾 IMMEDIATE GPU IMPACT:")
        immediate_gpu_changes = analysis.get('immediate_gpu_changes')
        if immediate_gpu_changes is None:
            print("   ❌ ERROR: Immediate GPU changes not available")
        else:
            print(f"   Allocated Change: {immediate_gpu_changes.get('allocated_change_mb', 0):+.1f} MB")
            print(f"   Reserved Change: {immediate_gpu_changes.get('reserved_change_mb', 0):+.1f} MB")
        
        # Memory Impact
        print(f"\n💾 MEMORY IMPACT:")
        memory_impact = analysis.get('memory_impact')
        
        if memory_impact is None:
            print("   ❌ ERROR: Memory impact analysis not available")
        elif 'error' not in memory_impact:
            # RAM Changes
            if 'ram' in memory_impact:
                ram = memory_impact['ram']
                print(f"\n   🖥️  RAM CHANGES:")
                print(f"      Used: {ram.get('used_change_mb', 0):+.1f} MB ({ram.get('baseline_used_mb', 0):.1f} → {ram.get('current_used_mb', 0):.1f} MB)")
                print(f"      Available: {ram.get('available_change_mb', 0):+.1f} MB ({ram.get('baseline_available_mb', 0):.1f} → {ram.get('current_available_mb', 0):.1f} MB)")
                print(f"      Usage: {ram.get('baseline_percent_used', 0):.1f}% → {ram.get('current_percent_used', 0):.1f}%")
            
            # GPU Changes
            if 'gpu' in memory_impact and memory_impact['gpu']:
                gpu = memory_impact['gpu']
                print(f"\n   🎮 GPU CHANGES:")
                print(f"      Allocated: {gpu.get('allocated_change_mb', 0):+.1f} MB ({gpu.get('baseline_allocated_mb', 0):.1f} → {gpu.get('current_allocated_mb', 0):.1f} MB)")
                print(f"      Reserved: {gpu.get('reserved_change_mb', 0):+.1f} MB ({gpu.get('baseline_reserved_mb', 0):.1f} → {gpu.get('current_reserved_mb', 0):.1f} MB)")
                print(f"      Total VRAM: {gpu.get('current_total_mb', 0):.1f} MB")
                print(f"      Allocated Change: {gpu.get('allocated_change_pct', 0):+.1f}%")
                print(f"      Reserved Change: {gpu.get('reserved_change_pct', 0):+.1f}%")
        else:
            print(f"   ❌ Memory calculation failed: {memory_impact.get('error', 'Unknown error')}")
        
        # Peak Memory Information
        peak_memory = analysis.get('peak_memory')
        if peak_memory is None:
            print(f"\n📊 PEAK MEMORY DURING MODEL SAMPLING:")
            print("   ❌ ERROR: Peak memory information not available")
        else:
            print(f"\n📊 PEAK MEMORY DURING MODEL SAMPLING:")
            print(f"   🖥️  RAM Peak: {peak_memory.get('ram_peak_mb', 0):.1f} MB")
            print(f"   🎮 GPU Allocated Peak: {peak_memory.get('gpu_allocated_peak_mb', 0):.1f} MB")
            print(f"   🎮 GPU Reserved Peak: {peak_memory.get('gpu_reserved_peak_mb', 0):.1f} MB")
            
            # Show peak timestamps if available
            peak_timestamps = peak_memory.get('peak_timestamps')
            if peak_timestamps:
                print(f"   ⏱️  Peak Timestamps:")
                for peak in peak_memory['peak_timestamps'][:5]:  # Show first 5 peaks
                    print(f"      {peak.get('type', 'unknown')}: {peak.get('value_mb', 0):.1f} MB at {peak.get('timestamp', 0):.2f}s")
        
        print("=" * 80)
    
    # === LATENT GENERATION MONITORING METHODS ===
    
    def capture_latent_generation_baseline(self, vae_model, video_data):
        """Capture VAE and GPU state before latent generation"""
        # Capture RAM baseline
        ram_baseline = psutil.virtual_memory()
        
        # Capture GPU baseline
        gpu_baseline = None
        if torch.cuda.is_available():
            gpu_baseline = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'total': torch.cuda.get_device_properties(0).total_memory,
                'device_name': torch.cuda.get_device_name(0)
            }
        
        baseline = {
            'timestamp': time.time(),
            'ram': {
                'used_mb': ram_baseline.used / (1024**2),
                'available_mb': ram_baseline.available / (1024**2),
                'total_mb': ram_baseline.total / (1024**2),
                'percent_used': ram_baseline.percent
            },
            'gpu': gpu_baseline,
            'vae': {
                'model_id': id(vae_model),
                'class': type(vae_model).__name__,
                'device': getattr(vae_model, 'device', None)
            }
        }
        return baseline
    
    def analyze_latent_generation_results(self, baseline, init_latent, elapsed_time, strategy_used, immediate_gpu_changes, video_length, video_height, video_width):
        """Analyze the results of latent generation (VAE encoding)"""
        
        # Check if baseline is available
        if baseline is None:
            print("⚠️  WARNING: Baseline not available - using limited analysis")
        
        # Get current memory state
        current_ram = psutil.virtual_memory()
        current_gpu = None
        if torch.cuda.is_available():
            current_gpu = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'total': torch.cuda.get_device_properties(0).total_memory
            }
        
        # Analyze latent output
        latent_analysis = self._analyze_latent_output(init_latent, video_length, video_height, video_width)
        
        # Calculate memory changes
        memory_impact = self._calculate_latent_generation_memory_change(baseline, current_ram, current_gpu)
        
        # Analyze VAE encoding performance
        performance_analysis = self._analyze_vae_encoding_performance(elapsed_time, strategy_used, video_length, video_height, video_width)
        
        analysis = {
            'encoding_success': init_latent is not None,
            'elapsed_time': elapsed_time,
            'strategy_used': strategy_used,
            'latent_analysis': latent_analysis,
            'performance_analysis': performance_analysis,
            'memory_impact': memory_impact,
            'immediate_gpu_changes': immediate_gpu_changes,
            'peak_memory': self.get_peak_memory_summary(),
            'baseline': baseline,
            'current_memory': {
                'ram': current_ram,
                'gpu': current_gpu
            }
        }
        
        return analysis
    
    def _analyze_latent_output(self, init_latent, video_length, video_height, video_width):
        """Analyze the generated latent output"""
        if init_latent is None:
            return {
                'status': 'failed',
                'error': 'No latent output generated'
            }
        
        try:
            # Analyze latent shape and dimensions
            latent_shape = init_latent.shape
            latent_dtype = str(init_latent.dtype)
            latent_device = str(init_latent.device)
            
            # Calculate latent size in MB
            num_elements = init_latent.numel()
            size_mb = (num_elements * init_latent.element_size()) / (1024**2)
            
            # Analyze latent structure
            if len(latent_shape) == 4:
                # Standard image latent: (batch, channels, height, width)
                latent_type = "Image Latent"
                batch_size = latent_shape[0]
                channels = latent_shape[1]
                height = latent_shape[2]
                width = latent_shape[3]
            elif len(latent_shape) == 5:
                # Video latent: (batch, frames, channels, height, width)
                latent_type = "Video Latent"
                batch_size = latent_shape[0]
                frames = latent_shape[1]
                channels = latent_shape[2]
                height = latent_shape[3]
                width = latent_shape[4]
            else:
                latent_type = f"Unknown Latent (rank {len(latent_shape)})"
                batch_size = frames = channels = height = width = "N/A"
            
            # Calculate compression ratios
            if isinstance(video_height, (int, float)) and isinstance(video_width, (int, float)):
                spatial_compression = (video_height * video_width) / (height * width) if height != "N/A" and width != "N/A" else "N/A"
            else:
                spatial_compression = "N/A"
            
            if isinstance(video_length, (int, float)) and frames != "N/A":
                temporal_compression = video_length / frames if frames > 0 else "N/A"
            else:
                temporal_compression = "N/A"
            
            return {
                'status': 'success',
                'latent_type': latent_type,
                'shape': latent_shape,
                'dtype': latent_dtype,
                'device': latent_device,
                'size_mb': size_mb,
                'num_elements': num_elements,
                'batch_size': batch_size,
                'frames': frames if 'frames' in locals() else "N/A",
                'channels': channels,
                'height': height,
                'width': width,
                'spatial_compression': spatial_compression,
                'temporal_compression': temporal_compression
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': f'Latent analysis failed: {str(e)}'
            }
    
    def _analyze_vae_encoding_performance(self, elapsed_time, strategy_used, video_length, video_height, video_width):
        """Analyze VAE encoding performance metrics"""
        try:
            # Calculate performance metrics
            total_pixels = video_length * video_height * video_width
            total_mb = (total_pixels * 3) / (1024**2)  # 3 channels (RGB)
            
            # Performance metrics
            if elapsed_time and elapsed_time > 0:
                pixels_per_second = total_pixels / elapsed_time
                mb_per_second = total_mb / elapsed_time
                frames_per_second = video_length / elapsed_time
            else:
                pixels_per_second = mb_per_second = frames_per_second = "N/A"
            
            return {
                'total_pixels': total_pixels,
                'total_mb': total_mb,
                'elapsed_time': elapsed_time,
                'strategy_used': strategy_used,
                'pixels_per_second': pixels_per_second,
                'mb_per_second': mb_per_second,
                'frames_per_second': frames_per_second,
                'video_dimensions': {
                    'length': video_length,
                    'height': video_height,
                    'width': video_width
                }
            }
            
        except Exception as e:
            return {
                'error': f'Performance analysis failed: {str(e)}'
            }
    
    def _calculate_latent_generation_memory_change(self, baseline, current_ram, current_gpu):
        """Calculate memory usage changes during latent generation"""
        try:
            # Check if baseline is available
            if baseline is None:
                return {'error': 'Baseline not available for memory calculation'}
            
            # Calculate RAM changes
            ram_changes = {
                'used_change_mb': (current_ram.used - baseline.get('ram', {}).get('used_mb', 0) * (1024**2)) / (1024**2),
                'available_change_mb': (current_ram.available - baseline.get('ram', {}).get('available_mb', 0) * (1024**2)) / (1024**2),
                'current_used_mb': current_ram.used / (1024**2),
                'current_available_mb': current_ram.available / (1024**2),
                'current_total_mb': current_ram.total / (1024**2),
                'current_percent_used': current_ram.percent,
                'baseline_used_mb': baseline.get('ram', {}).get('used_mb', 0),
                'baseline_available_mb': baseline.get('ram', {}).get('available_mb', 0),
                'baseline_percent_used': baseline.get('ram', {}).get('percent_used', 0)
            }
            
            # Calculate GPU changes
            gpu_changes = None
            if current_gpu and baseline.get('gpu'):
                baseline_gpu = baseline['gpu']
                allocated_change = current_gpu['allocated'] - baseline_gpu.get('allocated', 0)
                reserved_change = current_gpu['reserved'] - baseline_gpu.get('reserved', 0)
                
                gpu_changes = {
                    'allocated_change_mb': allocated_change / (1024**2),
                    'reserved_change_mb': reserved_change / (1024**2),
                    'current_allocated_mb': current_gpu['allocated'] / (1024**2),
                    'current_reserved_mb': current_gpu['reserved'] / (1024**2),
                    'current_total_mb': current_gpu['total'] / (1024**2),
                    'baseline_allocated_mb': baseline_gpu.get('allocated', 0) / (1024**2),
                    'baseline_reserved_mb': baseline_gpu.get('reserved', 0) / (1024**2),
                    'baseline_total_mb': baseline_gpu.get('total', 0) / (1024**2),
                    'allocated_change_pct': (allocated_change / baseline_gpu.get('allocated', 1) * 100) if baseline_gpu.get('allocated', 0) > 0 else 0,
                    'reserved_change_pct': (reserved_change / baseline_gpu.get('reserved', 1) * 100) if baseline_gpu.get('reserved', 0) > 0 else 0
                }
            
            return {
                'ram': ram_changes,
                'gpu': gpu_changes
            }
        except Exception as e:
            return {'error': f'Memory calculation failed: {e}'}
    
    def print_latent_generation_analysis_summary(self, analysis):
        """Print comprehensive latent generation analysis"""
        print(f"\n🔍 LATENT GENERATION ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Check if analysis is None or invalid
        if analysis is None:
            print("❌ ERROR: Latent generation analysis is None")
            print("   This indicates the latent generation step failed or analysis was not generated")
            print("=" * 80)
            return
        
        # Basic success info
        print(f"✅ VAE Encoding Success: {'YES' if analysis.get('encoding_success', False) else 'NO'}")
        elapsed_time = analysis.get('elapsed_time', 0)
        if elapsed_time is not None:
            print(f"⏱️  Total Execution Time: {elapsed_time:.3f} seconds")
        else:
            print(f"⏱️  Total Execution Time: N/A")
        print(f"🎯 Strategy Used: {analysis.get('strategy_used', 'Unknown')}")
        
        # Latent Output Analysis
        print(f"\n📐 LATENT OUTPUT ANALYSIS:")
        latent_analysis = analysis.get('latent_analysis')
        if latent_analysis is None:
            print("   ❌ ERROR: Latent analysis not available")
        elif latent_analysis.get('status') == 'success':
            print(f"   ✅ Status: SUCCESS")
            print(f"   🎯 Latent Type: {latent_analysis.get('latent_type', 'N/A')}")
            print(f"   📐 Shape: {latent_analysis.get('shape', 'N/A')}")
            print(f"   🏷️  Data Type: {latent_analysis.get('dtype', 'N/A')}")
            print(f"   📱 Device: {latent_analysis.get('device', 'N/A')}")
            size_mb = latent_analysis.get('size_mb', 0)
            if size_mb is not None:
                print(f"   💾 Size: {size_mb:.2f} MB")
            else:
                print(f"   💾 Size: N/A")
            num_elements = latent_analysis.get('num_elements', 0)
            if num_elements is not None:
                print(f"   🔢 Elements: {num_elements:,}")
            else:
                print(f"   🔢 Elements: N/A")
            
            # Show compression ratios
            spatial_comp = latent_analysis.get('spatial_compression')
            temporal_comp = latent_analysis.get('temporal_compression')
            if spatial_comp != "N/A" and spatial_comp is not None:
                print(f"   📏 Spatial Compression: {spatial_comp:.1f}x")
            if temporal_comp != "N/A" and temporal_comp is not None:
                print(f"   ⏱️  Temporal Compression: {temporal_comp:.1f}x")
        else:
            print(f"   ❌ Status: FAILED")
            print(f"   Error: {latent_analysis.get('error', 'Unknown error')}")
        
        # Performance Analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        performance_analysis = analysis.get('performance_analysis')
        if performance_analysis is None:
            print("   ❌ ERROR: Performance analysis not available")
        elif 'error' not in performance_analysis:
            video_dims = performance_analysis.get('video_dimensions', {})
            length = video_dims.get('length', 0)
            height = video_dims.get('height', 0)
            width = video_dims.get('width', 0)
            print(f"   📊 Video Dimensions: {length} frames, {height}x{width}")
            
            total_pixels = performance_analysis.get('total_pixels', 0)
            if total_pixels is not None:
                print(f"   📊 Total Pixels: {total_pixels:,}")
            else:
                print(f"   📊 Total Pixels: N/A")
                
            total_mb = performance_analysis.get('total_mb', 0)
            if total_mb is not None:
                print(f"   📊 Total Data: {total_mb:.2f} MB")
            else:
                print(f"   📊 Total Data: N/A")
            
            # Performance metrics
            fps = performance_analysis.get('frames_per_second')
            mbps = performance_analysis.get('mb_per_second')
            if fps != "N/A" and fps is not None:
                print(f"   ⚡ Processing Speed: {fps:.1f} frames/second")
            if mbps != "N/A" and mbps is not None:
                print(f"   ⚡ Data Throughput: {mbps:.1f} MB/second")
        else:
            print(f"   ❌ Performance analysis failed: {performance_analysis.get('error', 'Unknown error')}")
        
        # Immediate GPU Changes
        print(f"\n💾 IMMEDIATE GPU IMPACT:")
        immediate_gpu_changes = analysis.get('immediate_gpu_changes')
        if immediate_gpu_changes is None:
            print("   ❌ ERROR: Immediate GPU changes not available")
        else:
            allocated_change = immediate_gpu_changes.get('allocated_change_mb', 0)
            reserved_change = immediate_gpu_changes.get('reserved_change_mb', 0)
            if allocated_change is not None:
                print(f"   Allocated Change: {allocated_change:+.1f} MB")
            else:
                print(f"   Allocated Change: N/A")
            if reserved_change is not None:
                print(f"   Reserved Change: {reserved_change:+.1f} MB")
            else:
                print(f"   Reserved Change: N/A")
        
        # Memory Impact
        print(f"\n💾 MEMORY IMPACT:")
        memory_impact = analysis.get('memory_impact')
        
        if memory_impact is None:
            print("   ❌ ERROR: Memory impact analysis not available")
        elif 'error' not in memory_impact:
            # RAM Changes
            if 'ram' in memory_impact:
                ram = memory_impact['ram']
                used_change = ram.get('used_change_mb', 0)
                baseline_used = ram.get('baseline_used_mb', 0)
                current_used = ram.get('current_used_mb', 0)
                available_change = ram.get('available_change_mb', 0)
                baseline_available = ram.get('baseline_available_mb', 0)
                current_available = ram.get('current_available_mb', 0)
                baseline_percent = ram.get('baseline_percent_used', 0)
                current_percent = ram.get('current_percent_used', 0)
                
                print(f"\n   🖥️  RAM CHANGES:")
                if used_change is not None and baseline_used is not None and current_used is not None:
                    print(f"      Used: {used_change:+.1f} MB ({baseline_used:.1f} → {current_used:.1f} MB)")
                else:
                    print(f"      Used: N/A")
                if available_change is not None and baseline_available is not None and current_available is not None:
                    print(f"      Available: {available_change:+.1f} MB ({baseline_available:.1f} → {current_available:.1f} MB)")
                else:
                    print(f"      Available: N/A")
                if baseline_percent is not None and current_percent is not None:
                    print(f"      Usage: {baseline_percent:.1f}% → {current_percent:.1f}%")
                else:
                    print(f"      Usage: N/A")
            
            # GPU Changes
            if 'gpu' in memory_impact and memory_impact['gpu']:
                gpu = memory_impact['gpu']
                allocated_change = gpu.get('allocated_change_mb', 0)
                reserved_change = gpu.get('reserved_change_mb', 0)
                current_total = gpu.get('current_total_mb', 0)
                allocated_pct = gpu.get('allocated_change_pct', 0)
                reserved_pct = gpu.get('reserved_change_pct', 0)
                
                print(f"\n   🎮 GPU CHANGES:")
                if allocated_change is not None:
                    baseline_allocated = gpu.get('baseline_allocated_mb', 0)
                    current_allocated = gpu.get('current_allocated_mb', 0)
                    if baseline_allocated is not None and current_allocated is not None:
                        print(f"      Allocated: {allocated_change:+.1f} MB ({baseline_allocated:.1f} → {current_allocated:.1f} MB)")
                    else:
                        print(f"      Allocated: {allocated_change:+.1f} MB")
                else:
                    print(f"      Allocated: N/A")
                    
                if reserved_change is not None:
                    baseline_reserved = gpu.get('baseline_reserved_mb', 0)
                    current_reserved = gpu.get('current_reserved_mb', 0)
                    if baseline_reserved is not None and current_reserved is not None:
                        print(f"      Reserved: {reserved_change:+.1f} MB ({baseline_reserved:.1f} → {current_reserved:.1f} MB)")
                    else:
                        print(f"      Reserved: {reserved_change:+.1f} MB")
                else:
                    print(f"      Reserved: N/A")
                    
                if current_total is not None:
                    print(f"      Total VRAM: {current_total:.1f} MB")
                else:
                    print(f"      Total VRAM: N/A")
                    
                if allocated_pct is not None:
                    print(f"      Allocated Change: {allocated_pct:+.1f}%")
                else:
                    print(f"      Allocated Change: N/A")
                    
                if reserved_pct is not None:
                    print(f"      Reserved Change: {reserved_pct:+.1f}%")
                else:
                    print(f"      Reserved Change: N/A")
        else:
            print(f"   ❌ Memory calculation failed: {memory_impact.get('error', 'Unknown error')}")
        
        # Peak Memory Information
        peak_memory = analysis.get('peak_memory')
        if peak_memory is None:
            print(f"\n📊 PEAK MEMORY DURING LATENT GENERATION:")
            print("   ❌ ERROR: Peak memory information not available")
        else:
            print(f"\n📊 PEAK MEMORY DURING LATENT GENERATION:")
            ram_peak = peak_memory.get('ram_peak_mb', 0)
            gpu_allocated_peak = peak_memory.get('gpu_allocated_peak_mb', 0)
            gpu_reserved_peak = peak_memory.get('gpu_reserved_peak_mb', 0)
            
            if ram_peak is not None:
                print(f"   🖥️  RAM Peak: {ram_peak:.1f} MB")
            else:
                print(f"   🖥️  RAM Peak: N/A")
                
            if gpu_allocated_peak is not None:
                print(f"   🎮 GPU Allocated Peak: {gpu_allocated_peak:.1f} MB")
            else:
                print(f"   🎮 GPU Allocated Peak: N/A")
                
            if gpu_reserved_peak is not None:
                print(f"   🎮 GPU Reserved Peak: {gpu_reserved_peak:.1f} MB")
            else:
                print(f"   🎮 GPU Reserved Peak: N/A")
            
            # Show peak timestamps if available
            peak_timestamps = peak_memory.get('peak_timestamps')
            if peak_timestamps:
                print(f"   ⏱️  Peak Timestamps:")
                for peak in peak_memory['peak_timestamps'][:5]:  # Show first 5 peaks
                    peak_type = peak.get('type', 'unknown')
                    peak_value = peak.get('value_mb', 0)
                    peak_time = peak.get('timestamp', 0)
                    if peak_value is not None and peak_time is not None:
                        print(f"      {peak_type}: {peak_value:.1f} MB at {peak_time:.2f}s")
                    else:
                        print(f"      {peak_type}: N/A")
        
        print("=" * 80)

# Initialize the monitor
model_monitor = ModelLoadingMonitor()


def find_safu_files():
    """Find safu.jpg and safu.mp4 files in various possible locations"""
    possible_locations = [
        ".",  # Current directory
        "./input",  # Input directory
        "..",  # Parent directory
        "../..",  # Grandparent directory
        os.path.dirname(os.path.abspath(__file__)),  # Script directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),  # Script parent
    ]
    
    safu_files = {}
    
    for location in possible_locations:
        try:
            if os.path.exists(location):
                safu_mp4 = os.path.join(location, "safu.mp4")
                safu_jpg = os.path.join(location, "safu.jpg")
                
                if os.path.exists(safu_mp4) and "mp4" not in safu_files:
                    safu_files["mp4"] = os.path.abspath(safu_mp4)
                    print(f"   ✅ Found safu.mp4 at: {safu_files['mp4']}")
                
                if os.path.exists(safu_jpg) and "jpg" not in safu_files:
                    safu_files["jpg"] = os.path.abspath(safu_jpg)
                    print(f"   ✅ Found safu.jpg at: {safu_files['jpg']}")
                
                if len(safu_files) == 2:
                    break
        except Exception as e:
            continue
    
    return safu_files


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
        print("✅ Successfully imported extra_config from main")
    except ImportError:
        try:
            from utils.extra_config import load_extra_path_config
            print("✅ Successfully imported extra_config from utils.extra_config")
        except ImportError:
            print("⚠️  Warning: Could not import extra_config, skipping extra model paths")
            print("   This is normal if extra_model_paths.yaml is not configured")
            return

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        try:
            load_extra_path_config(extra_model_paths)
            print(f"✅ Successfully loaded extra model paths from: {extra_model_paths}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load extra model paths: {e}")
    else:
        print("ℹ️  No extra_model_paths.yaml found - using default paths only")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()

# VHS import will be attempted through custom nodes system instead of direct import
# attempt_vhs_import()


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
            print(f"⚠️  Warning: Custom node import fallback failed: {e}")
            pass
        
        finally:
            # Restore original working directory and sys.path
            os.chdir(original_cwd)
            sys.path = original_sys_path
    
    return {}


def main():
    print("🔍 Starting main function...")
    
    # Load custom nodes FIRST, before importing NODE_CLASS_MAPPINGS
    print("🔍 Attempting to import custom nodes...")
    custom_node_mappings = import_custom_nodes()
    print(f"🔍 Custom node mappings result: {type(custom_node_mappings)}")
    
    # Initialize VHS variables
    VHS_AVAILABLE = False
    VHS_LoadVideoPath = None
    VHS_LoadVideoUpload = None
    
    if custom_node_mappings:
        print(f"🔍 Found {len(custom_node_mappings)} custom nodes")
        # Show some of the custom node names
        custom_node_names = list(custom_node_mappings.keys())[:10]
        print(f"🔍 Custom node examples: {custom_node_names}")
        
        # Check if VHS nodes are in the custom node mappings
        vhs_nodes = [name for name in custom_node_mappings.keys() if 'video' in name.lower() or 'vhs' in name.lower()]
        if vhs_nodes:
            print(f"🔍 Found VHS-related nodes: {vhs_nodes}")
            VHS_AVAILABLE = True
            # Try to get VHS classes from custom node mappings
            for node_name in vhs_nodes:
                if 'load' in node_name.lower() and 'video' in node_name.lower():
                    if 'path' in node_name.lower():
                        VHS_LoadVideoPath = custom_node_mappings[node_name]
                        print(f"✅ Found VHS_LoadVideoPath: {node_name}")
                    elif 'upload' in node_name.lower():
                        VHS_LoadVideoUpload = custom_node_mappings[node_name]
                        print(f"✅ Found VHS_LoadVideoUpload: {node_name}")
        else:
            print("⚠️  No VHS-related nodes found in custom node mappings")
    else:
        print("⚠️  No custom node mappings returned")
        print("🔍 VHS will not be available - will use fallback methods")
    
    # Show VHS import status at the beginning
    print("\n🔍 VHS IMPORT STATUS:")
    print(f"   VHS_AVAILABLE: {VHS_AVAILABLE}")
    print(f"   VHS_LoadVideoPath: {VHS_LoadVideoPath}")
    print(f"   VHS_LoadVideoUpload: {VHS_LoadVideoUpload}")
    
    if VHS_AVAILABLE:
        print("   ✅ VHS classes successfully imported - will use for video loading")
    else:
        print("   ⚠️  VHS classes not available - will use fallback methods")
    
    # Now import NODE_CLASS_MAPPINGS after custom nodes are loaded
    print("🔍 Importing core ComfyUI nodes...")
    try:
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
        print("✅ Successfully imported core ComfyUI nodes")
    except ImportError as e:
        print(f"❌ ERROR importing core nodes: {e}")
        return
    
    # Manually merge custom nodes if they weren't merged automatically
    if custom_node_mappings:
        NODE_CLASS_MAPPINGS.update(custom_node_mappings)
        print(f"✅ Merged {len(custom_node_mappings)} custom nodes into NODE_CLASS_MAPPINGS")
    
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
    print(f"🔍 Checking NODE_CLASS_MAPPINGS...")
    print(f"🔍 Total available nodes: {len(NODE_CLASS_MAPPINGS)}")
    
    # Check for VHS nodes in the merged mappings
    vhs_node_names = [name for name in NODE_CLASS_MAPPINGS.keys() if 'video' in name.lower() or 'vhs' in name.lower()]
    if vhs_node_names:
        print(f"✅ Found VHS nodes in NODE_CLASS_MAPPINGS: {vhs_node_names}")
        # Update VHS availability based on what's found
        for node_name in vhs_node_names:
            if 'load' in node_name.lower() and 'video' in node_name.lower():
                if 'path' in node_name.lower():
                    VHS_LoadVideoPath = NODE_CLASS_MAPPINGS[node_name]
                    VHS_AVAILABLE = True
                    print(f"✅ VHS_LoadVideoPath available: {node_name}")
                elif 'upload' in node_name.lower():
                    VHS_LoadVideoUpload = NODE_CLASS_MAPPINGS[node_name]
                    VHS_AVAILABLE = True
                    print(f"✅ VHS_LoadVideoUpload available: {node_name}")
    else:
        print("⚠️  WARNING: No VHS nodes found in NODE_CLASS_MAPPINGS!")
        print("🔍 This is expected if custom nodes aren't loaded. Continuing with model loading debugging...")
        print("📋 Available nodes: " + ", ".join(list(NODE_CLASS_MAPPINGS.keys())[:10]) + "...")
    
    with torch.inference_mode():
        # === STEP 1 START: MODEL LOADING ===
        print("1. Loading diffusion model components...")
        
        # Load video and reference image using custom nodes if available
        print("\n🔍 SEARCHING FOR SAFU FILES...")
        safu_files = find_safu_files()
        
        # Use found files or fall back to current directory
        if "mp4" in safu_files:
            video_file = safu_files["mp4"]
            # Ensure the path is properly resolved
            if not os.path.isabs(video_file):
                video_file = os.path.abspath(video_file)
            print(f"   🎥 Using video file: {video_file}")
        else:
            video_file = "safu.mp4"
            print(f"   ⚠️  Video file not found, will look for: {video_file}")
        
        if "jpg" in safu_files:
            image_file = safu_files["jpg"]
            # Ensure the path is properly resolved
            if not os.path.isabs(image_file):
                image_file = os.path.abspath(image_file)
            print(f"   🖼️  Using image file: {image_file}")
        else:
            image_file = "safu.jpg"
            print(f"   ⚠️  Image file not found, will look for: {image_file}")
        
        # Debug: Show current working directory and file search paths
        print(f"\n🔍 FILE SEARCH DEBUG:")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Looking for video: {os.path.abspath(video_file)}")
        print(f"   Looking for image: {os.path.abspath(image_file)}")
        
        # Check if files exist with better debugging
        video_file_exists = os.path.exists(video_file)
        image_file_exists = os.path.exists(image_file)
        
        print(f"   Video file exists: {'✅ YES' if video_file_exists else '❌ NO'}")
        print(f"   Image file exists: {'✅ YES' if image_file_exists else '❌ NO'}")
        
        # List files in current directory and input directory for debugging
        try:
            current_files = os.listdir('.')
            video_files = [f for f in current_files if f.endswith('.mp4')]
            image_files = [f for f in current_files if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   Available video files: {video_files if video_files else 'None'}")
            print(f"   Available image files: {image_files if image_files else 'None'}")
            
            # Also check input directory
            if os.path.exists('./input'):
                try:
                    input_files = os.listdir('./input')
                    input_video_files = [f for f in input_files if f.endswith('.mp4')]
                    input_image_files = [f for f in input_files if f.endswith(('.jpg', '.jpeg', '.png'))]
                    print(f"   Input directory video files: {input_video_files if input_video_files else 'None'}")
                    print(f"   Input directory image files: {input_image_files if input_image_files else 'None'}")
                except Exception as e:
                    print(f"   Could not list input directory contents: {e}")
            else:
                print(f"   Input directory './input' does not exist")
        except Exception as e:
            print(f"   Could not list directory contents: {e}")
        
        # Always try to load video, using custom nodes if available
        if video_file_exists:
            print(f"\n🎥 Loading video: {video_file} - EXISTS ({os.path.getsize(video_file) / (1024**2):.2f} MB)")
            
            # Show VHS import status
            print(f"   🔍 VHS Import Status:")
            print(f"      VHS_AVAILABLE: {VHS_AVAILABLE}")
            print(f"      VHS_LoadVideoPath: {VHS_LoadVideoPath}")
            print(f"      VHS_LoadVideoUpload: {VHS_LoadVideoUpload}")
            
            if VHS_AVAILABLE and VHS_LoadVideoPath is not None:
                print("   🔧 Using custom node VHS LoadVideoPath for video loading...")
                try:
                    # Create instance of the custom node VHS class
                    video_loader = VHS_LoadVideoPath()
                    print(f"   🔍 VHS LoadVideoPath instance created: {type(video_loader).__name__}")
                    print(f"   🔍 VHS LoadVideoPath methods: {[m for m in dir(video_loader) if not m.startswith('_')][:10]}")
                    
                    # Check if load_video method exists
                    if hasattr(video_loader, 'load_video'):
                        print(f"   ✅ load_video method found")
                        
                        # Call the load_video method directly
                        vhs_loadvideo_1 = video_loader.load_video(
                            video=video_file,
                            force_rate=0,
                            custom_width=0,
                            custom_height=0,
                            frame_load_cap=0,
                            skip_first_frames=0,
                            select_every_nth=1
                        )
                        print("✅ Video loaded successfully using custom node VHS LoadVideoPath")
                        print(f"   🔍 Video data type: {type(vhs_loadvideo_1).__name__}")
                        if isinstance(vhs_loadvideo_1, (list, tuple)) and len(vhs_loadvideo_1) > 0:
                            print(f"   🔍 Video data length: {len(vhs_loadvideo_1)}")
                            if hasattr(vhs_loadvideo_1[0], 'shape'):
                                print(f"   🔍 Video tensor shape: {vhs_loadvideo_1[0].shape}")
                    else:
                        print(f"   ❌ load_video method not found in VHS_LoadVideoPath")
                        print(f"   🔍 Available methods: {[m for m in dir(video_loader) if not m.startswith('_')]}")
                        vhs_loadvideo_1 = None
                        
                except Exception as e:
                    print(f"❌ Error loading video with custom node VHS: {e}")
                    print("   Falling back to manual video loading...")
                    import traceback
                    traceback.print_exc()
                    vhs_loadvideo_1 = None
            else:
                print("   ⚠️  VHS not available - will use fallback video loading methods")
                vhs_loadvideo_1 = None
            
            # Fallback: Manual video loading if VHS fails or not available
            if vhs_loadvideo_1 is None:
                print("   🔧 Attempting manual video loading...")
                
                if TORCHVISION_AVAILABLE:
                    print("   🔧 Using torchvision for video loading...")
                    try:
                        # Load video using torchvision
                        video_tensor, audio, info = tvio.read_video(
                            video_file,
                            start_pts=0,
                            end_pts=None,
                            pts_unit='pts'
                        )
                        
                        # Convert to expected format: (frames, height, width, channels)
                        if len(video_tensor.shape) == 4:  # (frames, height, width, channels)
                            vhs_loadvideo_1 = [video_tensor, info['video_fps']]
                            print(f"✅ Video loaded with torchvision: {video_tensor.shape} frames at {info['video_fps']:.2f} fps")
                        else:
                            raise ValueError(f"Unexpected video tensor shape: {video_tensor.shape}")
                            
                    except Exception as tv_error:
                        print(f"   ❌ torchvision video loading failed: {tv_error}")
                        vhs_loadvideo_1 = None
                
                if vhs_loadvideo_1 is None and PIL_AVAILABLE:
                    print("   🔧 Trying PIL fallback for video loading...")
                    try:
                        # Load first frame as reference (simple fallback)
                        with Image.open(video_file) as img:
                            # Convert to tensor format
                            img_array = np.array(img)
                            if len(img_array.shape) == 3:  # (height, width, channels)
                                # Create a single frame video
                                video_tensor = torch.from_numpy(img_array).float() / 255.0
                                video_tensor = video_tensor.unsqueeze(0)  # Add frame dimension
                                vhs_loadvideo_1 = [video_tensor, 30.0]  # Assume 30 fps
                                print(f"✅ Video loaded with PIL fallback: {video_tensor.shape} (single frame)")
                            else:
                                raise ValueError(f"Unexpected image shape: {img_array.shape}")
                                
                    except Exception as pil_error:
                        print(f"   ❌ PIL fallback failed: {pil_error}")
                        vhs_loadvideo_1 = None
                
                if vhs_loadvideo_1 is None:
                    print("   ⚠️  All video loading methods failed, will create dummy data in Step 5")
        else:
            print(f"\n⏭️  Video file not found: {video_file}")
            print(f"   💡 Make sure '{video_file}' exists in the current directory")
            print(f"   💡 Current directory: {os.getcwd()}")
            print(f"   💡 Absolute path: {os.path.abspath(video_file)}")
            vhs_loadvideo_1 = None
        
        # Load reference image
        if image_file_exists:
            print(f"\n📷 Loading reference image: {image_file} - EXISTS ({os.path.getsize(image_file) / (1024**2):.2f} MB)")
            try:
                loadimage = LoadImage()
                loadimage_4 = loadimage.load_image(image=image_file)
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
        else:
            print(f"\n⏭️  Skipping reference image loading (image file not found)")
            print(f"   📁 Image file: {image_file} - NOT FOUND")
            print(f"   💡 Make sure '{image_file}' exists in the current directory")
            print(f"   💡 Current directory: {os.getcwd()}")
            print(f"   💡 Absolute path: {os.path.abspath(image_file)}")
            loadimage_4 = None
        
        # Debug: Show video loading results
        # if vhs_loadvideo_1:
        #     print(f"   Video type: {type(vhs_loadvideo_1)}")
        #     if isinstance(vhs_loadvideo_1, (list, tuple)) and len(vhs_loadvideo_1) > 0:
        #         print(f"   Video data type: {type(vhs_loadvideo_1[0])}")
        #         if hasattr(vhs_loadvideo_1[0], 'shape'):
        #         print(f"   Video shape: {vhs_loadvideo_1[0].shape}")
        #     if len(vhs_loadvideo_1) > 1:
        #         print(f"   Frame count: {vhs_loadvideo_1[1]}")
        # else:
        #     print("   Video: Not loaded")

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
            # model_monitor.start_monitoring("vae_loading")
            vaeloader = VAELoader()
            vaeloader_7 = vaeloader.load_vae(vae_name="vae.safetensors")
            # model_monitor.end_monitoring("vae_loading", vaeloader_7, "VAE")
            print("✅ VAE loaded successfully")
        except Exception as e:
            print(f"❌ ERROR loading VAE: {e}")
            vaeloader_7 = None

        # Load UNET with monitoring
        try:
            # model_monitor.start_monitoring("unet_loading")
            unetloader = UNETLoader()
            unetloader_27 = unetloader.load_unet(
                unet_name="model.safetensors", weight_dtype="default"
            )
            # model_monitor.end_monitoring("unet_loading", unetloader_27, "UNET")
            print("✅ UNET loaded successfully")
        except Exception as e:
            print(f"❌ ERROR loading UNET: {e}")
            unetloader_27 = None

        # Load CLIP with monitoring
        try:
            # model_monitor.start_monitoring("clip_loading")
            cliploader = CLIPLoader()
            cliploader_23 = cliploader.load_clip(
                clip_name="clip.safetensors", type="wan", device="default"
            )
            # model_monitor.end_monitoring("clip_loading", cliploader_23, "CLIP")
            print("✅ CLIP loaded successfully")
        except Exception as e:
            print(f"❌ ERROR loading CLIP: {e}")
            cliploader_23 = None
        
        # Print comprehensive summary of all model loading steps
        # model_monitor.print_summary()
        
        print("✅ Step 1 completed: Model Loading")
        # === STEP 1 END: MODEL LOADING ===
        
        # Debug execution stopped - continuing to step 2
        # print("\n🔍 Step 1 debugging complete - continuing to LoRA application...")

        # === STEP 2 START: LORA APPLICATION ===
        print("2. Applying LoRA...")
        
        # Capture baseline state before LoRA application
        print("\n🔍 CAPTURING BASELINE STATE BEFORE LORA APPLICATION...")
        unet_model_baseline = get_value_at_index(unetloader_27, 0)
        clip_model_baseline = get_value_at_index(cliploader_23, 0)
        
        # Check LoRA file status
        lora_filename = "models/loras/lora.safetensors"
        lora_file_exists = os.path.exists(lora_filename)
        lora_file_size = os.path.getsize(lora_filename) if lora_file_exists else 0
        
        print(f"   📁 LoRA File: {lora_filename}")
        print(f"   📁 File Exists: {'✅ YES' if lora_file_exists else '❌ NO'}")
        if lora_file_exists:
            print(f"   📁 File Size: {lora_file_size / (1024**2):.2f} MB")
        else:
            print(f"   ⚠️  LoRA file not found - will attempt to continue but may fail")
            print(f"   💡 Make sure 'lora.safetensors' exists in the current directory")
        
        # lora_baseline = model_monitor.capture_lora_baseline(
        #     unet_model_baseline, 
        #     clip_model_baseline
        # )
        
        # print(f"   ✅ UNET Baseline captured - ID: {lora_baseline['unet']['model_id']}, Patches: {lora_baseline['unet']['patches_count']}")
        # print(f"   ✅ CLIP Baseline captured - ID: {lora_baseline['clip']['model_id']}, Patches: {lora_baseline['clip']['patcher_patches_count']}")
        
        # Display baseline memory information
        # print(f"\n   💾 BASELINE MEMORY STATE:")
        # print(f"      🖥️  RAM: {lora_baseline['ram']['used_mb']:.1f} MB used / {lora_baseline['ram']['total_mb']:.1f} MB total ({lora_baseline['ram']['percent_used']:.1f}%)")
        # if lora_baseline['gpu']:
        #     print(f"      🎮 GPU: {lora_baseline['gpu']['allocated'] / (1024**2):.1f} MB allocated / {lora_baseline['gpu']['total'] / (1024**2):.1f} MB total")
        #     print(f"      🎮 GPU Device: {lora_baseline['gpu']['device_name']}")
        
        # Check if we can proceed with LoRA application
        if not lora_file_exists:
            print(f"\n❌ CANNOT PROCEED: LoRA file '{lora_filename}' not found!")
            print(f"🔍 Please ensure the LoRA file exists before running this script.")
            print(f"📁 Expected location: {os.path.abspath(lora_filename)}")
            print(f"💡 The LoRA file should be in: models/loras/lora.safetensors")
            return
        
        # Apply LoRA with monitoring
        try:
            print("\n🔧 APPLYING LORA TO MODELS...")
            
            # Start monitoring peak memory during LoRA application
            # model_monitor.start_monitoring("lora_application")
            
            # Update peak memory before LoRA application
            # model_monitor.update_peak_memory()
            
            loraloader = LoraLoader()
            loraloader_24 = loraloader.load_lora(
                lora_name="lora.safetensors",
                strength_model=0.5000000000000001,
                strength_clip=1,
                model=unet_model_baseline,
                clip=clip_model_baseline,
            )
            
            # Update peak memory after LoRA application
            # model_monitor.update_peak_memory()
            
            # Extract modified models from result
            modified_unet = get_value_at_index(loraloader_24, 0)
            modified_clip = get_value_at_index(loraloader_24, 1)
            
            # Update peak memory after model extraction
            # model_monitor.update_peak_memory()
            
            # End monitoring and get peak memory summary
            # elapsed_time = model_monitor.end_monitoring("lora_application", loraloader_24, "LoRA_Result")
            # peak_memory_summary = model_monitor.get_peak_memory_summary()
            
            # Analyze LoRA application results
            # print("\n🔍 ANALYZING LORA APPLICATION RESULTS...")
            # lora_analysis = model_monitor.analyze_lora_application_results(
            #     lora_baseline, 
            #     modified_unet, 
            #     modified_clip, 
            #     loraloader_24
            # )
            
            # Print comprehensive analysis
            # model_monitor.print_lora_analysis_summary(lora_analysis)
            
            # Print peak memory information
            # print(f"\n📊 PEAK MEMORY DURING LORA APPLICATION:")
            # print(f"   🖥️  RAM Peak: {peak_memory_summary['ram_peak_mb']:.1f} MB")
            # print(f"   🎮 GPU Allocated Peak: {peak_memory_summary['gpu_allocated_peak_mb']:.1f} MB")
            # print(f"   🎮 GPU Reserved Peak: {peak_memory_summary['gpu_reserved_peak_mb']:.1f} MB")
            # print(f"   ⏱️  Total Time: {elapsed_time:.3f} seconds")
            
            print("✅ Step 2 completed: LoRA Application")
            
        except Exception as e:
            print(f"❌ ERROR during LoRA application: {e}")
            print("🔍 LoRA application failed - check error details above")
            loraloader_24 = None
            
        # === STEP 2 END: LORA APPLICATION ===

        # === STEP 3 START: TEXT ENCODING ===
        print("3. Encoding text prompts...")
        
        # Initialize text encoding analysis variable
        text_encoding_analysis = None
        
        # Capture baseline state before text encoding
        # print("\n🔍 CAPTURING BASELINE STATE BEFORE TEXT ENCODING...")
        
        # Get the modified models from LoRA application
        if 'modified_unet' in locals() and 'modified_clip' in locals():
            # try:
            #     text_encoding_baseline = model_monitor.capture_text_encoding_baseline(
            #         modified_unet, 
            #         modified_clip
            #     )
            #     
            #     print(f"   ✅ UNET Baseline captured - ID: {text_encoding_baseline.get('unet', {}).get('model_id', 'N/A')}")
            #     print(f"   ✅ CLIP Baseline captured - ID: {text_encoding_baseline.get('clip', {}).get('model_id', 'N/A')}")
            #     
            #     # Display baseline memory information
            #     print(f"\n   💾 BASELINE MEMORY STATE:")
            #     ram_info = text_encoding_baseline.get('ram', {})
            #     print(f"      🖥️  RAM: {ram_info.get('used_mb', 0):.1f} MB used / {ram_info.get('total_mb', 0):.1f} MB total ({ram_info.get('percent_used', 0):.1f}%)")
            #     gpu_info = text_encoding_baseline.get('gpu')
            #     if gpu_info:
            #         print(f"      🎮 GPU: {gpu_info.get('allocated', 0) / (1024**2):.1f} MB allocated / {gpu_info.get('total', 0) / (1024**2):.1f} MB total")
            # except Exception as e:
            #     print(f"❌ ERROR during baseline capture: {e}")
            #     print("🔍 Baseline capture failed - will continue with limited monitoring")
            #     text_encoding_baseline = None
            pass
        else:
            print("❌ ERROR: Modified models not available from LoRA application")
            print("🔍 Cannot proceed with text encoding - check LoRA application step")
            return
        
        # Define text prompts for encoding
        positive_prompt = "a beautiful landscape with mountains and trees, high quality, detailed"
        negative_prompt = "blurry, low quality, distorted, ugly"
        
        print(f"\n📝 TEXT PROMPTS FOR ENCODING:")
        print(f"   Positive: '{positive_prompt}'")
        print(f"   Negative: '{negative_prompt}'")
        
        # Apply text encoding with monitoring
        try:
            print("\n🔧 ENCODING TEXT PROMPTS...")
            
            # Start monitoring peak memory during text encoding
            # model_monitor.start_monitoring("text_encoding")
            
            # Update peak memory before text encoding
            # model_monitor.update_peak_memory()
            
            # Create text encoder and encode positive prompt
            print("   🔤 Encoding positive prompt...")
            cliptextencode = CLIPTextEncode()
            positive_cond_tuple = cliptextencode.encode(modified_clip, positive_prompt)
            
            # Debug: Show what was returned
            print(f"   🔍 Positive encoding result type: {type(positive_cond_tuple).__name__}")
            if positive_cond_tuple is not None:
                print(f"   🔍 Positive encoding result length: {len(positive_cond_tuple) if hasattr(positive_cond_tuple, '__len__') else 'N/A'}")
                print(f"   🔍 Positive encoding result repr: {repr(positive_cond_tuple)[:200]}...")
            
            # Update peak memory after positive encoding
            # model_monitor.update_peak_memory()
            
            # Encode negative prompt
            print("   🔤 Encoding negative prompt...")
            negative_cond_tuple = cliptextencode.encode(modified_clip, negative_prompt)
            
            # Debug: Show what was returned
            print(f"   🔍 Negative encoding result type: {type(negative_cond_tuple).__name__}")
            if negative_cond_tuple is not None:
                print(f"   🔍 Negative encoding result length: {len(negative_cond_tuple) if hasattr(negative_cond_tuple, '__len__') else 'N/A'}")
                print(f"   🔍 Negative encoding result repr: {repr(negative_cond_tuple)[:200]}...")
            
            # Update peak memory after negative encoding
            # model_monitor.update_peak_memory()
            
            # Extract conditioning from tuples - try different extraction strategies
            # print("   🔍 Extracting conditioning tensors...")
            
            # Strategy 1: Try direct extraction
            # positive_cond = positive_cond_tuple
            # negative_cond = negative_cond_tuple
            
            # Strategy 2: If it's a list/tuple, try first element
            # if isinstance(positive_cond_tuple, (list, tuple)) and len(positive_cond_tuple) > 0:
            #     positive_cond = positive_cond_tuple[0]
            #     print(f"   🔍 Extracted positive_cond from positive_cond_tuple[0]")
            
            # if isinstance(negative_cond_tuple, (list, tuple)) and len(negative_cond_tuple) > 0:
            #     negative_cond = negative_cond_tuple[0]
            #     print(f"   🔍 Extracted negative_cond from negative_cond_tuple[0]")
            
            # Strategy 3: If it's a dict, look for tensor-like objects
            # if isinstance(positive_cond, dict):
            #     for key, value in positive_cond.items():
            #         if hasattr(value, 'shape'):
            #             positive_cond = value
            #             print(f"   🔍 Found positive tensor in dict['{key}']")
            #             break
            
            # if isinstance(negative_cond, dict):
            #     for key, value in negative_cond.items():
            #         if hasattr(value, 'shape'):
            #             negative_cond = value
            #             print(f"   🔍 Found negative tensor in dict['{key}']")
            #             break
            
            # Strategy 4: Recursive deep search for nested structures
            # if (positive_cond is None or not hasattr(positive_cond, 'shape')) and positive_cond_tuple is not None:
            #     print(f"   🔍 Strategy 4: Recursive search for positive tensor...")
            #     positive_cond, reason = model_monitor._extract_tensor_recursively(positive_cond_tuple)
            #     if positive_cond is not None:
            #         print(f"   ✅ Strategy 4 SUCCESS for positive: {reason}")
            #     else:
            #         print(f"   ❌ Strategy 4 FAILED for positive: {reason}")
            
            # if (negative_cond is None or not hasattr(negative_cond, 'shape')) and negative_cond_tuple is not None:
            #     print(f"   🔍 Strategy 4: Recursive search for negative tensor...")
            #     negative_cond, reason = model_monitor._extract_tensor_recursively(negative_cond_tuple)
            #     if negative_cond is not None:
            #         print(f"   ✅ Strategy 4 SUCCESS for negative: {reason}")
            #     else:
            #         print(f"   ❌ Strategy 4 FAILED for negative: {reason}")
            
            # print(f"   🔍 Final positive_cond type: {type(positive_cond).__name__}")
            # print(f"   🔍 Final negative_cond type: {type(negative_cond).__name__}")
            
            # if positive_cond is not None and hasattr(positive_cond, 'shape'):
            #     print(f"   ✅ Positive conditioning tensor found: {positive_cond.shape}")
            # else:
            #     print(f"   ❌ Positive conditioning tensor not found or invalid")
            
            # if negative_cond is not None and hasattr(negative_cond, 'shape'):
            #     print(f"   ✅ Negative conditioning tensor found: {negative_cond.shape}")
            # else:
            #     print(f"   ❌ Negative conditioning tensor not found or invalid")
            
            # End monitoring and get peak memory summary
            # elapsed_time = model_monitor.end_monitoring("text_encoding", [positive_cond, negative_cond], "TextEncoding_Result")
            # peak_memory_summary = model_monitor.get_peak_memory_summary()
            
            # Analyze text encoding results
            print("\n🔍 ANALYZING TEXT ENCODING RESULTS...")
            try:
                # text_encoding_analysis = model_monitor.analyze_text_encoding_results(
                #     text_encoding_baseline, 
                #     positive_cond, 
                #     negative_cond, 
                #     positive_prompt,
                #     negative_prompt,
                #     elapsed_time
                # )
                
                # Print comprehensive analysis
                # model_monitor.print_text_encoding_analysis_summary(text_encoding_analysis)
                print("   ℹ️  Text encoding analysis disabled for debugging")
                text_encoding_analysis = None
            except Exception as e:
                print(f"❌ ERROR during text encoding analysis: {e}")
                print("🔍 Text encoding analysis failed - will show error summary")
                print("🔍 Error details:", str(e))
                import traceback
                traceback.print_exc()
                text_encoding_analysis = None
            
            # Print peak memory information
            print(f"\n📊 PEAK MEMORY DURING TEXT ENCODING:")
            # if peak_memory_summary:
            #     print(f"   🖥️  RAM Peak: {peak_memory_summary.get('ram_peak_mb', 0):.1f} MB")
            #     print(f"   🎮 GPU Allocated Peak: {peak_memory_summary.get('gpu_allocated_peak_mb', 0):.1f} MB")
            #     print(f"   🎮 GPU Reserved Peak: {peak_memory_summary.get('gpu_reserved_peak_mb', 0):.1f} MB")
            # else:
            #     print("   ❌ ERROR: Peak memory summary not available")
            # if elapsed_time is not None and isinstance(elapsed_time, (int, float)):
            #     print(f"   ⏱️  Total Time: {elapsed_time:.3f} seconds")
            # else:
            #     print("   ⏱️  Total Time: N/A")
            print("   ℹ️  Peak memory monitoring disabled for debugging")
            
            print("✅ Step 3 completed: Text Encoding (monitoring disabled)")
            
            # Check output directory contents
            # print(f"\n📁 CHECKING OUTPUT DIRECTORY CONTENTS:")
            # output_dir = "./W_out/step3"
            # if os.path.exists(output_dir):
            #     try:
            #         files = os.listdir(output_dir)
            #         if files:
            #         print(f"   📂 Directory '{output_dir}' contains {len(files)} files:")
            #         for file in sorted(files):
            #         filepath = os.path.join(output_dir, file)
            #         if os.path.isfile(filepath):
            #         size_mb = os.path.getsize(filepath) / (1024**2)
            #         print(f"      📄 {file} ({size_mb:.4f} MB)")
            #         else:
            #         print(f"      📁 {file} (directory)")
            #         else:
            #         print(f"   📂 Directory '{output_dir}' is empty")
            #         except Exception as e:
            #         print(f"   ⚠️  Could not read directory '{output_dir}': {e}")
            #         else:
            #         print(f"   ❌ Directory '{output_dir}' does not exist")
            #         print(f"   💡 Current working directory: {os.getcwd()}")
            #         print(f"   💡 Absolute path: {os.path.abspath(output_dir)}")
            
        except Exception as e:
            print(f"❌ ERROR during text encoding: {e}")
            print("🔍 Text encoding failed - check error details above")
            # positive_cond = None
            # negative_cond = None
            text_encoding_analysis = None
            
        # === STEP 3 END: TEXT ENCODING ===

        # === STEP 4 START: MODEL SAMPLING ===
        print("4. Applying ModelSamplingSD3 to UNET...")
        
        # Get the modified models from LoRA application
        if 'modified_unet' in locals() and 'modified_clip' in locals():
            try:
                # Apply ModelSamplingSD3 (using existing node)
                try:
                    # Import ModelSamplingSD3 if available
                    from comfy_extras.nodes_model_advanced import ModelSamplingSD3
                    print("   ✅ ModelSamplingSD3 imported successfully")
                    
                    # Create and apply the sampling modification
                    model_sampling = ModelSamplingSD3()
                    modified_unet_sampled_tuple = model_sampling.patch(modified_unet, shift=8.0)
                    
                    # Extract the actual model from the tuple
                    if isinstance(modified_unet_sampled_tuple, (list, tuple)) and len(modified_unet_sampled_tuple) > 0:
                        modified_unet_sampled = modified_unet_sampled_tuple[0]
                    else:
                        modified_unet_sampled = modified_unet_sampled_tuple
                    
                    print("   ✅ ModelSamplingSD3 applied successfully")
                    
                except ImportError:
                    print("   ⚠️  ModelSamplingSD3 not available, using fallback approach")
                    # Fallback: just clone the model to simulate the effect
                    modified_unet_sampled = modified_unet
                    print("   ℹ️  Using fallback model cloning (no actual sampling applied)")
                
                print("✅ Step 4 completed: Model Sampling")
                
            except Exception as e:
                print(f"❌ ERROR during model sampling: {e}")
                print("🔍 Model sampling failed - check error details above")
                modified_unet_sampled = None
                return
        else:
            print("❌ ERROR: Modified models not available from LoRA application")
            print("🔍 Cannot proceed with model sampling - check LoRA application step")
            return
            
        # === STEP 4 END: MODEL SAMPLING ===

        # === STEP 5 START: INITIAL LATENT GENERATION ===
        print("5. Preparing inputs for WanVaceToVideo node...")
        
        # Initialize latent generation analysis variable
        latent_generation_analysis = None
        
        # Capture baseline state before latent generation
        print("\n🔍 CAPTURING BASELINE STATE BEFORE LATENT GENERATION...")
        
        # Get the modified models from previous steps
        if 'modified_unet_sampled' in locals() and 'vaeloader_7' in locals():
            try:
                # Capture comprehensive baseline for Step 5
                latent_generation_baseline = model_monitor.capture_latent_generation_baseline(
                    vaeloader_7,  # VAE model
                    None  # Will create dummy video data
                )
                
                print(f"   ✅ VAE Baseline captured - Model: {type(vaeloader_7).__name__}")
                print(f"   ✅ GPU Baseline captured - Allocated: {latent_generation_baseline['gpu']['allocated'] / (1024**2):.1f} MB, Reserved: {latent_generation_baseline['gpu']['reserved'] / (1024**2):.1f} MB")
                
                # Display baseline memory information
                print(f"\n   💾 BASELINE MEMORY STATE:")
                print(f"      🎮 GPU: {latent_generation_baseline['gpu']['allocated'] / (1024**2):.1f} MB allocated / {latent_generation_baseline['gpu']['reserved'] / (1024**2):.1f} MB reserved")
                print(f"      🎮 Available VRAM: {(latent_generation_baseline['gpu']['total'] - latent_generation_baseline['gpu']['reserved']) / (1024**2):.1f} MB")
                print(f"      🎮 Total VRAM: {latent_generation_baseline['gpu']['total'] / (1024**2):.1f} MB")
                print(f"      🎮 Device: {latent_generation_baseline['gpu']['device_name']}")
                
            except Exception as e:
                print(f"❌ ERROR during baseline capture: {e}")
                print("🔍 Baseline capture failed - will continue with limited monitoring")
                latent_generation_baseline = None
        else:
            print("❌ ERROR: Required models not available from previous steps")
            print("🔍 Cannot proceed with latent generation - check previous steps")
            return
        
        # Prepare inputs for WanVaceToVideo node (like ComfyUI web workflow)
        print("\n🎬 PREPARING INPUTS FOR WANVACETOVIDEO NODE...")
        try:
            # Check if we have actual video data from Step 1 (this will be the control video)
            if 'vhs_loadvideo_1' in locals() and vhs_loadvideo_1 is not None:
                print("   🎥 Using actual video data from Step 1 as CONTROL VIDEO...")
                actual_video_data = vhs_loadvideo_1
                
                # Extract video dimensions from actual data
                if isinstance(actual_video_data, (list, tuple)) and len(actual_video_data) > 0:
                    video_tensor = actual_video_data[0]
                    if hasattr(video_tensor, 'shape'):
                        video_shape = video_tensor.shape
                        print(f"   📐 Actual Video Shape: {video_shape}")
                        
                        # Determine dimensions based on actual video
                        if len(video_shape) == 4:  # (frames, height, width, channels)
                            actual_length = video_shape[0]
                            actual_height = video_shape[1]
                            actual_width = video_shape[2]
                            actual_channels = video_shape[3]
                        elif len(video_shape) == 3:  # (height, width, channels) - single frame
                            actual_length = 1
                            actual_height = video_shape[0]
                            actual_width = video_shape[1]
                            actual_channels = video_shape[2]
                        else:
                            raise ValueError(f"Unexpected video shape: {video_shape}")
                        
                        print(f"   📐 Actual Video Dimensions: {actual_length} frames, {actual_height}x{actual_width}, {actual_channels} channels")
                        
                        # Use actual video data as control video
                        control_video = video_tensor
                        
                        # For reference image, use the first frame of the video
                        reference_image = video_tensor[0] if actual_length > 0 else video_tensor
                        
                        print("   ✅ Using actual video data as control video, first frame as reference")
                        print("   💡 This matches the ComfyUI web workflow: WanVaceToVideo node will handle VAE encoding")
                        use_actual_data = True
                    else:
                        raise ValueError("Video tensor has no shape attribute")
                else:
                    raise ValueError("Video data structure is invalid")
                    
            # Check if we have actual image data from Step 1 (this will be used as reference image)
            elif 'loadimage_4' in locals() and loadimage_4 is not None:
                print("   🖼️  Using actual image data from Step 1 as REFERENCE IMAGE...")
                actual_image_data = loadimage_4
                
                # Extract image dimensions from actual data
                if isinstance(actual_image_data, (list, tuple)) and len(actual_image_data) > 0:
                    image_tensor = actual_image_data[0]
                    if hasattr(image_tensor, 'shape'):
                        image_shape = image_tensor.shape
                        print(f"   📐 Actual Image Shape: {image_shape}")
                        
                        # Determine dimensions based on actual image
                        if len(image_shape) == 3:  # (height, width, channels)
                            actual_length = 1  # Single frame
                            actual_height = image_shape[0]
                            actual_width = image_shape[1]
                            actual_channels = image_shape[2]
                        elif len(image_shape) == 4:  # (batch, height, width, channels)
                            actual_length = 1  # Single frame
                            actual_height = image_shape[1]
                            actual_width = image_shape[2]
                            actual_channels = image_shape[3]
                        else:
                            raise ValueError(f"Unexpected image shape: {image_shape}")
                        
                        print(f"   📐 Actual Image Dimensions: {actual_length} frame, {actual_height}x{actual_width}, {actual_channels} channels")
                        
                        # IMPORTANT: This is the REFERENCE IMAGE for the WanVaceToVideo node
                        reference_image = image_tensor
                        
                        # Create control video by repeating the reference image (this is what the node expects)
                        # The WanVaceToVideo node needs both inputs:
                        # - reference_image: single image (this)
                        # - control_video: video sequence (we'll create from this image)
                        control_video = image_tensor.unsqueeze(0).repeat(16, 1, 1, 1) if len(image_tensor.shape) == 3 else image_tensor.repeat(16, 1, 1, 1)
                        
                        print("   ✅ Using actual image as reference, creating control video from reference")
                        print("   💡 This matches the ComfyUI web workflow: WanVaceToVideo node will handle VAE encoding")
                        use_actual_data = True
                    else:
                        raise ValueError("Image tensor has no shape attribute")
                else:
                    raise ValueError("Image data structure is invalid")
                    
            else:
                print("   ⚠️  No actual video/image data available, creating dummy data...")
                # Create dummy video data with realistic dimensions
                actual_length = 16  # 16 frames
                actual_height = 512  # 512 pixels
                actual_width = 512   # 512 pixels
                actual_channels = 3  # RGB
                
                print(f"   📐 Dummy Video Dimensions: {actual_length} frames, {actual_height}x{actual_width}, {actual_channels} channels")
                
                # Create dummy control video tensor
                control_video = torch.randn((actual_length, actual_height, actual_width, actual_channels), device='cuda') * 0.5 + 0.5
                print(f"   🎥 Dummy control video created: {control_video.shape}")
                
                # Create dummy reference image tensor
                reference_image = torch.randn((1, actual_height, actual_width, actual_channels), device='cuda') * 0.5 + 0.5
                print(f"   🖼️  Dummy reference image created: {reference_image.shape}")
                
                print("   ✅ Dummy video data created successfully")
                use_actual_data = False
            
        except Exception as e:
            print(f"❌ ERROR preparing video data: {e}")
            print("🔍 Cannot proceed with latent generation")
            return
        
        # Monitor the input preparation process
        try:
            print("\n🔧 MONITORING INPUT PREPARATION FOR WANVACETOVIDEO NODE...")
            
            # Start monitoring peak memory during input preparation
            model_monitor.start_monitoring("input_preparation_step5")
            
            # Update peak memory before input preparation
            model_monitor.update_peak_memory()
            
            # Display GPU memory before input preparation
            current_gpu_allocated = torch.cuda.memory_allocated() / (1024**2)
            current_gpu_reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"   GPU Memory Before: {current_gpu_allocated:.1f} MB allocated, {current_gpu_reserved:.1f} MB reserved")
            
            # Show summary of what data was prepared
            print(f"\n📊 INPUT PREPARATION SUMMARY:")
            if 'use_actual_data' in locals() and use_actual_data:
                if 'vhs_loadvideo_1' in locals() and vhs_loadvideo_1 is not None:
                    print(f"   🎯 Data Source: ACTUAL VIDEO data from Step 1")
                else:
                    print(f"   🎯 Data Source: ACTUAL IMAGE data from Step 1 (converted to video)")
                print(f"   📐 Control Video Dimensions: {actual_length} frames, {actual_height}x{actual_width}, {actual_channels} channels")
                print(f"   📐 Reference Image Dimensions: {reference_image.shape}")
                print(f"   💾 Control Video Size: {(actual_length * actual_height * actual_width * actual_channels * 4) / (1024**2):.2f} MB")
                print(f"   💾 Reference Image Size: {(reference_image.numel() * reference_image.element_size()) / (1024**2):.2f} MB")
            else:
                print(f"   🎯 Data Source: DUMMY data (no actual files found)")
                print(f"   📐 Input Dimensions: {actual_length} frames, {actual_height}x{actual_width}, {actual_channels} channels")
                print(f"   💾 Input Size: {(actual_length * actual_height * actual_width * actual_channels * 4) / (1024**2):.2f} MB")
            
            print(f"   🔧 Input Strategy: {'Actual Data' if use_actual_data else 'Dummy Data'}")
            print(f"   📐 Control Video Shape: {control_video.shape}")
            print(f"   📐 Reference Image Shape: {reference_image.shape}")
            
            # Simulate what the WanVaceToVideo node would do (just for monitoring purposes)
            print(f"\n💡 SIMULATING WANVACETOVIDEO NODE INPUTS:")
            print(f"   🎯 Reference Image: {reference_image.shape} - Ready for node")
            print(f"   🎯 Control Video: {control_video.shape} - Ready for node")
            print(f"   💡 Note: In ComfyUI web, the WanVaceToVideo node would now:")
            print(f"      - Take these inputs")
            print(f"      - Apply VAE encoding internally")
            print(f"      - Produce latents automatically")
            print(f"      - No manual VAE encoding needed!")
            
            # Check immediate GPU impact
            gpu_after_preparation = {
                'allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'reserved_mb': torch.cuda.memory_reserved() / (1024**2)
            }
            
            print(f"   GPU Memory After Preparation: {gpu_after_preparation['allocated_mb']:.1f} MB allocated, {gpu_after_preparation['reserved_mb']:.1f} MB reserved")
            
            # Calculate immediate GPU changes
            immediate_gpu_changes = {
                'allocated_change_mb': gpu_after_preparation['allocated_mb'] - current_gpu_allocated,
                'reserved_change_mb': gpu_after_preparation['reserved_mb'] - current_gpu_reserved
            }
            
            print(f"   📊 Immediate GPU Impact: {immediate_gpu_changes['allocated_change_mb']:+.1f} MB allocated, {immediate_gpu_changes['reserved_change_mb']:+.1f} MB reserved")
            
            # Update peak memory after input preparation
            model_monitor.update_peak_memory()
            
            # End monitoring and get peak memory summary
            elapsed_time = model_monitor.end_monitoring("input_preparation_step5", [control_video, reference_image], "InputPreparation_Result")
            peak_memory_summary = model_monitor.get_peak_memory_summary()
            
            # Create a simple analysis result for monitoring
            latent_generation_analysis = {
                'encoding_success': True,  # Inputs are ready
                'elapsed_time': elapsed_time if elapsed_time is not None else 0.0,
                'strategy_used': 'Input Preparation Only',
                'latent_analysis': {
                    'status': 'success',
                    'latent_type': 'Inputs Ready for WanVaceToVideo Node',
                    'shape': f"Control: {control_video.shape}, Reference: {reference_image.shape}",
                    'dtype': f"Control: {control_video.dtype}, Reference: {reference_image.dtype}",
                    'device': f"Control: {control_video.device}, Reference: {reference_image.device}",
                    'size_mb': (control_video.numel() * control_video.element_size() + reference_image.numel() * reference_image.element_size()) / (1024**2),
                    'num_elements': control_video.numel() + reference_image.numel()
                },
                'performance_analysis': {
                    'total_pixels': actual_length * actual_height * actual_width,
                    'total_mb': (actual_length * actual_height * actual_width * actual_channels * 4) / (1024**2),
                    'elapsed_time': elapsed_time if elapsed_time is not None else 0.0,
                    'strategy_used': 'Input Preparation Only'
                },
                'memory_impact': {
                    'ram': {'used_change_mb': 0, 'available_change_mb': 0},
                    'gpu': immediate_gpu_changes if immediate_gpu_changes is not None else {'allocated_change_mb': 0, 'reserved_change_mb': 0}
                },
                'immediate_gpu_changes': immediate_gpu_changes if immediate_gpu_changes is not None else {'allocated_change_mb': 0, 'reserved_change_mb': 0},
                'peak_memory': peak_memory_summary if peak_memory_summary is not None else {}
            }
            
            # Print comprehensive analysis
            model_monitor.print_latent_generation_analysis_summary(latent_generation_analysis)
            
            # Print peak memory information
            print(f"\n📊 PEAK MEMORY DURING INPUT PREPARATION:")
            if peak_memory_summary:
                print(f"   🖥️  RAM Peak: {peak_memory_summary.get('ram_peak_mb', 0):.1f} MB")
                print(f"   🎮 GPU Allocated Peak: {peak_memory_summary.get('gpu_allocated_peak_mb', 0):.1f} MB")
                print(f"   🎮 GPU Reserved Peak: {peak_memory_summary.get('gpu_reserved_peak_mb', 0):.1f} MB")
            else:
                print("   ❌ ERROR: Peak memory summary not available")
            if elapsed_time is not None and isinstance(elapsed_time, (int, float)):
                print(f"   ⏱️  Total Time: {elapsed_time:.3f} seconds")
            else:
                print("   ⏱️  Total Time: N/A")
            
            print("✅ Step 5 completed: Input Preparation for WanVaceToVideo Node")
            
        except Exception as e:
            print(f"❌ ERROR during input preparation: {e}")
            print("🔍 Input preparation failed - check error details above")
            latent_generation_analysis = None
            
        # === STEP 5 END: INITIAL LATENT GENERATION ===

        # Stop execution after step 5 for debugging purposes
        print("\n🛑 STOPPING EXECUTION AFTER STEP 5 (INPUT PREPARATION)")
        print("🔍 All input preparation debugging information has been displayed above.")
        print("📊 Check the monitoring data above to analyze input preparation performance.")
        print("🔍 Step 1: Model Loading - COMPLETED (monitoring disabled)")
        print("🔍 Step 2: LoRA Application - COMPLETED (monitoring disabled)")
        print("🔍 Step 3: Text Encoding - COMPLETED (monitoring disabled)")
        print("🔍 Step 4: Model Sampling - COMPLETED (monitoring disabled)")
        print("🔍 Step 5: Input Preparation for WanVaceToVideo Node - COMPLETED (FULL monitoring)")
        print("🔍 Steps 6-9: SKIPPED for debugging purposes")
        
        # === FINAL MONITORING SUMMARY ===
        print("\n" + "="*80)
        print("🔍 FINAL WORKFLOW MONITORING SUMMARY")
        print("="*80)
        
        # Use comprehensive summary if analyses are available
        if 'latent_generation_analysis' in locals() and latent_generation_analysis is not None:
            try:
                model_monitor.print_comprehensive_summary(None, None, None, latent_generation_analysis)
            except Exception as summary_error:
                print(f"\n⚠️  Error in comprehensive summary: {summary_error}")
                print("   Showing basic completion status instead...")
                print(f"\n🔍 BASIC WORKFLOW COMPLETION STATUS:")
                print(f"   ✅ Step 1: Model Loading - COMPLETED")
                print(f"   ✅ Step 2: LoRA Application - COMPLETED")
                print(f"   ✅ Step 3: Text Encoding - COMPLETED")
                print(f"   ✅ Step 4: Model Sampling - COMPLETED")
                print(f"   ✅ Step 5: Input Preparation - COMPLETED")
        else:
            print("\n⚠️  Input preparation analysis not available - step 5 may not have completed")
            print(f"\n🔍 BASIC WORKFLOW COMPLETION STATUS:")
            print(f"   ✅ Step 1: Model Loading - COMPLETED")
            print(f"   ✅ Step 2: LoRA Application - COMPLETED")
            print(f"   ✅ Step 3: Text Encoding - COMPLETED")
            print(f"   ✅ Step 4: Model Sampling - COMPLETED")
            print(f"   ❌ Step 5: Input Preparation - FAILED or INCOMPLETE")
        
        print("="*80)
        
        return


if __name__ == "__main__":
    main()
