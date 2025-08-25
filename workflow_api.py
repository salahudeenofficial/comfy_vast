import os
import random
import sys
import time
import psutil
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np

# Direct imports for VHS functionality
VHS_AVAILABLE = False
VHS_LoadVideoUpload = None
VHS_LoadVideoPath = None

def attempt_vhs_import():
    """Attempt to import VHS classes with multiple fallback strategies"""
    global VHS_AVAILABLE, VHS_LoadVideoUpload, VHS_LoadVideoPath
    
    # Strategy 1: Try to find ComfyUI directory and import from there
    comfyui_path = find_path("ComfyUI")
    if comfyui_path and os.path.isdir(comfyui_path):
        # Try multiple import paths
        import_paths = [
            os.path.join(comfyui_path, 'custom_nodes', 'comfyui-videohelpersuite'),
            os.path.join(comfyui_path, 'custom_nodes', 'VideoHelperSuite'),
            os.path.join(comfyui_path, 'custom_nodes', 'videohelpersuite'),
            os.path.join(os.path.dirname(__file__), 'custom_nodes', 'comfyui-videohelpersuite'),
            os.path.join(os.path.dirname(__file__), 'custom_nodes', 'VideoHelperSuite'),
            os.path.join(os.path.dirname(__file__), 'custom_nodes', 'videohelpersuite')
        ]
        
        for import_path in import_paths:
            if os.path.exists(import_path):
                try:
                    if import_path not in sys.path:
                        sys.path.insert(0, import_path)
                    
                    # Try different import patterns
                    try:
                        from videohelpersuite.load_video_nodes import LoadVideoUpload, LoadVideoPath
                        print(f"✅ Successfully imported VHS from: {import_path}")
                        VHS_AVAILABLE = True
                        return True
                    except ImportError:
                        pass
                        
                    try:
                        from load_video_nodes import LoadVideoUpload, LoadVideoPath
                        print(f"✅ Successfully imported VHS from: {import_path}")
                        VHS_AVAILABLE = True
                        return True
                    except ImportError:
                        pass
                        
                except Exception as e:
                    continue
    
    # Strategy 2: Try to import from already loaded custom nodes
    try:
        # Check if VHS is already available in NODE_CLASS_MAPPINGS
        if 'VHS_LoadVideo' in globals().get('NODE_CLASS_MAPPINGS', {}):
            print("✅ VHS already available in NODE_CLASS_MAPPINGS")
            VHS_AVAILABLE = True
            return True
    except:
        pass
    
    # Strategy 3: Try pip-installed package
    try:
        from videohelpersuite.load_video_nodes import LoadVideoUpload, LoadVideoPath
        print("✅ Successfully imported VHS from pip-installed package")
        VHS_AVAILABLE = True
        return True
    except ImportError:
        pass
    
    print("⚠️  Warning: Could not import VHS classes with any strategy")
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
    
    def print_comprehensive_summary(self, lora_analysis=None, text_encoding_analysis=None):
        """Print comprehensive summary including model loading, LoRA application, and text encoding"""
        print(f"\n📊 COMPREHENSIVE WORKFLOW MONITORING SUMMARY")
        print("=" * 80)
        
        # Step 1: Model Loading Summary
        print(f"🔍 STEP 1: MODEL LOADING")
        print("   ✅ Models loaded successfully (detailed monitoring disabled)")
        # self.print_summary()
        
        # Step 2: LoRA Application Summary
        print(f"\n🔍 STEP 2: LORA APPLICATION")
        if lora_analysis:
            self.print_lora_analysis_summary(lora_analysis)
        else:
            print("   ✅ LoRA applied successfully (detailed monitoring disabled)")
        
        # Step 3: Text Encoding Summary
        if text_encoding_analysis:
            print(f"\n🔍 STEP 3: TEXT ENCODING")
            self.print_text_encoding_analysis_summary(text_encoding_analysis)
        else:
            print(f"\n🔍 STEP 3: TEXT ENCODING - No analysis available")
        
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
            
            # Strategy 1: Direct tensor
            if hasattr(conditioning, 'shape'):
                tensor_data = conditioning
                metadata = {}
                print(f"   ✅ Found direct tensor with shape: {tensor_data.shape}")
            
            # Strategy 2: List/tuple format [tensor, metadata]
            elif isinstance(conditioning, (list, tuple)) and len(conditioning) > 0:
                if hasattr(conditioning[0], 'shape'):
                    tensor_data = conditioning[0]
                    metadata = conditioning[1] if len(conditioning) > 1 else {}
                    print(f"   ✅ Found tensor in list[0] with shape: {tensor_data.shape}")
                else:
                    # Try to find tensor in the list
                    for i, item in enumerate(conditioning):
                        if hasattr(item, 'shape'):
                            tensor_data = item
                            metadata = {f'list_index_{i}': item for j, item in enumerate(conditioning) if j != i}
                            print(f"   ✅ Found tensor in list[{i}] with shape: {tensor_data.shape}")
                            break
            
            # Strategy 3: Dictionary format
            elif isinstance(conditioning, dict):
                # Look for tensor-like objects in the dictionary
                for key, value in conditioning.items():
                    if hasattr(value, 'shape'):
                        tensor_data = value
                        metadata = {k: v for k, v in conditioning.items() if k != key}
                        print(f"   ✅ Found tensor in dict['{key}'] with shape: {tensor_data.shape}")
                        break
            
            # If we found tensor data, analyze it
            if tensor_data is not None and hasattr(tensor_data, 'shape'):
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
                
                # Generate filename for tensor dump
                timestamp = int(time.time())
                tensor_filename = f"{tensor_type.lower()}_conditioning_{timestamp}.npy"
                tensor_filepath = os.path.join(output_dir, tensor_filename)
                
                # Store tensor for later comparison (dump to file)
                try:
                    # Convert to numpy and save
                    if hasattr(tensor_data, 'detach'):
                        numpy_tensor = tensor_data.detach().cpu().numpy()
                    elif hasattr(tensor_data, 'cpu'):
                        numpy_tensor = tensor_data.cpu().numpy()
                    else:
                        numpy_tensor = tensor_data.numpy()
                    
                    # Save tensor to file
                    np.save(tensor_filepath, numpy_tensor)
                    
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
                for peak in peak_timestamps[:5]:  # Show first 5 peaks
                    print(f"      {peak.get('type', 'unknown')}: {peak.get('value_mb', 0):.1f} MB at {peak.get('timestamp', 0):.2f}s")
        
        print("=" * 80)
    
    def print_lora_analysis_summary(self, analysis):
        """Print comprehensive LoRA application analysis"""
        print(f"\n🔍 LORA APPLICATION ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Basic success info
        print(f"✅ LoRA Application Success: {'YES' if analysis['lora_application_success'] else 'NO'}")
        print(f"📦 Models Returned: {analysis['models_returned']}")
        
        # UNET Changes
        print(f"\n🔧 UNET MODEL CHANGES:")
        unet_changes = analysis['unet_changes']
        print(f"   Model Cloned: {'✅ YES' if unet_changes['model_cloned'] else '❌ NO'}")
        print(f"   Class Changed: {'✅ YES' if unet_changes['class_changed'] else '❌ NO'}")
        print(f"   Patches Added: {unet_changes['patches_added']}")
        print(f"   UUID Changed: {'✅ YES' if unet_changes['uuid_changed'] else '❌ NO'}")
        print(f"   Original Patches: {unet_changes['original_patch_count']}")
        print(f"   Modified Patches: {unet_changes['modified_patch_count']}")
        
        # CLIP Changes
        print(f"\n🔧 CLIP MODEL CHANGES:")
        clip_changes = analysis['clip_changes']
        print(f"   Model Cloned: {'✅ YES' if clip_changes['model_cloned'] else '❌ NO'}")
        print(f"   Class Changed: {'✅ YES' if clip_changes['class_changed'] else '❌ NO'}")
        print(f"   Patches Added: {clip_changes['patches_added']}")
        print(f"   UUID Changed: {'✅ YES' if clip_changes['uuid_changed'] else '❌ NO'}")
        print(f"   Original Patches: {clip_changes['original_patch_count']}")
        print(f"   Modified Patches: {clip_changes['modified_patch_count']}")
        
        # LoRA Patches Analysis
        print(f"\n🔧 LORA PATCHES ANALYSIS:")
        unet_patches = analysis['unet_lora_patches']
        clip_patches = analysis['clip_lora_patches']
        
        if 'error' not in unet_patches:
            print(f"   UNET Patched Keys: {unet_patches['total_patched_keys']}")
        else:
            print(f"   UNET Patches: {unet_patches['error']}")
            
        if 'error' not in clip_patches:
            print(f"   CLIP Patched Keys: {clip_patches['total_patched_keys']}")
        else:
            print(f"   CLIP Patches: {clip_patches['error']}")
        
        # Model Placement Changes
        print(f"\n📍 MODEL PLACEMENT CHANGES:")
        placement_changes = analysis['placement_changes']
        
        # UNET Placement
        unet_placement = placement_changes['unet']
        print(f"   🔧 UNET PLACEMENT:")
        print(f"      Model Device Changed: {'✅ YES' if unet_placement['model_device_changed'] else '❌ NO'}")
        print(f"      Load Device Changed: {'✅ YES' if unet_placement['load_device_changed'] else '❌ NO'}")
        print(f"      Offload Device Changed: {'✅ YES' if unet_placement['offload_device_changed'] else '❌ NO'}")
        print(f"      Original Device: {unet_placement['original_device']}")
        print(f"      Modified Device: {unet_placement['modified_device']}")
        
        # CLIP Placement
        clip_placement = placement_changes['clip']
        print(f"   🔧 CLIP PLACEMENT:")
        print(f"      Model Device Changed: {'✅ YES' if clip_placement['model_device_changed'] else '❌ NO'}")
        print(f"      Load Device Changed: {'✅ YES' if clip_placement['load_device_changed'] else '❌ NO'}")
        print(f"      Offload Device Changed: {'✅ YES' if clip_placement['offload_device_changed'] else '❌ NO'}")
        print(f"      Original Device: {clip_placement['original_device']}")
        print(f"      Modified Device: {clip_placement['modified_device']}")
        
        # CLIP-specific device information
        if clip_placement['clip_model_device']:
            print(f"      CLIP Model Device: {clip_placement['clip_model_device']}")
        if clip_placement['clip_patcher_devices']:
            print(f"      CLIP Patcher Load Device: {clip_placement['clip_patcher_devices']['load']}")
            print(f"      CLIP Patcher Offload Device: {clip_placement['clip_patcher_devices']['offload']}")
        
        # Memory Impact
        print(f"\n💾 MEMORY IMPACT:")
        memory_impact = analysis['memory_impact']
        
        if 'error' not in memory_impact:
            # RAM Changes
            if 'ram' in memory_impact:
                ram = memory_impact['ram']
                print(f"\n   🖥️  RAM CHANGES:")
                print(f"      Used: {ram['used_change_mb']:+.1f} MB ({ram['baseline_used_mb']:.1f} → {ram['current_used_mb']:.1f} MB)")
                print(f"      Available: {ram['available_change_mb']:+.1f} MB ({ram['baseline_available_mb']:.1f} → {ram['current_available_mb']:.1f} MB)")
                print(f"      Total: {ram['current_total_mb']:.1f} MB")
                print(f"      Usage: {ram['baseline_percent_used']:.1f}% → {ram['current_percent_used']:.1f}%")
            
            # GPU Changes
            if 'gpu' in memory_impact and memory_impact['gpu']:
                gpu = memory_impact['gpu']
                print(f"\n   🎮 GPU CHANGES:")
                print(f"      Allocated: {gpu['allocated_change_mb']:+.1f} MB ({gpu['baseline_allocated_mb']:.1f} → {gpu['current_allocated_mb']:.1f} MB)")
                print(f"      Reserved: {gpu['reserved_change_mb']:+.1f} MB ({gpu['baseline_reserved_mb']:.1f} → {gpu['current_reserved_mb']:.1f} MB)")
                print(f"      Total VRAM: {gpu['current_total_mb']:.1f} MB")
                print(f"      Allocated Change: {gpu['allocated_change_pct']:+.1f}%")
                print(f"      Reserved Change: {gpu['reserved_change_pct']:+.1f}%")
        else:
            print(f"   ❌ Memory calculation failed: {memory_impact['error']}")
        
        # Peak Memory Information
        if 'peak_memory' in analysis:
            print(f"\n📊 PEAK MEMORY DURING LORA APPLICATION:")
            peak_memory = analysis['peak_memory']
            print(f"   🖥️  RAM Peak: {peak_memory['ram_peak_mb']:.1f} MB")
            print(f"   🎮 GPU Allocated Peak: {peak_memory['gpu_allocated_peak_mb']:.1f} MB")
            print(f"   🎮 GPU Reserved Peak: {peak_memory['gpu_reserved_peak_mb']:.1f} MB")
            
            # Show peak timestamps if available
            if peak_memory['peak_timestamps']:
                print(f"   ⏱️  Peak Timestamps:")
                for peak in peak_memory['peak_timestamps'][:5]:  # Show first 5 peaks
                    print(f"      {peak['type']}: {peak['value_mb']:.1f} MB at {peak['timestamp']:.2f}s")
        
        # Detailed Memory Breakdown
        print(f"\n💾 DETAILED MEMORY BREAKDOWN:")
        memory_impact = analysis['memory_impact']
        
        if 'error' not in memory_impact:
            # RAM Breakdown
            if 'ram' in memory_impact:
                ram = memory_impact['ram']
                print(f"   🖥️  RAM MEMORY BREAKDOWN:")
                print(f"      Baseline State:")
                print(f"         Used: {ram['baseline_used_mb']:.1f} MB")
                print(f"         Available: {ram['baseline_available_mb']:.1f} MB")
                print(f"         Usage: {ram['baseline_percent_used']:.1f}%")
                print(f"      Current State:")
                print(f"         Used: {ram['current_used_mb']:.1f} MB")
                print(f"         Available: {ram['current_available_mb']:.1f} MB")
                print(f"         Usage: {ram['current_percent_used']:.1f}%")
                print(f"      Changes:")
                print(f"         Used Change: {ram['used_change_mb']:+.1f} MB ({ram['used_change_mb']/ram['baseline_used_mb']*100:+.1f}%)")
                print(f"         Available Change: {ram['available_change_mb']:+.1f} MB")
                print(f"         Usage Change: {ram['current_percent_used'] - ram['baseline_percent_used']:+.1f}%")
            
            # GPU Breakdown
            if 'gpu' in memory_impact and memory_impact['gpu']:
                gpu = memory_impact['gpu']
                print(f"\n   🎮 GPU MEMORY BREAKDOWN:")
                print(f"      Baseline State:")
                print(f"         Allocated: {gpu['baseline_allocated_mb']:.1f} MB")
                print(f"         Reserved: {gpu['baseline_reserved_mb']:.1f} MB")
                print(f"         Total VRAM: {gpu['baseline_total_mb']:.1f} MB")
                print(f"         Available VRAM: {gpu['baseline_total_mb'] - gpu['baseline_reserved_mb']:.1f} MB")
                print(f"      Current State:")
                print(f"         Allocated: {gpu['current_allocated_mb']:.1f} MB")
                print(f"         Reserved: {gpu['current_reserved_mb']:.1f} MB")
                print(f"         Total VRAM: {gpu['current_total_mb']:.1f} MB")
                print(f"         Available VRAM: {gpu['current_total_mb'] - gpu['current_reserved_mb']:.1f} MB")
                print(f"      Changes:")
                print(f"         Allocated Change: {gpu['allocated_change_mb']:+.1f} MB ({gpu['allocated_change_pct']:+.1f}%)")
                print(f"         Reserved Change: {gpu['reserved_change_mb']:+.1f} MB ({gpu['reserved_change_pct']:+.1f}%)")
                print(f"         Available VRAM Change: {(gpu['current_total_mb'] - gpu['current_reserved_mb']) - (gpu['baseline_total_mb'] - gpu['baseline_reserved_mb']):+.1f} MB")
                
                # Memory efficiency analysis
                baseline_efficiency = (gpu['baseline_allocated_mb'] / gpu['baseline_reserved_mb'] * 100) if gpu['baseline_reserved_mb'] > 0 else 0
                current_efficiency = (gpu['current_allocated_mb'] / gpu['current_reserved_mb'] * 100) if gpu['current_reserved_mb'] > 0 else 0
                print(f"      Memory Efficiency:")
                print(f"         Baseline: {baseline_efficiency:.1f}% (allocated/reserved)")
                print(f"         Current: {current_efficiency:.1f}% (allocated/reserved)")
                print(f"         Efficiency Change: {current_efficiency - baseline_efficiency:+.1f}%")
        
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

# Now attempt VHS import after find_path is available
attempt_vhs_import()


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
        video_file = "safu.mp4"
        video_file_exists = os.path.exists(video_file)
        
        if VHS_AVAILABLE and VHS_LoadVideoPath is not None and video_file_exists:
            print("🎥 Loading video using direct VHS imports...")
            print(f"   📁 Video file: {video_file} - EXISTS ({os.path.getsize(video_file) / (1024**2):.2f} MB)")
            try:
                # Use LoadVideoPath for direct file loading
                video_loader = VHS_LoadVideoPath()
                vhs_loadvideo_1 = video_loader.load_video(
                    video=video_file,
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
                print("   This may be due to VHS compatibility issues or corrupted video file")
                vhs_loadvideo_1 = None
        elif not video_file_exists:
            print("⏭️  Skipping video loading (video file not found)")
            print(f"   📁 Video file: {video_file} - NOT FOUND")
            print(f"   💡 Make sure '{video_file}' exists in the current directory")
            vhs_loadvideo_1 = None
        else:
            print("⏭️  Skipping video loading (VHS not available)")
            print("   VHS_AVAILABLE:", VHS_AVAILABLE)
            print("   VHS_LoadVideoPath:", "Available" if VHS_LoadVideoPath is not None else "None")
            vhs_loadvideo_1 = None
        
        # Load reference image
        image_file = "safu.jpg"
        image_file_exists = os.path.exists(image_file)
        
        if image_file_exists:
            print(f"📷 Loading reference image: {image_file} - EXISTS ({os.path.getsize(image_file) / (1024**2):.2f} MB)")
            try:
                loadimage = LoadImage()
                loadimage_4 = loadimage.load_image(image=image_file)
                print("✅ Reference image loaded successfully")
                
                # Debug: Show what was loaded
                # if loadimage_4:
                #     print(f"   Image type: {type(loadimage_4)}")
                #     if isinstance(loadimage_4, (list, tuple)) and len(loadimage_4) > 0:
                #         print(f"   Image data type: {type(loadimage_4[0])}")
                #         if hasattr(loadimage_4[0], 'shape'):
                #         print(f"   Image shape: {loadimage_4[0].shape}")
            except Exception as e:
                print(f"❌ Error loading reference image: {e}")
                loadimage_4 = None
        else:
            print(f"⏭️  Skipping reference image loading (image file not found)")
            print(f"   📁 Image file: {image_file} - NOT FOUND")
            print(f"   💡 Make sure '{image_file}' exists in the current directory")
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
        print("\n🔍 CAPTURING BASELINE STATE BEFORE TEXT ENCODING...")
        
        # Get the modified models from LoRA application
        if 'modified_unet' in locals() and 'modified_clip' in locals():
            try:
                text_encoding_baseline = model_monitor.capture_text_encoding_baseline(
                    modified_unet, 
                    modified_clip
                )
                
                print(f"   ✅ UNET Baseline captured - ID: {text_encoding_baseline.get('unet', {}).get('model_id', 'N/A')}")
                print(f"   ✅ CLIP Baseline captured - ID: {text_encoding_baseline.get('clip', {}).get('model_id', 'N/A')}")
                
                # Display baseline memory information
                print(f"\n   💾 BASELINE MEMORY STATE:")
                ram_info = text_encoding_baseline.get('ram', {})
                print(f"      🖥️  RAM: {ram_info.get('used_mb', 0):.1f} MB used / {ram_info.get('total_mb', 0):.1f} MB total ({ram_info.get('percent_used', 0):.1f}%)")
                gpu_info = text_encoding_baseline.get('gpu')
                if gpu_info:
                    print(f"      🎮 GPU: {gpu_info.get('allocated', 0) / (1024**2):.1f} MB allocated / {gpu_info.get('total', 0) / (1024**2):.1f} MB total")
            except Exception as e:
                print(f"❌ ERROR during baseline capture: {e}")
                print("🔍 Baseline capture failed - will continue with limited monitoring")
                text_encoding_baseline = None
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
            model_monitor.start_monitoring("text_encoding")
            
            # Update peak memory before text encoding
            model_monitor.update_peak_memory()
            
            # Create text encoder and encode positive prompt
            print("   🔤 Encoding positive prompt...")
            cliptextencode = CLIPTextEncode()
            positive_cond_tuple = cliptextencode.encode(modified_clip, positive_prompt)
            
            # Update peak memory after positive encoding
            model_monitor.update_peak_memory()
            
            # Encode negative prompt
            print("   🔤 Encoding negative prompt...")
            negative_cond_tuple = cliptextencode.encode(modified_clip, negative_prompt)
            
            # Update peak memory after negative encoding
            model_monitor.update_peak_memory()
            
            # Extract conditioning from tuples
            positive_cond = positive_cond_tuple[0] if positive_cond_tuple else None
            negative_cond = negative_cond_tuple[0] if negative_cond_tuple else None
            
            # End monitoring and get peak memory summary
            elapsed_time = model_monitor.end_monitoring("text_encoding", [positive_cond, negative_cond], "TextEncoding_Result")
            peak_memory_summary = model_monitor.get_peak_memory_summary()
            
            # Analyze text encoding results
            print("\n🔍 ANALYZING TEXT ENCODING RESULTS...")
            try:
                text_encoding_analysis = model_monitor.analyze_text_encoding_results(
                    text_encoding_baseline, 
                    positive_cond, 
                    negative_cond, 
                    positive_prompt,
                    negative_prompt,
                    elapsed_time
                )
                
                # Print comprehensive analysis
                model_monitor.print_text_encoding_analysis_summary(text_encoding_analysis)
            except Exception as e:
                print(f"❌ ERROR during text encoding analysis: {e}")
                print("🔍 Text encoding analysis failed - will show error summary")
                print("🔍 Error details:", str(e))
                import traceback
                traceback.print_exc()
                text_encoding_analysis = None
            
            # Print peak memory information
            print(f"\n📊 PEAK MEMORY DURING TEXT ENCODING:")
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
            
            print("✅ Step 3 completed: Text Encoding with comprehensive monitoring")
            
        except Exception as e:
            print(f"❌ ERROR during text encoding: {e}")
            print("🔍 Text encoding failed - check error details above")
            positive_cond = None
            negative_cond = None
            text_encoding_analysis = None
            
        # === STEP 3 END: TEXT ENCODING ===

        # Stop execution after step 3 for debugging purposes
        print("\n🛑 STOPPING EXECUTION AFTER STEP 3 (TEXT ENCODING)")
        print("🔍 All text encoding debugging information has been displayed above.")
        print("📊 Check the monitoring data above to analyze text encoding performance.")
        print("🔍 Step 1: Model Loading - COMPLETED")
        print("🔍 Step 2: LoRA Application - COMPLETED")
        print("🔍 Step 3: Text Encoding - COMPLETED")
        print("🔍 Steps 4-9: SKIPPED for debugging purposes")
        
        # === FINAL MONITORING SUMMARY ===
        print("\n" + "="*80)
        print("🔍 FINAL WORKFLOW MONITORING SUMMARY")
        print("="*80)
        
        # Use comprehensive summary if analyses are available
        if 'text_encoding_analysis' in locals():
            model_monitor.print_comprehensive_summary(None, text_encoding_analysis)
        else:
            print("\n⚠️  Text encoding analysis not available - step 3 may not have completed")
        
        print("="*80)
        
        return


if __name__ == "__main__":
    main()
