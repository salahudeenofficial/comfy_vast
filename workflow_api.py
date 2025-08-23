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
        
        # Enhanced model information extraction
        print(f"🔧 ENHANCED MODEL INFORMATION:")
        self._extract_enhanced_model_info(loader_result, model_type)
        
        print("=" * 60)
    
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
    
    def print_comprehensive_summary(self, lora_analysis=None):
        """Print comprehensive summary including both model loading and LoRA application"""
        print(f"\n📊 COMPREHENSIVE WORKFLOW MONITORING SUMMARY")
        print("=" * 80)
        
        # Step 1: Model Loading Summary
        print(f"🔍 STEP 1: MODEL LOADING")
        self.print_summary()
        
        # Step 2: LoRA Application Summary
        if lora_analysis:
            print(f"\n🔍 STEP 2: LORA APPLICATION")
            self.print_lora_analysis_summary(lora_analysis)
        else:
            print(f"\n🔍 STEP 2: LORA APPLICATION - No analysis available")
        
        print("=" * 80)
    
    # === LORA APPLICATION MONITORING METHODS ===
    
    def capture_lora_baseline(self, unet_model, clip_model):
        """Capture baseline state before LoRA application"""
        baseline = {
            'timestamp': time.time(),
            'unet': {
                'model_id': id(unet_model),
                'class': type(unet_model).__name__,
                'device': getattr(unet_model, 'device', None),
                'patches_count': len(getattr(unet_model, 'patches', {})),
                'patches_uuid': getattr(unet_model, 'patches_uuid', None),
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            },
            'clip': {
                'model_id': id(clip_model),
                'class': type(clip_model).__name__,
                'device': getattr(clip_model, 'device', None),
                'patcher_patches_count': len(getattr(clip_model.patcher, 'patches', {})) if hasattr(clip_model, 'patcher') else 0,
                'patcher_patches_uuid': getattr(clip_model.patcher, 'patches_uuid', None) if hasattr(clip_model, 'patcher') else None,
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            }
        }
        return baseline
    
    def track_model_identity_changes(self, original_model, modified_model, model_type):
        """Track changes in model identity and structure"""
        
        # 1. Model Instance Changes
        model_cloned = original_model is not modified_model
        model_class_changed = type(original_model) != type(modified_model)
        
        # 2. ModelPatcher Changes (for UNET)
        original_patch_count = 0
        modified_patch_count = 0
        patches_added = 0
        original_uuid = None
        modified_uuid = None
        uuid_changed = False
        
        if hasattr(original_model, 'patches') and hasattr(modified_model, 'patches'):
            original_patch_count = len(original_model.patches)
            modified_patch_count = len(modified_model.patches)
            patches_added = modified_patch_count - original_patch_count
            
            # 3. Patch UUID Changes
            original_uuid = getattr(original_model, 'patches_uuid', None)
            modified_uuid = getattr(modified_model, 'patches_uuid', None)
            uuid_changed = original_uuid != modified_uuid
        
        return {
            'model_cloned': model_cloned,
            'class_changed': model_class_changed,
            'patches_added': patches_added,
            'uuid_changed': uuid_changed,
            'original_patch_count': original_patch_count,
            'modified_patch_count': modified_patch_count
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
        
        if not hasattr(modified_model, 'patches'):
            return {'error': 'Model has no patches attribute'}
        
        patches = modified_model.patches
        lora_patches = {}
        
        for key, patch_list in patches.items():
            if patch_list:  # If patches exist for this key
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
            'patch_details': lora_patches,
            'model_type': model_type
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
            current_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            current_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            
            allocated_change = current_allocated - baseline_info['memory_allocated']
            reserved_change = current_reserved - baseline_info['memory_reserved']
            
            return {
                'allocated_change_mb': allocated_change / (1024**2),
                'reserved_change_mb': reserved_change / (1024**2),
                'current_allocated_mb': current_allocated / (1024**2),
                'current_reserved_mb': current_reserved / (1024**2)
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
            'memory_impact': {
                'unet_memory_change': self.calculate_memory_change(
                    baseline['unet'], modified_unet
                ),
                'clip_memory_change': self.calculate_memory_change(
                    baseline['clip'], modified_clip
                )
            }
        }
        
        return analysis
    
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
        
        # Memory Impact
        print(f"\n💾 MEMORY IMPACT:")
        unet_memory = analysis['memory_impact']['unet_memory_change']
        clip_memory = analysis['memory_impact']['clip_memory_change']
        
        if 'error' not in unet_memory:
            print(f"   UNET Memory Change: {unet_memory['allocated_change_mb']:+.1f} MB allocated, {unet_memory['reserved_change_mb']:+.1f} MB reserved")
        if 'error' not in clip_memory:
            print(f"   CLIP Memory Change: {clip_memory['allocated_change_mb']:+.1f} MB allocated, {clip_memory['reserved_change_mb']:+.1f} MB reserved")
        
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
        try:
            from utils.extra_config import load_extra_path_config
        except ImportError:
            print("⚠️  Warning: Could not import extra_config, skipping extra model paths")
            return

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        try:
            load_extra_path_config(extra_model_paths)
        except Exception as e:
            print(f"⚠️  Warning: Could not load extra model paths: {e}")


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
        
        # Debug execution stopped - continuing to step 2
        print("\n🔍 Step 1 debugging complete - continuing to LoRA application...")

        # === STEP 2 START: LORA APPLICATION ===
        print("2. Applying LoRA...")
        
        # Capture baseline state before LoRA application
        print("\n🔍 CAPTURING BASELINE STATE BEFORE LORA APPLICATION...")
        unet_model_baseline = get_value_at_index(unetloader_27, 0)
        clip_model_baseline = get_value_at_index(cliploader_23, 0)
        
        # Check LoRA file status
        lora_filename = "lora.safetensors"
        lora_file_exists = os.path.exists(lora_filename)
        lora_file_size = os.path.getsize(lora_filename) if lora_file_exists else 0
        
        print(f"   📁 LoRA File: {lora_filename}")
        print(f"   📁 File Exists: {'✅ YES' if lora_file_exists else '❌ NO'}")
        if lora_file_exists:
            print(f"   📁 File Size: {lora_file_size / (1024**2):.2f} MB")
        
        lora_baseline = model_monitor.capture_lora_baseline(
            unet_model_baseline, 
            clip_model_baseline
        )
        
        print(f"   ✅ UNET Baseline captured - ID: {lora_baseline['unet']['model_id']}, Patches: {lora_baseline['unet']['patches_count']}")
        print(f"   ✅ CLIP Baseline captured - ID: {lora_baseline['clip']['model_id']}, Patches: {lora_baseline['clip']['patcher_patches_count']}")
        
        # Apply LoRA with monitoring
        try:
            print("\n🔧 APPLYING LORA TO MODELS...")
            loraloader = LoraLoader()
            loraloader_24 = loraloader.load_lora(
                lora_name="lora.safetensors",
                strength_model=0.5000000000000001,
                strength_clip=1,
                model=unet_model_baseline,
                clip=clip_model_baseline,
            )
            
            # Extract modified models from result
            modified_unet = get_value_at_index(loraloader_24, 0)
            modified_clip = get_value_at_index(loraloader_24, 1)
            
            # Analyze LoRA application results
            print("\n🔍 ANALYZING LORA APPLICATION RESULTS...")
            lora_analysis = model_monitor.analyze_lora_application_results(
                lora_baseline, 
                modified_unet, 
                modified_clip, 
                loraloader_24
            )
            
            # Print comprehensive analysis
            model_monitor.print_lora_analysis_summary(lora_analysis)
            
            print("✅ Step 2 completed: LoRA Application with comprehensive monitoring")
            
        except Exception as e:
            print(f"❌ ERROR during LoRA application: {e}")
            print("🔍 LoRA application failed - check error details above")
            loraloader_24 = None
            
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
        
        # Use comprehensive summary if LoRA analysis is available
        if 'lora_analysis' in locals():
            model_monitor.print_comprehensive_summary(lora_analysis)
        else:
            model_monitor.print_summary()
            print("\n⚠️  LoRA analysis not available - step 2 may not have completed")
        
        print("="*80)
        
        # === STEP 9 END: VIDEO EXPORT ===


if __name__ == "__main__":
    main()
