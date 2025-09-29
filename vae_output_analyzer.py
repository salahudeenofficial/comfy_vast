#!/usr/bin/env python3
"""
VAE Output Analyzer for WanVaceToVideo
=====================================

This module provides comprehensive analysis of VAE outputs for:
- Reactive latent outputs
- Inactive latent outputs  
- Reference image latent outputs
- Combined vace_frames outputs

Focus areas: Mean and Range analysis with detailed statistics
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os


class VAEOutputAnalyzer:
    """
    Specialized analyzer for VAE outputs in WanVaceToVideo pipeline
    
    Analyzes:
    1. Reactive latents (foreground regions)
    2. Inactive latents (background regions) 
    3. Reference image latents
    4. Combined vace_frames (concatenated control video latents)
    """
    
    def __init__(self):
        self.analysis_results = []
        self.current_analysis_id = 0
        
    def analyze_vae_outputs(self, 
                          reactive_latent: Optional[torch.Tensor] = None,
                          inactive_latent: Optional[torch.Tensor] = None, 
                          reference_latent: Optional[torch.Tensor] = None,
                          combined_vace_frames: Optional[torch.Tensor] = None,
                          analysis_context: str = "unknown") -> Dict[str, Any]:
        """
        Comprehensive analysis of all VAE outputs
        
        Args:
            reactive_latent: Foreground latent tensor [frames, channels, height, width]
            inactive_latent: Background latent tensor [frames, channels, height, width]
            reference_latent: Reference image latent tensor [1, channels, height, width]
            combined_vace_frames: Combined control video latent [frames, 2*channels, height, width]
            analysis_context: Context description for this analysis
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        
        self.current_analysis_id += 1
        analysis_id = f"vae_analysis_{self.current_analysis_id}"
        
        print(f"\n🔍 VAE OUTPUT ANALYSIS #{self.current_analysis_id}")
        print(f"   Context: {analysis_context}")
        print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize analysis result
        analysis_result = {
            'analysis_id': analysis_id,
            'context': analysis_context,
            'timestamp': time.time(),
            'reactive_analysis': None,
            'inactive_analysis': None,
            'reference_analysis': None,
            'combined_analysis': None,
            'comparative_analysis': None,
            'summary': {}
        }
        
        # Analyze individual components
        if reactive_latent is not None:
            print(f"\n📊 REACTIVE LATENT ANALYSIS:")
            analysis_result['reactive_analysis'] = self._analyze_latent_component(
                reactive_latent, "reactive", "foreground regions"
            )
            
        if inactive_latent is not None:
            print(f"\n📊 INACTIVE LATENT ANALYSIS:")
            analysis_result['inactive_analysis'] = self._analyze_latent_component(
                inactive_latent, "inactive", "background regions"
            )
            
        if reference_latent is not None:
            print(f"\n📊 REFERENCE IMAGE LATENT ANALYSIS:")
            analysis_result['reference_analysis'] = self._analyze_latent_component(
                reference_latent, "reference", "reference image"
            )
            
        if combined_vace_frames is not None:
            print(f"\n📊 COMBINED VACE_FRAMES ANALYSIS:")
            analysis_result['combined_analysis'] = self._analyze_combined_vace_frames(
                combined_vace_frames, reactive_latent, inactive_latent
            )
        
        # Comparative analysis
        if any([reactive_latent is not None, inactive_latent is not None, reference_latent is not None]):
            print(f"\n📊 COMPARATIVE ANALYSIS:")
            analysis_result['comparative_analysis'] = self._perform_comparative_analysis(
                analysis_result
            )
        
        # Generate summary
        analysis_result['summary'] = self._generate_analysis_summary(analysis_result)
        
        # Store results
        self.analysis_results.append(analysis_result)
        
        # Print summary
        self._print_analysis_summary(analysis_result)
        
        return analysis_result
    
    def _analyze_latent_component(self, 
                                latent_tensor: torch.Tensor, 
                                component_name: str,
                                description: str) -> Dict[str, Any]:
        """
        Analyze a single latent component with focus on mean and range
        
        Args:
            latent_tensor: The latent tensor to analyze
            component_name: Name of the component (reactive, inactive, reference)
            description: Human-readable description
            
        Returns:
            Dictionary with comprehensive analysis
        """
        
        if latent_tensor is None:
            return None
            
        # Basic tensor information
        shape = latent_tensor.shape
        dtype = str(latent_tensor.dtype)
        device = str(latent_tensor.device)
        num_elements = latent_tensor.numel()
        size_mb = (num_elements * latent_tensor.element_size()) / (1024**2)
        
        print(f"   Component: {component_name.upper()} ({description})")
        print(f"   Shape: {shape}")
        print(f"   Dtype: {dtype}")
        print(f"   Device: {device}")
        print(f"   Size: {size_mb:.2f} MB ({num_elements:,} elements)")
        
        # Convert to float32 for calculations
        if latent_tensor.dtype != torch.float32:
            latent_float = latent_tensor.float()
        else:
            latent_float = latent_tensor
        
        # MEAN AND RANGE ANALYSIS (Primary Focus)
        mean_analysis = self._analyze_mean_statistics(latent_float, component_name)
        range_analysis = self._analyze_range_statistics(latent_float, component_name)
        
        # Channel-wise analysis
        channel_analysis = self._analyze_channels(latent_float, component_name)
        
        # Temporal analysis (for video latents)
        temporal_analysis = self._analyze_temporal_dimension(latent_float, component_name)
        
        # Spatial analysis
        spatial_analysis = self._analyze_spatial_dimensions(latent_float, component_name)
        
        # Value distribution analysis
        distribution_analysis = self._analyze_value_distribution(latent_float, component_name)
        
        # Quality checks
        quality_analysis = self._perform_quality_checks(latent_float, component_name)
        
        analysis = {
            'component_name': component_name,
            'description': description,
            'tensor_info': {
                'shape': shape,
                'dtype': dtype,
                'device': device,
                'num_elements': num_elements,
                'size_mb': size_mb
            },
            'mean_analysis': mean_analysis,
            'range_analysis': range_analysis,
            'channel_analysis': channel_analysis,
            'temporal_analysis': temporal_analysis,
            'spatial_analysis': spatial_analysis,
            'distribution_analysis': distribution_analysis,
            'quality_analysis': quality_analysis
        }
        
        return analysis
    
    def _analyze_mean_statistics(self, tensor: torch.Tensor, component_name: str) -> Dict[str, Any]:
        """Analyze mean statistics with detailed breakdown"""
        
        # Global mean
        global_mean = torch.mean(tensor).item()
        
        # Channel-wise means
        if len(tensor.shape) >= 2:
            # For 4D tensors [frames, channels, height, width], reduce over spatial dims
            if len(tensor.shape) == 4:
                # Reduce over height and width dimensions (2, 3)
                channel_means = torch.mean(torch.mean(tensor, dim=2), dim=2).flatten()
            else:
                # For other shapes, reduce over all non-channel dims one by one
                reduced_tensor = tensor
                for dim in range(2, len(tensor.shape)):
                    reduced_tensor = torch.mean(reduced_tensor, dim=dim)
                channel_means = reduced_tensor.flatten()
            channel_means_list = channel_means.tolist()
        else:
            channel_means_list = [global_mean]
        
        # Temporal means (for video latents)
        temporal_means = None
        if len(tensor.shape) >= 4:  # [frames, channels, height, width]
            # Reduce over channels, height, and width dimensions
            temporal_means = torch.mean(torch.mean(torch.mean(tensor, dim=1), dim=1), dim=1).tolist()
        
        # Spatial means
        spatial_means = None
        if len(tensor.shape) >= 4:
            # Reduce over frames and channels dimensions
            spatial_means = torch.mean(torch.mean(tensor, dim=0), dim=0).tolist()
        
        print(f"   📈 MEAN ANALYSIS:")
        print(f"      Global mean: {global_mean:.6f}")
        print(f"      Channel means: {len(channel_means_list)} channels")
        print(f"         Min channel mean: {min(channel_means_list):.6f}")
        print(f"         Max channel mean: {max(channel_means_list):.6f}")
        print(f"         Mean of channel means: {np.mean(channel_means_list):.6f}")
        print(f"         Std of channel means: {np.std(channel_means_list):.6f}")
        
        if temporal_means:
            print(f"      Temporal means: {len(temporal_means)} frames")
            print(f"         Min temporal mean: {min(temporal_means):.6f}")
            print(f"         Max temporal mean: {max(temporal_means):.6f}")
            print(f"         Mean of temporal means: {np.mean(temporal_means):.6f}")
            print(f"         Std of temporal means: {np.std(temporal_means):.6f}")
        
        return {
            'global_mean': global_mean,
            'channel_means': channel_means_list,
            'channel_mean_stats': {
                'min': min(channel_means_list),
                'max': max(channel_means_list),
                'mean': np.mean(channel_means_list),
                'std': np.std(channel_means_list)
            },
            'temporal_means': temporal_means,
            'temporal_mean_stats': {
                'min': min(temporal_means) if temporal_means else None,
                'max': max(temporal_means) if temporal_means else None,
                'mean': np.mean(temporal_means) if temporal_means else None,
                'std': np.std(temporal_means) if temporal_means else None
            },
            'spatial_means': spatial_means
        }
    
    def _analyze_range_statistics(self, tensor: torch.Tensor, component_name: str) -> Dict[str, Any]:
        """Analyze range statistics with detailed breakdown"""
        
        # Global range
        global_min = torch.min(tensor).item()
        global_max = torch.max(tensor).item()
        global_range = global_max - global_min
        
        # Channel-wise ranges
        if len(tensor.shape) >= 2:
            # For 4D tensors [frames, channels, height, width], reduce over spatial dims
            if len(tensor.shape) == 4:
                # Reduce over height and width dimensions (2, 3)
                channel_mins = torch.min(torch.min(tensor, dim=2).values, dim=2).values.flatten()
                channel_maxs = torch.max(torch.max(tensor, dim=2).values, dim=2).values.flatten()
            else:
                # For other shapes, reduce over all non-channel dims one by one
                reduced_tensor = tensor
                for dim in range(2, len(tensor.shape)):
                    reduced_tensor = torch.min(reduced_tensor, dim=dim).values
                channel_mins = reduced_tensor.flatten()
                
                reduced_tensor = tensor
                for dim in range(2, len(tensor.shape)):
                    reduced_tensor = torch.max(reduced_tensor, dim=dim).values
                channel_maxs = reduced_tensor.flatten()
            
            channel_ranges = channel_maxs - channel_mins
            channel_ranges_list = channel_ranges.tolist()
        else:
            channel_ranges_list = [global_range]
        
        # Temporal ranges (for video latents)
        temporal_ranges = None
        if len(tensor.shape) >= 4:  # [frames, channels, height, width]
            # Reduce over channels, height, and width dimensions
            frame_mins = torch.min(torch.min(torch.min(tensor, dim=1).values, dim=1).values, dim=1).values
            frame_maxs = torch.max(torch.max(torch.max(tensor, dim=1).values, dim=1).values, dim=1).values
            temporal_ranges = (frame_maxs - frame_mins).tolist()
        
        # Spatial ranges
        spatial_ranges = None
        if len(tensor.shape) >= 4:
            # Reduce over frames and channels dimensions
            spatial_mins = torch.min(torch.min(tensor, dim=0).values, dim=0).values
            spatial_maxs = torch.max(torch.max(tensor, dim=0).values, dim=0).values
            spatial_ranges = (spatial_maxs - spatial_mins).tolist()
        
        print(f"   📊 RANGE ANALYSIS:")
        print(f"      Global range: [{global_min:.6f}, {global_max:.6f}] (span: {global_range:.6f})")
        print(f"      Channel ranges: {len(channel_ranges_list)} channels")
        print(f"         Min channel range: {min(channel_ranges_list):.6f}")
        print(f"         Max channel range: {max(channel_ranges_list):.6f}")
        print(f"         Mean of channel ranges: {np.mean(channel_ranges_list):.6f}")
        print(f"         Std of channel ranges: {np.std(channel_ranges_list):.6f}")
        
        if temporal_ranges:
            print(f"      Temporal ranges: {len(temporal_ranges)} frames")
            print(f"         Min temporal range: {min(temporal_ranges):.6f}")
            print(f"         Max temporal range: {max(temporal_ranges):.6f}")
            print(f"         Mean of temporal ranges: {np.mean(temporal_ranges):.6f}")
            print(f"         Std of temporal ranges: {np.std(temporal_ranges):.6f}")
        
        return {
            'global_range': {
                'min': global_min,
                'max': global_max,
                'span': global_range
            },
            'channel_ranges': channel_ranges_list,
            'channel_range_stats': {
                'min': min(channel_ranges_list),
                'max': max(channel_ranges_list),
                'mean': np.mean(channel_ranges_list),
                'std': np.std(channel_ranges_list)
            },
            'temporal_ranges': temporal_ranges,
            'temporal_range_stats': {
                'min': min(temporal_ranges) if temporal_ranges else None,
                'max': max(temporal_ranges) if temporal_ranges else None,
                'mean': np.mean(temporal_ranges) if temporal_ranges else None,
                'std': np.std(temporal_ranges) if temporal_ranges else None
            },
            'spatial_ranges': spatial_ranges
        }
    
    def _analyze_channels(self, tensor: torch.Tensor, component_name: str) -> Dict[str, Any]:
        """Analyze channel-wise statistics"""
        
        if len(tensor.shape) < 2:
            return {'num_channels': 1, 'channel_analysis': 'Single channel tensor'}
        
        num_channels = tensor.shape[1] if len(tensor.shape) >= 4 else tensor.shape[0]
        
        # Channel-wise statistics
        channel_stats = []
        for i in range(num_channels):
            if len(tensor.shape) >= 4:  # [frames, channels, height, width]
                channel_data = tensor[:, i, :, :]
            else:  # [channels, height, width]
                channel_data = tensor[i, :, :]
            
            channel_mean = torch.mean(channel_data).item()
            channel_std = torch.std(channel_data).item()
            channel_min = torch.min(channel_data).item()
            channel_max = torch.max(channel_data).item()
            channel_range = channel_max - channel_min
            
            channel_stats.append({
                'channel_id': i,
                'mean': channel_mean,
                'std': channel_std,
                'min': channel_min,
                'max': channel_max,
                'range': channel_range
            })
        
        print(f"   🔢 CHANNEL ANALYSIS:")
        print(f"      Number of channels: {num_channels}")
        print(f"      Channel statistics summary:")
        for i, stats in enumerate(channel_stats[:5]):  # Show first 5 channels
            print(f"         Channel {i}: mean={stats['mean']:.6f}, range={stats['range']:.6f}")
        if num_channels > 5:
            print(f"         ... and {num_channels - 5} more channels")
        
        return {
            'num_channels': num_channels,
            'channel_stats': channel_stats,
            'channel_summary': {
                'mean_of_means': np.mean([s['mean'] for s in channel_stats]),
                'std_of_means': np.std([s['mean'] for s in channel_stats]),
                'mean_of_ranges': np.mean([s['range'] for s in channel_stats]),
                'std_of_ranges': np.std([s['range'] for s in channel_stats])
            }
        }
    
    def _analyze_temporal_dimension(self, tensor: torch.Tensor, component_name: str) -> Dict[str, Any]:
        """Analyze temporal dimension for video latents"""
        
        if len(tensor.shape) < 4:  # Not a video latent
            return {'is_video': False}
        
        num_frames = tensor.shape[0]
        
        # Frame-wise statistics
        frame_stats = []
        for i in range(num_frames):
            frame_data = tensor[i, :, :, :]
            frame_mean = torch.mean(frame_data).item()
            frame_std = torch.std(frame_data).item()
            frame_min = torch.min(frame_data).item()
            frame_max = torch.max(frame_data).item()
            frame_range = frame_max - frame_min
            
            frame_stats.append({
                'frame_id': i,
                'mean': frame_mean,
                'std': frame_std,
                'min': frame_min,
                'max': frame_max,
                'range': frame_range
            })
        
        print(f"   ⏱️  TEMPORAL ANALYSIS:")
        print(f"      Number of frames: {num_frames}")
        print(f"      Frame statistics summary:")
        for i, stats in enumerate(frame_stats[:3]):  # Show first 3 frames
            print(f"         Frame {i}: mean={stats['mean']:.6f}, range={stats['range']:.6f}")
        if num_frames > 3:
            print(f"         ... and {num_frames - 3} more frames")
        
        return {
            'is_video': True,
            'num_frames': num_frames,
            'frame_stats': frame_stats,
            'temporal_summary': {
                'mean_of_means': np.mean([s['mean'] for s in frame_stats]),
                'std_of_means': np.std([s['mean'] for s in frame_stats]),
                'mean_of_ranges': np.mean([s['range'] for s in frame_stats]),
                'std_of_ranges': np.std([s['range'] for s in frame_stats])
            }
        }
    
    def _analyze_spatial_dimensions(self, tensor: torch.Tensor, component_name: str) -> Dict[str, Any]:
        """Analyze spatial dimensions"""
        
        if len(tensor.shape) < 3:
            return {'spatial_dims': 'Not applicable'}
        
        if len(tensor.shape) >= 4:  # [frames, channels, height, width]
            height, width = tensor.shape[2], tensor.shape[3]
        else:  # [channels, height, width]
            height, width = tensor.shape[1], tensor.shape[2]
        
        # Spatial statistics
        if len(tensor.shape) >= 4:
            # Reduce over frames and channels dimensions
            spatial_mean = torch.mean(torch.mean(tensor, dim=0), dim=0).mean().item()
            spatial_std = torch.std(torch.std(tensor, dim=0), dim=0).mean().item()
        else:
            spatial_mean = torch.mean(tensor, dim=0).mean().item()
            spatial_std = torch.std(tensor, dim=0).mean().item()
        
        print(f"   🗺️  SPATIAL ANALYSIS:")
        print(f"      Dimensions: {height} x {width}")
        print(f"      Spatial mean: {spatial_mean:.6f}")
        print(f"      Spatial std: {spatial_std:.6f}")
        
        return {
            'height': height,
            'width': width,
            'spatial_mean': spatial_mean,
            'spatial_std': spatial_std
        }
    
    def _analyze_value_distribution(self, tensor: torch.Tensor, component_name: str) -> Dict[str, Any]:
        """Analyze value distribution characteristics"""
        
        # Flatten tensor for distribution analysis
        flat_tensor = tensor.flatten()
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = {}
        for p in percentiles:
            percentile_values[p] = torch.quantile(flat_tensor, p/100.0).item()
        
        # Value counts
        zero_count = (flat_tensor == 0).sum().item()
        negative_count = (flat_tensor < 0).sum().item()
        positive_count = (flat_tensor > 0).sum().item()
        total_count = flat_tensor.numel()
        
        # Check for extreme values
        has_nan = torch.isnan(flat_tensor).any().item()
        has_inf = torch.isinf(flat_tensor).any().item()
        
        print(f"   📊 DISTRIBUTION ANALYSIS:")
        print(f"      Percentiles: P1={percentile_values[1]:.6f}, P50={percentile_values[50]:.6f}, P99={percentile_values[99]:.6f}")
        print(f"      Value counts: {positive_count:,} positive, {negative_count:,} negative, {zero_count:,} zero")
        print(f"      Quality: NaN={has_nan}, Inf={has_inf}")
        
        return {
            'percentiles': percentile_values,
            'value_counts': {
                'positive': positive_count,
                'negative': negative_count,
                'zero': zero_count,
                'total': total_count
            },
            'value_ratios': {
                'positive_ratio': positive_count / total_count,
                'negative_ratio': negative_count / total_count,
                'zero_ratio': zero_count / total_count
            },
            'quality_flags': {
                'has_nan': has_nan,
                'has_inf': has_inf
            }
        }
    
    def _perform_quality_checks(self, tensor: torch.Tensor, component_name: str) -> Dict[str, Any]:
        """Perform quality checks on the latent tensor"""
        
        # Check for all zeros
        is_all_zeros = (tensor == 0).all().item()
        
        # Check for constant values
        is_constant = (tensor == tensor[0]).all().item()
        
        # Check for reasonable value ranges
        global_min = torch.min(tensor).item()
        global_max = torch.max(tensor).item()
        global_range = global_max - global_min
        
        # Typical VAE latent ranges (empirically observed)
        reasonable_min = -10.0
        reasonable_max = 10.0
        
        is_reasonable_range = reasonable_min <= global_min and global_max <= reasonable_max
        
        # Check for extreme values
        extreme_threshold = 50.0
        has_extreme_values = (torch.abs(tensor) > extreme_threshold).any().item()
        
        quality_score = 0
        if not is_all_zeros:
            quality_score += 1
        if not is_constant:
            quality_score += 1
        if is_reasonable_range:
            quality_score += 1
        if not has_extreme_values:
            quality_score += 1
        
        print(f"   ✅ QUALITY CHECKS:")
        print(f"      All zeros: {is_all_zeros}")
        print(f"      Constant values: {is_constant}")
        print(f"      Reasonable range: {is_reasonable_range}")
        print(f"      Extreme values: {has_extreme_values}")
        print(f"      Quality score: {quality_score}/4")
        
        return {
            'is_all_zeros': is_all_zeros,
            'is_constant': is_constant,
            'is_reasonable_range': is_reasonable_range,
            'has_extreme_values': has_extreme_values,
            'quality_score': quality_score,
            'quality_grade': ['F', 'D', 'C', 'B', 'A'][quality_score]
        }
    
    def _analyze_combined_vace_frames(self, 
                                    combined_tensor: torch.Tensor,
                                    reactive_latent: Optional[torch.Tensor] = None,
                                    inactive_latent: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Analyze the combined vace_frames tensor"""
        
        # Basic analysis of combined tensor
        combined_analysis = self._analyze_latent_component(combined_tensor, "combined", "vace_frames")
        
        # Verify concatenation
        concatenation_analysis = None
        if reactive_latent is not None and inactive_latent is not None:
            expected_channels = reactive_latent.shape[1] + inactive_latent.shape[1]
            actual_channels = combined_tensor.shape[1]
            
            concatenation_analysis = {
                'expected_channels': expected_channels,
                'actual_channels': actual_channels,
                'concatenation_correct': expected_channels == actual_channels,
                'reactive_channels': reactive_latent.shape[1],
                'inactive_channels': inactive_latent.shape[1]
            }
            
            print(f"   🔗 CONCATENATION ANALYSIS:")
            print(f"      Expected channels: {expected_channels}")
            print(f"      Actual channels: {actual_channels}")
            print(f"      Concatenation correct: {expected_channels == actual_channels}")
        
        # Channel separation analysis
        channel_separation = None
        if reactive_latent is not None and inactive_latent is not None:
            reactive_channels = reactive_latent.shape[1]
            inactive_channels = inactive_latent.shape[1]
            
            # Analyze first half (inactive) vs second half (reactive)
            if len(combined_tensor.shape) >= 4:
                inactive_part = combined_tensor[:, :inactive_channels, :, :]
                reactive_part = combined_tensor[:, inactive_channels:, :, :]
                
                inactive_mean = torch.mean(inactive_part).item()
                reactive_mean = torch.mean(reactive_part).item()
                inactive_range = torch.max(inactive_part).item() - torch.min(inactive_part).item()
                reactive_range = torch.max(reactive_part).item() - torch.min(reactive_part).item()
                
                channel_separation = {
                    'inactive_part_mean': inactive_mean,
                    'reactive_part_mean': reactive_mean,
                    'inactive_part_range': inactive_range,
                    'reactive_part_range': reactive_range,
                    'mean_difference': abs(inactive_mean - reactive_mean),
                    'range_difference': abs(inactive_range - reactive_range)
                }
                
                print(f"   🔀 CHANNEL SEPARATION ANALYSIS:")
                print(f"      Inactive part mean: {inactive_mean:.6f}")
                print(f"      Reactive part mean: {reactive_mean:.6f}")
                print(f"      Mean difference: {abs(inactive_mean - reactive_mean):.6f}")
                print(f"      Inactive part range: {inactive_range:.6f}")
                print(f"      Reactive part range: {reactive_range:.6f}")
                print(f"      Range difference: {abs(inactive_range - reactive_range):.6f}")
        
        combined_analysis['concatenation_analysis'] = concatenation_analysis
        combined_analysis['channel_separation'] = channel_separation
        
        return combined_analysis
    
    def _perform_comparative_analysis(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis between different latent components"""
        
        components = []
        if analysis_result['reactive_analysis']:
            components.append(('reactive', analysis_result['reactive_analysis']))
        if analysis_result['inactive_analysis']:
            components.append(('inactive', analysis_result['inactive_analysis']))
        if analysis_result['reference_analysis']:
            components.append(('reference', analysis_result['reference_analysis']))
        
        if len(components) < 2:
            return {'comparison_possible': False, 'reason': 'Insufficient components for comparison'}
        
        print(f"   🔄 COMPARATIVE ANALYSIS:")
        print(f"      Comparing {len(components)} components: {[c[0] for c in components]}")
        
        # Compare means
        means = {}
        ranges = {}
        for name, analysis in components:
            means[name] = analysis['mean_analysis']['global_mean']
            ranges[name] = analysis['range_analysis']['global_range']['span']
        
        print(f"      Global means: {means}")
        print(f"      Global ranges: {ranges}")
        
        # Calculate differences
        mean_differences = {}
        range_differences = {}
        
        for i, (name1, _) in enumerate(components):
            for j, (name2, _) in enumerate(components):
                if i < j:
                    key = f"{name1}_vs_{name2}"
                    mean_differences[key] = abs(means[name1] - means[name2])
                    range_differences[key] = abs(ranges[name1] - ranges[name2])
        
        print(f"      Mean differences: {mean_differences}")
        print(f"      Range differences: {range_differences}")
        
        return {
            'comparison_possible': True,
            'components_compared': [c[0] for c in components],
            'global_means': means,
            'global_ranges': ranges,
            'mean_differences': mean_differences,
            'range_differences': range_differences,
            'mean_consistency': np.std(list(means.values())),
            'range_consistency': np.std(list(ranges.values()))
        }
    
    def _generate_analysis_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the analysis results"""
        
        summary = {
            'analysis_id': analysis_result['analysis_id'],
            'context': analysis_result['context'],
            'components_analyzed': [],
            'key_findings': [],
            'quality_assessment': {},
            'recommendations': []
        }
        
        # Collect component information
        if analysis_result['reactive_analysis']:
            summary['components_analyzed'].append('reactive')
        if analysis_result['inactive_analysis']:
            summary['components_analyzed'].append('inactive')
        if analysis_result['reference_analysis']:
            summary['components_analyzed'].append('reference')
        if analysis_result['combined_analysis']:
            summary['components_analyzed'].append('combined')
        
        # Key findings
        for component in summary['components_analyzed']:
            if component in ['reactive', 'inactive', 'reference']:
                analysis = analysis_result[f'{component}_analysis']
                mean = analysis['mean_analysis']['global_mean']
                range_span = analysis['range_analysis']['global_range']['span']
                quality = analysis['quality_analysis']['quality_grade']
                
                summary['key_findings'].append(
                    f"{component.upper()}: mean={mean:.6f}, range={range_span:.6f}, quality={quality}"
                )
        
        # Quality assessment
        if analysis_result['comparative_analysis'] and analysis_result['comparative_analysis']['comparison_possible']:
            mean_consistency = analysis_result['comparative_analysis']['mean_consistency']
            range_consistency = analysis_result['comparative_analysis']['range_consistency']
            
            summary['quality_assessment'] = {
                'mean_consistency': mean_consistency,
                'range_consistency': range_consistency,
                'overall_consistency': 'Good' if mean_consistency < 0.1 and range_consistency < 0.1 else 'Moderate'
            }
        
        # Recommendations
        if summary['quality_assessment'].get('overall_consistency') == 'Moderate':
            summary['recommendations'].append("Consider investigating mean/range inconsistencies between components")
        
        return summary
    
    def _print_analysis_summary(self, analysis_result: Dict[str, Any]):
        """Print a summary of the analysis results"""
        
        summary = analysis_result['summary']
        
        print(f"\n📋 ANALYSIS SUMMARY:")
        print(f"   Analysis ID: {summary['analysis_id']}")
        print(f"   Context: {summary['context']}")
        print(f"   Components analyzed: {', '.join(summary['components_analyzed'])}")
        print(f"   Key findings:")
        for finding in summary['key_findings']:
            print(f"      • {finding}")
        
        if summary['quality_assessment']:
            print(f"   Quality assessment: {summary['quality_assessment']}")
        
        if summary['recommendations']:
            print(f"   Recommendations:")
            for rec in summary['recommendations']:
                print(f"      • {rec}")
    
    def save_analysis_results(self, filename: str = None):
        """Save analysis results to a JSON file"""
        
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"vae_analysis_results_{timestamp}.json"
        
        # Convert tensors to lists for JSON serialization
        serializable_results = []
        for result in self.analysis_results:
            serializable_result = self._make_serializable(result)
            serializable_results.append(serializable_result)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Analysis results saved to: {filename}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


# Example usage and testing
if __name__ == "__main__":
    # Create analyzer
    analyzer = VAEOutputAnalyzer()
    
    # Example tensors (replace with actual VAE outputs)
    reactive_latent = torch.randn(37, 16, 104, 60)  # Example reactive latent
    inactive_latent = torch.randn(37, 16, 104, 60)  # Example inactive latent
    reference_latent = torch.randn(1, 16, 104, 60)  # Example reference latent
    combined_vace_frames = torch.cat([inactive_latent, reactive_latent], dim=1)  # Combined
    
    # Perform analysis
    results = analyzer.analyze_vae_outputs(
        reactive_latent=reactive_latent,
        inactive_latent=inactive_latent,
        reference_latent=reference_latent,
        combined_vace_frames=combined_vace_frames,
        analysis_context="Example VAE output analysis"
    )
    
    # Save results
    analyzer.save_analysis_results()
    
    print("\n✅ VAE Output Analysis Complete!")
