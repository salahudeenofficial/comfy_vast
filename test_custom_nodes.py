#!/usr/bin/env python3
"""
Simple test file to debug custom node loading issues on Vast AI
"""

import os
import sys

def test_basic_imports():
    """Test basic imports"""
    print("=== Testing Basic Imports ===")
    
    try:
        # Test main ComfyUI nodes
        from nodes import NODE_CLASS_MAPPINGS
        print(f"✓ Main nodes loaded: {len(NODE_CLASS_MAPPINGS)} nodes")
        
        # Check for expected nodes
        expected = ['LoadImage', 'VAELoader', 'UNETLoader', 'CLIPLoader']
        for node in expected:
            if node in NODE_CLASS_MAPPINGS:
                print(f"  ✓ {node}")
            else:
                print(f"  ✗ {node}")
                
        return True
    except Exception as e:
        print(f"✗ Error loading main nodes: {e}")
        return False

def test_custom_node_files():
    """Test if custom node files exist"""
    print("\n=== Testing Custom Node Files ===")
    
    # Check custom_nodes directory
    if not os.path.exists('custom_nodes'):
        print("✗ custom_nodes directory not found")
        return False
    
    print("✓ custom_nodes directory exists")
    
    # List custom node directories
    custom_dirs = []
    for item in os.listdir('custom_nodes'):
        if os.path.isdir(os.path.join('custom_nodes', item)):
            custom_dirs.append(item)
    
    print(f"Found custom node directories: {custom_dirs}")
    
    # Check videohelpersuite specifically
    vhs_path = 'custom_nodes/comfyui-videohelpersuite'
    if os.path.exists(vhs_path):
        print("✓ videohelpersuite directory exists")
        
        # Check key files
        files_to_check = ['__init__.py', 'videohelpersuite/nodes.py']
        for file in files_to_check:
            full_path = os.path.join(vhs_path, file)
            if os.path.exists(full_path):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file}")
    else:
        print("✗ videohelpersuite directory not found")
        return False
    
    return True

def test_dependencies():
    """Test if required dependencies are available"""
    print("\n=== Testing Dependencies ===")
    
    # Test OpenCV
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not available")
        return False
    
    # Test imageio-ffmpeg
    try:
        import imageio_ffmpeg
        print("✓ imageio-ffmpeg")
    except ImportError:
        print("✗ imageio-ffmpeg not available")
        return False
    
    return True

def test_custom_node_loading():
    """Test if custom nodes can be loaded"""
    print("\n=== Testing Custom Node Loading ===")
    
    try:
        # Try to import the custom nodes module
        sys.path.insert(0, 'custom_nodes/comfyui-videohelpersuite')
        
        # Import using string to avoid linter issues
        exec("import videohelpersuite.nodes")
        print("✓ Successfully imported videohelpersuite.nodes")
        
        # Get the NODE_CLASS_MAPPINGS
        exec("from videohelpersuite.nodes import NODE_CLASS_MAPPINGS")
        
        # Check if VHS_LoadVideo exists
        if 'VHS_LoadVideo' in locals()['NODE_CLASS_MAPPINGS']:
            print("✓ VHS_LoadVideo found in custom nodes")
            return True
        else:
            print("✗ VHS_LoadVideo not found in custom nodes")
            return False
            
    except Exception as e:
        print(f"✗ Error loading custom nodes: {e}")
        return False

def main():
    """Run all tests"""
    print("Custom Node Loading Debug Test")
    print("=" * 50)
    
    tests = [
        ("Basic imports", test_basic_imports),
        ("Custom node files", test_custom_node_files),
        ("Dependencies", test_dependencies),
        ("Custom node loading", test_custom_node_loading)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        result = test_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    
    passed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed! Custom nodes should work properly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main() 