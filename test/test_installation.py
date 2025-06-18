# Test Installation Script
# File: test/test_installation.py

import sys

def test_imports():
    """Test if all required packages can be imported"""
    
    required_packages = [
        'torch',
        'numpy',
        'matplotlib',
        'seaborn', 
        'sklearn',
        'scipy',
        'torchdiffeq'
    ]
    
    print("Testing package imports...")
    print("-" * 40)
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
                print(f"✓ {package:<15} - Version: {sklearn.__version__}")
            elif package == 'torchdiffeq':
                import torchdiffeq
                print(f"✓ {package:<15} - Available")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
                print(f"✓ {package:<15} - Version: {version}")
        except ImportError as e:
            print(f"✗ {package:<15} - Failed: {e}")
            failed_imports.append(package)
    
    print("-" * 40)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True

def test_torch_functionality():
    """Test basic PyTorch functionality"""
    import torch
    import torch.nn as nn
    
    print("\nTesting PyTorch functionality...")
    print("-" * 40)
    
    try:
        # Test tensor creation
        x = torch.randn(5, 3)
        print(f"✓ Tensor creation: {x.shape}")
        
        # Test neural network
        net = nn.Linear(3, 1)
        y = net(x)
        print(f"✓ Neural network: Input {x.shape} -> Output {y.shape}")
        
        # Test gradient computation
        x.requires_grad_(True)
        y = torch.sum(x**2)
        y.backward()
        print(f"✓ Gradient computation: {x.grad.shape}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} device(s)")
        else:
            print("ℹ CUDA not available (CPU only)")
        
        print("✅ PyTorch functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch functionality test failed: {e}")
        return False

def test_ode_solver():
    """Test ODE solver functionality"""
    try:
        import torch
        from torchdiffeq import odeint
        
        print("\nTesting ODE solver...")
        print("-" * 40)
        
        # Simple ODE: dy/dt = -y, solution: y(t) = y0 * exp(-t)
        def simple_ode(t, y):
            return -y
        
        y0 = torch.tensor([1.0])
        t = torch.tensor([0., 1.])
        
        solution = odeint(simple_ode, y0, t, method='dopri5')
        expected = torch.exp(-t).unsqueeze(1)
        error = torch.abs(solution - expected).max()
        
        print(f"✓ ODE solution error: {error.item():.6f}")
        
        if error < 1e-3:
            print("✅ ODE solver test passed!")
            return True
        else:
            print("❌ ODE solver test failed: High error")
            return False
            
    except Exception as e:
        print(f"❌ ODE solver test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("ENGINEERING ANALYTICS ASSIGNMENT 3")
    print("INSTALLATION TEST")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Package imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: PyTorch functionality
    if test_torch_functionality():
        tests_passed += 1
    
    # Test 3: ODE solver
    if test_ode_solver():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! You're ready to run the assignment.")
        print("\nNext steps:")
        print("1. Run Question 1: python src/pinn_cardiac_activation.py")
        print("2. Run Question 2: python src/neural_ode_classification.py")
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Update PyTorch: pip install torch --upgrade")
        print("- Install torchdiffeq: pip install torchdiffeq")

if __name__ == "__main__":
    main()