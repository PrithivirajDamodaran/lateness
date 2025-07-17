#!/usr/bin/env python3
"""
Test script to verify ColBERT + Qdrant setup
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'numpy',
        'requests', 
        'tqdm',
        'onnxruntime',
        'tokenizers',
        'qdrant_client'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'qdrant_client':
                import qdrant_client
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_docker():
    """Check if Docker is available and running"""
    print("\nChecking Docker...")
    
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker installed: {result.stdout.strip()}")
        else:
            print("❌ Docker not found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Docker not found or not responding")
        return False
    
    try:
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Docker daemon is running")
        else:
            print("❌ Docker daemon not running")
            print("Start Docker first, then run this test again")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Docker daemon not responding")
        return False
    
    return True

def check_qdrant_connection():
    """Check if Qdrant is accessible"""
    print("\nChecking Qdrant connection...")
    
    try:
        import requests
        
        # Try multiple endpoints to check if Qdrant is running
        endpoints_to_try = [
            'http://localhost:6333/',
            'http://localhost:6333/health',
            'http://localhost:6333/collections'
        ]
        
        for endpoint in endpoints_to_try:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code in [200, 404]:  # 404 is OK for collections endpoint when empty
                    print("✅ Qdrant is running and accessible")
                    
                    # Try to check collections specifically
                    try:
                        collections_response = requests.get('http://localhost:6333/collections', timeout=5)
                        if collections_response.status_code == 200:
                            collections_data = collections_response.json()
                            if 'result' in collections_data and 'collections' in collections_data['result']:
                                collections = collections_data['result']['collections']
                                if collections:
                                    print(f"✅ Found {len(collections)} collection(s)")
                                else:
                                    print("✅ No collections yet (this is normal)")
                    except:
                        pass  # Collections check is optional
                    
                    return True
            except:
                continue
        
        print("❌ Cannot connect to Qdrant on any endpoint")
        print("Start Qdrant with: ./qdrant_manager.sh start")
        return False
        
    except ImportError:
        print("❌ requests library not available")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        return False

def test_modern_colbert():
    """Test if ModernColBERT class can be imported"""
    print("\nTesting ModernColBERT import...")
    
    try:
        # Try to import from lateness package
        from lateness import ModernColBERT
        print("✅ ModernColBERT imports successfully")
        
        # Test basic instantiation (without actually loading model)
        print("✅ ModernColBERT class found and accessible")
        return True
            
    except ImportError as e:
        print(f"❌ Cannot import ModernColBERT: {e}")
        print("Make sure lateness package is installed: pip install -e .")
        return False
    except Exception as e:
        print(f"❌ Error with ModernColBERT: {e}")
        return False

def check_model_download():
    """Check if model can be downloaded from Hugging Face"""
    print("\nChecking Hugging Face model access...")
    
    try:
        from huggingface_hub import model_info
        model_name = "prithivida/modern_colbert_base_en_v1"
        
        # Check if model exists on Hugging Face
        info = model_info(model_name)
        print(f"✅ Model found on Hugging Face: {model_name}")
        print(f"✅ Model ID: {info.modelId}")
        return True
        
    except ImportError:
        print("❌ huggingface_hub not installed")
        return False
    except Exception as e:
        print(f"❌ Cannot access model on Hugging Face: {e}")
        print("This might be due to network issues or model access restrictions")
        return False

def main():
    """Run all tests"""
    print("🔍 ColBERT + Qdrant Setup Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Docker", check_docker),
        ("Qdrant Connection", check_qdrant_connection),
        ("ModernColBERT Import", test_modern_colbert),
        ("Model Download", check_model_download),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! You're ready to run the evaluation.")
        print("\nNext steps:")
        print("1. Start Qdrant: ./qdrant_manager.sh start --clear")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please fix the issues above.")
        
        if not any(name == "Dependencies" and result for name, result in results):
            print("\n💡 Try: pip install -r requirements.txt")
        
        if not any(name == "Docker" and result for name, result in results):
            print("💡 Make sure Docker is installed and running")
        
        if not any(name == "Qdrant Connection" and result for name, result in results):
            print("💡 Start Qdrant with: ./qdrant_manager.sh start")

if __name__ == "__main__":
    main()
