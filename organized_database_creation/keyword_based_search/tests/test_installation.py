#!/usr/bin/env python3
"""
Test script for Agriculture Data Curator
Verifies installation and basic functionality
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    required_modules = [
        "requests",
        "ddgs", 
        "bs4",
        "json",
        "threading",
        "concurrent.futures"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0, failed_imports

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "agriculture_data_curator.py",
        "requirements.txt",
        "config.yaml",
        "setup.py",
        "README.md"
    ]
    
    missing_files = []
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name}")
            missing_files.append(file_name)
    
    return len(missing_files) == 0, missing_files

def test_ollama_connection():
    """Test Ollama connection"""
    print("\nTesting Ollama connection...")
    
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("  ✓ Ollama is running")
            print(f"  Available models:")
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line.strip():
                    model_name = line.split()[0]
                    if model_name != 'NAME':
                        print(f"    - {model_name}")
            return True
        else:
            print("  ✗ Ollama not responding")
            return False
            
    except FileNotFoundError:
        print("  ✗ Ollama not installed")
        return False
    except subprocess.TimeoutExpired:
        print("  ✗ Ollama connection timeout")
        return False
    except Exception as e:
        print(f"  ✗ Ollama test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without running full curation"""
    print("\nTesting basic functionality...")
    
    try:
        # Test imports from main module
        sys.path.insert(0, '.')
        from agriculture_data_curator import (
            AgricultureSearchQueries, 
            EnhancedWebSearch,
            AgricultureDataEntry
        )
        
        print("  ✓ Main module imports successful")
        
        # Test search queries
        queries = AgricultureSearchQueries.get_search_queries(5)
        print(f"  ✓ Generated {len(queries)} search queries")
        
        # Test data entry structure
        entry = AgricultureDataEntry(
            title="Test",
            author="Test Author",
            link="https://example.com",
            text_extracted="Test content",
            abstract="Test abstract",
            genre="article",
            tags=["test"],
            indian_regions=["Punjab"]
        )
        entry_dict = entry.to_dict()
        print("  ✓ Data entry structure works")
        
        # Test search engine initialization (without actual search)
        search_engine = EnhancedWebSearch(max_results=3)
        print("  ✓ Search engine initialization successful")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Basic functionality test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['curator', 'search_categories', 'priority_domains']
        for section in required_sections:
            if section in config:
                print(f"  ✓ Config section '{section}' found")
            else:
                print(f"  ✗ Config section '{section}' missing")
                return False
        
        return True
        
    except ImportError:
        print("  ! PyYAML not installed, skipping config test")
        print("    Install with: pip install pyyaml")
        return True  # Not critical for basic functionality
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Agriculture Data Curator - Installation Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Imports", test_imports),
        ("Configuration", test_configuration),
        ("Basic Functionality", test_basic_functionality),
        ("Ollama Connection", test_ollama_connection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 30)
        
        try:
            success, details = test_func() if test_func.__name__ in ['test_imports', 'test_file_structure'] else (test_func(), None)
            results.append((test_name, success, details))
        except Exception as e:
            print(f"Test error: {e}")
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success, details in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<20} {status}")
        
        if not success and details:
            if isinstance(details, list):
                print(f"  Missing: {', '.join(details)}")
            else:
                print(f"  Error: {details}")
        
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Ensure required models are pulled:")
        print("   ollama pull deepseek-r1:70b")
        print("   ollama pull gemma3:27b")
        print("\n2. Run the curator:")
        print("   python agriculture_data_curator.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please address the issues above.")
        
        if not results[1][1]:  # imports failed
            print("\nTo fix import issues:")
            print("   pip install -r requirements.txt")
        
        if not results[4][1]:  # ollama failed
            print("\nTo fix Ollama issues:")
            print("   Visit https://ollama.ai for installation instructions")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
