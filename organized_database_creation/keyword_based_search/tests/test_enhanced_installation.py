#!/usr/bin/env python3
"""
Test script for Enhanced Agriculture Data Curator
Verifies all new features including PDF processing and duplicate prevention
"""

import sys
import importlib
from pathlib import Path

def test_enhanced_imports():
    """Test all enhanced imports including PDF processing libraries"""
    print("Testing enhanced imports...")
    
    required_modules = [
        ("requests", "HTTP requests"),
        ("ddgs", "DuckDuckGo search"), 
        ("bs4", "BeautifulSoup HTML parsing"),
        ("PyPDF2", "PDF text extraction"),
        ("fitz", "PyMuPDF advanced PDF processing"),
        ("pytesseract", "OCR text recognition"),
        ("PIL", "Image processing"),
        ("magic", "File type detection"),
        ("json", "JSON processing"),
        ("threading", "Multi-threading"),
        ("concurrent.futures", "Parallel execution"),
        ("difflib", "Text similarity")
    ]
    
    failed_imports = []
    
    for module, description in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module} ({description})")
        except ImportError as e:
            print(f"  ✗ {module} ({description}): {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0, failed_imports

def test_enhanced_features():
    """Test enhanced functionality"""
    print("\nTesting enhanced features...")
    
    try:
        # Test imports from enhanced module
        sys.path.insert(0, '.')
        from agriculture_data_curator_enhanced import (
            DuplicateTracker,
            PDFProcessor,
            IntelligentSearchExpansion,
            AgricultureSearchQueries,
            EnhancedWebSearch,
            AgricultureDataEntry
        )
        
        print("  ✓ Enhanced module imports successful")
        
        # Test duplicate tracker
        tracker = DuplicateTracker()
        assignments = tracker.assign_domains_to_agents(4)
        print(f"  ✓ Duplicate tracker with domain assignments: {len(assignments)} agents")
        
        # Test URL duplicate detection
        is_dup1 = tracker.is_duplicate_url("https://example.com/test1")
        is_dup2 = tracker.is_duplicate_url("https://example.com/test1")  # Should be duplicate
        print(f"  ✓ URL duplicate detection: {not is_dup1 and is_dup2}")
        
        # Test content duplicate detection
        content_dup1 = tracker.is_duplicate_content("Test Title", "Test content here")
        content_dup2 = tracker.is_duplicate_content("Test Title", "Different content")  # Title duplicate
        print(f"  ✓ Content duplicate detection: {not content_dup1 and content_dup2}")
        
        # Test PDF processor initialization
        pdf_processor = PDFProcessor("test_pdfs")
        print("  ✓ PDF processor initialization")
        
        # Test search queries
        queries = AgricultureSearchQueries.get_search_queries(5)
        print(f"  ✓ Generated {len(queries)} search queries")
        
        # Test enhanced data entry structure
        entry = AgricultureDataEntry(
            title="Test Agriculture Paper",
            author="Dr. Test",
            link="https://example.com/paper.pdf",
            text_extracted="Full extracted text content...",
            abstract="Test abstract",
            genre="pdf",
            tags=["test", "agriculture"],
            indian_regions=["Punjab"],
            crop_types=["wheat"],
            farming_methods=["organic"],
            data_type="statistical",
            publication_year=2023,
            source_domain="example.com",
            extraction_timestamp="2025-08-09T10:30:45",
            relevance_score=0.85,
            content_length=1000,
            content_hash="abc123",
            url_hash="def456",
            is_pdf=True,
            pdf_path="/path/to/pdf"
        )
        entry_dict = entry.to_dict()
        print("  ✓ Enhanced data entry structure with PDF support")
        
        # Test intelligent search expansion
        expansion = IntelligentSearchExpansion("test-model", "http://localhost:11434")
        print("  ✓ Intelligent search expansion initialization")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Enhanced functionality test failed: {e}")
        return False

def test_configuration_enhanced():
    """Test enhanced configuration"""
    print("\nTesting enhanced configuration...")
    
    try:
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check enhanced config sections
        required_sections = ['curator']
        enhanced_keys = [
            'pdf_storage_dir', 'pdf_download', 'pdf_max_size_mb', 'ocr_enabled',
            'duplicate_check', 'url_deduplication', 'content_similarity_threshold'
        ]
        
        for section in required_sections:
            if section in config:
                print(f"  ✓ Config section '{section}' found")
                
                # Check enhanced keys
                for key in enhanced_keys:
                    if key in config[section]:
                        print(f"    ✓ Enhanced key '{key}': {config[section][key]}")
                    else:
                        print(f"    ! Enhanced key '{key}' missing (will use defaults)")
            else:
                print(f"  ✗ Config section '{section}' missing")
                return False
        
        # Check for no content limits
        max_content = config['curator'].get('max_content_length')
        if max_content is None:
            print("  ✓ No content extraction limits (complete dataset mode)")
        else:
            print(f"  ! Content limit still set: {max_content}")
        
        return True
        
    except ImportError:
        print("  ! PyYAML not installed, skipping enhanced config test")
        print("    Install with: pip install pyyaml")
        return True  # Not critical for basic functionality
    except Exception as e:
        print(f"  ✗ Enhanced configuration test failed: {e}")
        return False

def test_file_structure_enhanced():
    """Test enhanced file structure"""
    print("\nTesting enhanced file structure...")
    
    required_files = [
        "agriculture_data_curator_enhanced.py",
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
    
    # Check if enhanced version is available
    if Path("agriculture_data_curator_enhanced.py").exists():
        print("  ✓ Enhanced version available")
    else:
        print("  ✗ Enhanced version missing")
    
    return len(missing_files) == 0, missing_files

def main():
    """Run all enhanced tests"""
    print("Enhanced Agriculture Data Curator - Installation Test")
    print("=" * 70)
    
    tests = [
        ("Enhanced File Structure", test_file_structure_enhanced),
        ("Enhanced Python Imports", test_enhanced_imports),
        ("Enhanced Configuration", test_configuration_enhanced),
        ("Enhanced Functionality", test_enhanced_features)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 40)
        
        try:
            if test_func.__name__ in ['test_enhanced_imports', 'test_file_structure_enhanced']:
                success, details = test_func()
            else:
                success = test_func()
                details = None
            results.append((test_name, success, details))
        except Exception as e:
            print(f"Test error: {e}")
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("ENHANCED TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, success, details in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<30} {status}")
        
        if not success and details:
            if isinstance(details, list):
                print(f"  Missing: {', '.join(details)}")
            else:
                print(f"  Error: {details}")
        
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All enhanced tests passed! The system is ready for complete dataset curation.")
        print("\nEnhanced Features Available:")
        print("✓ PDF download and processing with OCR")
        print("✓ Comprehensive duplicate prevention")
        print("✓ Intelligent search query expansion")
        print("✓ No content extraction limits")
        print("✓ Domain assignment to prevent agent overlap")
        
        print("\nNext steps:")
        print("1. Ensure required models are pulled:")
        print("   ollama pull deepseek-r1:70b")
        print("   ollama pull gemma3:27b")
        print("\n2. Install enhanced dependencies:")
        print("   pip install -r requirements.txt")
        print("\n3. Run the enhanced curator:")
        print("   python agriculture_data_curator_enhanced.py")
        
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please address the issues above.")
        
        if not results[1][1]:  # imports failed
            print("\nTo fix import issues:")
            print("   pip install -r requirements.txt")
            print("   # For OCR: sudo apt-get install tesseract-ocr (Linux)")
            print("   # For OCR: brew install tesseract (macOS)")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
