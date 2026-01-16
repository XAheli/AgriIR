#!/usr/bin/env python3
"""
Unified Agriculture Data Curator with LLM Support
Combines the best features from both curators:
- Advanced PDF processing with OCR from agriculture_curator_fixed.py
- LLM-powered intelligent search expansion from agriculture_data_curator_enhanced.py
- Persistent duplicate tracking with immediate JSONL writing
- Configurable LLM models via YAML config
"""

import yaml
from pathlib import Path
from typing import Optional, Dict
import logging

# Import the enhanced curator
import sys
sys.path.insert(0, str(Path(__file__).parent))
from agriculture_data_curator_enhanced import AgricultureDataCurator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agriculture_curator_unified.log'),
        logging.StreamHandler()
    ]
)


def load_config(config_path: str = "config/llm_config.yaml") -> Dict:
    """Load configuration from YAML file"""
    config_file = Path(__file__).parent / config_path
    
    if not config_file.exists():
        logging.warning(f"Config file not found: {config_file}, using defaults")
        return get_default_config()
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"✅ Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}, using defaults")
        return get_default_config()


def get_default_config() -> Dict:
    """Get default configuration"""
    return {
        'model': {
            'name': 'deepseek-r1:70b',
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 2000
        },
        'agents': {
            'num_agents': 4,
            'enable_intelligent_expansion': True,
            'enable_pdf_download': True
        },
        'search': {
            'max_search_results': 25,
            'num_queries': None
        },
        'output': {
            'output_file': 'indian_agriculture_data_complete.jsonl',
            'pdf_storage_dir': 'downloaded_pdfs'
        },
        'ollama': {
            'base_url': 'http://localhost:11434',
            'port_start': 11434
        },
        'features': {
            'duplicate_prevention': True,
            'pdf_ocr': True,
            'metadata_extraction': True,
            'structured_fields': True
        }
    }


def main():
    """Main function to run the unified agriculture data curator"""
    
    # Load configuration
    config = load_config()
    
    # Extract configuration values
    model_config = config.get('model', {})
    agent_config = config.get('agents', {})
    search_config = config.get('search', {})
    output_config = config.get('output', {})
    
    print("=" * 80)
    print("UNIFIED AGRICULTURE DATA CURATOR WITH LLM SUPPORT")
    print("=" * 80)
    print(f"LLM Model: {model_config.get('name', 'deepseek-r1:70b')}")
    print(f"Number of Agents: {agent_config.get('num_agents', 4)}")
    print(f"PDF Download: {'Enabled' if agent_config.get('enable_pdf_download') else 'Disabled'}")
    print(f"Intelligent Expansion: {'Enabled' if agent_config.get('enable_intelligent_expansion') else 'Disabled'}")
    print(f"Output File: {output_config.get('output_file', 'indian_agriculture_data_complete.jsonl')}")
    print("=" * 80)
    
    # Create and run curator with loaded configuration
    curator = AgricultureDataCurator(
        num_agents=agent_config.get('num_agents', 4),
        model=model_config.get('name', 'deepseek-r1:70b'),
        output_file=output_config.get('output_file', 'indian_agriculture_data_complete.jsonl'),
        max_search_results=search_config.get('max_search_results', 25),
        pdf_storage_dir=output_config.get('pdf_storage_dir', 'downloaded_pdfs'),
        enable_pdf_download=agent_config.get('enable_pdf_download', True),
        enable_intelligent_expansion=agent_config.get('enable_intelligent_expansion', True)
    )
    
    try:
        summary = curator.start_curation(num_queries=search_config.get('num_queries'))
        
        print("\n" + "=" * 80)
        print("AGRICULTURE DATA CURATION COMPLETED")
        print("=" * 80)
        print(f"✅ Total entries curated: {summary.get('total_entries', 0)}")
        print(f"📄 PDF files processed: {summary.get('pdf_count', 0)}")
        print(f"⏱️  Execution time: {summary.get('execution_time', 0):.2f} seconds")
        print(f"💾 Output file: {summary.get('output_file', 'N/A')}")
        print(f"✅ Successful agents: {summary.get('successful_agents', 0)}")
        print(f"❌ Failed agents: {summary.get('failed_agents', 0)}")
        print(f"🔄 Duplicate prevention: {'Enabled' if summary.get('duplicate_prevention_enabled') else 'Disabled'}")
        print(f"🧠 Intelligent expansion: {'Enabled' if summary.get('intelligent_expansion_enabled') else 'Disabled'}")
        print(f"📥 PDF download: {'Enabled' if summary.get('pdf_download_enabled') else 'Disabled'}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Curation interrupted by user")
    except Exception as e:
        print(f"❌ Curation failed: {e}")
        logging.error(f"Main execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
