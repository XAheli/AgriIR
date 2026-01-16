#!/usr/bin/env python3
"""
Main script for creating embeddings from agricultural datasets
"""

import argparse
import sys
import os
import logging
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding_system import AgricultureEmbeddingSystem

def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('embedding_generation.log')
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate embeddings from agricultural JSONL dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python create_embeddings.py --input data.jsonl --output embeddings_output
  
  # Using configuration file
  python create_embeddings.py --config config/qwen_config.yaml --input data.jsonl
  
  # Process only first 1000 records
  python create_embeddings.py --input data.jsonl --max-records 1000
  
  # Use CPU instead of GPU
  python create_embeddings.py --input data.jsonl --device cpu
  
  # Custom chunk size
  python create_embeddings.py --input data.jsonl --chunk-size 512
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input JSONL file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='embeddings_output',
        help='Output directory for embeddings (default: embeddings_output)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-Embedding-8B',
        help='HuggingFace model name (default: Qwen/Qwen3-Embedding-8B)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=256,
        help='Maximum tokens per chunk (default: 256)'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=25,
        help='Overlap between chunks in tokens (default: 25)'
    )
    
    parser.add_argument(
        '--max-records',
        type=int,
        default=None,
        help='Maximum number of records to process (default: all)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for processing (default: auto)'
    )
    
    parser.add_argument(
        '--index-type',
        type=str,
        default='flat',
        choices=['flat', 'ivf'],
        help='FAISS index type (default: flat)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--filter-domain',
        type=str,
        default=None,
        help='Filter records by source domain'
    )
    
    return parser.parse_args()

def create_filter_function(filter_domain: str = None):
    """Create a filter function based on arguments"""
    if not filter_domain:
        return None
    
    def filter_func(record):
        return record.get('source_domain', '') == filter_domain
    
    return filter_func

def main():
    """Main function"""
    args = parse_arguments()
    
    # Load configuration if provided
    config = {}
    if args.config:
        try:
            config = load_config(args.config)
            print(f"✅ Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            sys.exit(1)
    
    # Merge config with command line arguments (CLI args take precedence)
    model_name = args.model or config.get('model', {}).get('name', 'Qwen/Qwen3-Embedding-8B')
    chunk_size = args.chunk_size or config.get('text_processing', {}).get('chunk_size', 256)
    chunk_overlap = args.chunk_overlap or config.get('text_processing', {}).get('chunk_overlap', 25)
    device = args.device or config.get('model', {}).get('device', 'auto')
    index_type = args.index_type or config.get('index', {}).get('type', 'flat')
    output_dir = args.output or config.get('output', {}).get('default_output_dir', 'embeddings_output')
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting embedding generation...")
    if args.config:
        logger.info(f"Configuration file: {args.config}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Chunk overlap: {chunk_overlap}")
    logger.info(f"Max records: {args.max_records or 'all'}")
    logger.info(f"Device: {device}")
    logger.info(f"Index type: {index_type}")
    
    try:
        # Initialize embedding system
        embedding_system = AgricultureEmbeddingSystem(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            device=device
        )
        
        # Create filter function
        filter_func = create_filter_function(args.filter_domain)
        if filter_func:
            logger.info(f"Filtering by domain: {args.filter_domain}")
        
        # Process dataset
        processed_records, total_chunks = embedding_system.process_dataset(
            args.input,
            max_records=args.max_records,
            filter_function=filter_func
        )
        
        if total_chunks == 0:
            logger.warning("No chunks created. Check your input data.")
            sys.exit(1)
        
        # Build FAISS index
        logger.info(f"Building {index_type} FAISS index...")
        embedding_system.build_faiss_index(index_type=index_type)
        
        # Save everything
        logger.info("Saving embeddings and metadata...")
        embedding_system.save_embeddings(output_dir)
        
        logger.info("Embedding generation completed successfully!")
        logger.info(f"- Processed {processed_records} records")
        logger.info(f"- Created {total_chunks} text chunks")
        logger.info(f"- Saved to {output_dir}/")
        logger.info(f"- FAISS index ready for similarity search")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()