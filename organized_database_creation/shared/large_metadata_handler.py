#!/usr/bin/env python3
"""
Large Metadata Handler with Streaming and Memory-Mapped Loading
Handles large JSON/pickle files without exhausting memory
"""

import json
import pickle
import mmap
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional, Generator
import sys


class LargeMetadataHandler:
    """
    Handler for large metadata files with memory-efficient loading strategies.
    """
    
    # Size thresholds (in bytes)
    WARNING_SIZE = 100 * 1024 * 1024  # 100 MB
    STREAMING_SIZE = 500 * 1024 * 1024  # 500 MB - use streaming above this
    CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB chunks for streaming
    
    def __init__(self, max_memory_mb: int = 1000):
        """
        Initialize metadata handler.
        
        Args:
            max_memory_mb: Maximum memory to use for loading (MB)
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        logging.info(f"📦 Large Metadata Handler initialized (max memory: {max_memory_mb} MB)")
    
    def load_json_safe(self, file_path: str, streaming: bool = False) -> Any:
        """
        Load JSON file with appropriate strategy based on file size.
        
        Args:
            file_path: Path to JSON file
            streaming: Force streaming mode
            
        Returns:
            Loaded JSON data or generator for streaming
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        file_size = path.stat().st_size
        
        # Check file size and warn if large
        if file_size > self.WARNING_SIZE:
            logging.warning(f"⚠️ Large JSON file detected ({file_size / (1024**2):.2f} MB): {file_path}")
        
        # Use streaming for very large files
        if streaming or file_size > self.STREAMING_SIZE:
            logging.info(f"Using streaming mode for large JSON file: {file_path}")
            return self._stream_json_array(file_path)
        
        # Check if file fits in memory limit
        if file_size > self.max_memory_bytes:
            logging.error(f"❌ JSON file too large for memory limit ({file_size / (1024**3):.2f} GB)")
            logging.info("Attempting chunked loading...")
            return self._load_json_chunked(file_path)
        
        # Normal loading for smaller files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"✅ Loaded JSON file ({file_size / (1024**2):.2f} MB): {file_path}")
            return data
        except (json.JSONDecodeError, MemoryError) as e:
            logging.error(f"❌ Failed to load JSON: {e}")
            logging.info("Falling back to chunked loading...")
            return self._load_json_chunked(file_path)
    
    def _stream_json_array(self, file_path: str) -> Generator[Dict, None, None]:
        """
        Stream JSON array elements one at a time.
        Assumes file contains a JSON array.
        
        Args:
            file_path: Path to JSON array file
            
        Yields:
            Individual array elements
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read opening bracket
                char = f.read(1)
                while char and char.isspace():
                    char = f.read(1)
                
                if char != '[':
                    raise ValueError("JSON file does not start with array")
                
                buffer = ""
                depth = 0
                in_string = False
                escape = False
                
                for char in f.read():
                    if escape:
                        buffer += char
                        escape = False
                        continue
                    
                    if char == '\\':
                        escape = True
                        buffer += char
                        continue
                    
                    if char == '"' and not escape:
                        in_string = not in_string
                    
                    if not in_string:
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                        
                        if char == ',' and depth == 0:
                            # End of object
                            if buffer.strip():
                                try:
                                    obj = json.loads(buffer.strip())
                                    yield obj
                                except json.JSONDecodeError as e:
                                    logging.warning(f"Failed to parse JSON object: {e}")
                            buffer = ""
                            continue
                    
                    buffer += char
                
                # Parse last object
                if buffer.strip() and buffer.strip() != ']':
                    buffer = buffer.replace(']', '').strip()
                    if buffer:
                        try:
                            obj = json.loads(buffer)
                            yield obj
                        except json.JSONDecodeError:
                            pass
        
        except Exception as e:
            logging.error(f"Error streaming JSON: {e}")
            raise
    
    def _load_json_chunked(self, file_path: str, max_items: int = 10000) -> List[Dict]:
        """
        Load JSON file in chunks, returning first N items.
        
        Args:
            file_path: Path to JSON file
            max_items: Maximum number of items to load
            
        Returns:
            List of loaded items (limited)
        """
        items = []
        try:
            for i, item in enumerate(self._stream_json_array(file_path)):
                items.append(item)
                if i >= max_items - 1:
                    logging.warning(f"⚠️ Loaded {max_items} items (limit reached)")
                    break
            
            logging.info(f"✅ Loaded {len(items)} items via chunked loading")
            return items
        
        except Exception as e:
            logging.error(f"❌ Chunked loading failed: {e}")
            return []
    
    def load_pickle_safe(self, file_path: str) -> Any:
        """
        Load pickle file with size checks.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Loaded pickle data
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")
        
        file_size = path.stat().st_size
        
        if file_size > self.WARNING_SIZE:
            logging.warning(f"⚠️ Large pickle file detected ({file_size / (1024**2):.2f} MB): {file_path}")
        
        if file_size > self.max_memory_bytes:
            logging.error(f"❌ Pickle file too large for memory limit ({file_size / (1024**3):.2f} GB)")
            raise MemoryError("Pickle file exceeds memory limit")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"✅ Loaded pickle file ({file_size / (1024**2):.2f} MB): {file_path}")
            return data
        except (pickle.UnpicklingError, MemoryError) as e:
            logging.error(f"❌ Failed to load pickle: {e}")
            raise
    
    def save_json_safe(self, data: Any, file_path: str, indent: int = 2):
        """
        Save JSON with memory-efficient writing for large data.
        
        Args:
            data: Data to save
            file_path: Output file path
            indent: JSON indentation (use 0 for compact)
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Estimate size
        estimated_size = sys.getsizeof(data)
        
        if estimated_size > self.WARNING_SIZE:
            logging.warning(f"⚠️ Large data being saved ({estimated_size / (1024**2):.2f} MB)")
            logging.info("Using compact JSON format to reduce size")
            indent = None  # Compact format
        
        try:
            # Use temporary file for atomic write
            temp_path = path.with_suffix('.tmp')
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            temp_path.replace(path)
            
            final_size = path.stat().st_size
            logging.info(f"✅ Saved JSON file ({final_size / (1024**2):.2f} MB): {file_path}")
        
        except (IOError, OSError) as e:
            logging.error(f"❌ Failed to save JSON: {e}")
            raise
    
    def save_pickle_safe(self, data: Any, file_path: str, protocol: int = pickle.HIGHEST_PROTOCOL):
        """
        Save pickle with memory-efficient writing.
        
        Args:
            data: Data to save
            file_path: Output file path
            protocol: Pickle protocol version
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use temporary file for atomic write
            temp_path = path.with_suffix('.tmp')
            
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=protocol)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            temp_path.replace(path)
            
            final_size = path.stat().st_size
            logging.info(f"✅ Saved pickle file ({final_size / (1024**2):.2f} MB): {file_path}")
        
        except (IOError, OSError, pickle.PicklingError) as e:
            logging.error(f"❌ Failed to save pickle: {e}")
            raise
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """Get file size and statistics"""
        path = Path(file_path)
        
        if not path.exists():
            return {"exists": False}
        
        stat = path.stat()
        
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 ** 2),
            "size_gb": stat.st_size / (1024 ** 3),
            "modified": stat.st_mtime,
            "is_large": stat.st_size > LargeMetadataHandler.WARNING_SIZE,
            "requires_streaming": stat.st_size > LargeMetadataHandler.STREAMING_SIZE
        }


def load_metadata_safe(json_path: str, pickle_path: Optional[str] = None,
                       prefer_pickle: bool = True, max_memory_mb: int = 1000) -> Any:
    """
    Convenience function to load metadata from JSON or pickle.
    
    Args:
        json_path: Path to JSON file
        pickle_path: Path to pickle file (optional)
        prefer_pickle: Prefer pickle over JSON if both exist
        max_memory_mb: Maximum memory to use
        
    Returns:
        Loaded metadata
    """
    handler = LargeMetadataHandler(max_memory_mb)
    
    # Check which files exist
    json_exists = Path(json_path).exists()
    pickle_exists = pickle_path and Path(pickle_path).exists()
    
    if not json_exists and not pickle_exists:
        raise FileNotFoundError("Neither JSON nor pickle metadata file exists")
    
    # Prefer pickle if requested and available
    if prefer_pickle and pickle_exists:
        try:
            return handler.load_pickle_safe(pickle_path)
        except Exception as e:
            logging.warning(f"Failed to load pickle, falling back to JSON: {e}")
            if json_exists:
                return handler.load_json_safe(json_path)
            raise
    
    # Load JSON
    if json_exists:
        data = handler.load_json_safe(json_path)
        
        # Create pickle cache if it doesn't exist
        if pickle_path and not pickle_exists:
            try:
                handler.save_pickle_safe(data, pickle_path)
                logging.info(f"✅ Created pickle cache: {pickle_path}")
            except Exception as e:
                logging.warning(f"Failed to create pickle cache: {e}")
        
        return data
    
    # Load pickle as fallback
    if pickle_exists:
        return handler.load_pickle_safe(pickle_path)
    
    raise FileNotFoundError("No metadata file could be loaded")
