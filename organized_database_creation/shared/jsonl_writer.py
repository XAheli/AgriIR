#!/usr/bin/env python3
"""
JSONL Writer - Shared utility for writing structured data to JSONL files
"""

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


class JSONLWriter:
    """Thread-safe JSONL file writer with atomic writes"""
    
    def __init__(self, output_file: str, use_duplicate_tracker: bool = False,
                 duplicate_tracker_file: Optional[str] = None):
        self.output_file = Path(output_file)
        self.lock = threading.Lock()
        self.entries_written = 0
        self.use_duplicate_tracker = use_duplicate_tracker
        self.duplicate_tracker = None
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize duplicate tracker if requested
        if use_duplicate_tracker:
            from duplicate_tracker import get_global_tracker
            persistence_file = duplicate_tracker_file or str(self.output_file.parent / "processed_urls.json")
            self.duplicate_tracker = get_global_tracker(persistence_file)
            logging.info(f"✅ Duplicate tracker enabled for JSONL writer")
    
    def write_entry(self, entry: Dict[str, Any], check_duplicates: bool = True) -> bool:
        """
        Write a single entry to JSONL file with atomic write.
        
        Args:
            entry: Data entry to write
            check_duplicates: Whether to check for duplicates before writing
            
        Returns:
            True if successfully written, False if duplicate or error
        """
        try:
            # Check for duplicates if tracker is enabled
            if check_duplicates and self.duplicate_tracker:
                url = entry.get('link') or entry.get('url', '')
                title = entry.get('title', '')
                content = entry.get('text_extracted', '') or entry.get('content', '')
                
                if url and self.duplicate_tracker.is_duplicate_url(url):
                    logging.debug(f"Skipping duplicate URL: {url}")
                    return False
                
                if title and content and self.duplicate_tracker.is_duplicate_content(title, content):
                    logging.debug(f"Skipping duplicate content: {title[:100]}")
                    return False
                
                # Mark as processed with success=True (entry is being written)
                self.duplicate_tracker.mark_processed(url, title, content, success=True)
            
            with self.lock:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f.flush()  # Force write to OS buffer
                    os.fsync(f.fileno())  # Force OS to write to disk
                self.entries_written += 1
            return True
        except Exception as e:
            logging.error(f"Error writing entry: {e}")
            return False
    
    def write_entries(self, entries: List[Dict[str, Any]], check_duplicates: bool = True) -> int:
        """
        Write multiple entries to JSONL file with atomic write.
        
        Args:
            entries: List of data entries to write
            check_duplicates: Whether to check for duplicates before writing
            
        Returns:
            Number of entries successfully written
        """
        written_count = 0
        with self.lock:
            try:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    for entry in entries:
                        # Check for duplicates if tracker is enabled
                        if check_duplicates and self.duplicate_tracker:
                            url = entry.get('link') or entry.get('url', '')
                            title = entry.get('title', '')
                            content = entry.get('text_extracted', '') or entry.get('content', '')
                            
                            if url and self.duplicate_tracker.is_duplicate_url(url):
                                continue
                            
                            if title and content and self.duplicate_tracker.is_duplicate_content(title, content):
                                continue
                            
                            # Mark as processed with success=True
                            self.duplicate_tracker.mark_processed(url, title, content, success=True)
                        
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        written_count += 1
                    
                    f.flush()
                    os.fsync(f.fileno())
                
                self.entries_written += written_count
            except Exception as e:
                logging.error(f"Error writing entries: {e}")
        
        return written_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get writer statistics"""
        stats = {
            'output_file': str(self.output_file),
            'entries_written': self.entries_written,
            'file_exists': self.output_file.exists(),
            'file_size': self.output_file.stat().st_size if self.output_file.exists() else 0
        }
        
        if self.duplicate_tracker:
            stats['duplicate_tracker'] = self.duplicate_tracker.get_stats()
        
        return stats


class ImmediateJSONLWriter(JSONLWriter):
    """Immediate JSONL writer for real-time data processing with duplicate checking"""
    
    def __init__(self, output_file: str, buffer_size: int = 1,
                 use_duplicate_tracker: bool = False,
                 duplicate_tracker_file: Optional[str] = None,
                 clear_file: bool = False):
        super().__init__(output_file, use_duplicate_tracker, duplicate_tracker_file)
        self.buffer_size = buffer_size
        self.buffer = []
        self.last_flush = datetime.now()
        
        # Clear file if requested (for backward compatibility)
        if clear_file:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                pass
    
    def write_entry_immediate(self, entry: Dict[str, Any], check_duplicates: bool = True) -> bool:
        """Write entry immediately without buffering"""
        return self.write_entry(entry, check_duplicates)
    
    def get_entries_count(self) -> int:
        """Get total number of entries written (for backward compatibility)"""
        return self.entries_written
    
    def write_entry_buffered(self, entry: Dict[str, Any], check_duplicates: bool = True) -> bool:
        """
        Write entry with buffering.
        
        Returns:
            True if entry was buffered or successfully written, False on error
        """
        with self.lock:
            self.buffer.append((entry, check_duplicates))
            
            if len(self.buffer) >= self.buffer_size:
                return self._flush_buffer()
        
        # Entry successfully buffered (not yet written but will be)
        return True
    
    def _flush_buffer(self) -> bool:
        """Flush buffer to file"""
        if not self.buffer:
            return True
        
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                entries_written = 0
                for entry, check_duplicates in self.buffer:
                    # Check for duplicates if tracker is enabled
                    if check_duplicates and self.duplicate_tracker:
                        url = entry.get('link') or entry.get('url', '')
                        title = entry.get('title', '')
                        content = entry.get('text_extracted', '') or entry.get('content', '')
                        
                        if url and self.duplicate_tracker.is_duplicate_url(url):
                            continue
                        
                        if title and content and self.duplicate_tracker.is_duplicate_content(title, content):
                            continue
                        
                        # Mark as processed with success=True
                        self.duplicate_tracker.mark_processed(url, title, content, success=True)
                    
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    entries_written += 1
                
                f.flush()
                os.fsync(f.fileno())
            
            self.entries_written += entries_written
            self.buffer.clear()
            self.last_flush = datetime.now()
            return True
        except Exception as e:
            logging.error(f"Error flushing buffer: {e}")
            return False
    
    def force_flush(self) -> bool:
        """Force flush buffer to file"""
        with self.lock:
            return self._flush_buffer()
    
    def __del__(self):
        """Ensure buffer is flushed on destruction"""
        if hasattr(self, 'buffer') and self.buffer:
            self._flush_buffer()