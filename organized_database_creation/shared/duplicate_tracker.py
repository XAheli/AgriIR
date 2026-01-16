#!/usr/bin/env python3
"""
Centralized Duplicate Tracker with Persistence
Handles URL and content deduplication across runs and processes
"""

import os
import json
import hashlib
import threading
import fcntl
from pathlib import Path
from typing import Set, Dict, Optional, Tuple
from datetime import datetime
import logging


class PersistentDuplicateTracker:
    """
    Centralized duplicate tracking with file-based persistence.
    Prevents duplicate writes across separate runs/processes.
    """
    
    def __init__(self, persistence_file: str = "processed_urls.json", 
                 lock_timeout: int = 30,
                 batch_size: int = 10,
                 max_memory_entries: int = 100000):
        """
        Initialize persistent duplicate tracker.
        
        Args:
            persistence_file: Path to JSON file storing processed URLs/hashes
            lock_timeout: Timeout for file lock acquisition (seconds)
            batch_size: Number of entries to accumulate before writing to disk
            max_memory_entries: Maximum entries in memory before cleanup (0 = unlimited)
        """
        self.persistence_file = Path(persistence_file)
        self.lock_timeout = lock_timeout
        self._memory_lock = threading.RLock()
        
        # Batching configuration
        self.batch_size = batch_size
        self.max_memory_entries = max_memory_entries
        self._pending_batch: list = []
        self._save_counter = 0
        
        # In-memory caches for fast lookup
        self.seen_urls: Set[str] = set()
        self.seen_url_hashes: Set[str] = set()
        self.seen_content_hashes: Set[str] = set()
        self.seen_titles: Set[str] = set()
        
        # Metadata tracking
        self.url_metadata: Dict[str, Dict] = {}
        
        # Ensure persistence directory exists
        self.persistence_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_from_disk()
        
        logging.info(f"📦 Persistent duplicate tracker initialized: {self.persistence_file}")
        logging.info(f"   Loaded {len(self.seen_urls)} URLs, {len(self.seen_content_hashes)} content hashes")
        logging.info(f"   Batch size: {batch_size}, Max memory: {max_memory_entries}")
    
    def _load_from_disk(self):
        """Load processed URLs and hashes from disk with file locking"""
        if not self.persistence_file.exists():
            logging.info(f"No existing persistence file found: {self.persistence_file}")
            return
        
        try:
            with open(self.persistence_file, 'r', encoding='utf-8') as f:
                # Acquire shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                    
                    self.seen_urls = set(data.get('seen_urls', []))
                    self.seen_url_hashes = set(data.get('seen_url_hashes', []))
                    self.seen_content_hashes = set(data.get('seen_content_hashes', []))
                    self.seen_titles = set(data.get('seen_titles', []))
                    self.url_metadata = data.get('url_metadata', {})
                    
                    logging.info(f"✅ Loaded {len(self.seen_urls)} processed URLs from disk")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
        except json.JSONDecodeError as e:
            logging.error(f"❌ Failed to parse persistence file: {e}")
            self._backup_and_reset()
        except Exception as e:
            logging.error(f"❌ Failed to load persistence file: {e}")
    
    def _save_to_disk(self, force: bool = False):
        """
        Save processed URLs and hashes to disk with atomic write and file locking.
        Uses double-buffering to avoid holding lock during disk I/O.
        
        Args:
            force: Force immediate save even if batch not full
        """
        # Only save if we have enough entries or forced
        if not force and len(self._pending_batch) < self.batch_size:
            return
        
        # CRITICAL FIX: Copy data under lock, then release lock before disk I/O
        with self._memory_lock:
            data_snapshot = {
                'seen_urls': list(self.seen_urls),
                'seen_url_hashes': list(self.seen_url_hashes),
                'seen_content_hashes': list(self.seen_content_hashes),
                'seen_titles': list(self.seen_titles),
                'url_metadata': self.url_metadata.copy(),
                'last_updated': datetime.now().isoformat(),
                'total_urls': len(self.seen_urls),
                'total_content_hashes': len(self.seen_content_hashes)
            }
            # Clear pending batch under lock
            self._pending_batch.clear()
        
        # Now perform disk I/O WITHOUT holding the memory lock
        try:
            # Write to temporary file first (atomic swap pattern)
            temp_file = self.persistence_file.with_suffix('.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                # Acquire exclusive lock for writing
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(data_snapshot, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                    
                    # Atomic rename INSIDE the lock to prevent race conditions
                    temp_file.replace(self.persistence_file)
                    
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            self._save_counter += 1
            logging.debug(f"💾 Saved {data_snapshot['total_urls']} URLs to persistence file (save #{self._save_counter})")
            
        except Exception as e:
            logging.error(f"❌ Failed to save persistence file: {e}")
    
    def _backup_and_reset(self):
        """Backup corrupted file and reset"""
        if self.persistence_file.exists():
            backup_file = self.persistence_file.with_suffix(f'.backup.{int(datetime.now().timestamp())}')
            self.persistence_file.rename(backup_file)
            logging.warning(f"⚠️ Backed up corrupted file to: {backup_file}")
        
        self.seen_urls = set()
        self.seen_url_hashes = set()
        self.seen_content_hashes = set()
        self.seen_titles = set()
        self.url_metadata = {}
    
    def is_duplicate_url(self, url: str) -> bool:
        """
        Check if URL is duplicate (already processed).
        Thread-safe and persistent across runs.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is duplicate, False otherwise
        """
        with self._memory_lock:
            url_normalized = self._normalize_url(url)
            url_hash = hashlib.md5(url_normalized.encode('utf-8')).hexdigest()
            
            if url in self.seen_urls or url_hash in self.seen_url_hashes:
                return True
            
            return False
    
    def is_duplicate_content(self, title: str, content: str) -> bool:
        """
        Check if content is duplicate based on title and content hash.
        
        Args:
            title: Content title
            content: Content text
            
        Returns:
            True if content is duplicate, False otherwise
        """
        with self._memory_lock:
            title_normalized = self._normalize_text(title)
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            if title_normalized in self.seen_titles or content_hash in self.seen_content_hashes:
                return True
            
            return False
    
    def mark_processed(self, url: str, title: str = "", content: str = "", 
                       metadata: Optional[Dict] = None, success: bool = True) -> bool:
        """
        Mark URL/content as processed and persist to disk (batched).
        
        Args:
            url: URL to mark as processed
            title: Content title
            content: Content text
            metadata: Additional metadata to store
            success: Whether processing succeeded (default True)
            
        Returns:
            True if successfully marked (not duplicate), False if duplicate
        """
        with self._memory_lock:
            # Check for duplicates first
            if self.is_duplicate_url(url):
                logging.debug(f"Duplicate URL detected: {url}")
                return False
            
            if title and content and self.is_duplicate_content(title, content):
                logging.debug(f"Duplicate content detected: {title[:100]}")
                return False
            
            # Mark as processed (even if failed - prevents infinite retries)
            url_normalized = self._normalize_url(url)
            url_hash = hashlib.md5(url_normalized.encode('utf-8')).hexdigest()
            
            self.seen_urls.add(url)
            self.seen_url_hashes.add(url_hash)
            
            if title:
                title_normalized = self._normalize_text(title)
                self.seen_titles.add(title_normalized)
            
            if content:
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                self.seen_content_hashes.add(content_hash)
            
            # Store metadata with success status
            metadata_entry = {
                'timestamp': datetime.now().isoformat(),
                'success': success
            }
            if metadata:
                metadata_entry.update(metadata)
            self.url_metadata[url] = metadata_entry
            
            # Add to pending batch
            self._pending_batch.append(url)
            
            # Check for memory cleanup
            self._check_memory_limits()
            
            # Persist to disk when batch is full
            self._save_to_disk(force=False)
            
            return True
    
    def _check_memory_limits(self):
        """Check and enforce memory limits with LRU-style cleanup"""
        if self.max_memory_entries <= 0:
            return
        
        total_entries = len(self.seen_urls)
        if total_entries > self.max_memory_entries:
            # CRITICAL FIX: Clean ALL data structures, not just seen_urls
            excess = total_entries - self.max_memory_entries
            urls_to_remove = list(self.seen_urls)[:excess]
            
            for url in urls_to_remove:
                self.seen_urls.discard(url)
                
                # Also remove from other structures
                url_normalized = self._normalize_url(url)
                url_hash = hashlib.md5(url_normalized.encode('utf-8')).hexdigest()
                self.seen_url_hashes.discard(url_hash)
                
                # Remove associated metadata
                if url in self.url_metadata:
                    metadata = self.url_metadata.pop(url, {})
                    
                    # Try to clean up associated content hashes and titles
                    # Note: This is best-effort since we don't have reverse mapping
                    # In future, consider maintaining bidirectional mapping
            
            # Clean up orphaned entries in other structures (approximation)
            # Keep proportional sizes across all structures
            if len(self.seen_content_hashes) > self.max_memory_entries:
                content_hashes_list = list(self.seen_content_hashes)
                excess_content = len(self.seen_content_hashes) - self.max_memory_entries
                for hash_to_remove in content_hashes_list[:excess_content]:
                    self.seen_content_hashes.discard(hash_to_remove)
            
            if len(self.seen_titles) > self.max_memory_entries:
                titles_list = list(self.seen_titles)
                excess_titles = len(self.seen_titles) - self.max_memory_entries
                for title_to_remove in titles_list[:excess_titles]:
                    self.seen_titles.discard(title_to_remove)
            
            logging.info(f"🧹 Memory cleanup: Removed {excess} URLs and associated data")
    
    def force_save(self):
        """Force immediate save to disk"""
        with self._memory_lock:
            self._save_to_disk(force=True)
    
    def get_stats(self) -> Dict:
        """Get tracker statistics"""
        with self._memory_lock:
            return {
                'total_urls': len(self.seen_urls),
                'total_url_hashes': len(self.seen_url_hashes),
                'total_content_hashes': len(self.seen_content_hashes),
                'total_titles': len(self.seen_titles),
                'persistence_file': str(self.persistence_file),
                'file_exists': self.persistence_file.exists(),
                'file_size': self.persistence_file.stat().st_size if self.persistence_file.exists() else 0
            }
    
    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL for comparison"""
        # Remove trailing slashes, convert to lowercase, remove www
        url = url.lower().strip()
        url = url.rstrip('/')
        url = url.replace('www.', '')
        
        # Remove common tracking parameters
        if '?' in url:
            base_url = url.split('?')[0]
            # Keep important query parameters, remove tracking
            # For now, just use base URL
            url = base_url
        
        return url
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase, remove extra whitespace
        import re
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def clear(self):
        """Clear all tracked data (use with caution)"""
        with self._memory_lock:
            self.seen_urls.clear()
            self.seen_url_hashes.clear()
            self.seen_content_hashes.clear()
            self.seen_titles.clear()
            self.url_metadata.clear()
            self._pending_batch.clear()
            self._save_to_disk(force=True)
            logging.warning("⚠️ Cleared all duplicate tracking data")
    
    def merge_from_jsonl(self, jsonl_file: str):
        """
        Merge processed URLs from existing JSONL file.
        Useful for bootstrapping tracker from existing data.
        
        Args:
            jsonl_file: Path to JSONL file
        """
        with self._memory_lock:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            url = entry.get('link') or entry.get('url')
                            title = entry.get('title', '')
                            content = entry.get('text_extracted', '') or entry.get('content', '')
                            
                            if url:
                                self.mark_processed(url, title, content)
                        except json.JSONDecodeError:
                            continue
                
                logging.info(f"✅ Merged URLs from JSONL: {jsonl_file}")
                self._save_to_disk(force=True)
            except Exception as e:
                logging.error(f"❌ Failed to merge from JSONL: {e}")


# Singleton instance for global use
_global_tracker: Optional[PersistentDuplicateTracker] = None
_tracker_lock = threading.Lock()


def get_global_tracker(persistence_file: str = "data/processed_urls.json",
                      batch_size: int = 10,
                      max_memory_entries: int = 100000) -> PersistentDuplicateTracker:
    """
    Get or create global singleton duplicate tracker instance.
    
    Args:
        persistence_file: Path to persistence file (only used on first call)
        batch_size: Number of entries before disk write (only used on first call)
        max_memory_entries: Maximum entries in memory (only used on first call)
        
    Returns:
        Global PersistentDuplicateTracker instance
    """
    global _global_tracker
    
    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = PersistentDuplicateTracker(
                persistence_file=persistence_file,
                batch_size=batch_size,
                max_memory_entries=max_memory_entries
            )
        return _global_tracker
