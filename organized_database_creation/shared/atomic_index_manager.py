#!/usr/bin/env python3
"""
Atomic Index Manager with Versioning and Safe Updates
Handles FAISS index and metadata updates with atomic swaps and rollback
"""

import os
import shutil
import fcntl
import logging
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import threading


class AtomicIndexManager:
    """
    Manager for atomic FAISS index and metadata updates.
    Prevents corruption during updates and allows rollback.
    """
    
    def __init__(self, index_dir: str, max_versions: int = 3):
        """
        Initialize atomic index manager.
        
        Args:
            index_dir: Directory containing index files
            max_versions: Maximum number of versions to keep
        """
        self.index_dir = Path(index_dir)
        self.max_versions = max_versions
        self.lock_file = self.index_dir / ".index.lock"
        self.versions_dir = self.index_dir / "versions"
        self._lock = threading.Lock()
        
        # Ensure directories exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"🔒 Atomic Index Manager initialized: {self.index_dir}")
        logging.info(f"   Max versions: {self.max_versions}")
    
    def acquire_lock(self, timeout: int = 300) -> bool:
        """
        Acquire exclusive lock for index updates.
        
        Args:
            timeout: Lock acquisition timeout (seconds)
            
        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Create lock file
                self.lock_file.touch(exist_ok=True)
                
                with open(self.lock_file, 'r+') as f:
                    # Try to acquire exclusive lock (non-blocking)
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    logging.info("🔒 Acquired index update lock")
                    return True
            
            except (IOError, OSError):
                # Lock is held by another process
                time.sleep(1)
        
        logging.error(f"❌ Failed to acquire lock after {timeout}s")
        return False
    
    def release_lock(self):
        """Release index update lock"""
        try:
            if self.lock_file.exists():
                with open(self.lock_file, 'r+') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                logging.info("🔓 Released index update lock")
        except Exception as e:
            logging.warning(f"Failed to release lock: {e}")
    
    def create_version_snapshot(self, index_file: str, metadata_file: str,
                                 version_name: Optional[str] = None) -> str:
        """
        Create versioned snapshot of current index and metadata.
        
        Args:
            index_file: Path to FAISS index file
            metadata_file: Path to metadata file (JSON or pickle)
            version_name: Optional version name (defaults to timestamp)
            
        Returns:
            Version identifier
        """
        with self._lock:
            # Generate version name
            if not version_name:
                version_name = f"v_{int(datetime.now().timestamp())}"
            
            version_dir = self.versions_dir / version_name
            version_dir.mkdir(exist_ok=True)
            
            # Copy index file
            index_path = Path(index_file)
            if index_path.exists():
                shutil.copy2(index_file, version_dir / index_path.name)
                logging.info(f"📦 Backed up index: {index_path.name} → {version_name}")
            
            # Copy metadata file
            metadata_path = Path(metadata_file)
            if metadata_path.exists():
                shutil.copy2(metadata_file, version_dir / metadata_path.name)
                logging.info(f"📦 Backed up metadata: {metadata_path.name} → {version_name}")
            
            # Save version metadata
            version_info = {
                "version": version_name,
                "created_at": datetime.now().isoformat(),
                "index_file": index_path.name,
                "metadata_file": metadata_path.name,
                "index_size": index_path.stat().st_size if index_path.exists() else 0,
                "metadata_size": metadata_path.stat().st_size if metadata_path.exists() else 0
            }
            
            with open(version_dir / "version_info.json", 'w') as f:
                json.dump(version_info, f, indent=2)
            
            # Clean up old versions
            self._cleanup_old_versions()
            
            logging.info(f"✅ Created version snapshot: {version_name}")
            return version_name
    
    def atomic_update(self, new_index_file: str, new_metadata_file: str,
                      current_index_file: str, current_metadata_file: str,
                      create_backup: bool = True) -> bool:
        """
        Perform atomic update of index and metadata files.
        
        Args:
            new_index_file: Path to new FAISS index file
            new_metadata_file: Path to new metadata file
            current_index_file: Path to current FAISS index file
            current_metadata_file: Path to current metadata file
            create_backup: Whether to create backup before update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Validate new files exist
            if not Path(new_index_file).exists():
                raise FileNotFoundError(f"New index file not found: {new_index_file}")
            if not Path(new_metadata_file).exists():
                raise FileNotFoundError(f"New metadata file not found: {new_metadata_file}")
            
            # Create backup if requested
            if create_backup and (Path(current_index_file).exists() or 
                                  Path(current_metadata_file).exists()):
                self.create_version_snapshot(current_index_file, current_metadata_file)
            
            # Atomic swap using rename
            # Step 1: Move new files to temporary names
            temp_index = Path(current_index_file).with_suffix('.new')
            temp_metadata = Path(current_metadata_file).with_suffix('.new')
            
            shutil.copy2(new_index_file, temp_index)
            shutil.copy2(new_metadata_file, temp_metadata)
            
            # Step 2: Atomic rename to final names
            temp_index.replace(current_index_file)
            temp_metadata.replace(current_metadata_file)
            
            logging.info(f"✅ Atomic update complete:")
            logging.info(f"   Index: {current_index_file}")
            logging.info(f"   Metadata: {current_metadata_file}")
            
            return True
        
        except Exception as e:
            logging.error(f"❌ Atomic update failed: {e}")
            return False
    
    def rollback_to_version(self, version_name: str, current_index_file: str,
                            current_metadata_file: str) -> bool:
        """
        Rollback index and metadata to a previous version.
        
        Args:
            version_name: Version to rollback to
            current_index_file: Path to current FAISS index file
            current_metadata_file: Path to current metadata file
            
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            version_dir = self.versions_dir / version_name
            
            if not version_dir.exists():
                logging.error(f"Version not found: {version_name}")
                return False
            
            # Load version info
            version_info_path = version_dir / "version_info.json"
            if not version_info_path.exists():
                logging.error(f"Version info not found: {version_name}")
                return False
            
            with open(version_info_path, 'r') as f:
                version_info = json.load(f)
            
            # Copy versioned files to current
            index_file = version_dir / version_info["index_file"]
            metadata_file = version_dir / version_info["metadata_file"]
            
            if not index_file.exists() or not metadata_file.exists():
                logging.error(f"Version files incomplete: {version_name}")
                return False
            
            # Atomic swap
            temp_index = Path(current_index_file).with_suffix('.rollback')
            temp_metadata = Path(current_metadata_file).with_suffix('.rollback')
            
            shutil.copy2(index_file, temp_index)
            shutil.copy2(metadata_file, temp_metadata)
            
            temp_index.replace(current_index_file)
            temp_metadata.replace(current_metadata_file)
            
            logging.info(f"✅ Rolled back to version: {version_name}")
            logging.info(f"   Created at: {version_info['created_at']}")
            
            return True
        
        except Exception as e:
            logging.error(f"❌ Rollback failed: {e}")
            return False
    
    def list_versions(self) -> List[Dict]:
        """
        List all available versions.
        
        Returns:
            List of version information dictionaries
        """
        versions = []
        
        for version_dir in sorted(self.versions_dir.iterdir(), reverse=True):
            if not version_dir.is_dir():
                continue
            
            version_info_path = version_dir / "version_info.json"
            if version_info_path.exists():
                try:
                    with open(version_info_path, 'r') as f:
                        info = json.load(f)
                    versions.append(info)
                except Exception as e:
                    logging.warning(f"Failed to load version info: {version_dir.name}: {e}")
        
        return versions
    
    def _cleanup_old_versions(self):
        """Clean up old versions keeping only max_versions"""
        versions = sorted(self.versions_dir.iterdir(), 
                         key=lambda p: p.stat().st_mtime, 
                         reverse=True)
        
        # Remove excess versions
        for i, version_dir in enumerate(versions):
            if i >= self.max_versions:
                try:
                    shutil.rmtree(version_dir)
                    logging.info(f"🗑️ Removed old version: {version_dir.name}")
                except Exception as e:
                    logging.warning(f"Failed to remove old version: {e}")
    
    def get_current_version_info(self, index_file: str, metadata_file: str) -> Dict:
        """Get information about current index and metadata"""
        index_path = Path(index_file)
        metadata_path = Path(metadata_file)
        
        return {
            "index_exists": index_path.exists(),
            "metadata_exists": metadata_path.exists(),
            "index_size": index_path.stat().st_size if index_path.exists() else 0,
            "metadata_size": metadata_path.stat().st_size if metadata_path.exists() else 0,
            "index_modified": index_path.stat().st_mtime if index_path.exists() else 0,
            "metadata_modified": metadata_path.stat().st_mtime if metadata_path.exists() else 0,
            "available_versions": len(list(self.versions_dir.iterdir()))
        }
    
    def __enter__(self):
        """Context manager entry - acquire lock"""
        self.acquire_lock()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release lock"""
        self.release_lock()
