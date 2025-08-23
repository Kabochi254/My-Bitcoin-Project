#!/usr/bin/env python3
"""
Bitcoin Address Extractor

This script extracts Bitcoin addresses from specified web pages using configurable GPU acceleration
and multiprocessing capabilities. It supports resumable operations and maintains progress tracking.

Features:
- GPU acceleration support (PyTorch/CuPy)
- Multiprocessing for parallel page processing
- Configurable page list management
- Progress tracking and resumable operations
- Efficient address extraction and validation
"""

import json
import os
import logging
import re
import argparse
import multiprocessing
import asyncio
import aiohttp
from datetime import datetime
from typing import List, Set, Dict, Optional, Any, Union
from pathlib import Path
import time
import random
import signal
import sys
import traceback
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bitcoin_search.log')
    ]
)
logger = logging.getLogger(__name__)

# Performance constants (turbo settings for non-browser use)
MAX_CONCURRENT_REQUESTS = 10000
CHUNK_SIZE = 100_000
MAX_RETRIES = 2
RATE_LIMIT_DELAY = 0.0
SAVE_INTERVAL = 100_000
CONNECTION_TIMEOUT = 10
RESPONSE_TIMEOUT = 20
BATCH_SIZE = 2000
INITIAL_RETRY_DELAY = 0.25
MAX_RETRY_DELAY = 4
MEMORY_LIMIT = 0.90

def deterministic_worker_task(args):
    """Top-level worker for deterministic mode (Windows-friendly picklable function)."""
    base_p, start_off, end_off, kpp = args
    # Local imports inside worker to avoid global state issues on Windows spawn
    import hashlib
    from Crypto.Hash import RIPEMD160  # requires pycryptodome
    from ecdsa import SigningKey, SECP256k1

    def _hash160(data: bytes) -> bytes:
        sha = hashlib.sha256(data).digest()
        ripe = RIPEMD160.new(sha).digest()
        return ripe

    def _b58encode(b: bytes) -> str:
        alphabet = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        n = int.from_bytes(b, 'big')
        res = bytearray()
        while n > 0:
            n, rem = divmod(n, 58)
            res.append(alphabet[rem])
        pad = 0
        for ch in b:
            if ch == 0:
                pad += 1
            else:
                break
        return (alphabet[0:1] * pad + res[::-1]).decode('ascii')

    def _b58check(payload: bytes) -> str:
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        return _b58encode(payload + checksum)

    SECP256K1_N_LOCAL = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    lines = []
    count = 0
    for page_offset in range(start_off, end_off):
        page_num = base_p + page_offset
        if page_num <= 0:
            continue
        start_idx = (page_num - 1) * kpp + 1
        for k in range(kpp):
            priv_int = start_idx + k
            if priv_int <= 0 or priv_int >= SECP256K1_N_LOCAL:
                continue
            # Derive public key (uncompressed)
            sk = SigningKey.from_secret_exponent(priv_int, curve=SECP256k1)
            vk = sk.verifying_key
            x = vk.pubkey.point.x()
            y = vk.pubkey.point.y()
            x_bytes = x.to_bytes(32, 'big')
            pubkey = b'\x04' + x_bytes + y.to_bytes(32, 'big')
            h160 = _hash160(pubkey)
            payload = b'\x00' + h160
            address = _b58check(payload)
            lines.append(f"{address},{page_num}\n")
            count += 1
    return (''.join(lines), count)

class GPUManager:
    """Manages GPU initialization and operations."""
    
    @staticmethod
    def init_gpu() -> Optional[Any]:
        """Initialize and return GPU device if available."""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.get_device_properties(0)
                # Set memory growth to prevent OOM
                torch.cuda.set_per_process_memory_fraction(MEMORY_LIMIT)
                torch.cuda.empty_cache()
                
                logger.info(f"GPU Device: {device.name}")
                logger.info(f"GPU Memory: {device.total_memory / 1024**3:.2f} GB")
                logger.info(f"Memory Limit: {MEMORY_LIMIT * 100}%")
                
                return torch.device("cuda")
        except ImportError:
            try:
                import cupy as cp
                device = cp.cuda.Device(0)
                device.set_memory_fraction(MEMORY_LIMIT)
                return device
            except ImportError:
                pass
        
        logger.warning("GPU acceleration not available")
        return None

class PageListManager:
    """Manages the list of pages to process."""
    
    def __init__(self, page_list_file: str = 'page_list.json'):
        self.page_list_file = Path(page_list_file)
        self.pages = []
        self.load_pages()
    
    def load_pages(self) -> None:
        """Load pages from file."""
        try:
            if self.page_list_file.exists():
                with open(self.page_list_file, 'r') as f:
                    data = json.load(f)
                    self.pages = [str(p) for p in data.get('pages', [])]  # Convert to strings for consistency
                    if not self.pages:
                        raise ValueError("No pages found in page list file")
                    logger.info(f"Loaded {len(self.pages)} pages from {self.page_list_file}")
            else:
                raise FileNotFoundError(f"Page list file {self.page_list_file} not found")
        except Exception as e:
            logger.error(f"Error loading page list: {str(e)}")
            raise

    def get_pages(self) -> List[str]:
        """Get the list of pages."""
        if not self.pages:
            self.load_pages()
        return self.pages

    def add_page(self, page: Union[str, int]) -> None:
        """Add a page to the list."""
        page_str = str(page)
        if page_str not in self.pages:
            self.pages.append(page_str)
            self._save_pages()

    def _save_pages(self) -> None:
        """Save pages to file."""
        with open(self.page_list_file, 'w') as f:
            json.dump({'pages': self.pages}, f, indent=4)

class AddressParser:
    """Parses Bitcoin addresses from HTML content with optional GPU acceleration."""
    
    def __init__(self, use_gpu: bool = False):
        # Fixed regex pattern to properly capture the address group
        self.address_pattern = re.compile(
            r'href=["\']https?://(?:www\.)?blockchain\.(?:info|com)(?:/btc)?/address/([13][a-km-zA-HJ-NP-Z1-9]{25,34})["\']',
            re.DOTALL
        )
        self.device = GPUManager.init_gpu() if use_gpu else None
        self.use_gpu = use_gpu and self.device is not None
        self.batch_size = BATCH_SIZE if self.use_gpu else BATCH_SIZE
        self.address_buffer = []
        # Precompile common patterns
        self._precompile_patterns()

    def _precompile_patterns(self):
        """Precompile commonly used patterns for better performance."""
        self.delimiter_pattern = re.compile(r'\n###DELIMITER###\n')
        self.cleanup_pattern = re.compile(r'[\x00-\x1F\x7F-\x9F]')  # Remove control characters

    async def extract_addresses_batch(self, html_contents: List[str]) -> List[List[str]]:
        """Process multiple HTML contents in a single GPU batch with optimized memory usage."""
        if not html_contents:
            return []

        try:
            # Clean and preprocess HTML content
            cleaned_contents = [
                self.cleanup_pattern.sub('', content) 
                for content in html_contents
            ]
            
            # Process in smaller sub-batches to optimize memory usage
            sub_batch_size = 100
            results = []
            
            for i in range(0, len(cleaned_contents), sub_batch_size):
                sub_batch = cleaned_contents[i:i + sub_batch_size]
                
                # Process each page in the sub-batch
                for content in sub_batch:
                    matches = self.address_pattern.finditer(content)
                    page_addresses = [match.group(1) for match in matches]
                    results.append(page_addresses[:self.batch_size] if page_addresses else [])
            
            # Ensure we have results for all input pages
            while len(results) < len(html_contents):
                results.append([])
            
            return results
                
        except Exception as e:
            logger.error(f"Error in address extraction: {str(e)}")
            return [[] for _ in html_contents]  # Return empty lists on error

class ProgressTracker:
    """Tracks and persists progress of address extraction."""
    
    def __init__(self):
        self.progress_file = Path('fetch_progress.json')
        self.current_offset = 0
        self.current_base_page = 0
        self.current_file_number = 1
        self.addresses_in_current_file = 0
        self.total_addresses = 0
        self.current_index_position = 0
        self.addresses_per_index = 10_000_000
        self.last_save_time = time.time()
        # Save less frequently to reduce overhead
        self.save_interval = 30
        # Track completed pages and their offsets
        self.completed_pages = {}  # Format: {page_index: max_offset}
        # Track the last file number used for each index
        self.last_file_numbers = {}  # Format: {page_index: last_file_number}
        self.load_progress()

    def load_progress(self) -> None:
        """Load progress from file if it exists."""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.current_offset = data.get('current_offset', 0)
                    self.current_base_page = data.get('current_base_page', 0)
                    self.current_file_number = data.get('current_file_number', 1)
                    self.addresses_in_current_file = data.get('addresses_in_current_file', 0)
                    self.total_addresses = data.get('total_addresses', 0)
                    self.current_index_position = data.get('current_index_position', 0)
                    self.completed_pages = data.get('completed_pages', {})
                    self.last_file_numbers = data.get('last_file_numbers', {})
                    
                    # Convert string keys back to integers for both dictionaries
                    self.completed_pages = {int(k): v for k, v in self.completed_pages.items()}
                    self.last_file_numbers = {int(k): v for k, v in self.last_file_numbers.items()}
                    
                    # Get the current page list size
                    page_manager = PageListManager()
                    num_pages = len(page_manager.get_pages())
                    
                    # Validate index position
                    if self.current_index_position >= num_pages:
                        self.current_index_position = 0  # Start from beginning but keep completed pages
                        # Find the highest file number used across all indices
                        if self.last_file_numbers:
                            self.current_file_number = max(self.last_file_numbers.values())
                    
                    logger.info(f"Resuming from index position {self.current_index_position}")
                    logger.info(f"Current offset: {self.current_offset}, Base page: {self.current_base_page}")
                    logger.info(f"Current file number: {self.current_file_number}")
                    logger.info(f"Total addresses collected: {self.total_addresses:,}")
                    logger.info(f"Completed pages: {len(self.completed_pages)}")
        except Exception as e:
            logger.error(f"Error loading progress: {str(e)}")
            self.reset_progress()

    def save_progress(self, force: bool = False) -> None:
        """Save progress to file."""
        current_time = time.time()
        if force or (current_time - self.last_save_time >= self.save_interval):
            try:
                with open(self.progress_file, 'w') as f:
                    json.dump({
                        'current_offset': self.current_offset,
                        'current_base_page': self.current_base_page,
                        'current_file_number': self.current_file_number,
                        'addresses_in_current_file': self.addresses_in_current_file,
                        'total_addresses': self.total_addresses,
                        'current_index_position': self.current_index_position,
                        'completed_pages': self.completed_pages,
                        'last_file_numbers': self.last_file_numbers
                    }, f, indent=4)
                self.last_save_time = current_time
            except Exception as e:
                logger.error(f"Error saving progress: {str(e)}")

    def reset_progress(self) -> None:
        """Reset progress tracking."""
        self.current_offset = 0
        self.current_base_page = 0
        self.current_file_number = 1
        self.addresses_in_current_file = 0
        self.total_addresses = 0
        self.current_index_position = 0
        self.completed_pages = {}
        self.last_file_numbers = {}
        self.save_progress(force=True)

    def update_addresses(self, new_addresses: int) -> None:
        """Update address counts and save progress."""
        self.total_addresses += new_addresses
        self.addresses_in_current_file += new_addresses
        
        # Update completed pages with current progress
        if self.current_index_position not in self.completed_pages or self.current_offset > self.completed_pages[self.current_index_position]:
            self.completed_pages[self.current_index_position] = self.current_offset
        
        if self.addresses_in_current_file >= self.addresses_per_index:
            self.current_file_number += 1
            # Store the last file number used for this index
            self.last_file_numbers[self.current_index_position] = self.current_file_number
            self.addresses_in_current_file = 0
        
        self.save_progress()

    def should_move_to_next_index(self) -> bool:
        """Check if we should move to the next index."""
        return self.current_offset >= self.addresses_per_index

    def move_to_next_index(self, total_indices: int) -> bool:
        """Move to the next index if available."""
        # Store the last file number used for the current index
        self.last_file_numbers[self.current_index_position] = self.current_file_number
        
        self.current_index_position += 1
        if self.current_index_position >= total_indices:
            self.current_index_position = 0
            # When wrapping around, use the highest file number from any index
            if self.last_file_numbers:
                self.current_file_number = max(self.last_file_numbers.values())
            logger.info(f"Wrapped back to index 0, continuing with file number {self.current_file_number}")
            
        # Set offset based on previously completed work
        self.current_offset = self.completed_pages.get(self.current_index_position, 0)
        return True

    def update_offset(self, new_offset: int) -> None:
        """Update the current offset."""
        self.current_offset = new_offset
        # Update completed pages with current progress
        if self.current_index_position not in self.completed_pages or new_offset > self.completed_pages[self.current_index_position]:
            self.completed_pages[self.current_index_position] = new_offset
        self.save_progress()

class AddressFetcher:
    """Main class for fetching and processing Bitcoin addresses."""
    
    def __init__(self, use_gpu: bool = False, processes: Optional[int] = None, chunk_size: int = CHUNK_SIZE, use_local: bool = False, local_path: Optional[str] = None, deterministic: bool = False, last_page: Optional[int] = None):
        self.page_manager = PageListManager()
        self.parser = AddressParser(use_gpu=use_gpu)
        self.progress = ProgressTracker()
        # Use as many workers as requested or default to aggressive CPU utilization.
        # Removed previous hard cap of 16 workers.
        self.max_workers = processes if processes else max(multiprocessing.cpu_count() - 1, 1) * 2
        self.chunk_size = chunk_size
      #  self.base_url = 'http://104.131.0.88'
        self.output_dir = Path('addresses')
        self.output_dir.mkdir(exist_ok=True)
        self.address_buffer = []
        # Large buffer to minimize disk I/O
        self.buffer_size = 200_000
        self.last_flush_time = time.time()
        self.flush_interval = SAVE_INTERVAL
        self.retry_count = 0
        self.running = True  # Flag to control execution
        self.use_local = use_local
        self.local_path = Path(local_path).resolve() if local_path else None
        if self.use_local:
            # Disable remote base URL when scraping locally
            self.base_url = None
        self.deterministic = deterministic
        # Optional global last page to process (supports very large integers)
        self.last_page = int(last_page) if last_page is not None else None
        
        # Session timeout settings
        self.session_timeout = aiohttp.ClientTimeout(
            total=None,
            connect=CONNECTION_TIMEOUT,
            sock_read=RESPONSE_TIMEOUT
        )
        
        # Performance metrics
        self.metrics = {
            'start_time': time.time(),
            'successful_requests': 0,
            'failed_requests': 0,
            'total_addresses': 0,
            'last_speed_check': time.time(),
            'last_addresses_count': 0
        }
        
        # Signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)

    @asynccontextmanager
    async def _local_browser(self):
        """Context manager to launch a headless browser for local site rendering."""
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            logger.error("Playwright is required for --use-local mode. Install with: pip install playwright && playwright install")
            raise

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            try:
                yield page
            finally:
                await context.close()
                await browser.close()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal, cleaning up...")
        self.running = False

    async def fetch_with_exponential_backoff(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        retry_delay = INITIAL_RETRY_DELAY
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url, timeout=self.session_timeout) as response:
                    if response.status == 200:
                        self.metrics['successful_requests'] += 1
                        self.retry_count = 0  # Reset retry count on success
                        return await response.text()
                    elif response.status == 429:  # Rate limit
                        retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.warning(f"Request failed with status {response.status}: {url}")
                        self.metrics['failed_requests'] += 1
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Request error: {str(e)}")
                self.metrics['failed_requests'] += 1
                
            retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
            await asyncio.sleep(retry_delay)
            
        return None

    async def fetch_pages_batch(self, session: aiohttp.ClientSession, pages: List[int]) -> List[str]:
        """Optimized batch fetching with connection pooling and improved error handling."""
        tasks = []
        for page in pages:
            url = f"{self.base_url}/{page}"
            task = asyncio.create_task(self.fetch_with_exponential_backoff(session, url))
            tasks.append(task)
        
        # Use gather with return_exceptions=True for better error handling
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses and handle errors
        valid_contents = []
        for response in responses:
            if isinstance(response, Exception):
                self.metrics['failed_requests'] += 1
                valid_contents.append("")  # Placeholder for failed request
            elif isinstance(response, str):
                self.metrics['successful_requests'] += 1
                valid_contents.append(response)
            else:
                self.metrics['failed_requests'] += 1
                valid_contents.append("")  # Placeholder for invalid response
        
        return valid_contents

    async def _goto_local_page(self, page_obj, page_number: int) -> Optional[str]:
        """Open local index.html (once) and navigate to a specific page number, then return HTML content."""
        # First load of the local file if not already at it
        if page_obj.url == 'about:blank':
            if not self.local_path or not self.local_path.exists():
                logger.error(f"Local index.html not found at: {self.local_path}")
                return None
            # Convert path to file:// URL
            local_url = self.local_path.as_uri()
            await page_obj.goto(local_url)
        try:
            # Try to call refresh_page if available
            await page_obj.evaluate("(n)=>{ try { return refresh_page(n); } catch(e) { return null; } }", page_number)
        except Exception:
            # Fallback to filling input and clicking goto
            try:
                await page_obj.fill('#page_num', str(page_number))
                await page_obj.click('#btn_goto')
            except Exception as e:
                logger.error(f"Failed to navigate to local page {page_number}: {e}")
                return None
        # Give the page a brief moment to render
        await asyncio.sleep(0.1)
        try:
            html = await page_obj.content()
            return html
        except Exception as e:
            logger.error(f"Failed to read local page content: {e}")
            return None

    async def process_chunk_local(self, page_obj, base_page: int, offsets: range) -> int:
        """Process a set of page offsets against the local rendered site."""
        addresses_found = 0
        for page_offset in offsets:
            actual_page = base_page + page_offset
            content = await self._goto_local_page(page_obj, actual_page)
            if not content:
                continue
            try:
                matches = self.parser.address_pattern.finditer(content)
                addresses = [match.group(1) for match in matches]
                addresses_found += len(addresses)
                self.address_buffer.extend((addr, actual_page) for addr in addresses)
                if len(self.address_buffer) >= self.buffer_size:
                    self.flush_address_buffer()
            except Exception as e:
                logger.error(f"Error processing local content from page {actual_page}: {str(e)}")
        return addresses_found

    # ---------------------------
    # Deterministic (offline) mode
    # ---------------------------
    @staticmethod
    def _int_to_bytes32(i: int) -> bytes:
        return i.to_bytes(32, byteorder='big')

    @staticmethod
    def _hash160(data: bytes) -> bytes:
        import hashlib
        from Crypto.Hash import RIPEMD160  # requires pycryptodome
        sha = hashlib.sha256(data).digest()
        ripe = RIPEMD160.new(sha).digest()
        return ripe

    @staticmethod
    def _b58encode(b: bytes) -> str:
        alphabet = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        n = int.from_bytes(b, 'big')
        res = bytearray()
        while n > 0:
            n, rem = divmod(n, 58)
            res.append(alphabet[rem])
        # handle leading zeros
        pad = 0
        for ch in b:
            if ch == 0:
                pad += 1
            else:
                break
        return (alphabet[0:1] * pad + res[::-1]).decode('ascii')

    @classmethod
    def _b58check(cls, payload: bytes) -> str:
        import hashlib
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        return cls._b58encode(payload + checksum)

    @classmethod
    def _priv_to_p2pkh_address(cls, priv_int: int, compressed: bool = False) -> str:
        try:
            from ecdsa import SigningKey, SECP256k1
        except ImportError:
            logger.error("ecdsa is required for --deterministic mode. Install with: pip install ecdsa pycryptodome")
            raise
        # Derive public key
        sk = SigningKey.from_secret_exponent(priv_int, curve=SECP256k1)
        vk = sk.verifying_key
        # Get affine coordinates
        x = vk.pubkey.point.x()
        y = vk.pubkey.point.y()
        x_bytes = x.to_bytes(32, 'big')
        if compressed:
            prefix = b'\x02' if (y % 2 == 0) else b'\x03'
            pubkey = prefix + x_bytes
        else:
            pubkey = b'\x04' + x_bytes + y.to_bytes(32, 'big')
        # hash160 and Base58Check with version 0x00 (mainnet)
        h160 = cls._hash160(pubkey)
        payload = b'\x00' + h160
        return cls._b58check(payload)

    async def process_chunk_deterministic(self, base_page: int, offsets: range, keys_per_page: int = 120) -> int:
        """Compute addresses locally without any HTTP/Browser.
        Assumes page P maps to keys starting at ((P-1) * keys_per_page) + 1.
        """
        # secp256k1 order n
        SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        addresses_found = 0
        # If only one worker, keep fast in-process path to avoid IPC overhead
        if self.max_workers <= 1:
            for page_offset in offsets:
                page_num = base_page + page_offset
                if page_num <= 0:
                    continue
                start_idx = (page_num - 1) * keys_per_page + 1  # +1 to start from 1, not 0
                for k in range(keys_per_page):
                    priv_int = start_idx + k
                    if priv_int <= 0 or priv_int >= SECP256K1_N:
                        continue
                    try:
                        # Use uncompressed public keys to match site generation
                        address = self._priv_to_p2pkh_address(priv_int, compressed=False)
                        # Buffer: follow existing format (address,page)
                        self.address_buffer.extend(((address, page_num),))
                        # Also count addresses
                        addresses_found += 1
                        if len(self.address_buffer) >= self.buffer_size:
                            self.flush_address_buffer()
                    except Exception as e:
                        logger.error(f"Deterministic derivation error for key {priv_int}: {e}")
            return addresses_found

        # Choose a task size that balances IPC overhead and CPU throughput
        task_size = 200  # number of page offsets per task
        tasks = []
        for start in range(offsets.start, offsets.stop, task_size):
            end = min(start + task_size, offsets.stop)
            tasks.append((base_page, start, end, keys_per_page))

        addresses_found_total = 0
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(deterministic_worker_task, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    out_chunk, cnt = fut.result()
                    addresses_found_total += cnt
                    # Write chunk directly to avoid main-process buffering overhead
                    output_file = self.output_dir / f'addresses_part_{self.progress.current_file_number}.txt'
                    with open(output_file, 'a', buffering=1 << 20) as f:  # 1MB buffer
                        f.write(out_chunk)
                except Exception as e:
                    logger.error(f"Deterministic worker error: {e}")

        # Also track totals and timing metrics
        return addresses_found_total

    async def process_chunk(self, session: aiohttp.ClientSession, pages: List[int]) -> int:
        """Process a chunk of pages with controlled concurrency."""
        addresses_found = 0
        current_index = self.page_manager.pages[self.progress.current_index_position]
        base_page = int(current_index)
        
        # Process pages in smaller batches to control concurrency
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async def fetch_with_semaphore(page_offset):
            async with semaphore:
                actual_page = base_page + page_offset
                url = f"{self.base_url}/{actual_page}"
                return await self.fetch_with_exponential_backoff(session, url), page_offset
        
        for i in range(0, len(pages), BATCH_SIZE):
            batch = pages[i:i + BATCH_SIZE]
            tasks = [fetch_with_semaphore(offset) for offset in batch]
            
            try:
                # Process batch with controlled concurrency
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process valid results
                for result in results:
                    if isinstance(result, tuple):  # Valid result
                        content, page_offset = result
                        if isinstance(content, str) and content.strip():
                            try:
                                matches = self.parser.address_pattern.finditer(content)
                                addresses = [match.group(1) for match in matches]
                                addresses_found += len(addresses)
                                
                                # Use actual page number when buffering addresses
                                actual_page = base_page + page_offset
                                self.address_buffer.extend((addr, actual_page) for addr in addresses)
                                
                                # Flush if buffer is full
                                if len(self.address_buffer) >= self.buffer_size:
                                    self.flush_address_buffer()
                            except Exception as e:
                                logger.error(f"Error processing content from page {base_page + page_offset}: {str(e)}")
                    elif isinstance(result, Exception):
                        logger.error(f"Error in batch task: {str(result)}")
                
                # Update metrics
                current_time = time.time()
                if current_time - self.metrics['last_speed_check'] >= 60:
                    addresses_per_minute = (addresses_found - self.metrics['last_addresses_count']) / (
                        (current_time - self.metrics['last_speed_check']) / 60
                    )
                    logger.info(f"Processing speed: {addresses_per_minute:.0f} addresses/minute")
                    logger.info(f"Current base page: {base_page}, Processing offsets: {batch[0]}-{batch[-1]}")
                    self.metrics['last_speed_check'] = current_time
                    self.metrics['last_addresses_count'] = addresses_found
                # No artificial delay between batches in turbo mode
                
            except Exception as e:
                logger.error(f"Error processing batch starting at page {base_page + batch[0]}: {str(e)}")
                await asyncio.sleep(1)  # Longer delay on batch error
        
        return addresses_found

    async def run(self) -> None:
        """Main execution loop with reduced overhead in non-HTTP modes."""
        connector = None
        try:
            # Only create HTTP connector/session when actually scraping over HTTP
            if not (self.deterministic or self.use_local):
                connector = aiohttp.TCPConnector(
                    limit=MAX_CONCURRENT_REQUESTS,
                    limit_per_host=200,
                    ttl_dns_cache=600,
                    use_dns_cache=True,
                    force_close=False,
                    enable_cleanup_closed=True,
                    keepalive_timeout=60
                )
                session_ctx = aiohttp.ClientSession(
                    connector=connector,
                    timeout=self.session_timeout,
                    raise_for_status=False
                )
            else:
                session_ctx = None

            async def iteration_loop(session):
                while self.running:
                    try:
                        current_index = self.page_manager.pages[self.progress.current_index_position]
                        base_page = int(current_index)
                        logger.info(f"Processing index {self.progress.current_index_position}: {current_index}")
                        logger.info(f"Base page number: {base_page}")
                        
                        start_offset = self.progress.current_offset
                        end_offset = start_offset + self.chunk_size
                        
                        # If a terminal last page is defined, clamp the end and exit when exceeded
                        max_offset: Optional[int] = None
                        if self.last_page is not None and self.last_page >= base_page:
                            max_offset = self.last_page - base_page
                            # If we've already gone past the last page, finish
                            if start_offset > max_offset:
                                logger.info("Reached the last page for this index. Completed processing all indices")
                                return
                            # Clamp end_offset to not exceed last page (+1 because range end is exclusive)
                            if end_offset > max_offset + 1:
                                end_offset = max_offset + 1
                        
                        # Only consider fast-forwarding when no explicit last_page is set
                        if self.last_page is None:
                            significant_chunk_threshold = self.chunk_size * 10
                            if (self.progress.current_index_position in self.progress.completed_pages and 
                                start_offset <= self.progress.completed_pages[self.progress.current_index_position] and
                                self.progress.completed_pages[self.progress.current_index_position] >= significant_chunk_threshold):
                                # Fast-forward to the next unprocessed chunk boundary to avoid per-chunk skip logs
                                completed = self.progress.completed_pages[self.progress.current_index_position]
                                # Only fast-forward when actually behind processed work
                                if start_offset < completed:
                                    next_offset = ((completed // self.chunk_size) + 1) * self.chunk_size
                                    # Set current_offset directly without marking it as completed again
                                    self.progress.current_offset = next_offset
                                    self.progress.save_progress()
                                    logger.info(f"Fast-forwarded to offset {next_offset} based on completed progress")
                                    continue
                        
                        page_offsets = range(start_offset, end_offset)
                        
                        logger.info(f"Processing offsets: {start_offset} to {end_offset}")
                        logger.info(f"Actual page range: {base_page + start_offset} to {base_page + end_offset}")
                        if self.deterministic:
                            addresses_found = await self.process_chunk_deterministic(base_page, page_offsets)
                        elif self.use_local:
                            async with self._local_browser() as local_page:
                                addresses_found = await self.process_chunk_local(local_page, base_page, page_offsets)
                        else:
                            addresses_found = await self.process_chunk(session, page_offsets)
                        self.progress.update_addresses(addresses_found)
                        
                        # With explicit last_page, never rotate indices; process linearly to last page
                        if self.last_page is not None:
                            self.progress.update_offset(end_offset)
                            if max_offset is not None and end_offset >= max_offset + 1:
                                logger.info("Reached the last page chunk. Completed processing all indices")
                                return
                        else:
                            if self.progress.should_move_to_next_index():
                                if not self.progress.move_to_next_index(len(self.page_manager.pages)):
                                    logger.info("Completed processing all indices")
                                    return
                            else:
                                self.progress.update_offset(end_offset)
                        
                        if self.address_buffer:
                            self.flush_address_buffer()
                    except asyncio.CancelledError:
                        logger.info("Task cancelled, cleaning up...")
                        return
                    except Exception as e:
                        logger.error(f"Error in main loop: {str(e)}")
                        await asyncio.sleep(1)
                    
                    if not self.running:
                        logger.info("Shutdown requested, stopping gracefully...")
                        return

            if session_ctx is not None:
                async with session_ctx as session:
                    await iteration_loop(session)
            else:
                await iteration_loop(None)
                        
        except asyncio.CancelledError:
            logger.info("Main task cancelled, performing cleanup...")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            traceback.print_exc()
        finally:
            logger.info("Saving final progress...")
            self.progress.save_progress(force=True)
            if self.address_buffer:
                self.flush_address_buffer()
            if connector and not connector.closed:
                await connector.close()

    def flush_address_buffer(self) -> None:
        """Flush buffered addresses to file."""
        if not self.address_buffer:
            return

        output_file = self.output_dir / f'addresses_part_{self.progress.current_file_number}.txt'
        with open(output_file, 'a', buffering=1 << 20) as f:  # Use 1MB buffer to reduce I/O overhead
            for address, page in self.address_buffer:
                f.write(f"{address},{page}\n")
        
        self.address_buffer = []
        self.last_flush_time = time.time()

def main() -> None:
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description='Bitcoin Address Extractor')
    parser.add_argument('--processes', type=int, help='Number of worker processes')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Chunk size for processing')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--use-local', action='store_true', help='Use local rendered index.html instead of HTTP scraping')
    parser.add_argument('--local-path', type=str, help='Path to local index.html (defaults to ./index.html when --use-local is set)')
    parser.add_argument('--deterministic', action='store_true', help='Compute keys/addresses offline deterministically (no HTTP/browser)')
    parser.add_argument('--last-page', type=str, help='Stop after reaching this absolute last page number (supports very large integers)')
    args = parser.parse_args()
    
    if sys.platform == 'win32':
        # Use selector event loop policy on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        local_path_arg = args.local_path if args.local_path else ('index.html' if args.use_local else None)
        fetcher = AddressFetcher(
            use_gpu=args.gpu,
            processes=args.processes,
            chunk_size=args.chunk_size,
            use_local=args.use_local,
            local_path=local_path_arg,
            deterministic=args.deterministic,
            last_page=args.last_page
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_fetcher():
            try:
                await fetcher.run()
            except asyncio.CancelledError:
                logger.info("Fetcher cancelled, cleaning up...")
            except Exception as e:
                logger.error(f"Error in fetcher: {str(e)}")
                raise
            finally:
                logger.info("Fetcher cleanup complete")
        
        try:
            loop.run_until_complete(run_fetcher())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
            fetcher.running = False
            # Allow time for cleanup
            loop.run_until_complete(asyncio.sleep(1))
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Wait for all tasks to complete
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            loop.close()
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
