import sqlite3
import os
import time
from datetime import datetime
import argparse
from multiprocessing import Process, Lock, Value, Queue
import ctypes
import re

def get_next_file_number(processed_files):
    """Get the number of the next file to process."""
    current_files = [f for f in os.listdir('.') if f.startswith('addresses_part_') 
                    and f.endswith('.txt')]
    
    if not current_files:
        return None
        
    file_numbers = []
    for file in current_files:
        match = re.match(r'addresses_part_(\d+)\.txt', file)
        if match:
            file_num = int(match.group(1))
            if file_num not in processed_files:
                file_numbers.append(file_num)
    
    return min(file_numbers) if file_numbers else None

def check_address_in_db(cursor, address):
    """Check if an address exists in the database."""
    cursor.execute("SELECT address FROM addresses WHERE address = ?", (address.split(',')[0] if ',' in address else address,))
    return cursor.fetchone() is not None

def process_file_chunk(file_path, start_pos, chunk_size, process_id, shared_counter, 
                      file_lock, match_queue):
    """Process a chunk of addresses from a file."""
    # Connect to database in read-only mode to avoid locks and reduce I/O issues
    try:
        conn = sqlite3.connect('file:bitcoin.db?mode=ro', uri=True, timeout=30)
    except Exception:
        # Fallback to normal connection if URI mode unsupported in environment
        conn = sqlite3.connect('bitcoin.db', timeout=30)
    cursor = conn.cursor()
    
    # Create temporary file for matches only
    temp_file = f"{file_path}.temp_{process_id}"
    matches_found = 0
    
    # Dictionary to store address:page_numbers mapping
    address_pages = {}
    
    try:
        with open(file_path, 'r') as f:
            # Move to start position
            f.seek(start_pos)
            
            # Stream-read and process addresses to minimize memory usage
            with open(temp_file, 'w') as temp:
                count = 0
                while count < chunk_size:
                    raw = f.readline()
                    if not raw:
                        break
                    address_line = raw.strip()
                    # Split address and page number
                    parts = address_line.split(',')
                    address = parts[0] if parts else address_line
                    page_num = parts[1] if len(parts) > 1 else "Unknown"

                    if check_address_in_db(cursor, address):
                        # Store or update page numbers for this address
                        if address not in address_pages:
                            address_pages[address] = set()
                        address_pages[address].add(page_num)

                        # Save match to temp file with page number
                        temp.write(f"{address},{page_num}\n")
                        matches_found += 1

                    # Update counter
                    with shared_counter.get_lock():
                        shared_counter.value += 1
                        if shared_counter.value % 1000 == 0:
                            print(f"\rProcess {process_id} | "
                                  f"Addresses checked: {shared_counter.value:,} | "
                                  f"Matches found: {matches_found}", end="")
                    count += 1
            
            # Write matches to matches.txt with all page numbers for each address
            if matches_found > 0:
                with file_lock:
                    with open("matches.txt", "a") as match_file:
                        for address, pages in address_pages.items():
                            match_file.write(f"Time: {datetime.now()}\n")
                            match_file.write(f"Address: {address}\n")
                            match_file.write(f"Found on Pages: {', '.join(sorted(pages))}\n")
                            match_file.write("-" * 50 + "\n")
            
            
    except Exception as e:
        print(f"\nError in process {process_id}: {str(e)}")
    finally:
        conn.close()
        # Always report matches_found so the parent does not block on queue.get()
        try:
            match_queue.put(matches_found)
        except Exception:
            pass
    
    return temp_file

def merge_temp_files(original_file, temp_files, file_lock, total_matches):
    """Merge temporary files back into the original file only if matches were found."""
    if total_matches > 0:
        # If matches were found, keep only the matching addresses
        with file_lock:
            with open(original_file, 'w') as outfile:
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        with open(temp_file, 'r') as infile:
                            outfile.write(infile.read())
    else:
        # If no matches were found, clear the file completely
        with file_lock:
            open(original_file, 'w').close()  # Clear the file
    
    # Clean up temp files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def wait_for_file_ready(file_path, min_size_kb=1):
    """Wait until the file is ready to be processed."""
    while True:
        if not os.path.exists(file_path):
            return False
            
        # Get file size in KB
        size_kb = os.path.getsize(file_path) / 1024
        
        if size_kb >= min_size_kb:
            # Check if file is being written to
            time.sleep(1)  # Wait 1 second
            new_size_kb = os.path.getsize(file_path) / 1024
            
            # If file size hasn't changed in 1 second, consider it ready
            if new_size_kb == size_kb:
                return True
        
        print(f"\rWaiting for {os.path.basename(file_path)} to be ready...", end="")
        time.sleep(2)

def process_single_file(file_path, num_processes, chunk_size):
    """Process a single file."""
    if not wait_for_file_ready(file_path):
        print(f"\nFile {file_path} not found")
        return
    
    print(f"\nProcessing {os.path.basename(file_path)}")
    
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        print(f"Skipping empty file: {file_path}")
        return
    
    # Shared resources
    shared_counter = Value(ctypes.c_uint64, 0)
    file_lock = Lock()
    match_queue = Queue()
    
    # Calculate chunks
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    chunk_size = min(chunk_size, total_lines // num_processes + 1)
    chunks = []
    
    with open(file_path, 'r') as f:
        start_pos = 0
        while True:
            chunks.append(start_pos)
            # Skip chunk_size lines
            for _ in range(chunk_size):
                if not f.readline():
                    break
            if f.tell() == start_pos:
                break
            start_pos = f.tell()
    
    # Process chunks in parallel but LIMIT concurrency to num_processes
    total_matches = 0
    processes = []
    temp_files = []
    
    for batch_start in range(0, len(chunks), num_processes):
        batch = chunks[batch_start:batch_start + num_processes]
        processes = []
        
        # Start a bounded batch of processes
        for j, start_pos in enumerate(batch):
            proc_id = batch_start + j
            p = Process(target=process_file_chunk,
                        args=(file_path, start_pos, chunk_size, proc_id,
                              shared_counter, file_lock, match_queue))
            processes.append(p)
            p.start()
            temp_files.append(f"{file_path}.temp_{proc_id}")
        
        # Wait for this batch to complete and collect results
        for p in processes:
            p.join()
        
        # Collect match counts for this batch (one per process)
        for _ in range(len(batch)):
            try:
                matches = match_queue.get(timeout=30)
            except Exception:
                matches = 0
            total_matches += matches
    
    # Merge temporary files
    merge_temp_files(file_path, temp_files, file_lock, total_matches)
    
    print(f"\nCompleted processing {os.path.basename(file_path)}")
    print(f"Total addresses checked: {shared_counter.value:,}")
    print(f"Total matches found: {total_matches:,}")
    if total_matches == 0:
        print(f"No matches found - file has been cleared")
    else:
        print(f"Matches found - only matching addresses kept in file")

def main():
    parser = argparse.ArgumentParser(description='Match addresses against bitcoin.db')
    parser.add_argument('--processes', type=int, default=8, help='Number of processes to use')
    parser.add_argument('--chunk-size', type=int, default=100000, 
                       help='Number of addresses to process per chunk')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuously monitor for new files')
    
    args = parser.parse_args()
    
    processed_files = set()
    
    try:
        while True:
            next_file_num = get_next_file_number(processed_files)
            
            if next_file_num is None:
                if not args.continuous:
                    print("\nNo more files to process!")
                    break
                print("\rWaiting for new files...", end="")
                time.sleep(5)
                continue
            
            file_name = f"addresses_part_{next_file_num}.txt"
            file_path = os.path.abspath(file_name)
            
            process_single_file(file_path, args.processes, args.chunk_size)
            processed_files.add(next_file_num)
            
            if not args.continuous:
                break
            
            print("\nWaiting for next file...")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nScript stopped by user")
    
    print("\nAll files processed!")

if __name__ == "__main__":
    main()
