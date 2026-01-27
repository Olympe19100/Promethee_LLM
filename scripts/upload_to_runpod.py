"""
Upload processed SEC corpus to RunPod storage

This script helps upload the sec_corpus_clean.jsonl file to RunPod
for use in LLM distillation training.

Prerequisites:
- runpodctl installed: pip install runpod
- RUNPOD_API_KEY environment variable set
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from loguru import logger

DEFAULT_CORPUS_PATH = r"C:\Users\ANTEC MSI\Desktop\pro\quantamentale-main\sec_corpus_clean.jsonl"


def check_runpod_cli():
    """Check if runpodctl is available"""
    try:
        result = subprocess.run(["runpodctl", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"runpodctl version: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    logger.error("runpodctl not found. Install with: pip install runpod")
    logger.info("Or download from: https://github.com/runpod/runpodctl/releases")
    return False


def get_file_size_mb(path: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(path) / (1024 * 1024)


def upload_to_runpod(file_path: str, pod_id: str = None):
    """Upload file to RunPod pod or network storage"""

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    size_mb = get_file_size_mb(file_path)
    logger.info(f"File size: {size_mb:.1f} MB")

    if pod_id:
        # Upload directly to a running pod
        logger.info(f"Uploading to pod {pod_id}...")
        cmd = ["runpodctl", "send", file_path, f"{pod_id}:/workspace/data/"]
    else:
        # Use runpodctl send (creates a temporary share link)
        logger.info("Creating shareable upload link...")
        cmd = ["runpodctl", "send", file_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.success("Upload successful!")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"Upload failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return False


def split_file_for_upload(file_path: str, max_size_mb: int = 500):
    """Split large file into chunks for easier upload"""
    size_mb = get_file_size_mb(file_path)

    if size_mb <= max_size_mb:
        logger.info(f"File is {size_mb:.1f} MB, no splitting needed")
        return [file_path]

    logger.info(f"Splitting {size_mb:.1f} MB file into {max_size_mb} MB chunks...")

    base_path = Path(file_path)
    output_dir = base_path.parent / "chunks"
    output_dir.mkdir(exist_ok=True)

    chunk_paths = []
    chunk_num = 0
    current_chunk = []
    current_size = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_size = len(line.encode('utf-8'))

            if current_size + line_size > max_size_mb * 1024 * 1024:
                # Write current chunk
                chunk_path = output_dir / f"{base_path.stem}_part{chunk_num:03d}.jsonl"
                with open(chunk_path, 'w', encoding='utf-8') as chunk_f:
                    chunk_f.writelines(current_chunk)
                chunk_paths.append(str(chunk_path))
                logger.info(f"Created {chunk_path.name} ({current_size / 1024 / 1024:.1f} MB)")

                chunk_num += 1
                current_chunk = []
                current_size = 0

            current_chunk.append(line)
            current_size += line_size

    # Write final chunk
    if current_chunk:
        chunk_path = output_dir / f"{base_path.stem}_part{chunk_num:03d}.jsonl"
        with open(chunk_path, 'w', encoding='utf-8') as chunk_f:
            chunk_f.writelines(current_chunk)
        chunk_paths.append(str(chunk_path))
        logger.info(f"Created {chunk_path.name} ({current_size / 1024 / 1024:.1f} MB)")

    logger.success(f"Split into {len(chunk_paths)} chunks")
    return chunk_paths


def main():
    parser = argparse.ArgumentParser(description="Upload SEC corpus to RunPod")
    parser.add_argument("--file", type=str, default=DEFAULT_CORPUS_PATH, help="File to upload")
    parser.add_argument("--pod-id", type=str, default=None, help="RunPod pod ID (optional)")
    parser.add_argument("--split", action="store_true", help="Split large files into chunks")
    parser.add_argument("--max-chunk-mb", type=int, default=500, help="Max chunk size in MB")
    args = parser.parse_args()

    if not check_runpod_cli():
        sys.exit(1)

    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        logger.info("Make sure you've run process_sec_data.py first")
        sys.exit(1)

    files_to_upload = [args.file]

    if args.split:
        files_to_upload = split_file_for_upload(args.file, args.max_chunk_mb)

    for file_path in files_to_upload:
        upload_to_runpod(file_path, args.pod_id)


if __name__ == "__main__":
    main()
