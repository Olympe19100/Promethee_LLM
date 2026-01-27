"""
Process SEC 10-K Filings into Clean Text Corpus

This script reads raw SEC filings (HTML/TXT) and extracts clean text
for use in LLM training.
"""

import os
import glob
import json
import re
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
from loguru import logger

DEFAULT_DATA_ROOT = "data/sec-edgar-filings"
DEFAULT_OUTPUT_FILE = "data/sec_corpus_clean.jsonl"
MIN_TEXT_LENGTH = 1000


def clean_html(html_content: str) -> str:
    """Remove HTML tags and extract clean text"""
    try:
        if not html_content or len(html_content) < 100:
            return ""

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator='\n')

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

    except Exception as e:
        logger.error(f"Error parsing HTML: {e}")
        return ""


def process_file(file_path: str) -> dict:
    """Process a single SEC filing file"""
    try:
        # Extract metadata from path
        # Structure: .../TICKER/TYPE/FILING_ID/full-submission.txt
        parts = file_path.replace("\\", "/").split("/")
        if len(parts) >= 4:
            ticker = parts[-4]
            filing_type = parts[-3]
            filing_id = parts[-2]
        else:
            ticker = "UNKNOWN"
            filing_type = "UNKNOWN"
            filing_id = "UNKNOWN"

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Parse SEC document structure
        documents = content.split('<DOCUMENT>')
        main_text = ""

        for doc in documents:
            if '<TYPE>10-K' in doc or '<TYPE>10-Q' in doc:
                start_match = re.search(r'<TEXT>', doc, re.IGNORECASE)
                end_match = re.search(r'</TEXT>', doc, re.IGNORECASE)

                if start_match and end_match:
                    raw_html = doc[start_match.end():end_match.start()]
                    cleaned = clean_html(raw_html)
                    if len(cleaned) > len(main_text):
                        main_text = cleaned

        # Fallback: clean entire content if no structured text found
        if not main_text and len(content) > 0:
            main_text = clean_html(content)

        if len(main_text) < MIN_TEXT_LENGTH:
            return None

        return {
            "ticker": ticker,
            "filing_type": filing_type,
            "filing_id": filing_id,
            "path": file_path,
            "text": main_text
        }

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Process SEC filings into clean text")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="SEC filings root directory")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE, help="Output JSONL file")
    parser.add_argument("--min-length", type=int, default=MIN_TEXT_LENGTH, help="Min text length to keep")
    args = parser.parse_args()

    global MIN_TEXT_LENGTH
    MIN_TEXT_LENGTH = args.min_length

    logger.info(f"Searching for filings in: {args.data_root}")

    # Find all submission files
    files = glob.glob(os.path.join(args.data_root, "**", "full-submission.txt"), recursive=True)
    logger.info(f"Found {len(files)} files")

    if not files:
        logger.warning("No files found. Check the data root path.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    processed = 0
    with open(args.output, 'w', encoding='utf-8') as out_f:
        for file_path in tqdm(files, desc="Processing"):
            result = process_file(file_path)
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed += 1

    logger.info(f"Done. Processed {processed} files.")
    logger.info(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
