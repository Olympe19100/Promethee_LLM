"""
Download SEC 10-K Filings

This script downloads SEC 10-K filings for specified tickers
using the sec-edgar-downloader library.
"""

import os
import argparse
from sec_edgar_downloader import Downloader
from loguru import logger

DEFAULT_DOWNLOAD_DIR = "data"
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "UNH", "HD",
    "DIS", "PYPL", "NFLX", "ADBE", "CRM"
]


def main():
    parser = argparse.ArgumentParser(description="Download SEC 10-K filings")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="List of tickers")
    parser.add_argument("--tickers-file", type=str, default=None, help="File with tickers (one per line)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_DOWNLOAD_DIR, help="Download directory")
    parser.add_argument("--amount", type=int, default=5, help="Number of filings per ticker (None=all)")
    parser.add_argument("--email", type=str, default="research@example.com", help="Your email for SEC")
    parser.add_argument("--name", type=str, default="Research Project", help="Your name/org for SEC")
    args = parser.parse_args()

    # Load tickers from file if specified
    if args.tickers_file and os.path.exists(args.tickers_file):
        with open(args.tickers_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        tickers = args.tickers

    logger.info(f"Will download 10-K filings for {len(tickers)} tickers")

    # Create download directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize downloader
    dl = Downloader(args.name, args.email, args.output_dir)

    for ticker in tickers:
        logger.info(f"Downloading {ticker}...")
        try:
            dl.get("10-K", ticker, limit=args.amount)
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")

    logger.info("Download complete!")
    logger.info(f"Files saved to: {args.output_dir}/sec-edgar-filings/")


if __name__ == "__main__":
    main()
