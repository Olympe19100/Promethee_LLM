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

# S&P 500 Tickers (as of 2024)
SP500_TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AVGO", "CSCO", "ADBE", "CRM",
    "ORCL", "ACN", "IBM", "INTC", "AMD", "QCOM", "TXN", "NOW", "INTU", "AMAT",
    "ADI", "LRCX", "MU", "KLAC", "SNPS", "CDNS", "MCHP", "APH", "MSI", "TEL",
    "FTNT", "PANW", "CRWD", "IT", "CTSH", "ANSS", "KEYS", "ON", "FSLR", "HPQ",
    "HPE", "NTAP", "WDC", "STX", "JNPR", "ZBRA", "TYL", "EPAM", "AKAM", "FFIV",
    "SWKS", "QRVO", "ENPH", "SEDG", "TER", "PAYC", "MPWR", "NXPI", "GEN", "TRMB",

    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI",
    "CB", "MMC", "PGR", "AON", "CME", "ICE", "MCO", "MET", "AIG", "PRU",
    "TRV", "ALL", "AFL", "AJG", "MSCI", "BK", "STT", "TROW", "NTRS", "FRC",
    "USB", "PNC", "TFC", "FITB", "CFG", "RF", "HBAN", "KEY", "MTB", "CMA",
    "ZION", "FDS", "MKTX", "CBOE", "NDAQ", "IVZ", "BEN", "LNC", "GL", "AIZ",
    "L", "RE", "WRB", "CINF", "RJF", "SIVB", "SBNY", "DFS", "SYF", "COF",

    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "MDT", "ELV", "CVS", "CI", "ISRG", "VRTX", "REGN", "SYK",
    "BDX", "ZTS", "BSX", "HUM", "MRNA", "MCK", "HCA", "DXCM", "IDXX", "IQV",
    "EW", "A", "MTD", "RMD", "ILMN", "ALGN", "BAX", "ZBH", "BIIB", "CAH",
    "CNC", "MOH", "HOLX", "COO", "TECH", "WST", "DGX", "LH", "PKI", "BIO",
    "CRL", "XRAY", "HSIC", "OGN", "VTRS", "CTLT", "INCY", "TFX", "STE", "WAT",

    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
    "ORLY", "AZO", "MAR", "HLT", "YUM", "DHI", "LEN", "ROST", "EBAY", "ETSY",
    "DRI", "POOL", "BBY", "ULTA", "APTV", "GRMN", "LVS", "WYNN", "MGM", "CZR",
    "RCL", "CCL", "NCLH", "EXPE", "F", "GM", "RIVN", "LCID", "NVR", "PHM",
    "GPC", "KMX", "AAP", "AN", "LAD", "SAH", "BWA", "LEA", "RL", "TPR",
    "VFC", "PVH", "HBI", "CPRI", "GPS", "PENN", "DKS", "HAS", "MAT", "WHR",

    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "EL",
    "KMB", "GIS", "KHC", "HSY", "K", "SJM", "MKC", "HRL", "CAG", "CPB",
    "CHD", "CLX", "TSN", "ADM", "BG", "KR", "SYY", "WBA", "TGT", "DG",
    "DLTR", "KSS", "M", "JWN", "COTY", "BF.B", "STZ", "TAP", "SAM", "MNST",

    # Industrials
    "RTX", "HON", "UPS", "UNP", "CAT", "DE", "BA", "LMT", "GE", "MMM",
    "GD", "NOC", "CSX", "NSC", "WM", "EMR", "ETN", "ITW", "PH", "ROK",
    "CMI", "PCAR", "OTIS", "CARR", "JCI", "TT", "IR", "AME", "FAST", "RSG",
    "VRSK", "CPRT", "CTAS", "PAYX", "ADP", "EFX", "LDOS", "LHX", "TDG", "HWM",
    "WAB", "PWR", "DOV", "ODFL", "CHRW", "JBHT", "DAL", "LUV", "AAL", "UAL",
    "FDX", "EXPD", "XPO", "SAIA", "GWW", "SNA", "SWK", "RHI", "NLOK", "NI",

    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "PXD", "OXY",
    "HES", "DVN", "FANG", "HAL", "BKR", "MRO", "APA", "EQT", "CTRA", "OVV",
    "TRGP", "WMB", "KMI", "OKE", "KINDER", "ET",

    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "PEG", "WEC",
    "ED", "ES", "EIX", "DTE", "FE", "PPL", "AWK", "AEE", "CMS", "EVRG",
    "LNT", "CNP", "NI", "ATO", "NRG", "PNW",

    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
    "EQR", "VICI", "VTR", "ARE", "MAA", "UDR", "ESS", "INVH", "SUI", "ELS",
    "CPT", "HST", "KIM", "REG", "FRT", "BXP", "VNO", "SLG", "CBRE", "JLL",

    # Materials
    "LIN", "APD", "SHW", "FCX", "ECL", "NEM", "NUE", "DD", "DOW", "PPG",
    "VMC", "MLM", "CTVA", "FMC", "IFF", "ALB", "CF", "MOS", "EMN", "CE",
    "IP", "PKG", "WRK", "AVY", "SEE", "BLL", "AMCR",

    # Communication Services
    "GOOG", "GOOGL", "META", "DIS", "NFLX", "CMCSA", "VZ", "T", "TMUS", "CHTR",
    "ATVI", "EA", "TTWO", "WBD", "PARA", "FOX", "FOXA", "NWS", "NWSA", "OMC",
    "IPG", "LYV", "MTCH", "PINS", "SNAP", "TWTR", "ZG", "RBLX"
]

DEFAULT_TICKERS = SP500_TICKERS


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
