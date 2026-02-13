import os
import json
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "JPM", "XOM"]
QUARTERS_PER_TICKER = 4

# buffer in calendar days so you can later slice to [-20, +10] trading days safely
CAL_START_BUFFER_DAYS = 90
CAL_END_BUFFER_DAYS = 30

OUT_DIR = "data"


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_recent_earnings_dates(ticker: str, n: int) -> list[str]:
    """
    yfinance provides earnings dates, but it can be incomplete/shifted.
    We use it for v1 only. You can replace later with a better source.
    """
    tk = yf.Ticker(ticker)
    ed = tk.get_earnings_dates(limit=max(n, 8))
    if ed is None or len(ed) == 0:
        return []

    # index is timestamps; keep most recent n dates, format YYYY-MM-DD
    dates = sorted([d.to_pydatetime().date() for d in ed.index], reverse=True)[:n]
    return [d.isoformat() for d in dates]


def download_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # Standardize columns
    keep = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[keep]
    df["Date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    return df


def build_event_folder(ticker: str, earnings_date: str) -> str:
    path = os.path.join(OUT_DIR, ticker, earnings_date)
    safe_mkdir(path)
    return path


def main():
    safe_mkdir(OUT_DIR)

    for ticker in TICKERS:
        print(f"\n== {ticker} ==")
        earnings_dates = get_recent_earnings_dates(ticker, QUARTERS_PER_TICKER)
        if not earnings_dates:
            print("  (no earnings dates found via yfinance)")
            continue

        for ed in earnings_dates:
            event_dir = build_event_folder(ticker, ed)

            # meta.json
            meta_path = os.path.join(event_dir, "meta.json")
            meta = {
                "ticker": ticker,
                "earnings_date": ed,
                "source": "yfinance.get_earnings_dates (v1)",
                "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # prices.csv (download with buffers)
            d = datetime.fromisoformat(ed).date()
            start = (d - timedelta(days=CAL_START_BUFFER_DAYS)).isoformat()
            end = (d + timedelta(days=CAL_END_BUFFER_DAYS)).isoformat()

            prices = download_prices(ticker, start, end)
            prices_path = os.path.join(event_dir, "prices.csv")
            prices.to_csv(prices_path, index=False)

            # text.txt placeholder (you fill later with transcript/summary)
            text_path = os.path.join(event_dir, "text.txt")
            if not os.path.exists(text_path):
                with open(text_path, "w") as f:
                    f.write(f"[PLACEHOLDER] Add earnings transcript/summary for {ticker} on {ed}\n")

            print(f"  saved: {event_dir}")

    print("\nDone. Next: add text, then compute features + returns.")


if __name__ == "__main__":
    main()
