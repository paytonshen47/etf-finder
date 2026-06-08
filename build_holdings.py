"""
Run this script to (re)build etf_holdings.json.
  python3 build_holdings.py

It fetches top holdings for ~150 popular ETFs from Yahoo Finance
and writes a reverse-lookup index: { "NVDA": ["SMH", "SOXX", "QQQ", ...], ... }
plus per-ETF metadata.
"""
import json
import time
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── The ETF universe ──────────────────────────────────────────────────────────
ETFS = [
    # Broad US
    "SPY","IVV","VOO","VTI","QQQ","IWM","IWF","IWD","MDY","IJH","IJR","DIA",
    "RSP","SCHB","SCHX","ITOT","VV","VXF",
    # Sector (SPDR)
    "XLK","XLF","XLV","XLE","XLI","XLY","XLP","XLB","XLU","XLRE","XLC",
    # Tech / Growth
    "VGT","FTEC","IGV","SOXX","SMH","SOXQ","QTEC","PSI","XSMO",
    "ARKK","ARKG","ARKW","ARKF","ARKQ","ARKX",
    "WCLD","BUG","CLOU","SKYY","IGN","FDN","IVES",
    # Semiconductors / AI
    "SOXL","FNGU","TECL","USD","NVDL","TQQQ",
    # Defense / Aerospace
    "ITA","XAR","PPA","SHLD","DFEN",
    # Energy
    "XLE","VDE","OIH","XOP","AMLP","EMLP",
    # Financials
    "XLF","KRE","KBE","IAI","VFH","KBWB",
    # Healthcare / Biotech
    "XLV","IBB","ARKG","XBI","LABU","BBH","IHF","PJP",
    # Consumer
    "XLY","XLP","VCR","VDC","IBUY","BETZ",
    # Industrials / Materials
    "XLI","XLB","VIS","VAW","PICK","GDX","GDXJ","SLX","COPX",
    # Real Estate
    "XLRE","VNQ","IYR","REM","REZ",
    # Bonds / Fixed Income (skip — no equity holdings)
    # International Developed
    "EFA","IEFA","VEA","SCHF","EWG","EWJ","EWU","EWL","EWQ","EWI","EWP","EWD","EWN",
    "EWA","EWC","EWH","EWS","EWZS","EPOL","EWW",
    # Europe / Defense heavy
    "VGK","HEDJ","FEZ","EZU","IEUR","FLGR","DFAI",
    # Emerging Markets
    "EEM","IEMG","VWO","SCHE",
    # China
    "FXI","KWEB","MCHI","ASHR","CQQQ","KBA",
    # India
    "INDA","EPI","PIN","INDY",
    # Latin America
    "EWZ","ILF","GXG",
    # Thematic
    "ICLN","QCLN","TAN","FAN","LIT","BATT","DRIV","IDRV",
    "ROBO","BOTZ","AIQ","IRBO","METV","UFO","YOLO",
    "MOO","SOIL","KROP","WOOD","TREQ",
    "XHB","ITB","PKB",
    "PBW","GRID","AMPS","HYDR",
    # Dividend
    "VYM","DVY","SCHD","HDV","DGRO","SDY","NOBL","VIG","DGRW",
    # Factor
    "MTUM","VLUE","USMV","QUAL","SIZE","EFAV","ACWV",
    # Small/Mid Cap
    "VBR","VBK","VO","VB","IWO","IWN","IWP","IWS",
]
ETFS = sorted(set(ETFS))  # deduplicate

def fetch_etf(symbol: str) -> dict | None:
    try:
        t = yf.Ticker(symbol)
        info = t.info
        fd = t.funds_data

        # Must be an ETF/fund
        qtype = info.get("quoteType", "")
        if qtype not in ("ETF", "MUTUALFUND"):
            return None

        # Top holdings
        holdings = []
        try:
            th = fd.top_holdings
            if th is not None and not th.empty:
                for sym, row in th.iterrows():
                    holdings.append({
                        "symbol": sym,
                        "name": str(row.get("Name", "")),
                        "pct": round(float(row.get("Holding Percent", 0)) * 100, 3),
                    })
        except Exception:
            pass

        if not holdings:
            return None

        # Expense ratio
        exp = None
        try:
            ops = fd.fund_operations
            if ops is not None and not ops.empty:
                exp = round(float(ops.loc["Annual Report Expense Ratio"].iloc[0]), 6)
        except Exception:
            pass

        return {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName") or symbol,
            "aum": info.get("totalAssets"),
            "expenseRatio": exp,
            "currency": info.get("currency", "USD"),
            "dividendYield": info.get("yield") or info.get("trailingAnnualDividendYield"),
            "category": info.get("category"),
            "holdings": holdings,
        }
    except Exception as e:
        print(f"  ERROR {symbol}: {e}")
        return None


def build():
    print(f"Fetching {len(ETFS)} ETFs …")
    etf_data: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_etf, sym): sym for sym in ETFS}
        done = 0
        for future in as_completed(futures):
            sym = futures[future]
            done += 1
            result = future.result()
            if result:
                etf_data[sym] = result
                h_count = len(result["holdings"])
                print(f"  [{done}/{len(ETFS)}] {sym:8s} — {h_count} holdings, AUM={result['aum']}")
            else:
                print(f"  [{done}/{len(ETFS)}] {sym:8s} — skipped")

    # Build reverse index: stock_symbol → [etf_symbol, ...]
    reverse: dict[str, list[str]] = {}
    for etf_sym, etf in etf_data.items():
        for h in etf["holdings"]:
            stock = h["symbol"].upper()
            reverse.setdefault(stock, [])
            if etf_sym not in reverse[stock]:
                reverse[stock].append(etf_sym)

    output = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "etfs": etf_data,
        "reverse": reverse,
    }

    with open("etf_holdings.json", "w") as f:
        json.dump(output, f, separators=(",", ":"))

    total_stocks = len(reverse)
    print(f"\nDone. {len(etf_data)} ETFs, {total_stocks} unique stocks indexed.")
    print("Saved → etf_holdings.json")


if __name__ == "__main__":
    build()
