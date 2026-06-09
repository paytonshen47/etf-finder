"""
ETF Finder — pre-built holdings DB + live Yahoo Finance supplement + direct ETF lookup.
"""
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import uvicorn
import yfinance as yf
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

app = FastAPI()
_pool = ThreadPoolExecutor(max_workers=8)

# ── Load holdings database ────────────────────────────────────────────────────
_DB_PATH = Path(__file__).parent / "etf_holdings.json"

def _load_db():
    if not _DB_PATH.exists():
        return {"etfs": {}, "reverse": {}}
    with open(_DB_PATH) as f:
        return json.load(f)

_DB = _load_db()

# ── Cache (TTL = 1 hour) ──────────────────────────────────────────────────────
_cache: dict[str, tuple[float, object]] = {}
CACHE_TTL = 3600

def _cache_get(key):
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return val
    return None

def _cache_set(key, val):
    _cache[key] = (time.time(), val)

async def _run(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pool, lambda: fn(*args, **kwargs))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/api/search")
async def search(q: str = Query(..., min_length=1)):
    key = f"search:{q.lower()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    def _search():
        results = yf.Search(q, max_results=10).quotes
        out = []
        for r in results:
            sym = r.get("symbol", "")
            if not sym:
                continue
            out.append({
                "symbol": sym,
                "name": r.get("longname") or r.get("shortname") or sym,
                "exchange": r.get("exchDisp") or r.get("exchange") or "",
                "type": r.get("quoteType") or "",
            })
        return out

    data = await _run(_search)
    _cache_set(key, data)
    return data


@app.get("/api/etfs/{symbol:path}")
async def etfs_for_stock(symbol: str):
    """Reverse lookup: which ETFs hold this stock?
    Stage 1 (instant): pre-built DB.
    Stage 2 (live):    mutualfund_holders to catch ETFs outside the top-10 snapshot.
    """
    symbol = symbol.upper()
    key = f"etfs:{symbol}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    reverse   = _DB.get("reverse", {})
    etfs_meta = _DB.get("etfs", {})

    # ── Stage 1: DB lookup ────────────────────────────────────────────────────
    db_syms = set(reverse.get(symbol, []))

    db_results = {}
    for etf_sym in db_syms:
        meta   = etfs_meta.get(etf_sym, {})
        weight = next(
            (h["pct"] for h in meta.get("holdings", []) if h["symbol"].upper() == symbol),
            None,
        )
        db_results[etf_sym] = {
            "symbol":       etf_sym,
            "name":         meta.get("name", etf_sym),
            "weight":       weight,
            "aum":          meta.get("aum"),
            "expenseRatio": meta.get("expenseRatio"),
            "currency":     meta.get("currency", "USD"),
            "dividendYield":meta.get("dividendYield"),
            "category":     meta.get("category"),
        }

    # ── Stage 2: live supplement via mutualfund_holders ───────────────────────
    def _live_holders():
        try:
            mfh = yf.Ticker(symbol).mutualfund_holders
            stock_info = yf.Ticker(symbol).info
            return mfh, stock_info
        except Exception:
            return None, {}

    mfh, stock_info = await _run(_live_holders)

    if mfh is not None and not mfh.empty:
        currency = stock_info.get("currency", "USD")

        def _resolve_one(row):
            holder_name: str = row.get("Holder", "")
            parts       = holder_name.split("-", 1)
            search_name = parts[1].strip() if len(parts) > 1 else holder_name.strip()
            value       = float(row.get("Value") or 0)

            try:
                quotes = yf.Search(search_name, max_results=4).quotes
                etf_sym = None
                for q in quotes:
                    s = q.get("symbol", "")
                    qt = q.get("quoteType", "")
                    if qt in ("ETF", "MUTUALFUND") and "." not in s and "-" not in s:
                        etf_sym = s
                        break
                if not etf_sym:
                    for q in quotes:
                        s = q.get("symbol", "")
                        if s and "." not in s and "-" not in s:
                            etf_sym = s
                            break
                if not etf_sym:
                    return None

                # Skip if already in DB results
                if etf_sym in db_results:
                    return None

                etf_info = yf.Ticker(etf_sym).info
                aum = etf_info.get("totalAssets") or 0

                # Compute approximate weight
                weight = None
                if aum and value:
                    fx = _usd_fx_sync(currency)
                    weight = round((value * fx / aum) * 100, 3)

                # Expense ratio
                exp = None
                try:
                    ops = yf.Ticker(etf_sym).funds_data.fund_operations
                    if ops is not None and not ops.empty:
                        exp = round(float(ops.loc["Annual Report Expense Ratio"].iloc[0]), 6)
                except Exception:
                    pass

                return {
                    "symbol":       etf_sym,
                    "name":         etf_info.get("longName") or etf_info.get("shortName") or etf_sym,
                    "weight":       weight,
                    "aum":          aum or None,
                    "expenseRatio": exp,
                    "currency":     etf_info.get("currency", "USD"),
                    "dividendYield":etf_info.get("yield") or etf_info.get("trailingAnnualDividendYield"),
                    "category":     etf_info.get("category"),
                }
            except Exception:
                return None

        rows = mfh.to_dict("records")
        live_tasks = [_run(_resolve_one, r) for r in rows]
        live_results = await asyncio.gather(*live_tasks)
        for r in live_results:
            if r:
                db_results[r["symbol"]] = r

    results = sorted(db_results.values(), key=lambda x: x.get("weight") or 0, reverse=True)
    _cache_set(key, results)
    return results


@app.get("/api/etf/{symbol:path}")
async def etf_detail(symbol: str):
    """Detail view for any ETF — works for ETFs in DB and ones discovered live."""
    symbol = symbol.upper()
    key    = f"etf_detail:{symbol}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    meta = _DB.get("etfs", {}).get(symbol, {})

    def _fetch_live():
        t    = yf.Ticker(symbol)
        info = t.info
        hist = t.history(period="6y", interval="1mo", auto_adjust=True)

        # If ETF not in DB, fetch holdings live
        holdings = meta.get("holdings", [])
        if not holdings:
            try:
                fd = t.funds_data
                th = fd.top_holdings
                if th is not None and not th.empty:
                    holdings = [
                        {"symbol": sym, "name": str(row.get("Name", "")),
                         "pct": round(float(row.get("Holding Percent", 0)) * 100, 3)}
                        for sym, row in th.iterrows()
                    ]
            except Exception:
                pass

        # Expense ratio
        exp = meta.get("expenseRatio")
        if exp is None:
            try:
                ops = t.funds_data.fund_operations
                if ops is not None and not ops.empty:
                    exp = round(float(ops.loc["Annual Report Expense Ratio"].iloc[0]), 6)
            except Exception:
                pass

        return info, hist, holdings, exp

    info, hist, holdings, exp = await _run(_fetch_live)
    performance = _calc_performance(hist)

    # Rename pct → weightPercentage for frontend compatibility
    holdings_out = [
        {"asset": h.get("symbol", ""), "name": h.get("name", ""),
         "weightPercentage": h.get("pct", h.get("weightPercentage", 0))}
        for h in holdings
    ]

    result = {
        "info": {
            "name":          info.get("longName") or info.get("shortName") or symbol,
            "aum":           info.get("totalAssets") or meta.get("aum"),
            "expenseRatio":  exp,
            "currency":      info.get("currency", "USD"),
            "dividendYield": info.get("yield") or info.get("trailingAnnualDividendYield"),
            "category":      info.get("category") or meta.get("category"),
            "description":   info.get("longBusinessSummary") or "",
        },
        "holdings":    holdings_out,
        "performance": performance,
    }
    _cache_set(key, result)
    return result


def _calc_performance(hist) -> dict:
    if hist is None or hist.empty:
        return {}
    try:
        prices = hist["Close"].dropna()
        if prices.empty:
            return {}
        today        = date.today()
        current_year = today.year
        result       = {}

        def price_near(target: date):
            for delta in range(0, 7):
                check   = (target + timedelta(days=delta * 30)).strftime("%Y-%m")
                matches = [float(p) for idx, p in prices.items() if idx.strftime("%Y-%m") == check]
                if matches:
                    return matches[0]
            for delta in range(1, 4):
                check   = (target - timedelta(days=delta * 30)).strftime("%Y-%m")
                matches = [float(p) for idx, p in prices.items() if idx.strftime("%Y-%m") == check]
                if matches:
                    return matches[0]
            return None

        latest  = float(prices.iloc[-1])
        p_start = price_near(date(current_year, 1, 1))
        if p_start:
            result["YTD"] = round((latest / p_start - 1) * 100, 2)

        for yr in range(current_year - 1, 2019, -1):   # most recent first
            p1 = price_near(date(yr, 1, 1))
            p2 = price_near(date(yr, 12, 1))
            if p1 and p2:
                result[str(yr)] = round((p2 / p1 - 1) * 100, 2)

        return result
    except Exception:
        return {}


def _usd_fx_sync(currency: str) -> float:
    if not currency or currency.upper() == "USD":
        return 1.0
    key    = f"fx:{currency.upper()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        rate = yf.Ticker(f"{currency.upper()}USD=X").info.get("regularMarketPrice") or 1.0
        _cache_set(key, float(rate))
        return float(rate)
    except Exception:
        return 1.0


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  ETF Finder → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
