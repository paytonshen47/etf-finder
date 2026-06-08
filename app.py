"""
ETF Finder — powered by a pre-built holdings database + Yahoo Finance for details.
Lookup is instant (JSON file); details fetched on demand.
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
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

app = FastAPI()
_pool = ThreadPoolExecutor(max_workers=6)

# ── Load holdings database ────────────────────────────────────────────────────
_DB_PATH = Path(__file__).parent / "etf_holdings.json"

def _load_db():
    if not _DB_PATH.exists():
        return {"etfs": {}, "reverse": {}}
    with open(_DB_PATH) as f:
        return json.load(f)

_DB = _load_db()

# ── Simple in-memory cache (TTL = 1 hour) ────────────────────────────────────
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
    p = Path(__file__).parent / "index.html"
    return p.read_text()


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
    symbol = symbol.upper()
    reverse = _DB.get("reverse", {})
    etfs_meta = _DB.get("etfs", {})

    etf_syms = reverse.get(symbol, [])
    if not etf_syms:
        return []

    results = []
    for etf_sym in etf_syms:
        meta = etfs_meta.get(etf_sym, {})
        # Find this stock's weight in the ETF
        weight = None
        for h in meta.get("holdings", []):
            if h["symbol"].upper() == symbol:
                weight = h["pct"]
                break
        results.append({
            "symbol": etf_sym,
            "name": meta.get("name", etf_sym),
            "weight": weight,
            "aum": meta.get("aum"),
            "expenseRatio": meta.get("expenseRatio"),
            "currency": meta.get("currency", "USD"),
            "dividendYield": meta.get("dividendYield"),
            "category": meta.get("category"),
        })

    # Sort by weight descending
    results.sort(key=lambda x: x.get("weight") or 0, reverse=True)
    return results


@app.get("/api/etf/{symbol:path}")
async def etf_detail(symbol: str):
    symbol = symbol.upper()
    key = f"etf_detail:{symbol}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    # Holdings from DB (fast)
    meta = _DB.get("etfs", {}).get(symbol, {})
    holdings = meta.get("holdings", [])

    def _fetch_live():
        t = yf.Ticker(symbol)
        info = t.info
        hist = t.history(period="6y", interval="1mo", auto_adjust=True)
        return info, hist

    info, hist = await _run(_fetch_live)
    performance = _calc_performance(hist)

    # Expense ratio: prefer DB (already fetched), fall back to live
    exp = meta.get("expenseRatio")

    result = {
        "info": {
            "name": info.get("longName") or info.get("shortName") or symbol,
            "aum": info.get("totalAssets") or meta.get("aum"),
            "expenseRatio": exp,
            "currency": info.get("currency", "USD"),
            "dividendYield": info.get("yield") or info.get("trailingAnnualDividendYield"),
            "category": info.get("category") or meta.get("category"),
            "description": info.get("longBusinessSummary") or "",
        },
        "holdings": holdings,
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
        today = date.today()
        current_year = today.year
        result = {}

        def price_near(target: date):
            for delta in range(0, 7):
                check = (target + timedelta(days=delta * 30)).strftime("%Y-%m")
                matches = [float(p) for idx, p in prices.items() if idx.strftime("%Y-%m") == check]
                if matches:
                    return matches[0]
            for delta in range(1, 4):
                check = (target - timedelta(days=delta * 30)).strftime("%Y-%m")
                matches = [float(p) for idx, p in prices.items() if idx.strftime("%Y-%m") == check]
                if matches:
                    return matches[0]
            return None

        latest = float(prices.iloc[-1])
        p_start = price_near(date(current_year, 1, 1))
        if p_start:
            result["YTD"] = round((latest / p_start - 1) * 100, 2)
        for yr in range(2020, current_year):
            p1 = price_near(date(yr, 1, 1))
            p2 = price_near(date(yr, 12, 1))
            if p1 and p2:
                result[str(yr)] = round((p2 / p1 - 1) * 100, 2)
        return result
    except Exception:
        return {}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  ETF Finder → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
