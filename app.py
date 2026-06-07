"""
ETF Finder — find ETFs by stock holding, powered by Yahoo Finance (yfinance).
No API key required.
"""
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import uvicorn
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

app = FastAPI()

# Thread pool for yfinance (blocking) calls
_pool = ThreadPoolExecutor(max_workers=8)

# Simple in-memory cache (TTL = 1 hour)
_cache: dict[str, tuple[float, object]] = {}
CACHE_TTL = 3600


def _cache_get(key: str):
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return val
    return None


def _cache_set(key: str, val):
    _cache[key] = (time.time(), val)


def _run(fn, *args, **kwargs):
    """Run a blocking function in the thread pool."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_pool, lambda: fn(*args, **kwargs))


# ── Routes ──────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")) as f:
        return f.read()


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


@app.get("/api/etfs/{symbol}")
async def etfs_for_stock(symbol: str):
    symbol = symbol.upper()
    key = f"etfs:{symbol}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    def _get_holders():
        ticker = yf.Ticker(symbol)
        mfh = ticker.mutualfund_holders
        info = ticker.info
        return mfh, info

    mfh, stock_info = await _run(_get_holders)

    if mfh is None or mfh.empty:
        _cache_set(key, [])
        return []

    stock_price_usd = stock_info.get("currentPrice") or stock_info.get("regularMarketPrice") or 0
    currency = stock_info.get("currency", "USD")

    rows = mfh.to_dict("records")

    async def resolve_etf(row: dict):
        holder_name: str = row.get("Holder", "")
        # Fund names come as "TRUST NAME-Fund Display Name"; take the part after the first dash
        parts = holder_name.split("-", 1)
        search_name = parts[1].strip() if len(parts) > 1 else holder_name.strip()

        etf_sym = await _find_etf_ticker(search_name)
        if not etf_sym:
            return None

        etf_info = await _get_etf_info(etf_sym)
        if not etf_info:
            return None

        # Calculate weight: value of holding / ETF total assets
        value = row.get("Value", 0) or 0
        aum = etf_info.get("aum") or 0
        weight = None
        if aum and value:
            # Yahoo reports Value in the local stock currency; convert to USD if needed
            fx = _usd_fx(currency)
            value_usd = value * fx
            weight = round((value_usd / aum) * 100, 3)

        return {
            "symbol": etf_sym,
            "name": etf_info.get("name") or search_name,
            "weight": weight,
            "pctHeld": round(row.get("pctHeld", 0) * 100, 4) if row.get("pctHeld") else None,
            "aum": aum,
            "expenseRatio": etf_info.get("expenseRatio"),
            "currency": etf_info.get("currency") or "USD",
            "dividendYield": etf_info.get("dividendYield"),
            "category": etf_info.get("category"),
        }

    tasks = [resolve_etf(r) for r in rows]
    results = await asyncio.gather(*tasks)
    results = [r for r in results if r]

    # Sort by weight descending
    results.sort(key=lambda x: x.get("weight") or 0, reverse=True)

    _cache_set(key, results)
    return results


async def _find_etf_ticker(fund_name: str) -> str | None:
    key = f"ticker:{fund_name.lower()[:60]}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    def _search():
        try:
            quotes = yf.Search(fund_name, max_results=5).quotes
            for q in quotes:
                sym = q.get("symbol", "")
                qtype = q.get("quoteType", "")
                exchDisp = q.get("exchDisp", "")
                # Prefer ETFs listed on US exchanges
                if qtype in ("ETF", "MUTUALFUND") and not any(c in sym for c in [".", "-"]):
                    return sym
            # fallback: first result without a dot
            for q in quotes:
                sym = q.get("symbol", "")
                if sym and "." not in sym and "-" not in sym:
                    return sym
        except Exception:
            pass
        return None

    result = await _run(_search)
    if result:
        _cache_set(key, result)
    return result


async def _get_etf_info(symbol: str) -> dict | None:
    key = f"etf_info:{symbol}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    def _fetch():
        try:
            t = yf.Ticker(symbol)
            info = t.info
            fd = t.funds_data

            # Expense ratio from fund_operations
            exp_ratio = None
            try:
                ops = fd.fund_operations
                if ops is not None and not ops.empty:
                    row = ops.loc["Annual Report Expense Ratio"]
                    exp_ratio = float(row.iloc[0])
            except Exception:
                pass

            aum = info.get("totalAssets")
            return {
                "name": info.get("longName") or info.get("shortName") or symbol,
                "aum": aum,
                "expenseRatio": exp_ratio,
                "currency": info.get("currency", "USD"),
                "dividendYield": info.get("yield") or info.get("trailingAnnualDividendYield"),
                "category": info.get("category") or info.get("fundFamily"),
                "numberOfHoldings": None,
                "description": info.get("longBusinessSummary") or "",
            }
        except Exception:
            return None

    result = await _run(_fetch)
    if result:
        _cache_set(key, result)
    return result


@app.get("/api/etf/{symbol}")
async def etf_detail(symbol: str):
    symbol = symbol.upper()
    key = f"etf_detail:{symbol}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    def _fetch():
        t = yf.Ticker(symbol)
        info = t.info
        fd = t.funds_data

        # Top holdings
        holdings = []
        try:
            th = fd.top_holdings
            if th is not None and not th.empty:
                for sym_idx, row in th.iterrows():
                    holdings.append({
                        "asset": sym_idx,
                        "name": row.get("Name", ""),
                        "weightPercentage": round(float(row.get("Holding Percent", 0)) * 100, 3),
                    })
        except Exception:
            pass

        # Expense ratio
        exp_ratio = None
        try:
            ops = fd.fund_operations
            if ops is not None and not ops.empty:
                exp_ratio = float(ops.loc["Annual Report Expense Ratio"].iloc[0])
        except Exception:
            pass

        # Historical prices for performance
        hist = t.history(period="6y", interval="1mo", auto_adjust=True)

        return info, holdings, hist, exp_ratio

    info, holdings, hist, exp_ratio = await _run(_fetch)

    performance = _calc_performance(hist)

    result = {
        "info": {
            "name": info.get("longName") or info.get("shortName") or symbol,
            "aum": info.get("totalAssets"),
            "expenseRatio": exp_ratio,
            "currency": info.get("currency", "USD"),
            "dividendYield": info.get("yield") or info.get("trailingAnnualDividendYield"),
            "category": info.get("category"),
            "inceptionDate": None,
            "description": info.get("longBusinessSummary") or "",
            "numberOfHoldings": info.get("holdings_count"),
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
            target_str = target.strftime("%Y-%m")
            for i in range(6):
                check = (target + timedelta(days=i * 30)).strftime("%Y-%m")
                matches = [p for idx, p in prices.items() if idx.strftime("%Y-%m") == check]
                if matches:
                    return matches[0]
            for i in range(1, 4):
                check = (target - timedelta(days=i * 30)).strftime("%Y-%m")
                matches = [p for idx, p in prices.items() if idx.strftime("%Y-%m") == check]
                if matches:
                    return matches[0]
            return None

        latest = float(prices.iloc[-1])

        # YTD
        p_start = price_near(date(current_year, 1, 1))
        if p_start:
            result["YTD"] = round((latest / float(p_start) - 1) * 100, 2)

        # Annual
        for yr in range(2020, current_year):
            p1 = price_near(date(yr, 1, 1))
            p2 = price_near(date(yr, 12, 1))
            if p1 and p2:
                result[str(yr)] = round((float(p2) / float(p1) - 1) * 100, 2)

        return result
    except Exception:
        return {}


def _usd_fx(currency: str) -> float:
    """Very rough FX — returns 1.0 for USD, otherwise fetches from Yahoo."""
    if not currency or currency.upper() == "USD":
        return 1.0
    key = f"fx:{currency.upper()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        pair = f"{currency.upper()}USD=X"
        rate = yf.Ticker(pair).info.get("regularMarketPrice") or 1.0
        _cache_set(key, rate)
        return float(rate)
    except Exception:
        return 1.0


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  ETF Finder → http://localhost:{port}\n  No API key required — powered by Yahoo Finance\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
