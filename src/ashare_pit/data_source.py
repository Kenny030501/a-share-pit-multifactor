#!/usr/bin/env python3
"""A 股 Point-in-Time 数据适配层。

统一读取全 A 股票名单、后复权日线、按报告期披露的财务数据和 TTM PE，并在进入
下游前转成普通 Python 字典。公开源的“最新公告日期”字段在本样本中不可靠，因此
财务可用日采用法定披露截止日的保守规则：一季报 4 月 30 日、半年报 8 月 31 日、
三季报 10 月 31 日、年报次年 4 月 30 日。该规则避免提前使用财报，但不声称还原
每份公告的真实发布时间。
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
from pathlib import Path
from typing import Any

from .cache import read_cache, write_cache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "interim" / "cache"
LEGACY_CACHE_PATH = PROJECT_ROOT / ".data_cache"
CACHE_DIR_DEFAULT = str(
    LEGACY_CACHE_PATH if LEGACY_CACHE_PATH.exists() and not DEFAULT_CACHE_PATH.exists() else DEFAULT_CACHE_PATH
)

# ---------------------------------------------------------------------------
# 报告期与保守可用日
# ---------------------------------------------------------------------------
_PERIOD_DEADLINE = {
    "0331": (0, 4, 30),   # Q1  -> same year Apr 30
    "0630": (0, 8, 31),   # H1  -> same year Aug 31
    "0930": (0, 10, 31),  # Q3  -> same year Oct 31
    "1231": (1, 4, 30),   # 年报 -> next year Apr 30
}


def report_periods(start_year: int, end_year: int) -> list[str]:
    """All quarterly report periods 'YYYYMMDD' from start_year to end_year."""
    periods: list[str] = []
    for year in range(start_year, end_year + 1):
        for month_day in ("0331", "0630", "0930", "1231"):
            periods.append(f"{year}{month_day}")
    return periods


def availability_date(period: str) -> dt.date:
    """Earliest date a report period could legally be public (deadline rule)."""
    if len(period) != 8 or period[4:] not in _PERIOD_DEADLINE:
        raise ValueError(f"Bad report period: {period!r}")
    year = int(period[:4])
    offset, month, day = _PERIOD_DEADLINE[period[4:]]
    return dt.date(year + offset, month, day)


# ---------------------------------------------------------------------------
# AkShare 访问与字典标准化：不让 pandas 类型泄漏到下游
# ---------------------------------------------------------------------------
def _ak():
    try:
        import akshare  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("akshare is required: pip install akshare") from exc
    return akshare


def _clean(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "isoformat"):  # pandas Timestamp
        try:
            return value.isoformat()
        except Exception:
            pass
    try:
        import pandas as pd  # noqa: PLC0415

        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except Exception:
            pass
    return value


def _records(df) -> list[dict[str, Any]]:
    return [{k: _clean(v) for k, v in rec.items()} for rec in df.to_dict("records")]


# ---------------------------------------------------------------------------
# 股票池
# ---------------------------------------------------------------------------
def load_universe(cache_dir: str | None = CACHE_DIR_DEFAULT, refresh: bool = False) -> list[dict[str, str]]:
    """Full A-share list as [{'code': '600519', 'name': '贵州茅台'}, ...]."""
    if not refresh:
        cached = read_cache(cache_dir, "cn_universe", "all", None)
        if cached:
            return cached
    df = _ak().stock_info_a_code_name()
    data = [{"code": str(r["code"]).zfill(6), "name": r["name"]} for r in _records(df)]
    write_cache(cache_dir, "cn_universe", "all", data)
    return data


def load_share_snapshot(
    cache_dir: str | None = CACHE_DIR_DEFAULT, refresh: bool = False
) -> dict[str, dict[str, float | None]]:
    """Current per-stock spot snapshot: {code: {'price': 最新价, 'mktcap': 总市值|None}}.

    One spot call, cached under cn_shares/snapshot (network only on first build /
    --refresh). It anchors the *unadjusted* price used for market-cap reconstruction:
    the hfq series are adjusted, and the cached `amount` is a hfq_close×volume proxy,
    so amount/volume is NOT the raw price — see ashare_pit.size. Sina (`stock_zh_a_spot`) is
    primary; the eastmoney host (`stock_zh_a_spot_em`) is often TLS-blocked here.
    """
    if not refresh:
        cached = read_cache(cache_dir, "cn_shares", "snapshot", None)
        if cached is not None:
            return cached.get("data", cached) if isinstance(cached, dict) and "data" in cached else cached
    ak = _ak()
    out: dict[str, dict[str, float | None]] = {}
    source = None
    for fn in ("stock_zh_a_spot", "stock_zh_a_spot_em"):
        try:
            recs = _records(getattr(ak, fn)())
        except Exception:
            continue
        cols = list(recs[0].keys()) if recs else []
        code_c = next((c for c in ("代码", "symbol", "code") if c in cols), None)
        price_c = next((c for c in ("最新价", "trade", "price") if c in cols), None)
        mc_c = "总市值" if "总市值" in cols else None
        if not code_c or not price_c:
            continue
        for r in recs:
            digits = "".join(ch for ch in str(r.get(code_c) or "") if ch.isdigit())
            if len(digits) < 6:
                continue
            try:
                price = float(r.get(price_c))
            except (TypeError, ValueError):
                continue
            if not price or price <= 0:
                continue
            mc: float | None = None
            if mc_c is not None:
                try:
                    mc = float(r.get(mc_c))
                except (TypeError, ValueError):
                    mc = None
            out[digits[-6:]] = {"price": price, "mktcap": mc}
        if out:
            source = fn
            break
    if not out:
        raise RuntimeError("load_share_snapshot: spot fetch failed on all sources (sina/eastmoney).")
    write_cache(cache_dir, "cn_shares", "snapshot", {"source": source, "data": out})
    return out


# ---------------------------------------------------------------------------
# 财务数据：按报告期批量读取业绩报表
# ---------------------------------------------------------------------------
_YJBB_MAP = {
    "revenue": "营业总收入-营业总收入",
    "revenue_yoy_percent": "营业总收入-同比增长",
    "net_profit": "净利润-净利润",
    "net_profit_yoy_percent": "净利润-同比增长",
    "roe_percent": "净资产收益率",
    "gross_margin_percent": "销售毛利率",
    "eps": "每股收益",
    "bps": "每股净资产",
    "industry": "所处行业",
}


def load_yjbb_period(
    period: str, cache_dir: str | None = CACHE_DIR_DEFAULT, refresh: bool = False
) -> dict[str, dict[str, Any]]:
    """One report period's fundamentals for the whole market: {code: {fields}}."""
    if not refresh:
        cached = read_cache(cache_dir, "cn_yjbb", period, None)
        if cached is not None:
            return cached
    try:
        df = _ak().stock_yjbb_em(date=period)
    except (TypeError, KeyError, ValueError):
        return {}  # period not yet reported / no data; do not cache empty
    out: dict[str, dict[str, Any]] = {}
    for rec in _records(df):
        code = str(rec.get("股票代码") or "").zfill(6)
        if not code or len(code) != 6:
            continue
        out[code] = {field: rec.get(src) for field, src in _YJBB_MAP.items()}
    write_cache(cache_dir, "cn_yjbb", period, out)
    return out


def build_financial_timeline(
    periods: list[str],
    cache_dir: str | None = CACHE_DIR_DEFAULT,
    refresh: bool = False,
    codes: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Per-code sorted list of {period, avail_date, financials} across periods."""
    today = dt.date.today()
    timeline: dict[str, list[dict[str, Any]]] = {}
    for period in periods:
        avail_dt = availability_date(period)
        if avail_dt > today:
            continue  # report not yet due (or future period) -> unusable for PIT
        avail = avail_dt.isoformat()
        period_data = load_yjbb_period(period, cache_dir, refresh)
        for code, financials in period_data.items():
            if codes is not None and code not in codes:
                continue
            timeline.setdefault(code, []).append(
                {"period": period, "avail_date": avail, "financials": financials}
            )
    for entries in timeline.values():
        entries.sort(key=lambda e: e["period"])
    return timeline


def pit_financials(entries: list[dict[str, Any]], as_of: dt.date | str) -> dict[str, Any] | None:
    """Latest report already available at as_of (avail_date <= as_of)."""
    as_of_iso = as_of.isoformat() if isinstance(as_of, dt.date) else as_of
    available = [e for e in entries if e["avail_date"] <= as_of_iso]
    if not available:
        return None
    return max(available, key=lambda e: e["period"])


# ---------------------------------------------------------------------------
# 后复权日线与 TTM PE
# ---------------------------------------------------------------------------
def _prefixed(code: str) -> str:
    """A-share code -> exchange-prefixed symbol (sina/tencent format)."""
    code = str(code).zfill(6)
    if code[0] == "6":
        return "sh" + code            # 上交所(含 688 科创板)
    if code[0] in ("0", "3"):
        return "sz" + code            # 深交所(000/002/300)
    if code[0] in ("4", "8") or code[:2] == "92":
        return "bj" + code            # 北交所
    return "sh" + code


def load_hfq_prices(
    code: str,
    start: str = "20140101",
    end: str | None = None,
    cache_dir: str | None = CACHE_DIR_DEFAULT,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """后复权 daily bars: [{'date','open','high','low','close','volume','amount'}, ...].

    Uses Sina (`stock_zh_a_daily`) as primary and Tencent (`stock_zh_a_hist_tx`)
    as fallback; the eastmoney source is avoided because its host is frequently
    blocked. Both sources return full OHLCV; the fields are kept verbatim (hfq
    prices, raw volume) so downstream can build the GTJA191/Alpha101 price-volume
    factors (价量背离/开盘缺口/异常成交量/量幅背离 …) that need open/high/low/volume.
    `amount` stays a turnover proxy (close×volume) for backward compatibility with
    the Amihud liquidity factor; Tencent lacks volume, so its bars carry
    volume=None and amount from the source's own 成交额 column.
    """
    end = end or dt.date.today().strftime("%Y%m%d")
    key = f"{code}_{start}_{end}"
    if not refresh:
        cached = read_cache(cache_dir, "cn_price_hfq", key, None)
        if cached is not None:
            return cached
    sym = _prefixed(code)
    ak = _ak()
    bars: list[dict[str, Any]] | None = None
    try:
        df = ak.stock_zh_a_daily(symbol=sym, start_date=start, end_date=end, adjust="hfq")
        bars = [
            {
                "date": r["date"],
                "open": r.get("open"),
                "high": r.get("high"),
                "low": r.get("low"),
                "close": r["close"],
                "volume": r.get("volume"),
                "amount": (r["close"] * r["volume"]) if r.get("volume") else None,
            }
            for r in _records(df)
        ]
    except Exception:
        df = ak.stock_zh_a_hist_tx(symbol=sym, start_date=start, end_date=end, adjust="hfq")
        bars = [
            {
                "date": r["date"],
                "open": r.get("open"),
                "high": r.get("high"),
                "low": r.get("low"),
                "close": r["close"],
                "volume": None,
                "amount": r.get("amount"),
            }
            for r in _records(df)
        ]
    write_cache(cache_dir, "cn_price_hfq", key, bars)
    return bars


def load_pe_ttm(
    code: str,
    period: str = "近五年",
    cache_dir: str | None = CACHE_DIR_DEFAULT,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """TTM PE time series: [{'date','pe_ttm'}, ...] (Baidu, limited history)."""
    if not refresh:
        cached = read_cache(cache_dir, "cn_pe_ttm", code, None)
        if cached is not None:
            return cached
    df = _ak().stock_zh_valuation_baidu(symbol=code, indicator="市盈率(TTM)", period=period)
    series = [{"date": r["date"], "pe_ttm": r["value"]} for r in _records(df)]
    write_cache(cache_dir, "cn_pe_ttm", code, series)
    return series


# ---------------------------------------------------------------------------
# 指数成分与基准价格
# ---------------------------------------------------------------------------
def load_index_constituents(
    index_code: str = "000300",
    cache_dir: str | None = CACHE_DIR_DEFAULT,
    refresh: bool = False,
) -> list[str]:
    """Current index members as 6-digit codes (e.g. 沪深300='000300').

    Note: this is the *current* membership, so backtests using it carry
    index-membership survivorship bias (point-in-time membership is not
    freely available). Fundamentals/prices themselves stay point-in-time.
    """
    if not refresh:
        cached = read_cache(cache_dir, "cn_index_cons", index_code, None)
        if cached:
            return cached
    df = _ak().index_stock_cons(symbol=index_code)
    codes = [str(r["品种代码"]).zfill(6) for r in _records(df) if r.get("品种代码")]
    write_cache(cache_dir, "cn_index_cons", index_code, codes)
    return codes


def load_index_close(
    symbol: str = "sh000300",
    cache_dir: str | None = CACHE_DIR_DEFAULT,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Benchmark index daily close: [{'date','close'}, ...] (Sina)."""
    if not refresh:
        cached = read_cache(cache_dir, "cn_index_close", symbol, None)
        if cached is not None:
            return cached
    df = _ak().stock_zh_index_daily(symbol=symbol)
    series = [{"date": r["date"], "close": r["close"]} for r in _records(df)]
    write_cache(cache_dir, "cn_index_close", symbol, series)
    return series


def load_benchmark(
    benchmark_code: str = "880008",
    cache_dir: str | None = CACHE_DIR_DEFAULT,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Benchmark index daily close series. Dispatches to the right loader.

    Recognised codes:
      - ``880008`` — TDX 全A等权 (xlsx → cached)
      - ``sh000300`` / ``000300`` — 沪深 300 (Sina)
      - ``sh000905`` / ``000905`` — 中证 500 (Sina)
      - any other ``shXXXXXX`` / ``szXXXXXX`` — Sina index family
    """
    if benchmark_code == "880008":
        return load_equal_weight_880008(cache_dir, refresh)
    if not (benchmark_code.startswith("sh") or benchmark_code.startswith("sz")):
        benchmark_code = f"sh{benchmark_code}"
    return load_index_close(benchmark_code, cache_dir, refresh)


def load_equal_weight_880008(
    cache_dir: str | None = CACHE_DIR_DEFAULT,
    refresh: bool = False,
    xlsx_path: str = "tdx_880008_quan_a_equal_weight_daily.xlsx",
) -> list[dict[str, Any]]:
    """通达信 880008「全A等权」指数日线基准: [{'date','close'}, ...].

    Read from the user-supplied xlsx once, then cached (xlsx not needed after).
    Unlike an index rebuilt from the *current* universe, 880008 is a real,
    real-time-edited index that already includes since-delisted names, so the
    benchmark side carries no survivorship bias. (The strategy's own current-全A
    universe still does — that asymmetry is noted where the benchmark is used.)
    """
    if not refresh:
        cached = read_cache(cache_dir, "cn_index_close", "880008", None)
        if cached is not None:
            return cached
    try:
        import openpyxl  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openpyxl required to read the 880008 xlsx: pip install openpyxl") from exc
    import os  # noqa: PLC0415
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"880008 benchmark file not found: {xlsx_path}")
    ws = openpyxl.load_workbook(xlsx_path, read_only=True).active
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    di, ci = header.index("date"), header.index("close")
    series = [{"date": str(r[di])[:10], "close": float(r[ci])}
              for r in rows if r[di] is not None and r[ci] is not None]
    write_cache(cache_dir, "cn_index_close", "880008", series)
    return series


# ---------------------------------------------------------------------------
# 统一离线快照：下游程序的单一数据入口
# ---------------------------------------------------------------------------
def _read_blob_data(path: str) -> Any:
    """Return the `data` payload of a data_cache blob file (or None)."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("data")
    except (OSError, ValueError):
        return None


def load_market_snapshot(
    cache_dir: str | None = CACHE_DIR_DEFAULT,
    with_prices: bool = True,
    with_pe: bool = True,
    benchmark_code: str = "880008",
) -> dict[str, Any]:
    """One offline call returning everything already in `data/interim/cache` — the
    single data-access entry point for downstream programs (no network, no
    per-stock sleep).

    Returns plain dicts, shaped exactly like the backtest already consumes:
      universe            [{code, name}, ...]
      prices              {code: [{date, close, amount}, ...]}   (hfq)
      financials          {code: [{period, avail_date, financials{...}}, ...]}  (PIT timeline)
      pe                  {code: [{date, pe_ttm}, ...]}
      benchmark           [{date, close}, ...]
      meta                counts / date ranges / cached report periods

    Prices are located by *globbing* the cache (picking the widest series per
    code) rather than reconstructing `{code}_{start}_{end}` keys — the end date
    is today by default, so key reconstruction silently misses the cache and
    re-fetches. Downstream code should use this function, not `load_hfq_prices`.
    """
    base = cache_dir or CACHE_DIR_DEFAULT

    universe = read_cache(cache_dir, "cn_universe", "all", None) or []

    # Financials: only report periods actually cached → build_financial_timeline
    # reads them from cache and never hits the network.
    period_files = sorted(glob.glob(os.path.join(base, "cn_yjbb", "*.json")))
    cached_periods = [os.path.basename(p)[:-5] for p in period_files]
    financials = (
        build_financial_timeline(cached_periods, cache_dir, refresh=False, codes=None)
        if cached_periods else {}
    )

    prices: dict[str, list[dict[str, Any]]] = {}
    if with_prices:
        best: dict[str, tuple[tuple[int, str], list]] = {}  # code -> ((nbars,last_date), series)
        for path in glob.glob(os.path.join(base, "cn_price_hfq", "*.json")):
            code = os.path.basename(path)[:-5].split("_", 1)[0]
            data = _read_blob_data(path) or []
            if not data:
                continue
            key = (len(data), data[-1]["date"])
            if code not in best or key > best[code][0]:
                best[code] = (key, data)
        prices = {c: v[1] for c, v in best.items()}

    pe: dict[str, list[dict[str, Any]]] = {}
    if with_pe:
        for path in glob.glob(os.path.join(base, "cn_pe_ttm", "*.json")):
            pe[os.path.basename(path)[:-5]] = _read_blob_data(path) or []

    benchmark = load_benchmark(benchmark_code, cache_dir)

    firsts = [s[0]["date"] for s in prices.values() if s]
    lasts = [s[-1]["date"] for s in prices.values() if s]
    meta = {
        "cache_dir": base,
        "n_universe": len(universe),
        "n_prices": len(prices),
        "n_pe": len([c for c, s in pe.items() if s]),
        "fin_periods": cached_periods,
        "price_date_range": [min(firsts), max(lasts)] if firsts else None,
        "benchmark_range": [benchmark[0]["date"], benchmark[-1]["date"]] if benchmark else None,
    }
    return {"universe": universe, "prices": prices, "financials": financials,
            "pe": pe, "benchmark": benchmark, "meta": meta}


def export_inventory_xlsx(
    path: str = "data_snapshot.xlsx",
    cache_dir: str | None = CACHE_DIR_DEFAULT,
) -> None:
    """Export a human-readable Excel snapshot of the cached data.

    Prices (~11M rows) cannot fit a sheet, so only a per-stock coverage summary
    is written; raw prices stay in JSON (use `load_market_snapshot`).
    """
    try:
        import openpyxl  # noqa: PLC0415
        from openpyxl.cell import WriteOnlyCell  # noqa: PLC0415
        from openpyxl.styles import Font, PatternFill  # noqa: PLC0415
        from openpyxl.utils import get_column_letter  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openpyxl required to export Excel: pip install openpyxl") from exc

    snap = load_market_snapshot(cache_dir, with_prices=True, with_pe=True)
    uni, prices, fin = snap["universe"], snap["prices"], snap["financials"]
    pe, bench, meta = snap["pe"], snap["benchmark"], snap["meta"]
    name_of = {r["code"]: r.get("name", "") for r in uni}

    wb = openpyxl.Workbook(write_only=True)
    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="305496")

    def header(ws, cols: list[str], widths: list[int]) -> None:
        row = []
        for c in cols:
            cell = WriteOnlyCell(ws, value=c)
            cell.font, cell.fill = head_font, head_fill
            row.append(cell)
        ws.append(row)
        ws.freeze_panes = "A2"
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w

    ws = wb.create_sheet("数据总览")
    header(ws, ["数据", "存储位置", "格式", "记录数", "日期范围", "调用/说明"], [16, 42, 8, 8, 24, 52])
    pr, br = meta["price_date_range"], meta["benchmark_range"]
    ws.append(["全A股票列表", "data/interim/cache/cn_universe/ALL.json", "JSON", meta["n_universe"], "",
               "load_market_snapshot()['universe']"])
    ws.append(["后复权日线", "data/interim/cache/cn_price_hfq/{code}_*.json", "JSON", meta["n_prices"],
               f"{pr[0]}~{pr[1]}" if pr else "", "['prices'][code](Excel仅覆盖摘要,原始留JSON)"])
    ws.append(["业绩报表财务", "data/interim/cache/cn_yjbb/{报告期}.json", "JSON", len(meta["fin_periods"]),
               f"{meta['fin_periods'][0]}~{meta['fin_periods'][-1]}" if meta["fin_periods"] else "",
               "['financials'][code];PIT用 cn.pit_financials(timeline, as_of)"])
    ws.append(["TTM PE", "data/interim/cache/cn_pe_ttm/{code}.json", "JSON", meta["n_pe"], "", "['pe'][code]"])
    ws.append(["880008 全A等权基准", "data/interim/cache/cn_index_close/880008.json", "JSON",
               len(bench) if bench else 0, f"{br[0]}~{br[1]}" if br else "", "['benchmark']"])
    ws.append(["→ 统一调用入口", "ashare_pit.data_source.load_market_snapshot()", "Python", "", "",
               "一次拿全、离线秒级;勿用 load_hfq_prices(默认end=今天→cache miss重抓)"])

    ws = wb.create_sheet("股票列表")
    header(ws, ["代码", "名称"], [12, 26])
    for r in uni:
        ws.append([r["code"], r.get("name", "")])

    ws = wb.create_sheet("财务全量")
    fkeys = ["revenue", "revenue_yoy_percent", "net_profit", "net_profit_yoy_percent",
             "roe_percent", "gross_margin_percent", "eps", "bps", "industry"]
    header(ws, ["代码", "名称", "报告期", "PIT可用日", "营业总收入", "营收同比%", "净利润", "净利同比%",
                "ROE%", "毛利率%", "每股收益", "每股净资产", "所处行业"],
           [12, 16, 10, 12, 16, 10, 16, 10, 8, 8, 10, 10, 16])
    for code in sorted(fin):
        nm = name_of.get(code, "")
        for e in fin[code]:
            f = e["financials"]
            ws.append([code, nm, e["period"], e["avail_date"]] + [f.get(k) for k in fkeys])

    ws = wb.create_sheet("PE覆盖")
    header(ws, ["代码", "名称", "首日", "末日", "点数"], [12, 16, 12, 12, 8])
    for code in sorted(pe):
        s = pe[code]
        if s:
            ws.append([code, name_of.get(code, ""), s[0]["date"], s[-1]["date"], len(s)])

    ws = wb.create_sheet("基准880008")
    header(ws, ["日期", "收盘"], [12, 12])
    for b in bench or []:
        ws.append([b["date"], b["close"]])

    ws = wb.create_sheet("价格覆盖")
    header(ws, ["代码", "名称", "首日", "末日", "条数", "最新后复权收盘"], [12, 16, 12, 12, 8, 14])
    for code in sorted(prices):
        s = prices[code]
        if s:
            ws.append([code, name_of.get(code, ""), s[0]["date"], s[-1]["date"], len(s), s[-1]["close"]])

    wb.save(path)


def export_price_panel_xlsx(
    path: str = "全市场日线面板_后复权收盘.xlsx",
    cache_dir: str | None = CACHE_DIR_DEFAULT,
    field: str = "close",
    start: str = "2021-01-01",
) -> None:
    """Full-market daily panel in **wide** form: rows = trading days (>= start),
    columns = every stock, cell = that day's value of `field` (close / amount),
    blank if the stock did not trade that day.

    A wide matrix fits Excel (≤16384 cols): ~1300 rows × ~5500 cols. The long
    form (day×stock) would be ~7M rows and does not fit. Values are located by
    walking a per-stock cursor along the shared 880008 trading calendar, so it
    is O(total bars) and memory-light.
    """
    try:
        import openpyxl  # noqa: PLC0415
        from openpyxl.cell import WriteOnlyCell  # noqa: PLC0415
        from openpyxl.styles import Font, PatternFill  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openpyxl required to export Excel: pip install openpyxl") from exc

    snap = load_market_snapshot(cache_dir, with_prices=True, with_pe=False)
    prices, bench = snap["prices"], snap["benchmark"]
    name_of = {r["code"]: r.get("name", "") for r in snap["universe"]}
    codes = sorted(prices)
    # Cap at the last day any stock actually traded (the index calendar may run a
    # day ahead), so there is no trailing all-blank row.
    last_stock_day = snap["meta"]["price_date_range"][1] if snap["meta"]["price_date_range"] else start
    dates = [b["date"] for b in bench if start <= b["date"] <= last_stock_day]

    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet(f"后复权{('收盘' if field == 'close' else field)}")
    head_font, head_fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="305496")

    def head_row(values: list) -> list:
        out = []
        for v in values:
            cell = WriteOnlyCell(ws, value=v)
            cell.font, cell.fill = head_font, head_fill
            out.append(cell)
        return out

    ws.append(head_row(["日期/代码"] + codes))
    ws.append(head_row(["名称"] + [name_of.get(c, "") for c in codes]))
    ws.freeze_panes = "B3"

    bars = {c: prices[c] for c in codes}
    ptr = {c: 0 for c in codes}
    for d in dates:
        row = [d]
        for c in codes:
            b, i = bars[c], ptr[c]
            n = len(b)
            while i < n and b[i]["date"] < d:
                i += 1
            ptr[c] = i
            row.append(b[i].get(field) if (i < n and b[i]["date"] == d) else None)
        ws.append(row)
    wb.save(path)


# ---------------------------------------------------------------------------
# 离线自检与联网演示
# ---------------------------------------------------------------------------
def self_test() -> None:
    assert availability_date("20231231") == dt.date(2024, 4, 30), "年报次年4-30"
    assert availability_date("20230331") == dt.date(2023, 4, 30), "一季报当年4-30"
    assert availability_date("20230630") == dt.date(2023, 8, 31), "半年报当年8-31"
    assert availability_date("20230930") == dt.date(2023, 10, 31), "三季报当年10-31"
    print("[PASS] availability_date 四类报告期截止日正确")

    entries = [
        {"period": "20230331", "avail_date": "2023-04-30", "financials": {"roe_percent": 5}},
        {"period": "20230630", "avail_date": "2023-08-31", "financials": {"roe_percent": 10}},
        {"period": "20230930", "avail_date": "2023-10-31", "financials": {"roe_percent": 15}},
        {"period": "20231231", "avail_date": "2024-04-30", "financials": {"roe_percent": 20}},
    ]
    # Before Q1 deadline: nothing available.
    assert pit_financials(entries, "2023-04-01") is None, "Q1截止前无可用"
    # Between Q1 and H1 deadline: only Q1.
    assert pit_financials(entries, "2023-06-01")["period"] == "20230331"
    # Right after annual deadline: annual is freshest.
    assert pit_financials(entries, "2024-05-01")["period"] == "20231231"
    # Just before annual deadline: still Q3 (annual not yet public).
    assert pit_financials(entries, "2024-04-15")["period"] == "20230930"
    print("[PASS] pit_financials 按法定截止日选最新可用报告(无前视)")
    print("ALL PASS")


def demo(cache_dir: str) -> None:
    universe = load_universe(cache_dir)
    print(f"全A 列表: {len(universe)} 只,示例 {universe[:3]}")

    periods = report_periods(2023, 2024)
    print(f"抓取 {len(periods)} 个报告期业绩报表(全市场批量,可能较慢)...")
    sample = {"600519", "000001"}
    timeline = build_financial_timeline(periods, cache_dir, codes=sample)

    for code in ("600519", "000001"):
        entries = timeline.get(code, [])
        print(f"\n=== {code} 财务时间线({len(entries)} 期)===")
        for as_of in ("2024-04-15", "2024-05-15", "2025-06-30"):
            pit = pit_financials(entries, as_of)
            if pit:
                fin = pit["financials"]
                print(
                    f"  截至 {as_of}: 用 {pit['period']} 期(可用日 {pit['avail_date']})"
                    f" 营收同比={fin['revenue_yoy_percent']}% ROE={fin['roe_percent']}"
                    f" 毛利率={fin['gross_margin_percent']}"
                )
            else:
                print(f"  截至 {as_of}: 无可用报告")

    bars = load_hfq_prices("600519", start="20240101", end="20240131", cache_dir=cache_dir)
    print(f"\n600519 后复权日线(2024-01):{len(bars)} 根,末根 {bars[-1] if bars else None}")

    pe = load_pe_ttm("600519", cache_dir=cache_dir)
    print(f"600519 TTM PE 序列:{len(pe)} 点,末点 {pe[-1] if pe else None}")


def _print_snapshot_summary(cache_dir: str | None) -> None:
    snap = load_market_snapshot(cache_dir)
    m = snap["meta"]
    print("数据快照(离线,来自 data/interim/cache):")
    print(f"  股票列表  : {m['n_universe']} 只")
    print(f"  后复权价  : {m['n_prices']} 只  日期 {m['price_date_range']}")
    print(f"  财务报告期: {len(m['fin_periods'])} 期  {m['fin_periods'][0]}~{m['fin_periods'][-1]}"
          if m["fin_periods"] else "  财务报告期: 0 期")
    print(f"  TTM PE    : {m['n_pe']} 只")
    print(f"  基准: {len(snap['benchmark'])} 天  日期 {m['benchmark_range']}")
    print("  调用: ashare_pit.data_source.load_market_snapshot()")


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share point-in-time data adapter (AkShare).")
    parser.add_argument("--self-test", action="store_true", help="Offline logic tests, no network.")
    parser.add_argument("--demo", action="store_true", help="Network demo on a few stocks.")
    parser.add_argument("--snapshot", action="store_true",
                        help="Print offline cached-data inventory summary.")
    parser.add_argument("--export-excel", nargs="?", const="data_snapshot.xlsx", default=None,
                        metavar="PATH", help="Export a human-readable Excel snapshot (default data_snapshot.xlsx).")
    parser.add_argument("--export-panel", nargs="?", const="全市场日线面板_后复权收盘.xlsx", default=None,
                        metavar="PATH", help="Export the full-market daily wide panel (dates × stocks).")
    parser.add_argument("--panel-field", default="close", choices=["close", "amount"],
                        help="Field for --export-panel (default close).")
    parser.add_argument("--panel-start", default="2021-01-01", help="First trading day for --export-panel.")
    parser.add_argument("--data-cache-dir", default=CACHE_DIR_DEFAULT)
    args = parser.parse_args()
    cache_dir = args.data_cache_dir or None
    if args.self_test:
        self_test()
    elif args.demo:
        demo(cache_dir)
    elif args.snapshot:
        _print_snapshot_summary(cache_dir)
    elif args.export_excel is not None:
        export_inventory_xlsx(args.export_excel, cache_dir)
        print(f"已导出 Excel 快照: {args.export_excel}")
    elif args.export_panel is not None:
        export_price_panel_xlsx(args.export_panel, cache_dir, args.panel_field, args.panel_start)
        print(f"已导出全市场日线面板: {args.export_panel}(字段 {args.panel_field},自 {args.panel_start})")
    else:
        parser.error("Use --self-test / --demo / --snapshot / --export-excel / --export-panel.")


if __name__ == "__main__":
    main()
