"""一次性脚本: 将 data/interim/cache JSON 缓存导入 SQLite 数据库。

输出: data/interim/a_share_research.sqlite (5 张表)
支持断点续传: 已存在的行自动跳过。
"""

import glob
import json
import os
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "interim" / "cache"
LEGACY_CACHE = ROOT / ".data_cache"
CACHE = str(LEGACY_CACHE if LEGACY_CACHE.exists() and not DEFAULT_CACHE.exists() else DEFAULT_CACHE)
DB_PATH = str(ROOT / "data" / "interim" / "a_share_research.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS universe (
    code TEXT PRIMARY KEY,
    name TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    code TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    amount REAL,
    PRIMARY KEY (code, date)
);

CREATE TABLE IF NOT EXISTS benchmark (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    close REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS financials (
    code TEXT NOT NULL,
    period TEXT NOT NULL,
    avail_date TEXT NOT NULL,
    revenue REAL,
    revenue_yoy REAL,
    net_profit REAL,
    net_profit_yoy REAL,
    roe REAL,
    gross_margin REAL,
    eps REAL,
    bps REAL,
    industry TEXT,
    PRIMARY KEY (code, period)
);

CREATE TABLE IF NOT EXISTS pe_ttm (
    code TEXT NOT NULL,
    date TEXT NOT NULL,
    pe_ttm REAL,
    PRIMARY KEY (code, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);
CREATE INDEX IF NOT EXISTS idx_benchmark_date ON benchmark(date);
CREATE INDEX IF NOT EXISTS idx_financials_avail_date ON financials(avail_date);
CREATE INDEX IF NOT EXISTS idx_pe_ttm_date ON pe_ttm(date);
"""


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
    db.executescript(SCHEMA)
    return db


def import_universe(db):
    path = os.path.join(CACHE, "cn_universe", "ALL.json")
    if not os.path.exists(path):
        log("universe: ALL.json 不存在,跳过")
        return
    with open(path) as f:
        rows = json.load(f).get("data", [])
    db.executemany("INSERT OR IGNORE INTO universe(code,name) VALUES(?,?)",
                   [(r["code"], r["name"]) for r in rows])
    db.commit()
    log(f"universe: {len(rows)} 行")


def import_benchmark(db):
    files = sorted(glob.glob(os.path.join(CACHE, "cn_index_close", "*.json")))
    total = 0
    for fp in files:
        symbol = os.path.basename(fp)[:-5]
        with open(fp) as f:
            rows = json.load(f).get("data", [])
        db.executemany("INSERT OR IGNORE INTO benchmark(symbol,date,close) VALUES(?,?,?)",
                       [(symbol, r["date"], r["close"]) for r in rows])
        total += len(rows)
    db.commit()
    log(f"benchmark: {total} 行 ({len(files)} 文件)")


def import_prices(db):
    files = sorted(glob.glob(os.path.join(CACHE, "cn_price_hfq", "*.json")))
    total = 0
    batch = []
    t0 = time.time()
    for i, fp in enumerate(files):
        with open(fp) as f:
            rows = json.load(f).get("data", [])
        code = os.path.basename(fp).split("_", 1)[0]
        for r in rows:
            batch.append((code, r["date"], r.get("open"), r.get("high"),
                          r.get("low"), r["close"], r.get("volume"), r.get("amount")))
        if len(batch) >= 50000:
            db.executemany(
                "INSERT OR IGNORE INTO prices(code,date,open,high,low,close,volume,amount) "
                "VALUES(?,?,?,?,?,?,?,?)", batch)
            total += len(batch)
            batch.clear()
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            log(f"prices: {i+1}/{len(files)} 文件, {total} 行, {el:.1f}s")
    if batch:
        db.executemany(
            "INSERT OR IGNORE INTO prices(code,date,open,high,low,close,volume,amount) "
            "VALUES(?,?,?,?,?,?,?,?)", batch)
        total += len(batch)
    db.commit()
    log(f"prices: {total} 行 ({len(files)} 文件), {time.time()-t0:.1f}s")


def _avail_date(period: str) -> str:
    """法定披露截止日（与 ashare_pit.data_source.availability_date 一致）。"""
    deadline = {"0331": (0,4,30), "0630": (0,8,31), "0930": (0,10,31), "1231": (1,4,30)}
    y = int(period[:4])
    off, m, d = deadline[period[4:]]
    return f"{y+off}-{m:02d}-{d:02d}"


def import_financials(db):
    files = sorted(glob.glob(os.path.join(CACHE, "cn_yjbb", "*.json")))
    total = 0
    for fp in files:
        with open(fp) as f:
            period_data = json.load(f).get("data", {})
        period = os.path.basename(fp)[:-5]
        ad = _avail_date(period)
        batch = []
        for code, d in period_data.items():
            batch.append((code, period, ad,
                          d.get("revenue"), d.get("revenue_yoy_percent"),
                          d.get("net_profit"), d.get("net_profit_yoy_percent"),
                          d.get("roe_percent"), d.get("gross_margin_percent"),
                          d.get("eps"), d.get("bps"), d.get("industry")))
        db.executemany(
            "INSERT OR IGNORE INTO financials(code,period,avail_date,"
            "revenue,revenue_yoy,net_profit,net_profit_yoy,"
            "roe,gross_margin,eps,bps,industry) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        total += len(batch)
    db.commit()
    log(f"financials: {total} 行 ({len(files)} 文件)")


def import_pe(db):
    files = sorted(glob.glob(os.path.join(CACHE, "cn_pe_ttm", "*.json")))
    total = 0
    batch = []
    for fp in files:
        with open(fp) as f:
            rows = json.load(f).get("data", [])
        code = os.path.basename(fp)[:-5]
        for r in rows:
            batch.append((code, r["date"], r.get("pe_ttm")))
        if len(batch) >= 10000:
            db.executemany("INSERT OR IGNORE INTO pe_ttm(code,date,pe_ttm) VALUES(?,?,?)", batch)
            total += len(batch)
            batch.clear()
    if batch:
        db.executemany("INSERT OR IGNORE INTO pe_ttm(code,date,pe_ttm) VALUES(?,?,?)", batch)
        total += len(batch)
    db.commit()
    log(f"pe_ttm: {total} 行 ({len(files)} 文件)")


def verify(db):
    counts = {}
    for table in ["universe", "prices", "benchmark", "financials", "pe_ttm"]:
        row = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = row[0]
    log(f"验证: {counts}")

    sample = db.execute("SELECT code, date, close FROM prices WHERE code='000001' ORDER BY date LIMIT 3").fetchall()
    log(f"000001 样本: {sample}")

    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    log(f"DB 文件大小: {size_mb:.0f} MB")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log(f"开始导入 → {DB_PATH}")
    db = init_db()
    import_universe(db)
    import_benchmark(db)
    import_financials(db)
    import_pe(db)
    import_prices(db)
    verify(db)
    db.close()
    log("完成")


if __name__ == "__main__":
    main()
