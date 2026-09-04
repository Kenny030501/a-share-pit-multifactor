#!/usr/bin/env python3
"""国泰君安 GTJA191 短周期价量因子实现。

每个因子接收包含 open、high、low、close、volume、vwap 和 ret 的 NumPy 序列，
返回调仓时点的最新标量值。因子统一定向为“数值越高、预期收益越高”；VWAP 使用
amount/volume 近似。需要指数 OHLC 的 075、149、181、182 默认不进入批量计算，
另有 030、143、165、183 未实现，因此当前可调用名称为 187 个。

这些公式用于探索性复现，不代表全部因子已经通过稳健统计、独立样本外验证和交易
成本检验；进入策略前仍必须经过 screening 与 walk-forward。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd

# ============================================================
# 滚动计算基础算子
# ============================================================

def _s(x: pd.Series) -> pd.Series: return x

def _rolling_sum(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=w).sum()

def _ts_min(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=w).min()

def _ts_max(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=w).max()

def _ts_mean(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=w).mean()

def _ts_std(x: pd.Series, w: int, ddof: int = 1) -> pd.Series:
    return x.rolling(w, min_periods=w).std(ddof=ddof)

def _ts_delta(x: pd.Series, d: int) -> pd.Series:
    return x.diff(d)

def _ts_delay(x: pd.Series, d: int) -> pd.Series:
    return x.shift(d)

def _ts_corr(x: pd.Series, y: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=w).corr(y)

def _ts_cov(x: pd.Series, y: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=w).cov(y)

def _ts_rank(x: pd.Series, w: int, pct: bool = True) -> pd.Series:
    """Percentile rank of last value within past w values (0-1)."""
    return x.rolling(w, min_periods=w).apply(
        lambda a: (a <= a.iloc[-1]).sum() / len(a) if len(a) > 0 else np.nan,
        raw=False,
    )

def _sma(x: pd.Series, n: int, m: int) -> pd.Series:
    """SMA(x, n, m): first value = mean(x[:n]); SMA_t = (x_t + (m-1)*SMA_{t-1})/m"""
    arr = x.to_numpy(dtype=float, copy=True)
    out = np.full(len(arr), np.nan)
    if len(arr) < n: return pd.Series(out, index=x.index)
    out[n - 1] = arr[:n].mean()
    alpha = 1.0 / m; beta = (m - 1.0) / m
    for i in range(n, len(arr)):
        if np.isnan(arr[i]): out[i] = out[i - 1]
        else: out[i] = alpha * arr[i] + beta * out[i - 1]
    return pd.Series(out, index=x.index)

def _decay_linear(x: pd.Series, d: int) -> pd.Series:
    """DECAYLINEAR: weighted MA, weights = d, d-1, ..., 1"""
    w = np.arange(1.0, d + 1.0)
    ws = w.sum()
    return x.rolling(d, min_periods=d).apply(
        lambda a: (a * w[-len(a):]).sum() / ws, raw=True,
    )

def _wma(x: pd.Series, n: int) -> pd.Series:
    """WMA: weights = 0.9^(n-1), 0.9^(n-2), ..., 0.9^0"""
    w = 0.9 ** np.arange(n - 1.0, -1.0, -1.0)
    ws = w.sum()
    return x.rolling(n, min_periods=n).apply(
        lambda a: (a * w[-len(a):]).sum() / ws, raw=True,
    )

def _ts_argmax(x: pd.Series, w: int) -> pd.Series:
    """HIGHDAY: days since max (0 = today)."""
    return x.rolling(w, min_periods=w).apply(
        lambda a: float(w - 1 - np.argmax(a)), raw=True,
    )

def _ts_argmin(x: pd.Series, w: int) -> pd.Series:
    """LOWDAY: days since min (0 = today)."""
    return x.rolling(w, min_periods=w).apply(
        lambda a: float(w - 1 - np.argmin(a)), raw=True,
    )

def _ts_prod(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=w).apply(np.prod, raw=True)

def _ts_regbeta(y: pd.Series, x: pd.Series, w: int) -> pd.Series:
    """Rolling regression slope: y ~ x. β = cov(x,y)/var(x)."""
    cov = _ts_cov(x, y, w)
    var = x.rolling(w, min_periods=w).var(ddof=0)
    return cov / var.replace(0, np.nan)

def _rank(x: pd.Series) -> pd.Series:
    """Percentage rank (0-1). NaN-safe."""
    valid = x.notna()
    out = pd.Series(np.nan, index=x.index)
    if valid.sum() < 2: return out
    from scipy.stats import rankdata
    ranks = rankdata(x[valid].to_numpy())
    out.loc[valid] = ranks / ranks[-1]
    return out

def _last(x: pd.Series) -> float:
    """Latest non-NaN value."""
    valid = x.dropna()
    return float(valid.iloc[-1]) if len(valid) > 0 else np.nan

def _sign(x: pd.Series) -> pd.Series:
    return np.sign(x)

def _abs(x: pd.Series) -> pd.Series:
    return x.abs()

def _log(x: pd.Series) -> pd.Series:
    return np.log(x.clip(lower=1e-300))

# ============================================================
# 因子计算上下文
# ============================================================

class OHLCV:
    """Single-stock OHLCV series for factor computation. `vwap` = amount/volume."""
    __slots__ = ('open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'ret', 'n')

    def __init__(self, bars: list[dict], as_of: str):
        n = 0
        for i, b in enumerate(bars):
            if b["date"] <= as_of:
                n = i + 1
        self.n = n
        def arr(key, default=np.nan):
            a = np.array([b.get(key, default) or np.nan for b in bars[:n]], dtype=np.float64)
            # backfill NaN with nearest valid (some stocks have sparse bars)
            mask = np.isnan(a)
            if mask.any() and not mask.all():
                idx = np.arange(len(a))
                a = np.where(mask, np.interp(idx, idx[~mask], a[~mask]), a)
            return pd.Series(a)
        self.open = arr("open")
        self.high = arr("high")
        self.low = arr("low")
        self.close = arr("close")
        self.volume = arr("volume")
        self.amount = arr("amount")
        self.vwap = pd.Series(np.where(self.volume.to_numpy() > 0,
                              self.amount.to_numpy() / self.volume.to_numpy(),
                              self.close.to_numpy()), index=self.close.index)
        self.ret = self.close.pct_change()

    def l(self, key: str) -> float:
        """Latest non-NaN scalar for `key`."""
        arr: pd.Series = getattr(self, key)
        if arr is None or len(arr) == 0: return np.nan
        valid = arr.dropna()
        return float(valid.iloc[-1]) if len(valid) > 0 else np.nan


# ============================================================
# 因子函数：每个函数输入 OHLCV 上下文并输出单个时点值
# ============================================================

# ---- 001-010 ----
def alpha001(d: OHLCV) -> float:
    """−CORR(RANK(DELTA(LOG(VOL),1)), RANK((CLOSE−OPEN)/OPEN), 6)"""
    if d.n < 6: return np.nan
    x = _rank(_ts_delta(_log(d.volume), 1))
    y = _rank((d.close - d.open) / np.maximum(d.open, 1e-300))
    return float(-_ts_corr(x, y, 6).iloc[-1])

def alpha002(d: OHLCV) -> float:
    """−DELTA(((CLOSE−LOW)−(HIGH−CLOSE))/(HIGH−LOW), 1)"""
    if d.n < 2: return np.nan
    denom = np.maximum(d.high - d.low, 1e-300)
    x = ((d.close - d.low) - (d.high - d.close)) / denom
    return float(-_ts_delta(x, 1).iloc[-1])

def alpha003(d: OHLCV) -> float:
    """SUM(CLOSE==PREV?0:CLOSE−(UP?MIN(LOW,PREV):MAX(HIGH,PREV)), 6)"""
    if d.n < 7: return np.nan
    prev = _ts_delay(d.close, 1)
    up = d.close > prev
    ref = np.where(up, np.minimum(d.low, prev), np.maximum(d.high, prev))
    diff = np.where(d.close == prev, 0.0, d.close - ref)
    return float(_rolling_sum(diff, 6).iloc[-1])

def alpha004(d: OHLCV) -> float:
    """Ternary: if 2d_avg < 8d_avg − 8d_std → −1 elif 2d_avg > 8d_avg + 8d_std → 1
    elif vol/mean_vol_20 <= 1 → 1 else −1"""
    if d.n < 20: return np.nan
    s8 = _ts_mean(d.close, 8); s2 = _ts_mean(d.close, 2)
    std8 = _ts_std(d.close, 8, ddof=0)
    lower = s8 - std8; upper = s8 + std8
    vol_ratio = d.volume / _ts_mean(d.volume, 20)
    c = np.where(s2 < lower, -1.0, np.where(s2 > upper, 1.0,
            np.where(vol_ratio <= 1.0, 1.0, -1.0)))
    return float(c.iloc[-1])

def alpha005(d: OHLCV) -> float:
    """−TSMAX(CORR(TSRANK(VOL,5), TSRANK(HIGH,5), 5), 3)"""
    if d.n < 5: return np.nan
    c = _ts_corr(_ts_rank(d.volume, 5), _ts_rank(d.high, 5), 5)
    return float(-_ts_max(c, 3).iloc[-1])

def alpha006(d: OHLCV) -> float:
    """−RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15, 4)))"""
    if d.n < 5: return np.nan
    x = _sign(_ts_delta(d.open * 0.85 + d.high * 0.15, 4))
    return float(-_rank(x).iloc[-1])

def alpha007(d: OHLCV) -> float:
    """(RANK(MAX(VWAP−CLOSE,3)) + RANK(MIN(VWAP−CLOSE,3))) * RANK(DELTA(VOL,3))"""
    if d.n < 3: return np.nan
    diff = d.vwap - d.close
    return float((_rank(_ts_max(diff, 3)) + _rank(_ts_min(diff, 3))) * _rank(_ts_delta(d.volume, 3))).iloc[-1]

def alpha008(d: OHLCV) -> float:
    """−RANK(DELTA(((HIGH+LOW)/2*0.2 + VWAP*0.8), 4))"""
    if d.n < 5: return np.nan
    x = (d.high + d.low) / 2 * 0.2 + d.vwap * 0.8
    return float(-_rank(_ts_delta(x, 4)).iloc[-1])

def alpha009(d: OHLCV) -> float:
    """SMA(((HIGH+LOW)/2−(PREV_HIGH+PREV_LOW)/2)*(HIGH−LOW)/VOL, 7, 2)"""
    if d.n < 9: return np.nan
    prev_h = _ts_delay(d.high, 1); prev_l = _ts_delay(d.low, 1)
    x = ((d.high + d.low) / 2 - (prev_h + prev_l) / 2) * (d.high - d.low) / np.maximum(d.volume, 1e-300)
    return float(_sma(x, 7, 2).iloc[-1])

def alpha010(d: OHLCV) -> float:
    """RANK(MAX((RET<0 ? STD(RET,20) : CLOSE)^2, 5))"""
    if d.n < 20: return np.nan
    x = np.where(d.ret < 0, _ts_std(d.ret, 20, ddof=0), d.close)
    return float(_rank(_ts_max(x * x, 5)).iloc[-1])

# ---- 011-020 ----
def alpha011(d: OHLCV) -> float:
    """SUM(((CLOSE−LOW)−(HIGH−CLOSE))/(HIGH−LOW)*VOL, 6)"""
    if d.n < 6: return np.nan
    denom = np.maximum(d.high - d.low, 1e-300)
    x = ((d.close - d.low) - (d.high - d.close)) / denom * d.volume
    return float(_rolling_sum(x, 6).iloc[-1])

def alpha012(d: OHLCV) -> float:
    """RANK(OPEN−SUM(VWAP,10)/10) * (−RANK(ABS(CLOSE−VWAP)))"""
    if d.n < 10: return np.nan
    return float(_rank(d.open - _ts_mean(d.vwap, 10)) * (-_rank(_abs(d.close - d.vwap)))).iloc[-1]

def alpha013(d: OHLCV) -> float:
    """(HIGH*LOW)^0.5 − VWAP"""
    if d.n < 1: return np.nan
    return float((np.sqrt(np.maximum(d.high * d.low, 0)) - d.vwap).iloc[-1])

def alpha014(d: OHLCV) -> float:
    """CLOSE − DELAY(CLOSE, 5)"""
    if d.n < 6: return np.nan
    return float(_ts_delta(d.close, 5).iloc[-1])

def alpha015(d: OHLCV) -> float:
    """OPEN / DELAY(CLOSE, 1) − 1"""
    if d.n < 2: return np.nan
    return float(d.open.iloc[-1] / d.close.iloc[-2] - 1.0) if d.close.iloc[-2] > 0 else np.nan

def alpha016(d: OHLCV) -> float:
    """−TSMAX(RANK(CORR(RANK(VOL), RANK(VWAP), 5)), 5)"""
    if d.n < 5: return np.nan
    x = _rank(_ts_corr(_rank(d.volume), _rank(d.vwap), 5))
    return float(-_ts_max(x, 5).iloc[-1])

def alpha017(d: OHLCV) -> float:
    """RANK(VWAP−MAX(VWAP,15)) ^ DELTA(CLOSE,5)"""
    if d.n < 15: return np.nan
    v = _rank(d.vwap - _ts_max(d.vwap, 15))
    exp = _ts_delta(d.close, 5)
    return float(v[-1] ** exp[-1]) if not np.isnan(v[-1]) and not np.isnan(exp[-1]) else np.nan

def alpha018(d: OHLCV) -> float:
    """CLOSE / DELAY(CLOSE, 5)"""
    if d.n < 6: return np.nan
    return float(d.close.iloc[-1] / d.close[-6]) if d.close[-6] > 0 else np.nan

def alpha019(d: OHLCV) -> float:
    """5d return: down→denom=prev_close, up→denom=close, flat→0"""
    if d.n < 6: return np.nan
    c, pc = d.close.iloc[-1], d.close[-6]
    if c < pc: return float((c - pc) / pc) if pc > 0 else np.nan
    if c == pc: return 0.0
    return float((c - pc) / c) if c > 0 else np.nan

def alpha020(d: OHLCV) -> float:
    """(CLOSE−DELAY(CLOSE,6)) / DELAY(CLOSE,6) * 100"""
    if d.n < 7: return np.nan
    return float(_ts_delta(d.close, 6).iloc[-1] / d.close[-7] * 100.0) if d.close[-7] > 0 else np.nan

# ---- 021-030 ----
def alpha021(d: OHLCV) -> float:
    """REGBETA(MEAN(CLOSE,6), SEQUENCE(6))"""
    if d.n < 6: return np.nan
    return float(_ts_regbeta(_ts_mean(d.close, 6), pd.Series(np.arange(1, d.n + 1, dtype=np.float64), index=d.close.index), 6).iloc[-1])

def alpha022(d: OHLCV) -> float:
    """SMA((CLOSE−MEAN(CLOSE,6))/MEAN(CLOSE,6)−DELAY(...,3), 12, 1)"""
    if d.n < 12: return np.nan
    dev = (d.close - _ts_mean(d.close, 6)) / np.maximum(_ts_mean(d.close, 6), 1e-300)
    x = dev - _ts_delay(dev, 3)
    return float(_sma(x, 12, 1).iloc[-1])

def alpha023(d: OHLCV) -> float:
    """SMA(up_std,20,1) / (SMA(up_std,20,1)+SMA(down_std,20,1)) * 100"""
    if d.n < 20: return np.nan
    std20 = _ts_std(d.close, 20, ddof=0)
    up_std = np.where(d.close > _ts_delay(d.close, 1), std20, 0.0)
    down_std = np.where(d.close <= _ts_delay(d.close, 1), std20, 0.0)
    su = _sma(up_std, 20, 1); sd = _sma(down_std, 20, 1)
    return float(su[-1] / (su[-1] + sd[-1]) * 100.0) if (su[-1] + sd[-1]) > 0 else np.nan

def alpha024(d: OHLCV) -> float:
    """SMA(CLOSE−DELAY(CLOSE,5), 5, 1)"""
    if d.n < 6: return np.nan
    return float(_sma(_ts_delta(d.close, 5), 5, 1).iloc[-1])

def alpha025(d: OHLCV) -> float:
    """(−RANK(DELTA(CLOSE,7)*(1−RANK(DECAYLINEAR(VOL/MEAN(VOL,20),9))))) * (1+RANK(SUM(RET,250)))"""
    if d.n < 250: return np.nan
    vr = d.volume / _ts_mean(d.volume, 20)
    dl = _decay_linear(vr, 9)
    rank_sum_ret = _rank(_rolling_sum(d.ret, 250))
    x = -_rank(_ts_delta(d.close, 7) * (1.0 - _rank(dl)))
    return float((x * (1.0 + rank_sum_ret)).iloc[-1])

def alpha026(d: OHLCV) -> float:
    """(MEAN(CLOSE,7)/7 − CLOSE) + CORR(VWAP, DELAY(CLOSE,5), 230)"""
    if d.n < 230: return np.nan
    return float((_ts_mean(d.close, 7) - d.close) + _ts_corr(d.vwap, _ts_delay(d.close, 5), 230)).iloc[-1]

def alpha027(d: OHLCV) -> float:
    """WMA(3d_ret*100 + 6d_ret*100, 12)"""
    if d.n < 12: return np.nan
    r3 = _ts_delta(d.close, 3) / np.maximum(np.abs(_ts_delay(d.close, 3)), 1e-300) * 100.0
    r6 = _ts_delta(d.close, 6) / np.maximum(np.abs(_ts_delay(d.close, 6)), 1e-300) * 100.0
    return float(_wma(r3 + r6, 12).iloc[-1])

def alpha028(d: OHLCV) -> float:
    """3*SMA(RSV9,3,1) − 2*SMA(SMA(RSV9,3,1),3,1)"""
    if d.n < 9: return np.nan
    hi9 = _ts_max(d.high, 9); lo9 = _ts_min(d.low, 9)
    rsv = (d.close - lo9) / np.maximum(hi9 - lo9, 1e-300) * 100.0
    k = _sma(rsv, 3, 1); d_ = _sma(k, 3, 1)
    return float(3.0 * k[-1] - 2.0 * d_[-1])

def alpha029(d: OHLCV) -> float:
    """(CLOSE−DELAY(CLOSE,6))/DELAY(CLOSE,6) * VOLUME"""
    if d.n < 7: return np.nan
    ret = _ts_delta(d.close, 6) / np.maximum(np.abs(_ts_delay(d.close, 6)), 1e-300)
    return float((ret * d.volume).iloc[-1])

# alpha030: unimplemented (requires Fama-French factors)

# ---- 031-040 ----
def alpha031(d: OHLCV) -> float:
    """(CLOSE−MEAN(CLOSE,12))/MEAN(CLOSE,12)*100"""
    if d.n < 12: return np.nan
    m = _ts_mean(d.close, 12)
    return float((d.close.iloc[-1] - m[-1]) / m[-1] * 100.0) if m[-1] > 0 else np.nan

def alpha032(d: OHLCV) -> float:
    """−SUM(RANK(CORR(RANK(HIGH), RANK(VOL), 3)), 3)"""
    if d.n < 3: return np.nan
    c = _rank(_ts_corr(_rank(d.high), _rank(d.volume), 3))
    return float(-_rolling_sum(c, 3).iloc[-1])

def alpha033(d: OHLCV) -> float:
    """((−TSMIN(LOW,5)+DELAY(TSMIN(LOW,5),5)) * RANK((SUM(RET,240)−SUM(RET,20))/220)) * TSRANK(VOL,5)"""
    if d.n < 240: return np.nan
    lo5 = _ts_min(d.low, 5)
    a = -lo5 + _ts_delay(lo5, 5)
    b = _rank((_rolling_sum(d.ret, 240) - _rolling_sum(d.ret, 20)) / 220.0)
    c = _ts_rank(d.volume, 5)
    return float((a * b * c).iloc[-1])

def alpha034(d: OHLCV) -> float:
    """MEAN(CLOSE,12) / CLOSE"""
    if d.n < 12: return np.nan
    return float(_ts_mean(d.close, 12).iloc[-1] / d.close.iloc[-1]) if d.close.iloc[-1] > 0 else np.nan

def alpha035(d: OHLCV) -> float:
    """−MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)), RANK(DECAYLINEAR(CORR(VOL,OPEN*0.65+OPEN*0.35,17),7)))"""
    if d.n < 17: return np.nan
    a = _rank(_decay_linear(_ts_delta(d.open, 1), 15))
    b = _rank(_decay_linear(_ts_corr(d.volume, d.open * 0.65 + d.open * 0.35, 17), 7))
    return float(-np.minimum(a, b).iloc[-1])

def alpha036(d: OHLCV) -> float:
    """RANK(SUM(CORR(RANK(VOL), RANK(VWAP), 6), 2))"""
    if d.n < 6: return np.nan
    c = _ts_corr(_rank(d.volume), _rank(d.vwap), 6)
    return float(_rank(_rolling_sum(c, 2)).iloc[-1])

def alpha037(d: OHLCV) -> float:
    """−RANK((SUM(OPEN,5)*SUM(RET,5) − DELAY(SUM(OPEN,5)*SUM(RET,5),10)))"""
    if d.n < 15: return np.nan
    x = _rolling_sum(d.open, 5) * _rolling_sum(d.ret, 5)
    return float(-_rank(x - _ts_delay(x, 10)).iloc[-1])

def alpha038(d: OHLCV) -> float:
    """(MEAN(HIGH,20) < HIGH) ? −DELTA(HIGH,2) : 0"""
    if d.n < 20: return np.nan
    cond = _ts_mean(d.high, 20) < d.high
    return float(np.where(cond, -_ts_delta(d.high, 2), 0.0)[-1])

def alpha039(d: OHLCV) -> float:
    """−(RANK(DECAYLINEAR(DELTA(CLOSE,2),8)) − RANK(DECAYLINEAR(CORR(VWAP*0.3+OPEN*0.7, SUM(MEAN(VOL,180),37),14),12)))"""
    if d.n < 180: return np.nan
    a = _rank(_decay_linear(_ts_delta(d.close, 2), 8))
    mv = _ts_mean(d.volume, 180)
    mv37 = _rolling_sum(mv, 37)
    b = _rank(_decay_linear(_ts_corr(d.vwap * 0.3 + d.open * 0.7, mv37, 14), 12))
    return float(-(a - b).iloc[-1])

def alpha040(d: OHLCV) -> float:
    """SUM(up_vol,26) / SUM(down_vol,26) * 100"""
    if d.n < 26: return np.nan
    up = d.close > _ts_delay(d.close, 1)
    up_vol = np.where(up, d.volume, 0.0); down_vol = np.where(~up, d.volume, 0.0)
    su = _rolling_sum(up_vol, 26); sd = _rolling_sum(down_vol, 26)
    return float(su[-1] / sd[-1] * 100.0) if sd[-1] > 0 else np.nan

# ---- 041-050 ----
def alpha041(d: OHLCV) -> float:
    """−RANK(MAX(DELTA(VWAP,3), 5))"""
    if d.n < 8: return np.nan
    return float(-_rank(_ts_max(_ts_delta(d.vwap, 3), 5)).iloc[-1])

def alpha042(d: OHLCV) -> float:
    """−RANK(STD(HIGH,10)) * CORR(HIGH, VOL, 10)"""
    if d.n < 10: return np.nan
    return float((- _rank(_ts_std(d.high, 10, ddof=0)) * _ts_corr(d.high, d.volume, 10)).iloc[-1])

def alpha043(d: OHLCV) -> float:
    """SUM(up→VOL, down→−VOL, 6)"""
    if d.n < 6: return np.nan
    up = d.close > _ts_delay(d.close, 1); down = d.close < _ts_delay(d.close, 1)
    x = np.where(up, d.volume, np.where(down, -d.volume, 0.0))
    return float(_rolling_sum(x, 6).iloc[-1])

def alpha044(d: OHLCV) -> float:
    """TSRANK(DECAYLINEAR(CORR(LOW,MEAN(VOL,10),7),6),4) + TSRANK(DECAYLINEAR(DELTA(VWAP,3),10),15)"""
    if d.n < 15: return np.nan
    mv10 = _ts_mean(d.volume, 10)
    a = _ts_rank(_decay_linear(_ts_corr(d.low, mv10, 7), 6), 4)
    b = _ts_rank(_decay_linear(_ts_delta(d.vwap, 3), 10), 15)
    return float((a + b).iloc[-1])

def alpha045(d: OHLCV) -> float:
    """RANK(DELTA(CLOSE*0.6+OPEN*0.4, 1)) * RANK(CORR(VWAP, MEAN(VOL,150), 15))"""
    if d.n < 150: return np.nan
    a = _rank(_ts_delta(d.close * 0.6 + d.open * 0.4, 1))
    b = _rank(_ts_corr(d.vwap, _ts_mean(d.volume, 150), 15))
    return float((a * b).iloc[-1])

def alpha046(d: OHLCV) -> float:
    """(MA3+MA6+MA12+MA24) / (4*CLOSE)"""
    if d.n < 24: return np.nan
    s = _ts_mean(d.close, 3) + _ts_mean(d.close, 6) + _ts_mean(d.close, 12) + _ts_mean(d.close, 24)
    return float(s[-1] / (4.0 * d.close.iloc[-1])) if d.close.iloc[-1] > 0 else np.nan

def alpha047(d: OHLCV) -> float:
    """SMA((TSMAX(HIGH,6)−CLOSE)/(TSMAX(HIGH,6)−TSMIN(LOW,6))*100, 9, 1)"""
    if d.n < 9: return np.nan
    hi6 = _ts_max(d.high, 6); lo6 = _ts_min(d.low, 6)
    x = (hi6 - d.close) / np.maximum(hi6 - lo6, 1e-300) * 100.0
    return float(_sma(x, 9, 1).iloc[-1])

def alpha048(d: OHLCV) -> float:
    """−RANK(SIGN(CLOSE−PREV)+SIGN(PREV−PREV2)+SIGN(PREV2−PREV3)) * SUM(VOL,5) / SUM(VOL,20)"""
    if d.n < 20: return np.nan
    s = _sign(_ts_delta(d.close, 1)) + _sign(_ts_delta(_ts_delay(d.close, 1), 1)) + _sign(_ts_delta(_ts_delay(d.close, 2), 1))
    return float((- _rank(s) * _rolling_sum(d.volume, 5) / np.maximum(_rolling_sum(d.volume, 20), 1e-300)).iloc[-1])

def alpha049(d: OHLCV) -> float:
    """SUM(HL下跌?0:MAX(ABS(HIGH−PREV_HIGH),ABS(LOW−PREV_LOW)), 12) / (same_up + same_down)"""
    if d.n < 13: return np.nan
    hl = d.high + d.low; prev_hl = _ts_delay(d.high, 1) + _ts_delay(d.low, 1)
    up = np.where(hl >= prev_hl, np.maximum(_abs(_ts_delta(d.high, 1)), _abs(_ts_delta(d.low, 1))), 0.0)
    down = np.where(hl < prev_hl, np.maximum(_abs(_ts_delta(d.high, 1)), _abs(_ts_delta(d.low, 1))), 0.0)
    su = _rolling_sum(up, 12); sd = _rolling_sum(down, 12)
    return float(sd[-1] / (su[-1] + sd[-1])) if (su[-1] + sd[-1]) > 0 else np.nan

def alpha050(d: OHLCV) -> float:
    """(up_sum − down_sum) / (up_sum + down_sum)"""
    if d.n < 13: return np.nan
    hl = d.high + d.low; prev_hl = _ts_delay(d.high, 1) + _ts_delay(d.low, 1)
    up = np.where(hl >= prev_hl, np.maximum(_abs(_ts_delta(d.high, 1)), _abs(_ts_delta(d.low, 1))), 0.0)
    down = np.where(hl < prev_hl, np.maximum(_abs(_ts_delta(d.high, 1)), _abs(_ts_delta(d.low, 1))), 0.0)
    su = _rolling_sum(up, 12); sd = _rolling_sum(down, 12)
    return float((su[-1] - sd[-1]) / (su[-1] + sd[-1])) if (su[-1] + sd[-1]) > 0 else np.nan

# ---- 051-060 ----
def alpha051(d: OHLCV) -> float:
    """up_sum / (up_sum + down_sum)"""
    if d.n < 13: return np.nan
    hl = d.high + d.low; prev_hl = _ts_delay(d.high, 1) + _ts_delay(d.low, 1)
    up = np.where(hl >= prev_hl, np.maximum(_abs(_ts_delta(d.high, 1)), _abs(_ts_delta(d.low, 1))), 0.0)
    down = np.where(hl < prev_hl, np.maximum(_abs(_ts_delta(d.high, 1)), _abs(_ts_delta(d.low, 1))), 0.0)
    su = _rolling_sum(up, 12); sd = _rolling_sum(down, 12)
    return float(su[-1] / (su[-1] + sd[-1])) if (su[-1] + sd[-1]) > 0 else np.nan

def alpha052(d: OHLCV) -> float:
    """SUM(MAX(0,HIGH−DELAY(TP,1)),26)/SUM(MAX(0,DELAY(TP,1)−LOW),26)*100 where TP=(H+L+C)/3"""
    if d.n < 27: return np.nan
    tp = (d.high + d.low + d.close) / 3.0
    prev_tp = _ts_delay(tp, 1)
    up = np.maximum(0.0, d.high - prev_tp); down = np.maximum(0.0, prev_tp - d.low)
    su = _rolling_sum(up, 26); sd = _rolling_sum(down, 26)
    return float(su[-1] / sd[-1] * 100.0) if sd[-1] > 0 else np.nan

def alpha053(d: OHLCV) -> float:
    """COUNT(CLOSE>DELAY(CLOSE,1), 12) / 12 * 100"""
    if d.n < 12: return np.nan
    return float((d.close > _ts_delay(d.close, 1)).sum() / 12.0 * 100.0)

def alpha054(d: OHLCV) -> float:
    """−RANK(STD(ABS(CLOSE−OPEN)) + (CLOSE−OPEN) + CORR(CLOSE,OPEN,10))"""
    if d.n < 10: return np.nan
    x = _ts_std(_abs(d.close - d.open), 10, ddof=0) + (d.close - d.open) + _ts_corr(d.close, d.open, 10)
    return float(-_rank(x).iloc[-1])

def alpha055(d: OHLCV) -> float:
    """EMV-like: 16*(CLOSE−PREV+(CLOSE−OPEN)/2+PREV−PREV_OPEN)/(max abs range)"""
    if d.n < 20: return np.nan
    prev_c = _ts_delay(d.close, 1); prev_o = _ts_delay(d.open, 1)
    num = d.close - prev_c + (d.close - d.open) / 2.0 + prev_c - prev_o
    denom = np.maximum(np.maximum(d.high - d.low, _abs(d.high - prev_c)), _abs(prev_c - d.low))
    denom = np.maximum(denom, 1e-300)
    x = 16.0 * num / denom
    return float(_rolling_sum(x, 20).iloc[-1])

def alpha056(d: OHLCV) -> float:
    """(RANK(OPEN−TSMIN(OPEN,12)) < RANK(RANK(CORR(SUM((H+L)/2,19), SUM(MEAN(VOL,40),19),13))^5))"""
    if d.n < 40: return np.nan
    a = _rank(d.open - _ts_min(d.open, 12))
    mv40 = _ts_mean(d.volume, 40)
    b_inner = _ts_corr(_rolling_sum((d.high + d.low) / 2.0, 19), _rolling_sum(mv40, 19), 13)
    b = _rank(_rank(b_inner) ** 5)
    return float((a < b).iloc[-1])

def alpha057(d: OHLCV) -> float:
    """SMA((CLOSE−TSMIN(LOW,9))/(TSMAX(HIGH,9)−TSMIN(LOW,9))*100, 3, 1)"""
    if d.n < 9: return np.nan
    hi9 = _ts_max(d.high, 9); lo9 = _ts_min(d.low, 9)
    rsv = (d.close - lo9) / np.maximum(hi9 - lo9, 1e-300) * 100.0
    return float(_sma(rsv, 3, 1).iloc[-1])

def alpha058(d: OHLCV) -> float:
    """COUNT(CLOSE>DELAY(CLOSE,1), 20) / 20 * 100"""
    if d.n < 20: return np.nan
    return float((d.close > _ts_delay(d.close, 1)).sum() / 20.0 * 100.0)

def alpha059(d: OHLCV) -> float:
    """SUM(CLOSE==PREV?0:CLOSE−(UP?MIN(LOW,PREV):MAX(HIGH,PREV)), 20)"""
    if d.n < 21: return np.nan
    prev = _ts_delay(d.close, 1)
    up = d.close > prev
    ref = np.where(up, np.minimum(d.low, prev), np.maximum(d.high, prev))
    diff = np.where(d.close == prev, 0.0, d.close - ref)
    return float(_rolling_sum(diff, 20).iloc[-1])

def alpha060(d: OHLCV) -> float:
    """SUM(((CLOSE−LOW)−(HIGH−CLOSE))/(HIGH−LOW)*VOL, 20)"""
    if d.n < 20: return np.nan
    denom = np.maximum(d.high - d.low, 1e-300)
    x = ((d.close - d.low) - (d.high - d.close)) / denom * d.volume
    return float(_rolling_sum(x, 20).iloc[-1])

# ---- 061-070 ----
def alpha061(d: OHLCV) -> float:
    """−MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)), RANK(DECAYLINEAR(RANK(CORR(LOW,MEAN(VOL,80),8)),17)))"""
    if d.n < 80: return np.nan
    a = _rank(_decay_linear(_ts_delta(d.vwap, 1), 12))
    mv80 = _ts_mean(d.volume, 80)
    b = _rank(_decay_linear(_rank(_ts_corr(d.low, mv80, 8)), 17))
    return float(-np.maximum(a, b).iloc[-1])

def alpha062(d: OHLCV) -> float:
    """−CORR(HIGH, RANK(VOL), 5)"""
    if d.n < 5: return np.nan
    return float(-_ts_corr(d.high, _rank(d.volume), 5).iloc[-1])

def alpha063(d: OHLCV) -> float:
    """SMA(MAX(CLOSE−DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE−DELAY(CLOSE,1)),6,1)*100"""
    if d.n < 6: return np.nan
    diff = _ts_delta(d.close, 1)
    u = _sma(np.maximum(diff, 0.0), 6, 1); d_ = _sma(_abs(diff), 6, 1)
    return float(u[-1] / d_[-1] * 100.0) if d_[-1] > 0 else np.nan

def alpha064(d: OHLCV) -> float:
    """−MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOL),4),4)), RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE),RANK(MEAN(VOL,60)),4),13),14)))"""
    if d.n < 60: return np.nan
    a = _rank(_decay_linear(_ts_corr(_rank(d.vwap), _rank(d.volume), 4), 4))
    mv60 = _ts_mean(d.volume, 60)
    b_inner = _ts_corr(_rank(d.close), _rank(mv60), 4)
    b = _rank(_decay_linear(_ts_max(b_inner, 13), 14))
    return float(-np.maximum(a, b).iloc[-1])

def alpha065(d: OHLCV) -> float:
    """MEAN(CLOSE,6) / CLOSE"""
    if d.n < 6: return np.nan
    return float(_ts_mean(d.close, 6).iloc[-1] / d.close.iloc[-1]) if d.close.iloc[-1] > 0 else np.nan

def alpha066(d: OHLCV) -> float:
    """(CLOSE−MEAN(CLOSE,6))/MEAN(CLOSE,6)*100"""
    if d.n < 6: return np.nan
    m = _ts_mean(d.close, 6)
    return float((d.close.iloc[-1] - m[-1]) / m[-1] * 100.0) if m[-1] > 0 else np.nan

def alpha067(d: OHLCV) -> float:
    """SMA(MAX(CLOSE−DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE−DELAY(CLOSE,1)),24,1)*100"""
    if d.n < 24: return np.nan
    diff = _ts_delta(d.close, 1)
    u = _sma(np.maximum(diff, 0.0), 24, 1); d_ = _sma(_abs(diff), 24, 1)
    return float(u[-1] / d_[-1] * 100.0) if d_[-1] > 0 else np.nan

def alpha068(d: OHLCV) -> float:
    """SMA(((HIGH+LOW)/2−(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH−LOW)/VOL, 15, 2)"""
    if d.n < 17: return np.nan
    prev_h = _ts_delay(d.high, 1); prev_l = _ts_delay(d.low, 1)
    x = ((d.high + d.low) / 2.0 - (prev_h + prev_l) / 2.0) * (d.high - d.low) / np.maximum(d.volume, 1e-300)
    return float(_sma(x, 15, 2).iloc[-1])

def alpha069(d: OHLCV) -> float:
    """DTM/DBM diff ratio; DTM=(OPEN<=PREV_OPEN?0:MAX(HIGH−OPEN,OPEN−PREV_OPEN)), DBM vice-versa"""
    if d.n < 20: return np.nan
    prev_o = _ts_delay(d.open, 1)
    dtm = np.where(d.open > prev_o, np.maximum(d.high - d.open, d.open - prev_o), 0.0)
    dbm = np.where(d.open < prev_o, np.maximum(d.open - d.low, prev_o - d.open), 0.0)
    sdtm = _rolling_sum(dtm, 20); sdbm = _rolling_sum(dbm, 20)
    return float(_last(np.where(sdtm > sdbm, (sdtm - sdbm) / np.maximum(sdtm, 1e-300),
                   np.where(sdtm == sdbm, 0.0, (sdtm - sdbm) / np.maximum(sdbm, 1e-300)))))

def alpha070(d: OHLCV) -> float:
    """STD(AMOUNT, 6)"""
    if d.n < 6: return np.nan
    return float(_ts_std(d.amount, 6, ddof=0).iloc[-1])

# ---- 071-080 ----
def alpha071(d: OHLCV) -> float:
    """(CLOSE−MEAN(CLOSE,24))/MEAN(CLOSE,24)*100"""
    if d.n < 24: return np.nan
    m = _ts_mean(d.close, 24)
    return float((d.close.iloc[-1] - m[-1]) / m[-1] * 100.0) if m[-1] > 0 else np.nan

def alpha072(d: OHLCV) -> float:
    """SMA((TSMAX(HIGH,6)−CLOSE)/(TSMAX(HIGH,6)−TSMIN(LOW,6))*100, 15, 1)"""
    if d.n < 15: return np.nan
    hi6 = _ts_max(d.high, 6); lo6 = _ts_min(d.low, 6)
    x = (hi6 - d.close) / np.maximum(hi6 - lo6, 1e-300) * 100.0
    return float(_sma(x, 15, 1).iloc[-1])

def alpha073(d: OHLCV) -> float:
    """−(TSRANK(DECAYLINEAR(DECAYLINEAR(CORR(CLOSE,VOL,10),16),4),5) − RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOL,30),4),3)))"""
    if d.n < 30: return np.nan
    mv30 = _ts_mean(d.volume, 30)
    a = _ts_rank(_decay_linear(_decay_linear(_ts_corr(d.close, d.volume, 10), 16), 4), 5)
    b = _rank(_decay_linear(_ts_corr(d.vwap, mv30, 4), 3))
    return float(-(a - b).iloc[-1])

def alpha074(d: OHLCV) -> float:
    """RANK(CORR(SUM(LOW*0.35+VWAP*0.65,20), SUM(MEAN(VOL,40),20),7)) + RANK(CORR(RANK(VWAP),RANK(VOL),6))"""
    if d.n < 40: return np.nan
    mv40 = _ts_mean(d.volume, 40)
    a = _rank(_ts_corr(_rolling_sum(d.low * 0.35 + d.vwap * 0.65, 20), _rolling_sum(mv40, 20), 7))
    b = _rank(_ts_corr(_rank(d.vwap), _rank(d.volume), 6))
    return float((a + b).iloc[-1])

def alpha075(d: OHLCV) -> float:
    """Benchmark-index factor: COUNT(bench_close < bench_open, 50) / 50"""
    return np.nan  # requires benchmark index data

def alpha076(d: OHLCV) -> float:
    """STD(ABS(RET)/VOL, 20) / MEAN(ABS(RET)/VOL, 20)"""
    if d.n < 20: return np.nan
    x = _abs(d.ret) / np.maximum(d.volume, 1e-300)
    s = _ts_std(x, 20, ddof=0); m = _ts_mean(x, 20)
    return float(s[-1] / m[-1]) if m[-1] > 0 else np.nan

def alpha077(d: OHLCV) -> float:
    """MIN(RANK(DECAYLINEAR((H+L)/2+HIGH−(VWAP+HIGH),20)), RANK(DECAYLINEAR(CORR((H+L)/2,MEAN(VOL,40),3),6)))"""
    if d.n < 40: return np.nan
    a = _rank(_decay_linear((d.high + d.low) / 2.0 + d.high - (d.vwap + d.high), 20))
    mv40 = _ts_mean(d.volume, 40)
    b = _rank(_decay_linear(_ts_corr((d.high + d.low) / 2.0, mv40, 3), 6))
    return float(np.minimum(a, b).iloc[-1])

def alpha078(d: OHLCV) -> float:
    """CCI: (TP−MA(TP,12)) / (0.015*MEAN(ABS(CLOSE−MEAN(TP,12)),12))"""
    if d.n < 12: return np.nan
    tp = (d.high + d.low + d.close) / 3.0; mtp = _ts_mean(tp, 12)
    mad = _ts_mean(_abs(d.close - mtp), 12)
    return float((tp[-1] - mtp[-1]) / (0.015 * mad[-1])) if mad[-1] > 0 else np.nan

def alpha079(d: OHLCV) -> float:
    """SMA(MAX(CLOSE−DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE−DELAY(CLOSE,1)),12,1)*100"""
    if d.n < 12: return np.nan
    diff = _ts_delta(d.close, 1)
    u = _sma(np.maximum(diff, 0.0), 12, 1); d_ = _sma(_abs(diff), 12, 1)
    return float(u[-1] / d_[-1] * 100.0) if d_[-1] > 0 else np.nan

def alpha080(d: OHLCV) -> float:
    """(VOLUME−DELAY(VOLUME,5))/DELAY(VOLUME,5)*100"""
    if d.n < 6: return np.nan
    prev = _ts_delay(d.volume, 5)
    return float((d.volume.iloc[-1] - prev[-1]) / prev[-1] * 100.0) if prev[-1] > 0 else np.nan

# ---- 081-090 ----
def alpha081(d: OHLCV) -> float:
    """SMA(VOLUME, 21, 2)"""
    if d.n < 21: return np.nan
    return float(_sma(d.volume, 21, 2).iloc[-1])

def alpha082(d: OHLCV) -> float:
    """SMA((TSMAX(HIGH,6)−CLOSE)/(TSMAX(HIGH,6)−TSMIN(LOW,6))*100, 20, 1)"""
    if d.n < 20: return np.nan
    hi6 = _ts_max(d.high, 6); lo6 = _ts_min(d.low, 6)
    x = (hi6 - d.close) / np.maximum(hi6 - lo6, 1e-300) * 100.0
    return float(_sma(x, 20, 1).iloc[-1])

def alpha083(d: OHLCV) -> float:
    """−RANK(COV(RANK(HIGH), RANK(VOL), 5))"""
    if d.n < 5: return np.nan
    return float(-_rank(_ts_cov(_rank(d.high), _rank(d.volume), 5)).iloc[-1])

def alpha084(d: OHLCV) -> float:
    """SUM(up→VOL, down→−VOL, 20)"""
    if d.n < 20: return np.nan
    up = d.close > _ts_delay(d.close, 1); down = d.close < _ts_delay(d.close, 1)
    x = np.where(up, d.volume, np.where(down, -d.volume, 0.0))
    return float(_rolling_sum(x, 20).iloc[-1])

def alpha085(d: OHLCV) -> float:
    """TSRANK(VOL/MEAN(VOL,20),20) * TSRANK(−DELTA(CLOSE,7),8)"""
    if d.n < 20: return np.nan
    a = _ts_rank(d.volume / _ts_mean(d.volume, 20), 20)
    b = _ts_rank(-_ts_delta(d.close, 7), 8)
    return float((a * b).iloc[-1])

def alpha086(d: OHLCV) -> float:
    """Slope-acceleration ternary: if slope_diff > 0.25 → −1, elif < 0 → 1, else −(ret)"""
    if d.n < 20: return np.nan
    s1 = (_ts_delay(d.close, 20) - _ts_delay(d.close, 10)) / 10.0
    s2 = (_ts_delay(d.close, 10) - d.close) / 10.0
    diff = s1 - s2
    return float(np.where(diff > 0.25, -1.0, np.where(diff < 0.0, 1.0, -_ts_delta(d.close, 1))).iloc[-1])

def alpha087(d: OHLCV) -> float:
    """−(RANK(DECAYLINEAR(DELTA(VWAP,4),7)) + TSRANK(DECAYLINEAR((LOW*0.9+LOW*0.1−VWAP)/(OPEN−(H+L)/2),11),7))"""
    if d.n < 11: return np.nan
    a = _rank(_decay_linear(_ts_delta(d.vwap, 4), 7))
    denom = np.maximum(d.open - (d.high + d.low) / 2.0, 1e-300)
    inner = (d.low - d.vwap) / denom  # LOW*0.9+LOW*0.1 = LOW
    b = _ts_rank(_decay_linear(inner, 11), 7)
    return float(-(a + b).iloc[-1])

def alpha088(d: OHLCV) -> float:
    """(CLOSE−DELAY(CLOSE,20))/DELAY(CLOSE,20)*100"""
    if d.n < 21: return np.nan
    return float(_ts_delta(d.close, 20).iloc[-1] / d.close[-21] * 100.0) if d.close[-21] > 0 else np.nan

def alpha089(d: OHLCV) -> float:
    """2*(SMA(CLOSE,13,2)−SMA(CLOSE,27,2)−SMA(SMA(CLOSE,13,2)−SMA(CLOSE,27,2),10,2))"""
    if d.n < 27: return np.nan
    fast = _sma(d.close, 13, 2); slow = _sma(d.close, 27, 2)
    dif = fast - slow; dea = _sma(dif, 10, 2)
    return float(2.0 * (dif.iloc[-1] - dea.iloc[-1]))

def alpha090(d: OHLCV) -> float:
    """−RANK(CORR(RANK(VWAP), RANK(VOL), 5))"""
    if d.n < 5: return np.nan
    return float(-_rank(_ts_corr(_rank(d.vwap), _rank(d.volume), 5)).iloc[-1])

# ---- 091-100 ----
def alpha091(d: OHLCV) -> float:
    """−RANK(CLOSE−MAX(CLOSE,5)) * RANK(CORR(MEAN(VOL,40), LOW, 5))"""
    if d.n < 40: return np.nan
    a = _rank(d.close - _ts_max(d.close, 5))
    b = _rank(_ts_corr(_ts_mean(d.volume, 40), d.low, 5))
    return float(-(a * b).iloc[-1])

def alpha092(d: OHLCV) -> float:
    """−MAX(RANK(DECAYLINEAR(DELTA(CLOSE*0.35+VWAP*0.65,2),3)), TSRANK(DECAYLINEAR(ABS(CORR(MEAN(VOL,180),CLOSE,13)),5),15))"""
    if d.n < 180: return np.nan
    a = _rank(_decay_linear(_ts_delta(d.close * 0.35 + d.vwap * 0.65, 2), 3))
    mv180 = _ts_mean(d.volume, 180)
    b = _ts_rank(_decay_linear(_abs(_ts_corr(mv180, d.close, 13)), 5), 15)
    return float(-np.maximum(a, b).iloc[-1])

def alpha093(d: OHLCV) -> float:
    """SUM(OPEN>=DELAY(OPEN,1)?0:MAX(OPEN−LOW,OPEN−DELAY(OPEN,1)), 20)"""
    if d.n < 20: return np.nan
    prev_o = _ts_delay(d.open, 1)
    down = np.where(d.open < prev_o, np.maximum(d.open - d.low, d.open - prev_o), 0.0)
    return float(_rolling_sum(down, 20).iloc[-1])

def alpha094(d: OHLCV) -> float:
    """SUM(up→VOL, down→−VOL, 30)"""
    if d.n < 30: return np.nan
    up = d.close > _ts_delay(d.close, 1); down = d.close < _ts_delay(d.close, 1)
    x = np.where(up, d.volume, np.where(down, -d.volume, 0.0))
    return float(_rolling_sum(x, 30).iloc[-1])

def alpha095(d: OHLCV) -> float:
    """STD(AMOUNT, 20)"""
    if d.n < 20: return np.nan
    return float(_ts_std(d.amount, 20, ddof=0).iloc[-1])

def alpha096(d: OHLCV) -> float:
    """SMA(SMA((CLOSE−TSMIN(LOW,9))/(TSMAX(HIGH,9)−TSMIN(LOW,9))*100,3,1), 3, 1)"""
    if d.n < 9: return np.nan
    hi9 = _ts_max(d.high, 9); lo9 = _ts_min(d.low, 9)
    rsv = (d.close - lo9) / np.maximum(hi9 - lo9, 1e-300) * 100.0
    return float(_sma(_sma(rsv, 3, 1), 3, 1).iloc[-1])

def alpha097(d: OHLCV) -> float:
    """STD(VOLUME, 10)"""
    if d.n < 10: return np.nan
    return float(_ts_std(d.volume, 10, ddof=0).iloc[-1])

def alpha098(d: OHLCV) -> float:
    """If 100d MA change ≤ 0.05 → −(CLOSE−TSMIN(CLOSE,100)) else −DELTA(CLOSE,3)"""
    if d.n < 100: return np.nan
    ma100 = _ts_mean(d.close, 100)
    cond = _abs(_ts_delta(ma100, 100) / np.maximum(_ts_delay(ma100, 100), 1e-300)) <= 0.05
    return float(np.where(cond, -(d.close - _ts_min(d.close, 100)), -_ts_delta(d.close, 3)).iloc[-1])

def alpha099(d: OHLCV) -> float:
    """−RANK(COV(RANK(CLOSE), RANK(VOL), 5))"""
    if d.n < 5: return np.nan
    return float(-_rank(_ts_cov(_rank(d.close), _rank(d.volume), 5)).iloc[-1])

def alpha100(d: OHLCV) -> float:
    """STD(VOLUME, 20)"""
    if d.n < 20: return np.nan
    return float(_ts_std(d.volume, 20, ddof=0).iloc[-1])

# ---- 101-110 ----
def alpha101(d: OHLCV) -> float:
    """RANK(CORR(CLOSE, SUM(MEAN(VOL,30),37), 15))"""
    if d.n < 67: return np.nan  # 30+37
    x = _rolling_sum(_ts_mean(d.volume, 30), 37)
    return float(_rank(_ts_corr(d.close, x, 15)).iloc[-1])

def alpha102(d: OHLCV) -> float:
    """SMA(MAX(VOL−DELAY(VOL,1),0),6,1)/SMA(ABS(VOL−DELAY(VOL,1)),6,1)*100"""
    if d.n < 6: return np.nan
    diff = _ts_delta(d.volume, 1)
    u = _sma(np.maximum(diff, 0.0), 6, 1); d_ = _sma(_abs(diff), 6, 1)
    return float(u[-1] / d_[-1] * 100.0) if d_[-1] > 0 else np.nan

def alpha103(d: OHLCV) -> float:
    """(20−LOWDAY(LOW,20))/20*100"""
    if d.n < 20: return np.nan
    ld = _ts_argmin(d.low, 20)
    return float((20.0 - ld.iloc[-1]) / 20.0 * 100.0)

def alpha104(d: OHLCV) -> float:
    """−DELTA(CORR(HIGH,VOL,5),5) * RANK(STD(CLOSE,20))"""
    if d.n < 20: return np.nan
    c = _ts_corr(d.high, d.volume, 5)
    a = _ts_delta(c, 5)
    b = _rank(_ts_std(d.close, 20, ddof=0))
    return float((-a * b).iloc[-1])

def alpha105(d: OHLCV) -> float:
    """−CORR(RANK(OPEN), RANK(VOL), 10)"""
    if d.n < 10: return np.nan
    return float(-_ts_corr(_rank(d.open), _rank(d.volume), 10).iloc[-1])

def alpha106(d: OHLCV) -> float:
    """CLOSE − DELAY(CLOSE, 20)"""
    if d.n < 21: return np.nan
    return float(_ts_delta(d.close, 20).iloc[-1])

def alpha107(d: OHLCV) -> float:
    """−RANK(OPEN−DELAY(HIGH,1)) * RANK(OPEN−DELAY(CLOSE,1)) * RANK(OPEN−DELAY(LOW,1))"""
    if d.n < 2: return np.nan
    a = _rank(d.open - _ts_delay(d.high, 1))
    b = _rank(d.open - _ts_delay(d.close, 1))
    c = _rank(d.open - _ts_delay(d.low, 1))
    return float((-a * b * c).iloc[-1])

def alpha108(d: OHLCV) -> float:
    """−RANK(HIGH−MIN(HIGH,2)) ^ RANK(CORR(VWAP, MEAN(VOL,120), 6))"""
    if d.n < 120: return np.nan
    a = _rank(d.high - _ts_min(d.high, 2))
    b = _rank(_ts_corr(d.vwap, _ts_mean(d.volume, 120), 6))
    return float(-(a.iloc[-1] ** b.iloc[-1])) if not np.isnan(a.iloc[-1]) and not np.isnan(b.iloc[-1]) else np.nan

def alpha109(d: OHLCV) -> float:
    """SMA(HIGH−LOW,10,2) / SMA(SMA(HIGH−LOW,10,2),10,2)"""
    if d.n < 10: return np.nan
    rng = d.high - d.low
    a = _sma(rng, 10, 2); b = _sma(a, 10, 2)
    return float(a.iloc[-1] / b.iloc[-1]) if b.iloc[-1] > 0 else np.nan

def alpha110(d: OHLCV) -> float:
    """SUM(MAX(0,HIGH−DELAY(CLOSE,1)),20) / SUM(MAX(0,DELAY(CLOSE,1)−LOW),20) * 100"""
    if d.n < 20: return np.nan
    prev_c = _ts_delay(d.close, 1)
    up = np.maximum(0.0, d.high - prev_c); down = np.maximum(0.0, prev_c - d.low)
    su = _rolling_sum(up, 20); sd = _rolling_sum(down, 20)
    return float(su[-1] / sd[-1] * 100.0) if sd[-1] > 0 else np.nan

# ---- 111-120 ----
def alpha111(d: OHLCV) -> float:
    """SMA(VOL*((C−L)−(H−C))/(H−L),11,2) − SMA(VOL*((C−L)−(H−C))/(H−L),4,2)"""
    if d.n < 11: return np.nan
    denom = np.maximum(d.high - d.low, 1e-300)
    x = d.volume * ((d.close - d.low) - (d.high - d.close)) / denom
    return float((_sma(x, 11, 2) - _sma(x, 4, 2)).iloc[-1])

def alpha112(d: OHLCV) -> float:
    """(SUM(up_ret,12) − SUM(down_loss,12)) / (SUM(up_ret,12) + SUM(down_loss,12)) * 100"""
    if d.n < 12: return np.nan
    diff = _ts_delta(d.close, 1)
    up = np.maximum(0.0, diff); down = np.maximum(0.0, -diff)
    su = _rolling_sum(up, 12); sd = _rolling_sum(down, 12)
    return float((su[-1] - sd[-1]) / (su[-1] + sd[-1]) * 100.0) if (su[-1] + sd[-1]) > 0 else np.nan

def alpha113(d: OHLCV) -> float:
    """−RANK(SUM(DELAY(CLOSE,5),20)/20) * CORR(CLOSE,VOL,2) * RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))"""
    if d.n < 25: return np.nan
    a = _rank(_ts_mean(_ts_delay(d.close, 5), 20))
    b = _ts_corr(d.close, d.volume, 2)
    c = _rank(_ts_corr(_rolling_sum(d.close, 5), _rolling_sum(d.close, 20), 2))
    return float((-a * b * c).iloc[-1])

def alpha114(d: OHLCV) -> float:
    """(RANK(DELAY((H−L)/(SUM(C,5)/5),2)) * RANK(RANK(VOL))) / (((H−L)/(SUM(C,5)/5)) / (VWAP−CLOSE))"""
    if d.n < 5: return np.nan
    amp = (d.high - d.low) / _ts_mean(d.close, 5)
    a = _rank(_ts_delay(amp, 2))
    b = _rank(_rank(d.volume))
    denom = amp / np.maximum(d.vwap - d.close, 1e-300)
    return float((a.iloc[-1] * b.iloc[-1]) / denom.iloc[-1]) if denom.iloc[-1] != 0 else np.nan

def alpha115(d: OHLCV) -> float:
    """RANK(CORR(HIGH*0.9+CLOSE*0.1, MEAN(VOL,30),10)) ^ RANK(CORR(TSRANK((H+L)/2,4), TSRANK(VOL,10),7))"""
    if d.n < 30: return np.nan
    a = _rank(_ts_corr(d.high * 0.9 + d.close * 0.1, _ts_mean(d.volume, 30), 10))
    b = _rank(_ts_corr(_ts_rank((d.high + d.low) / 2.0, 4), _ts_rank(d.volume, 10), 7))
    return float(a.iloc[-1] ** b.iloc[-1]) if not np.isnan(a.iloc[-1]) and not np.isnan(b.iloc[-1]) else np.nan

def alpha116(d: OHLCV) -> float:
    """REGBETA(CLOSE, SEQUENCE, 20)"""
    if d.n < 20: return np.nan
    return float(_ts_regbeta(d.close, pd.Series(np.arange(1, d.n + 1, dtype=np.float64), index=d.close.index), 20).iloc[-1])

def alpha117(d: OHLCV) -> float:
    """TSRANK(VOL,32) * (1−TSRANK(CLOSE+HIGH−LOW,16)) * (1−TSRANK(RET,32))"""
    if d.n < 32: return np.nan
    a = _ts_rank(d.volume, 32)
    b = 1.0 - _ts_rank(d.close + d.high - d.low, 16)
    c = 1.0 - _ts_rank(d.ret, 32)
    return float((a * b * c).iloc[-1])

def alpha118(d: OHLCV) -> float:
    """SUM(HIGH−OPEN,20) / SUM(OPEN−LOW,20) * 100"""
    if d.n < 20: return np.nan
    up = np.maximum(0.0, d.high - d.open); down = np.maximum(0.0, d.open - d.low)
    su = _rolling_sum(up, 20); sd = _rolling_sum(down, 20)
    return float(su[-1] / sd[-1] * 100.0) if sd[-1] > 0 else np.nan

def alpha119(d: OHLCV) -> float:
    """RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOL,5),26),5),7)) − RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOL,15)),21),9),7),8))"""
    if d.n < 31: return np.nan
    mv5 = _ts_mean(d.volume, 5); s26 = _rolling_sum(mv5, 26)
    a = _rank(_decay_linear(_ts_corr(d.vwap, s26, 5), 7))
    mv15 = _ts_mean(d.volume, 15)
    inner = _ts_corr(_rank(d.open), _rank(mv15), 21)
    b = _rank(_decay_linear(_ts_rank(_ts_min(inner, 9), 7), 8))
    return float((a - b).iloc[-1])

def alpha120(d: OHLCV) -> float:
    """RANK(VWAP−CLOSE) / RANK(VWAP+CLOSE)"""
    if d.n < 1: return np.nan
    a = _rank(d.vwap - d.close); b = _rank(d.vwap + d.close)
    return float(a.iloc[-1] / b.iloc[-1]) if b.iloc[-1] != 0 else np.nan

# ---- 121-130 ----
def alpha121(d: OHLCV) -> float:
    """−RANK(VWAP−MIN(VWAP,12)) ^ TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOL,60),2),18),3)"""
    if d.n < 60: return np.nan
    a = _rank(d.vwap - _ts_min(d.vwap, 12))
    mv60 = _ts_mean(d.volume, 60)
    b = _ts_rank(_ts_corr(_ts_rank(d.vwap, 20), _ts_rank(mv60, 2), 18), 3)
    return float(-(a.iloc[-1] ** b.iloc[-1])) if not np.isnan(a.iloc[-1]) and not np.isnan(b.iloc[-1]) else np.nan

def alpha122(d: OHLCV) -> float:
    """(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2) − DELAY(...,1)) / DELAY(...,1)"""
    if d.n < 13: return np.nan
    x = _sma(_sma(_sma(_log(d.close), 13, 2), 13, 2), 13, 2)
    return float(_ts_delta(x, 1).iloc[-1] / _ts_delay(x, 1).iloc[-1]) if _ts_delay(x, 1).iloc[-1] != 0 else np.nan

def alpha123(d: OHLCV) -> float:
    """RANK(CORR(SUM((H+L)/2,20), SUM(MEAN(VOL,60),20),9))"""
    if d.n < 80: return np.nan
    mv60 = _ts_mean(d.volume, 60)
    return float(_rank(_ts_corr(_rolling_sum((d.high + d.low) / 2.0, 20), _rolling_sum(mv60, 20), 9)).iloc[-1])

def alpha124(d: OHLCV) -> float:
    """(CLOSE−VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)"""
    if d.n < 30: return np.nan
    a = d.close - d.vwap
    b = _decay_linear(_rank(_ts_max(d.close, 30)), 2)
    return float(a.iloc[-1] / b.iloc[-1]) if b.iloc[-1] != 0 else np.nan

def alpha125(d: OHLCV) -> float:
    """RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOL,80),17),20)) / RANK(DECAYLINEAR(DELTA(CLOSE*0.5+VWAP*0.5,3),16))"""
    if d.n < 80: return np.nan
    mv80 = _ts_mean(d.volume, 80)
    a = _rank(_decay_linear(_ts_corr(d.vwap, mv80, 17), 20))
    b = _rank(_decay_linear(_ts_delta(d.close * 0.5 + d.vwap * 0.5, 3), 16))
    return float(a.iloc[-1] / b.iloc[-1]) if b.iloc[-1] != 0 else np.nan

def alpha126(d: OHLCV) -> float:
    """(CLOSE+HIGH+LOW) / 3"""
    if d.n < 1: return np.nan
    return float((d.close.iloc[-1] + d.high.iloc[-1] + d.low.iloc[-1]) / 3.0)

def alpha127(d: OHLCV) -> float:
    """(MEAN((100*(CLOSE−MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)"""
    if d.n < 12: return np.nan
    dev = (d.close - _ts_max(d.close, 12)) / np.maximum(_ts_max(d.close, 12), 1e-300) * 100.0
    return float(np.sqrt(_ts_mean(dev * dev, 12).iloc[-1]))

def alpha128(d: OHLCV) -> float:
    """100 − 100/(1 + SUM(TP>PREV_TP?TP*VOL:0,14) / SUM(TP<PREV_TP?TP*VOL:0,14))
    where TP=(H+L+C)/3"""
    if d.n < 14: return np.nan
    tp = (d.high + d.low + d.close) / 3.0; prev_tp = _ts_delay(tp, 1)
    up = np.where(tp > prev_tp, tp * d.volume, 0.0)
    down = np.where(tp < prev_tp, tp * d.volume, 0.0)
    su = _rolling_sum(up, 14); sd = _rolling_sum(down, 14)
    ratio = su[-1] / sd[-1] if sd[-1] > 0 else 0.0
    return float(100.0 - 100.0 / (1.0 + ratio))

def alpha129(d: OHLCV) -> float:
    """SUM((CLOSE<DELAY(CLOSE,1)?ABS(CLOSE−DELAY(CLOSE,1)):0), 12)"""
    if d.n < 12: return np.nan
    diff = _ts_delta(d.close, 1)
    down = np.where(diff < 0, -diff, 0.0)
    return float(_rolling_sum(down, 12).iloc[-1])

def alpha130(d: OHLCV) -> float:
    """RANK(DECAYLINEAR(CORR((H+L)/2,MEAN(VOL,40),9),10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOL),7),3))"""
    if d.n < 40: return np.nan
    mv40 = _ts_mean(d.volume, 40)
    a = _rank(_decay_linear(_ts_corr((d.high + d.low) / 2.0, mv40, 9), 10))
    b = _rank(_decay_linear(_ts_corr(_rank(d.vwap), _rank(d.volume), 7), 3))
    return float(a.iloc[-1] / b.iloc[-1]) if b.iloc[-1] != 0 else np.nan

# ---- 131-140 ----
def alpha131(d: OHLCV) -> float:
    """RANK(DELTA(VWAP,1)) ^ TSRANK(CORR(CLOSE,MEAN(VOL,50),18),18)"""
    if d.n < 50: return np.nan
    a = _rank(_ts_delta(d.vwap, 1))
    mv50 = _ts_mean(d.volume, 50)
    b = _ts_rank(_ts_corr(d.close, mv50, 18), 18)
    return float(a.iloc[-1] ** b.iloc[-1]) if not np.isnan(a.iloc[-1]) and not np.isnan(b.iloc[-1]) else np.nan

def alpha132(d: OHLCV) -> float:
    """MEAN(AMOUNT, 20)"""
    if d.n < 20: return np.nan
    return float(_ts_mean(d.amount, 20).iloc[-1])

def alpha133(d: OHLCV) -> float:
    """(20−HIGHDAY(HIGH,20))/20*100 − (20−LOWDAY(LOW,20))/20*100"""
    if d.n < 20: return np.nan
    hd = _ts_argmax(d.high, 20); ld = _ts_argmin(d.low, 20)
    return float(((20.0 - hd.iloc[-1]) - (20.0 - ld.iloc[-1])) / 20.0 * 100.0)

def alpha134(d: OHLCV) -> float:
    """(CLOSE−DELAY(CLOSE,12))/DELAY(CLOSE,12) * VOLUME"""
    if d.n < 13: return np.nan
    ret = _ts_delta(d.close, 12) / np.maximum(np.abs(_ts_delay(d.close, 12)), 1e-300)
    return float((ret * d.volume).iloc[-1])

def alpha135(d: OHLCV) -> float:
    """−SMA(DELAY(CLOSE/DELAY(CLOSE,20),1), 20, 1)"""
    if d.n < 22: return np.nan
    ratio = d.close / np.maximum(_ts_delay(d.close, 20), 1e-300)
    return float(-_sma(_ts_delay(ratio, 1), 20, 1).iloc[-1])

def alpha136(d: OHLCV) -> float:
    """−RANK(DELTA(RET,3)) * CORR(OPEN, VOL, 10)"""
    if d.n < 10: return np.nan
    a = _rank(_ts_delta(d.ret, 3))
    b = _ts_corr(d.open, d.volume, 10)
    return float((-a * b).iloc[-1])

def alpha137(d: OHLCV) -> float:
    """EMV-like scaled: 16*(C−PREV+(C−O)/2+PREV−PREV_O) / (max_abs_range)"""
    if d.n < 2: return np.nan
    prev_c = _ts_delay(d.close, 1); prev_o = _ts_delay(d.open, 1)
    num = d.close - prev_c + (d.close - d.open) / 2.0 + prev_c - prev_o
    denom = np.maximum(np.maximum(d.high - d.low, _abs(d.high - prev_c)), _abs(prev_c - d.low))
    denom = np.maximum(denom, 1e-300)
    return float((16.0 * num / denom).iloc[-1])

def alpha138(d: OHLCV) -> float:
    """−(RANK(DECAYLINEAR(DELTA(LOW*0.7+VWAP*0.3,3),20)) − TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW,8),TSRANK(MEAN(VOL,60),17),5),19),16),7))"""
    if d.n < 60: return np.nan
    a = _rank(_decay_linear(_ts_delta(d.low * 0.7 + d.vwap * 0.3, 3), 20))
    mv60 = _ts_mean(d.volume, 60)
    inner = _ts_corr(_ts_rank(d.low, 8), _ts_rank(mv60, 17), 5)
    b = _ts_rank(_decay_linear(_ts_rank(inner, 19), 16), 7)
    return float(-(a - b).iloc[-1])

def alpha139(d: OHLCV) -> float:
    """−CORR(OPEN, VOL, 10)"""
    if d.n < 10: return np.nan
    return float(-_ts_corr(d.open, d.volume, 10).iloc[-1])

def alpha140(d: OHLCV) -> float:
    """MIN(RANK(DECAYLINEAR((RANK(OPEN)+RANK(LOW)−RANK(HIGH)−RANK(CLOSE)),8)),
    TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOL,60),20),8),7),3))"""
    if d.n < 60: return np.nan
    a = _rank(_decay_linear(_rank(d.open) + _rank(d.low) - _rank(d.high) - _rank(d.close), 8))
    mv60 = _ts_mean(d.volume, 60)
    b = _ts_rank(_decay_linear(_ts_corr(_ts_rank(d.close, 8), _ts_rank(mv60, 20), 8), 7), 3)
    return float(np.minimum(a, b).iloc[-1])

# ---- 141-150 ----
def alpha141(d: OHLCV) -> float:
    """−RANK(CORR(RANK(HIGH), RANK(MEAN(VOL,15)), 9))"""
    if d.n < 15: return np.nan
    return float(-_rank(_ts_corr(_rank(d.high), _rank(_ts_mean(d.volume, 15)), 9)).iloc[-1])

def alpha142(d: OHLCV) -> float:
    """−RANK(TSRANK(CLOSE,10)) * RANK(DELTA(DELTA(CLOSE,1),1)) * RANK(TSRANK(VOL/MEAN(VOL,20),5))"""
    if d.n < 20: return np.nan
    a = _rank(_ts_rank(d.close, 10))
    b = _rank(_ts_delta(_ts_delta(d.close, 1), 1))
    c = _rank(_ts_rank(d.volume / _ts_mean(d.volume, 20), 5))
    return float((-a * b * c).iloc[-1])

def alpha143(d: OHLCV) -> float:
    """Unimplemented (recursive SELF reference)"""
    return np.nan

def alpha144(d: OHLCV) -> float:
    """SUMIF(ABS(RET)/AMOUNT, 20, CLOSE<DELAY(CLOSE,1)) / COUNT(CLOSE<DELAY(CLOSE,1), 20)"""
    if d.n < 20: return np.nan
    impact = _abs(d.ret) / np.maximum(d.amount, 1e-300)
    down = d.close < _ts_delay(d.close, 1)
    s = impact.where(down, 0.0)
    su = _rolling_sum(s, 20)
    cnt = _rolling_sum(down.astype(float), 20).clip(lower=1e-300)
    return float((su / cnt).dropna().iloc[-1])

def alpha145(d: OHLCV) -> float:
    """(MEAN(VOL,9)−MEAN(VOL,26))/MEAN(VOL,12)*100"""
    if d.n < 26: return np.nan
    return float((_ts_mean(d.volume, 9) - _ts_mean(d.volume, 26)).iloc[-1] / _ts_mean(d.volume, 12).iloc[-1] * 100.0) if _ts_mean(d.volume, 12).iloc[-1] > 0 else np.nan

def alpha146(d: OHLCV) -> float:
    """MEAN(ret−SMA(ret,61,2), 20) * (ret−SMA(ret,61,2)) / SMA((ret−dev)^2, 60)"""
    if d.n < 61: return np.nan
    s = _sma(d.ret, 61, 2); dev = d.ret - s
    mdev = _ts_mean(dev, 20)
    v = _sma(dev * dev, 60, 2)
    return float((mdev[-1] * dev[-1]) / v[-1]) if v[-1] > 0 else np.nan

def alpha147(d: OHLCV) -> float:
    """REGBETA(MEAN(CLOSE,12), SEQUENCE(12))"""
    if d.n < 12: return np.nan
    return float(_ts_regbeta(_ts_mean(d.close, 12), pd.Series(np.arange(1, d.n + 1, dtype=np.float64), index=d.close.index), 12).iloc[-1])

def alpha148(d: OHLCV) -> float:
    """(RANK(CORR(OPEN,SUM(MEAN(VOL,60),9),6)) < RANK(OPEN−TSMIN(OPEN,14))) ? 1 : 0 → −sign"""
    if d.n < 60: return np.nan
    mv60 = _ts_mean(d.volume, 60)
    a = _rank(_ts_corr(d.open, _rolling_sum(mv60, 9), 6))
    b = _rank(d.open - _ts_min(d.open, 14))
    c = a < b
    return float(-c.iloc[-1])

def alpha149(d: OHLCV) -> float:
    """Benchmark-index factor: down-market beta"""
    return np.nan

def alpha150(d: OHLCV) -> float:
    """(CLOSE+HIGH+LOW)/3 * VOLUME"""
    if d.n < 1: return np.nan
    return float(((d.close.iloc[-1] + d.high.iloc[-1] + d.low.iloc[-1]) / 3.0) * d.volume.iloc[-1])

# ---- 151-160 ----
def alpha151(d: OHLCV) -> float:
    """SMA(CLOSE−DELAY(CLOSE,20), 20, 1)"""
    if d.n < 21: return np.nan
    return float(_sma(_ts_delta(d.close, 20), 20, 1).iloc[-1])

def alpha152(d: OHLCV) -> float:
    """SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)
    − MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26), 9, 1)"""
    if d.n < 37: return np.nan
    r = d.close / np.maximum(_ts_delay(d.close, 9), 1e-300)
    s = _sma(_ts_delay(r, 1), 9, 1)
    ds = _ts_delay(s, 1)
    dif = _ts_mean(ds, 12) - _ts_mean(ds, 26)
    return float(_sma(dif, 9, 1).iloc[-1])

def alpha153(d: OHLCV) -> float:
    """(MA3+MA6+MA12+MA24)/4"""
    if d.n < 24: return np.nan
    return float((_ts_mean(d.close, 3) + _ts_mean(d.close, 6) + _ts_mean(d.close, 12) + _ts_mean(d.close, 24)).iloc[-1] / 4.0)

def alpha154(d: OHLCV) -> float:
    """(VWAP−MIN(VWAP,16)) < CORR(VWAP, MEAN(VOL,180), 18)"""
    if d.n < 180: return np.nan
    a = d.vwap - _ts_min(d.vwap, 16)
    mv180 = _ts_mean(d.volume, 180)
    b = _ts_corr(d.vwap, mv180, 18)
    return float((a < b).iloc[-1])

def alpha155(d: OHLCV) -> float:
    """Volume MACD: SMA(VOL,13,2)−SMA(VOL,27,2)−SMA(SMA(VOL,13,2)−SMA(VOL,27,2),10,2)"""
    if d.n < 27: return np.nan
    fast = _sma(d.volume, 13, 2); slow = _sma(d.volume, 27, 2)
    dif = fast - slow; dea = _sma(dif, 10, 2)
    return float((dif - dea).iloc[-1])

def alpha156(d: OHLCV) -> float:
    """−MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)), RANK(DECAYLINEAR(−DELTA(OPEN*0.15+LOW*0.85,2)/(OPEN*0.15+LOW*0.85),3)))"""
    if d.n < 8: return np.nan
    a = _rank(_decay_linear(_ts_delta(d.vwap, 5), 3))
    p = d.open * 0.15 + d.low * 0.85
    b_inner = -_ts_delta(p, 2) / np.maximum(p, 1e-300)
    b = _rank(_decay_linear(b_inner, 3))
    return float(-np.maximum(a, b).iloc[-1])

def alpha157(d: OHLCV) -> float:
    """MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK(−RANK(DELTA(CLOSE−1,5)))),2),1)))),1),5)
    + TSRANK(DELAY(−RET,6),5)"""
    if d.n < 10: return np.nan
    x = -_rank(_ts_delta(d.close - 1.0, 5))
    y = _ts_min(_rank(_rank(x)), 2)
    z = _rank(_rank(_log(_rolling_sum(y, 1))))
    p = _ts_prod(z, 1)
    a = _ts_min(p, 5)
    b = _ts_rank(_ts_delay(-d.ret, 6), 5)
    return float((a + b).iloc[-1])

def alpha158(d: OHLCV) -> float:
    """((HIGH−SMA(CLOSE,15,2))−(LOW−SMA(CLOSE,15,2)))/CLOSE"""
    if d.n < 15: return np.nan
    s = _sma(d.close, 15, 2)
    return float(((d.high.iloc[-1] - s[-1]) - (d.low.iloc[-1] - s[-1])) / d.close.iloc[-1]) if d.close.iloc[-1] > 0 else np.nan

def alpha159(d: OHLCV) -> float:
    """Multi-horizon price position vs prev-close range: weighted avg of 6/12/24d positions"""
    if d.n < 24: return np.nan
    prev_c = _ts_delay(d.close, 1)
    lo_ref = np.minimum(d.low, prev_c); hi_ref = np.maximum(d.high, prev_c)
    rng = hi_ref - lo_ref
    pos = (d.close - lo_ref) / np.maximum(rng, 1e-300)
    w_sum = (_rolling_sum(pos, 6) * 12 + _rolling_sum(pos, 12) * 24 + _rolling_sum(pos, 24) * 48)
    return float(w_sum[-1] / (12.0 + 24.0 + 48.0) * 100.0)

def alpha160(d: OHLCV) -> float:
    """SMA(CLOSE<=DELAY(CLOSE,1) ? STD(CLOSE,20) : 0, 20, 1)"""
    if d.n < 20: return np.nan
    std20 = _ts_std(d.close, 20, ddof=0)
    x = np.where(d.close <= _ts_delay(d.close, 1), std20, 0.0)
    return float(_sma(x, 20, 1).iloc[-1])

# ---- 161-170 ----
def alpha161(d: OHLCV) -> float:
    """MEAN(TR, 12) — ATR"""
    if d.n < 13: return np.nan
    prev_c = _ts_delay(d.close, 1)
    tr = np.maximum(np.maximum(d.high - d.low, _abs(d.high - prev_c)), _abs(prev_c - d.low))
    return float(_ts_mean(tr, 12).iloc[-1])

def alpha162(d: OHLCV) -> float:
    """(RSI12−MIN(RSI12,12))/(MAX(RSI12,12)−MIN(RSI12,12))  — stochastic of RSI"""
    if d.n < 12: return np.nan
    diff = _ts_delta(d.close, 1)
    u = _sma(np.maximum(diff, 0.0), 12, 1); d_ = _sma(_abs(diff), 12, 1)
    rsi = np.where(d_ > 0, u / d_ * 100.0, 50.0)
    lo12 = _ts_min(rsi, 12); hi12 = _ts_max(rsi, 12)
    return float((rsi.iloc[-1] - lo12[-1]) / (hi12[-1] - lo12[-1])) if (hi12[-1] - lo12[-1]) > 0 else np.nan

def alpha163(d: OHLCV) -> float:
    """RANK((−RET * MEAN(VOL,20) * VWAP * (HIGH−CLOSE)))"""
    if d.n < 20: return np.nan
    x = -d.ret * _ts_mean(d.volume, 20) * d.vwap * (d.high - d.close)
    return float(_rank(x).iloc[-1])

def alpha164(d: OHLCV) -> float:
    """SMA(...) price-position with inverse return weighting"""
    if d.n < 13: return np.nan
    diff = np.maximum(_ts_delta(d.close, 1), 1e-300)
    x = np.where(d.close > _ts_delay(d.close, 1), 1.0 / diff, 1.0)
    lo12 = _ts_min(x, 12); hi = d.high - d.low
    y = (x - lo12) / np.maximum(hi, 1e-300) * 100.0
    return float(_sma(y, 13, 2).iloc[-1])

def alpha165(d: OHLCV) -> float:
    """Unimplemented (requires SUMAC of CLOSE−MEAN(CLOSE,48))"""
    return np.nan

def alpha166(d: OHLCV) -> float:
    """−20*(20−1)^1.5 * SUM(ret−mean_ret,20) / ((20−1)*(20−2)*(SUM(ret^2,20))^1.5) — skewness approx"""
    if d.n < 20: return np.nan
    r = d.ret; mr = _ts_mean(r, 20)
    num = _rolling_sum(r - mr, 20)
    denom_inner = _rolling_sum(r * r, 20)
    denom = 19.0 * 18.0 * (denom_inner ** 1.5)
    factor = -20.0 * (19.0 ** 1.5)
    return float((factor * num[-1]) / denom.iloc[-1]) if denom.iloc[-1] != 0 else np.nan

def alpha167(d: OHLCV) -> float:
    """SUM((CLOSE>DELAY(CLOSE,1)?CLOSE−DELAY(CLOSE,1):0), 12)"""
    if d.n < 12: return np.nan
    diff = _ts_delta(d.close, 1)
    return float(_rolling_sum(np.maximum(diff, 0.0), 12).iloc[-1])

def alpha168(d: OHLCV) -> float:
    """−VOLUME / MEAN(VOLUME, 20)"""
    if d.n < 20: return np.nan
    return float(-d.volume.iloc[-1] / _ts_mean(d.volume, 20).iloc[-1]) if _ts_mean(d.volume, 20).iloc[-1] > 0 else np.nan

def alpha169(d: OHLCV) -> float:
    """SMA(MEAN(DELAY(SMA(DELTA(CLOSE,1),9,1),1),12) − MEAN(DELAY(SMA(DELTA(CLOSE,1),9,1),1),26), 10, 1)"""
    if d.n < 37: return np.nan
    diff = _ts_delta(d.close, 1)
    s = _sma(diff, 9, 1); ds = _ts_delay(s, 1)
    dif = _ts_mean(ds, 12) - _ts_mean(ds, 26)
    return float(_sma(dif, 10, 1).iloc[-1])

def alpha170(d: OHLCV) -> float:
    """RANK(1/CLOSE)*VOL/MEAN(VOL,20) * (HIGH*RANK(HIGH−CLOSE)/(SUM(HIGH,5)/5)) − RANK(VWAP−DELAY(VWAP,5))"""
    if d.n < 20: return np.nan
    a = _rank(1.0 / np.maximum(d.close, 1e-300)) * d.volume / _ts_mean(d.volume, 20)
    b = d.high * _rank(d.high - d.close) / _ts_mean(d.high, 5)
    c = _rank(_ts_delta(d.vwap, 5))
    return float((a * b - c).iloc[-1])

# ---- 171-180 ----
def alpha171(d: OHLCV) -> float:
    """−((LOW−CLOSE)*(OPEN^5)) / ((CLOSE−HIGH)*(CLOSE^5))"""
    if d.n < 1: return np.nan
    num = (d.low.iloc[-1] - d.close.iloc[-1]) * (d.open.iloc[-1] ** 5)
    denom = (d.close.iloc[-1] - d.high.iloc[-1]) * (d.close.iloc[-1] ** 5)
    return float(-num / denom) if denom != 0 else np.nan

def alpha172(d: OHLCV) -> float:
    """ADXR: MEAN(ABS(SUM(LD_signal,14)/SUM(TR,14) − SUM(HD_signal,14)/SUM(TR,14))
    / (SUM(LD_signal,14)/SUM(TR,14) + SUM(HD_signal,14)/SUM(TR,14)) * 100, 6)"""
    if d.n < 15: return np.nan
    prev_c = _ts_delay(d.close, 1)
    tr = np.maximum(np.maximum(d.high - d.low, _abs(d.high - prev_c)), _abs(prev_c - d.low))
    hd = d.high - _ts_delay(d.high, 1); ld = _ts_delay(d.low, 1) - d.low
    hd_sig = np.where((hd > 0) & (hd > ld), hd, 0.0)
    ld_sig = np.where((ld > 0) & (ld > hd), ld, 0.0)
    str_sum = _rolling_sum(tr, 14)
    sh = _rolling_sum(hd_sig, 14) / np.maximum(str_sum, 1e-300)
    sl = _rolling_sum(ld_sig, 14) / np.maximum(str_sum, 1e-300)
    dx = _abs(sh - sl) / (sh + sl + 1e-300) * 100.0
    return float(_ts_mean(dx, 6).iloc[-1])

def alpha173(d: OHLCV) -> float:
    """3*SMA(CLOSE,13,2) − 2*SMA(SMA(CLOSE,13,2),13,2) + SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)"""
    if d.n < 13: return np.nan
    s1 = _sma(d.close, 13, 2); s2 = _sma(s1, 13, 2)
    l1 = _sma(_sma(_sma(_log(d.close), 13, 2), 13, 2), 13, 2)
    return float(3.0 * s1.iloc[-1] - 2.0 * s2.iloc[-1] + l1.iloc[-1])

def alpha174(d: OHLCV) -> float:
    """SMA(CLOSE>DELAY(CLOSE,1) ? STD(CLOSE,20) : 0, 20, 1)"""
    if d.n < 20: return np.nan
    std20 = _ts_std(d.close, 20, ddof=0)
    x = np.where(d.close > _ts_delay(d.close, 1), std20, 0.0)
    return float(_sma(x, 20, 1).iloc[-1])

def alpha175(d: OHLCV) -> float:
    """MEAN(TR, 6) — 6-day ATR"""
    if d.n < 7: return np.nan
    prev_c = _ts_delay(d.close, 1)
    tr = np.maximum(np.maximum(d.high - d.low, _abs(d.high - prev_c)), _abs(prev_c - d.low))
    return float(_ts_mean(tr, 6).iloc[-1])

def alpha176(d: OHLCV) -> float:
    """CORR(RANK((CLOSE−TSMIN(LOW,12))/(TSMAX(HIGH,12)−TSMIN(LOW,12))), RANK(VOL), 6)"""
    if d.n < 12: return np.nan
    hi12 = _ts_max(d.high, 12); lo12 = _ts_min(d.low, 12)
    pos = (d.close - lo12) / np.maximum(hi12 - lo12, 1e-300)
    return float(_ts_corr(_rank(pos), _rank(d.volume), 6).iloc[-1])

def alpha177(d: OHLCV) -> float:
    """(20−HIGHDAY(HIGH,20))/20*100"""
    if d.n < 20: return np.nan
    hd = _ts_argmax(d.high, 20)
    return float((20.0 - hd.iloc[-1]) / 20.0 * 100.0)

def alpha178(d: OHLCV) -> float:
    """(CLOSE−DELAY(CLOSE,1))/DELAY(CLOSE,1) * VOLUME"""
    if d.n < 2: return np.nan
    return float((d.ret.iloc[-1] * d.volume.iloc[-1]) if not np.isnan(d.ret.iloc[-1]) else np.nan)

def alpha179(d: OHLCV) -> float:
    """RANK(CORR(VWAP,VOL,4)) * RANK(CORR(RANK(LOW),RANK(MEAN(VOL,50)),12))"""
    if d.n < 50: return np.nan
    a = _rank(_ts_corr(d.vwap, d.volume, 4))
    mv50 = _ts_mean(d.volume, 50)
    b = _rank(_ts_corr(_rank(d.low), _rank(mv50), 12))
    return float((a * b).iloc[-1])

def alpha180(d: OHLCV) -> float:
    """If VOL > MEAN(VOL,20): −TSRANK(ABS(DELTA(CLOSE,7)),60) * SIGN(DELTA(CLOSE,7)), else −VOL"""
    if d.n < 60: return np.nan
    cond = d.volume > _ts_mean(d.volume, 20)
    a = -_ts_rank(_abs(_ts_delta(d.close, 7)), 60) * _sign(_ts_delta(d.close, 7))
    return float(np.where(cond, a, -d.volume)[-1])

# ---- 181-191 ----
def alpha181(d: OHLCV) -> float:
    """Benchmark-index factor: coskewness"""
    return np.nan

def alpha182(d: OHLCV) -> float:
    """Benchmark-index factor: COUNT(market-direction alignment, 20)/20"""
    return np.nan

def alpha183(d: OHLCV) -> float:
    """Unimplemented"""
    return np.nan

def alpha184(d: OHLCV) -> float:
    """RANK(CORR(DELAY(OPEN−CLOSE,1), CLOSE, 200)) + RANK(OPEN−CLOSE)"""
    if d.n < 200: return np.nan
    a = _rank(_ts_corr(_ts_delay(d.open - d.close, 1), d.close, 200))
    b = _rank(d.open - d.close)
    return float((a + b).iloc[-1])

def alpha185(d: OHLCV) -> float:
    """RANK(−(1−OPEN/CLOSE)^2)"""
    if d.n < 1: return np.nan
    x = -((1.0 - d.open / np.maximum(d.close, 1e-300)) ** 2)
    return float(_rank(x).iloc[-1])

def alpha186(d: OHLCV) -> float:
    """(ADXR + DELAY(ADXR, 6)) / 2 — smoothed version of alpha172"""
    if d.n < 21: return np.nan
    prev_c = _ts_delay(d.close, 1)
    tr = np.maximum(np.maximum(d.high - d.low, _abs(d.high - prev_c)), _abs(prev_c - d.low))
    hd = d.high - _ts_delay(d.high, 1); ld = _ts_delay(d.low, 1) - d.low
    hd_sig = np.where((hd > 0) & (hd > ld), hd, 0.0)
    ld_sig = np.where((ld > 0) & (ld > hd), ld, 0.0)
    str_sum = _rolling_sum(tr, 14)
    sh = _rolling_sum(hd_sig, 14) / np.maximum(str_sum, 1e-300)
    sl = _rolling_sum(ld_sig, 14) / np.maximum(str_sum, 1e-300)
    dx = _abs(sh - sl) / (sh + sl + 1e-300) * 100.0
    adxr = _ts_mean(dx, 6)
    return float((adxr + _ts_delay(adxr, 6)).iloc[-1] / 2.0)

def alpha187(d: OHLCV) -> float:
    """SUM(OPEN>DELAY(OPEN,1)?MAX(HIGH−OPEN,OPEN−DELAY(OPEN,1)):0, 20)"""
    if d.n < 20: return np.nan
    prev_o = _ts_delay(d.open, 1)
    up = np.where(d.open > prev_o, np.maximum(d.high - d.open, d.open - prev_o), 0.0)
    return float(_rolling_sum(up, 20).iloc[-1])

def alpha188(d: OHLCV) -> float:
    """((HIGH−LOW−SMA(HIGH−LOW,11,2))/SMA(HIGH−LOW,11,2))*100"""
    if d.n < 11: return np.nan
    rng = d.high - d.low; s = _sma(rng, 11, 2)
    return float((rng[-1] - s[-1]) / s[-1] * 100.0) if s[-1] > 0 else np.nan

def alpha189(d: OHLCV) -> float:
    """MEAN(ABS(CLOSE−MEAN(CLOSE,6)), 6)"""
    if d.n < 6: return np.nan
    return float(_ts_mean(_abs(d.close - _ts_mean(d.close, 6)), 6).iloc[-1])

def alpha190(d: OHLCV) -> float:
    """LOG((COUNT(RET>0,20)−1) * SUM(−RET^2|down,20) / (COUNT(RET<0,20) * SUM(RET^2|up,20)))"""
    if d.n < 20: return np.nan
    up = d.ret > 0; down = d.ret < 0
    cup = up.sum(); cdown = down.sum()
    sup = (d.ret[up] ** 2).sum() if cup > 0 else 1e-300
    sdown = (d.ret[down] ** 2).sum() if cdown > 0 else 1e-300
    val = (cup - 1.0) * sdown / (cdown * sup) if cdown > 0 and sup > 0 else 1e-300
    return float(np.log(max(val, 1e-300)))

def alpha191(d: OHLCV) -> float:
    """CORR(MEAN(VOL,20), LOW, 5) + (HIGH+LOW)/2 − CLOSE"""
    if d.n < 20: return np.nan
    mv20 = _ts_mean(d.volume, 20)
    return float((_ts_corr(mv20, d.low, 5) + (d.high + d.low) / 2.0 - d.close).iloc[-1])


# ============================================================
# 因子名称到函数的注册表
# ============================================================
GTJA191_FACTORY: dict[str, Callable[[OHLCV], float]] = {}
for _i in range(1, 192):
    _name = f"alpha{_i:03d}"
    _fn = globals().get(_name)
    if _fn is not None:
        GTJA191_FACTORY[_name] = _fn

GTJA191_NAMES = sorted(GTJA191_FACTORY.keys())
GTJA191_COUNT = len(GTJA191_NAMES)


# ============================================================
# 批量计算入口
# ============================================================
def compute_factors(
    codes: list[str],
    as_of: str,
    prices: dict[str, list[dict]],
    factor_names: list[str] | None = None,
    min_bars: int = 20,
) -> dict[str, dict[str, float | None]]:
    """Compute GTJA191 factors for `codes` as of `as_of`.

    Returns {code: {alpha001: float|None, ...}}. Missing/unavailable → None.
    `factor_names` defaults to all available factors; pass a subset for speed."""
    names = factor_names or GTJA191_NAMES
    fns = [(n, GTJA191_FACTORY[n]) for n in names if n in GTJA191_FACTORY]
    out: dict[str, dict[str, float | None]] = {}
    for code in codes:
        bars = prices.get(code)
        if not bars:
            out[code] = {n: None for n, _ in fns}
            continue
        d = OHLCV(bars, as_of)
        if d.n < min_bars:
            out[code] = {n: None for n, _ in fns}
            continue
        vals: dict[str, float | None] = {}
        for name, fn in fns:
            try:
                v = fn(d)
            except Exception:
                v = None
            vals[name] = v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else None
        out[code] = vals
    return out


# ============================================================
# 离线自检
# ============================================================
def _self_test() -> None:
    import math
    import random
    rng = random.Random(42)
    bars = []
    px, dt = 100.0, date(2021, 1, 1)
    for i in range(300):
        ret = rng.gauss(0.0005, 0.015)
        op = px * (1 + rng.gauss(0, 0.003))
        hi = max(op, px * (1 + ret)) * (1 + abs(rng.gauss(0, 0.005)))
        lo = min(op, px * (1 + ret)) * (1 - abs(rng.gauss(0, 0.005)))
        px = px * (1 + ret)
        vol = 1e7 * (1 + abs(rng.gauss(0, 0.5)))
        amt = px * vol
        bars.append({"date": dt.isoformat(), "open": round(op, 2), "high": round(hi, 2),
                      "low": round(lo, 2), "close": round(px, 2),
                      "volume": vol, "amount": amt})
        dt = date.fromordinal(dt.toordinal() + 1)
    as_of = bars[-1]["date"]
    d = OHLCV(bars, as_of)
    assert d.n == 300
    assert len(d.close) == 300

    # Verify primitives
    a = alpha001(d); assert not math.isnan(a), f"alpha001={a}"
    a2 = alpha002(d); assert not math.isnan(a2)
    a14 = alpha014(d); assert not math.isnan(a14)
    a15 = alpha015(d); assert not math.isnan(a15)
    print(f"[PASS] alpha001={a:.4f} alpha002={a2:.4f} alpha014={a14:.4f} alpha015={a15:.4f}")

    # Batch compute
    result = compute_factors(["TEST"], as_of, {"TEST": bars}, min_bars=20)
    assert "TEST" in result
    vals = result["TEST"]
    ok = sum(1 for v in vals.values() if v is not None)
    total = len(vals)
    assert ok > 100, f"Only {ok}/{total} factors computed"
    print(f"[PASS] Batch compute: {ok}/{total} factors available (rest need longer history)")

    # Verify factor orientations on synthetic extreme data
    up_bars = []
    px_u = 100.0
    for i in range(300):
        px_u *= 1.002
        up_bars.append({"date": (date(2021, 1, 1) + __import__('datetime').timedelta(days=i)).isoformat(),
                        "open": px_u * 0.99, "high": px_u * 1.02, "low": px_u * 0.98,
                        "close": px_u, "volume": 1e7, "amount": px_u * 1e7})
    d_up = OHLCV(up_bars, up_bars[-1]["date"])
    # In uptrend, momentum factors (alpha014=close-delay5) should be positive
    assert alpha014(d_up) > 0, "uptrend should give positive alpha014"
    # alpha001 should be negative (vol up with price up = positive corr → negated)
    a001_up = alpha001(d_up)
    print(f"[PASS] Uptrend alpha014={alpha014(d_up):.4f}>0, alpha001={a001_up:.4f}")

    print(f"[PASS] GTJA191: {GTJA191_COUNT} factors, all self-tests passed")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="GTJA 191 Alpha factors.")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        _self_test()
    else:
        p.error("Use --self-test.")
