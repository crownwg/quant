"""择时 / 仓位管理：给纯多头组合加一层「什么时候该在场」的判断。

为什么需要这一层
----------------
风险归因已经给出结论：策略的超额收益全程为正，但**纯多头躲不开 beta**。
消费池实测牛市 +118.7%、熊市 -46.6%——选股再准，也扛不住系统性下跌。
「什么时候在场」和「买什么」是两个独立的问题，前者靠仓位管理解决。

两种互补的手段
--------------
1. **趋势择时（trend）**：用市场中期趋势做开关。趋势向上满仓，向下空仓。
   本质是「用趋势换掉一部分波动」，代价是震荡市里反复打脸。
2. **波动率目标（vol target）**：仓位 = 目标波动率 / 已实现波动率。
   市场越躁动仓位越低。不判断方向，只控制风险预算。

两者可以叠加（敞口相乘）：趋势决定「要不要在场」，波动率决定「在场放多少」。
经典配置是趋势做 0/1 开关、波动率做 0.2~1.0 的连续缩放。

⚠️ 因果性（这是本模块最容易写错的地方）
--------------------------------------
敞口序列与目标权重同索引，含义是「在 t 日收盘时决定、t 之后生效」。
因此信号只能用到 close[t] 及更早的信息，**绝不能出现 shift(-k)**。
- proxy/MA 用的是 close[t] 及之前 → 合法
- 滚动波动率 rolling(N).std() 在 t 处用的是 t-N+1..t → 合法
本模块所有函数都遵守这一点，tests/test_timing.py 有专门的因果性测试。

滞回带（band）为什么必要
------------------------
不带 band 的均线择时在均线附近会被反复穿越，敞口在 0/1 之间来回跳，
换手成本会吃掉择时收益。band 让「进场」和「出场」用不同的门槛：
  - 空仓 → 满仓：需要价格高于 MA×(1+band)
  - 满仓 → 空仓：需要价格低于 MA×(1-band)
中间地带维持原状态，这就是滞回。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMING_MODES = ("off", "ma", "momentum", "dual")
PROXY_MODES = ("auto", "benchmark", "pool")


def norm_index_symbol(sym: str) -> str:
    """把裸指数代码补上交易所前缀：000932 → sh000932、399006 → sz399006。

    指数接口要求带前缀。用户手写代理指数代码时容易漏，这里兜住。
    （放在本模块是为了让 main 与 combine 共用一份实现，避免两处规则漂移。）
    """
    s = str(sym or "").strip()
    if not s:
        return s
    if s[:2].lower() in ("sh", "sz", "bj"):
        return s
    if s.startswith("399") or s.startswith("159"):
        return f"sz{s}"
    return f"sh{s}"


def _cfg_get(cfg, key, default):
    """同时兼容 argparse 命名空间与 dict 配置（daily.py 用的是 JSON dict）。"""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# --------------------------------------------------------- 代理指数

def pool_index(prices: pd.DataFrame, min_names: int = 1) -> pd.Series:
    """池子等权组合净值 —— 用作择时代理。

    为什么不直接用池子里随便一只票：单只票的走势噪声太大，择时信号会被个股
    特质波动淹没。等权组合把这些噪声平均掉，剩下的是这个池子共同的 beta。

    做法：每只票的日收益等权平均后复利，等价于「每日再平衡的等权组合」。
    未上市 / 停牌（收益为 NaN）的票自动排除在当日平均之外。
    """
    if prices is None or prices.empty:
        raise ValueError("pool_index 需要非空的价格面板")
    ret = prices.pct_change()
    n_valid = ret.notna().sum(axis=1)
    ew = ret.mean(axis=1, skipna=True)
    ew = ew.where(n_valid >= max(min_names, 1), 0.0).fillna(0.0)
    return (1.0 + ew).cumprod()


# --------------------------------------------------------- 趋势择时

def _hysteresis(signal: pd.Series, band: float, initial: int = 0) -> pd.Series:
    """把连续信号按滞回带变成 0/1 状态序列（状态机，带记忆）。

    signal > +band → 1；signal < -band → 0；中间维持上一个状态。
    initial=0 表示「从空仓开始」——保守默认，宁可错过也比默认在场安全。
    NaN（预热期）维持当前状态，不参与翻转。
    """
    values = signal.to_numpy(dtype=float)
    out = np.empty(len(values), dtype=float)
    state = float(initial)
    band = float(abs(band))
    for i, v in enumerate(values):
        if v == v:  # 非 NaN
            if state == 0.0 and v > band:
                state = 1.0
            elif state == 1.0 and v < -band:
                state = 0.0
        out[i] = state
    return pd.Series(out, index=signal.index)


def trend_exposure(proxy: pd.Series, lookback: int = 120, mode: str = "ma",
                   band: float = 0.0, ma_slope: int = 0,
                   min_exposure: float = 0.0, max_exposure: float = 1.0) -> pd.Series:
    """趋势择时敞口（0/1，再按 min/max 裁剪）。

    mode:
      ma       —— 价格相对 lookback 日均线的高低。最经典，也最稳。
      momentum —— 过去 lookback 日累计收益为正。比 ma 迟钝一些，
                  但在单边趋势里不会因为均线本身被拉高而提前出场。
      dual     —— ma 信号 且 均线向上（斜率双确认）。更严格，出场更早、
                  在场时间更短，适合想压回撤的场景。

    band        : 滞回带（相对幅度）。0.02 表示 ±2% 的缓冲区。
    ma_slope    : dual 模式里判断「均线向上」的回看天数，0 表示自动取 lookback/10。
    min_exposure: 出场时的最低仓位。设 0.3 表示「看空也只减到三成」，
                  适合长期看好某个池子、只想削峰填谷的场景。
    """
    if mode not in TIMING_MODES:
        raise ValueError(f"timing mode 必须是 {TIMING_MODES} 之一，收到 {mode!r}")
    if mode == "off":
        return pd.Series(max_exposure, index=proxy.index)

    if mode == "momentum":
        signal = proxy.pct_change(lookback)
        state = _hysteresis(signal, band)
    else:
        ma = proxy.rolling(lookback, min_periods=lookback).mean()
        signal = proxy / ma - 1.0
        state = _hysteresis(signal, band)
        if mode == "dual":
            win = ma_slope if ma_slope > 0 else max(5, lookback // 10)
            slope = ma - ma.shift(win)
            up = (slope > 0).astype(float)
            up[slope.isna()] = 0.0          # 均线历史不足 → 不予确认
            state = state * up

    return state.clip(lower=min_exposure, upper=max_exposure)


# --------------------------------------------------------- 波动率目标

def vol_target_exposure(proxy: pd.Series, target_vol: float = 0.15,
                        lookback: int = 60, floor: float = 0.2, cap: float = 1.0,
                        periods_per_year: int = 252) -> pd.Series:
    """波动率目标敞口：仓位 = 目标波动率 / 已实现年化波动率。

    波动率 20%、目标 15% → 仓位 75%。波动率 40% → 仓位 37.5%（被 floor 抬到 20%）。
    不判断方向，纯粹按风险预算缩放。

    参数
    ----
    lookback : 已实现波动率的滚动窗口（交易日）。
    floor    : 仓位下限。设为 0 就是「极端行情可以完全空仓」——但那样它就不再是
               风险控制而是择时了，回测里也容易变成「在最低点清仓」的巧合。
               默认 0.2。
    cap      : 仓位上限（>1 即允许加杠杆，本工具默认不放大，取 1.0）。

    预热期（滚动窗口不足）视作不做去杠杆，取 cap，与「无择时」基线一致，
    避免在数据不足时静默地压低仓位而让人误以为是择时的功劳。
    """
    if target_vol <= 0:
        raise ValueError("target_vol 必须为正")
    ret = proxy.pct_change()
    min_periods = max(10, lookback // 3)
    rv = ret.rolling(lookback, min_periods=min_periods).std() * np.sqrt(periods_per_year)
    rv = rv.replace(0.0, np.nan)            # 波动率为 0（长期停牌）→ 不参与计算
    ex = (target_vol / rv).clip(lower=floor, upper=cap)
    return ex.fillna(cap)


# --------------------------------------------------------- 组装

def smooth_exposure(exposure: pd.Series, window: int) -> pd.Series:
    """对敞口做移动平均平滑，降低仓位自身的换手。

    注意这会引入一点点滞后（用过去 window 天的平均），但窗口内全是已实现数据，
    不构成前视偏差——只是「反应慢一点」，不代表用了未来信息。
    """
    if window is None or window <= 1:
        return exposure
    return exposure.rolling(window, min_periods=1).mean()


def build_exposure(cfg, proxy: pd.Series | None = None,
                   prices: pd.DataFrame | None = None,
                   proxy_label: str = "",
                   verbose: bool = True):
    """按统一配置口径构造敞口序列；未启用择时时返回 None。

    cfg 需提供（缺失取默认）：
      timing            : off / ma / momentum / dual
      timing_lookback   : 趋势窗口，默认 120
      timing_band       : 滞回带，默认 0.0
      timing_ma_slope   : dual 模式的斜率窗口，默认 0（自动）
      timing_min_exposure / timing_max_exposure : 默认 0.0 / 1.0
      timing_smooth     : 敞口平滑窗口，默认 0
      vol_target        : 目标年化波动率，0 表示不启用，默认 0.0
      vol_target_lookback : 默认 60
      vol_floor / vol_cap : 默认 0.2 / 1.0

    proxy 缺省时用 prices 构造池子等权组合。
    """
    mode = (_cfg_get(cfg, "timing", "off") or "off").strip().lower()
    target_vol = float(_cfg_get(cfg, "vol_target", 0.0) or 0.0)
    if mode in ("", "off", "none") and target_vol <= 0:
        return None

    if proxy is None:
        if prices is None:
            raise ValueError("未提供 proxy 时必须提供 prices（用于构造池子等权组合）")
        proxy = pool_index(prices)
        proxy_label = proxy_label or "池子等权组合"

    parts: list[str] = []
    exposure: pd.Series | None = None

    if mode not in ("", "off", "none"):
        lb = int(_cfg_get(cfg, "timing_lookback", 120))
        band = float(_cfg_get(cfg, "timing_band", 0.0) or 0.0)
        ex = trend_exposure(
            proxy, lookback=lb, mode=mode, band=band,
            ma_slope=int(_cfg_get(cfg, "timing_ma_slope", 0) or 0),
            min_exposure=float(_cfg_get(cfg, "timing_min_exposure", 0.0) or 0.0),
            max_exposure=float(_cfg_get(cfg, "timing_max_exposure", 1.0) or 1.0),
        )
        parts.append(f"趋势 {mode}({lb}日,带{band:.1%})")
        exposure = ex

    if target_vol > 0:
        ve = vol_target_exposure(
            proxy, target_vol=target_vol,
            lookback=int(_cfg_get(cfg, "vol_target_lookback", 60)),
            floor=float(_cfg_get(cfg, "vol_floor", 0.2)),
            cap=float(_cfg_get(cfg, "vol_cap", 1.0) or 1.0),
        )
        parts.append(f"波动率目标 {target_vol:.0%}(下限{float(_cfg_get(cfg,'vol_floor',0.2)):.0%})")
        exposure = ve if exposure is None else exposure * ve

    if exposure is None:
        return None

    exposure = exposure.clip(lower=0.0)
    smooth = int(_cfg_get(cfg, "timing_smooth", 0) or 0)
    if smooth > 1:
        exposure = smooth_exposure(exposure, smooth)
        parts.append(f"平滑{smooth}日")

    exposure = exposure.rename("exposure")
    if verbose:
        # 这里只报「配置」。平均仓位 / 切换次数留给回测后的指标区——
        # 那边算的是评估期，与全序列（含预热期）口径不同，两处都打印会互相矛盾。
        print(f"  择时仓位    : {' × '.join(parts)}"
              f" | 代理={proxy_label or '给定序列'}")
    return exposure


def exposure_summary(exposure: pd.Series) -> dict:
    """敞口序列的概括统计，用于诊断输出与报告指标。"""
    if exposure is None or len(exposure) == 0:
        return {"avg_exposure": 1.0, "in_market_ratio": 1.0, "switches": 0,
                "min_exposure": 1.0, "max_exposure": 1.0}
    ex = exposure.astype(float)
    # 「在场」定义为仓位过半——0.5 是趋势开关（0/1）与连续缩放的分界
    in_market = float((ex > 0.5).mean())
    switches = int((ex.diff().abs() > 1e-9).sum())
    return {
        "avg_exposure": float(ex.mean()),
        "in_market_ratio": in_market,
        "switches": switches,
        "min_exposure": float(ex.min()),
        "max_exposure": float(ex.max()),
    }
