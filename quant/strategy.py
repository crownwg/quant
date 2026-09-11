"""组合构建：把「打分表」变成「每日目标权重矩阵」。

核心职责三件事：
  1. 决定什么时候调仓（月度 / 周度 / 季度）
  2. 调仓日按分数选前 top_n 名，等权配置
  3. 把目标权重改造成「实际可执行」的权重（受涨跌停 / 停牌约束）

关于权重矩阵的构造，有一个极易踩的坑，这里显式处理了：
    weights 必须用 NaN 初始化，调仓日整行显式写 0（非 NaN），再覆盖入选股票。
    原因见 factor_weights 内的注释。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_flags(index: pd.DatetimeIndex, freq: str = "M") -> pd.Series:
    """调仓日标记：每个周期的第一个交易日为 True。

    freq: "M" 月度（默认，A 股低频策略最常用）
          "W" 周度、"Q" 季度、"2M" 双月 等 pandas 支持的周期字符串
    """
    period = index.to_period(freq.upper())
    # 注意：pandas 3.0 中 PeriodIndex 的 `!=` 比较会返回全 True（破坏性变更），
    # 改用 duplicated 判定「每个周期的首个交易日」更稳健。
    flags = ~period.duplicated(keep="first")
    return pd.Series(flags, index=index).fillna(True)  # 首个交易日也要建仓


def factor_weights(score: pd.DataFrame, top_n: int = 10, freq: str = "M",
                   min_names: int = 1, buffer: int = 0,
                   can_buy: pd.DataFrame | None = None,
                   can_sell: pd.DataFrame | None = None,
                   exec_shift: int = 0) -> pd.DataFrame:
    """按因子分数定期选股，生成每日目标权重矩阵。

    参数
    ----
    score    : date × code 打分表，越大越看好，NaN 表示不具备资格
    top_n    : 每次调仓选前几名
    freq     : 调仓频率
    min_names: 有效候选少于该数量时当天不选股（保持空仓），
               避免只剩 1 只垃圾票时被迫满仓
    buffer   : 换手缓冲。已持有的股票只要排名还在 top_n + buffer 之内就继续持有，
               不因微弱的分差被换掉。0 表示不启用。
               典型取 1~2，能显著降低换手与交易成本。
    can_buy / can_sell : 可执行性面板（涨停/跌停/停牌/流动性）。
    exec_shift : 决策日到**实际成交日**之间隔了几根 K 线。
                 - 按收盘价成交 → 0（信号在 close[t] 算出、当刻成交）
                 - 按次日开盘价成交 → 1（信号在 close[t] 算出、最早 open[t+1] 成交）
                 可执行性面板会按该偏移后移，确保「用成交当日的状态」判断能否下单。
                 不传（默认 0）时，用决策日状态判断——在开盘成交模式下这是前视偏差。

    返回
    ----
    date × code 的权重矩阵。**只在调仓日发生变化**，非调仓日 ffill 沿用；
    每行之和通常等于 1，受可行性约束时可能小于 1（买不进的票当月持有现金）。

    关键设计（修复历史 bug）
    ------------------------
    可行性约束（涨跌停/停牌）只在「调仓日当天」生效，不做逐日迭代：
      - 调仓日想买但买不进（涨停/停牌）→ 该票当月权重置 0（现金），下月再评估；
      - 调仓日想清仓但卖不出（跌停）→ 保留上期权重继续持有。
    这样权重矩阵在非调仓日恒定，调仓次数不会被「一次调仓拆散到多天」而虚高。
    （旧版 apply_tradability 逐日 min/max 迭代会把单次调仓拆成多次，已废弃。）
    """
    if exec_shift:
        # 成交发生在 exec_shift 根之后，因此要取「成交当日」的可交易性。
        # shift(-n) 会把未来的行挪到当前位置；末行无未来可言，保守填 False。
        def _align(mask: pd.DataFrame | None) -> pd.DataFrame | None:
            if mask is None:
                return None
            return mask.shift(-exec_shift).fillna(False).astype(bool)

        can_buy = _align(can_buy)
        can_sell = _align(can_sell)

    flags = rebalance_flags(score.index, freq)

    # NaN 初始化：ffill 只填充 NaN，调仓日显式写值、非调仓日沿用。
    weights = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    held: list[str] = []
    for date in score.index[flags]:
        row = score.loc[date].dropna()
        ranked = row.sort_values(ascending=False)

        # 选股（含换手缓冲）
        picks: list[str] = []
        if buffer and held:
            rank_of = {code: i for i, code in enumerate(ranked.index)}
            survivors = [c for c in held
                         if c in rank_of and rank_of[c] < top_n + buffer]
            picks = survivors[:top_n]
        for code in ranked.index:
            if len(picks) >= top_n:
                break
            if code not in picks:
                picks.append(code)
        picks = picks[:top_n]

        w_date = pd.Series(0.0, index=score.columns)
        if len(picks) >= min_names:
            w_date[picks] = 1.0 / len(picks)

        # ---- 可行性约束（仅本调仓日生效，不跨日迭代）----
        if can_buy is not None or can_sell is not None:
            cb = (can_buy.loc[date].reindex(score.columns).fillna(False)
                  if can_buy is not None else None)
            cs = (can_sell.loc[date].reindex(score.columns).fillna(False)
                  if can_sell is not None else None)
            # 想买但买不进（涨停/停牌）→ 该票当月空仓
            if cb is not None:
                blocked_buy = (w_date > 0) & (~cb)
                w_date[blocked_buy] = 0.0
            # 想卖但卖不出（跌停）→ 保留上期权重继续持有
            if cs is not None and held:
                prev_w = 1.0 / len(held)
                for old in held:
                    if old not in picks and not bool(cs.get(old, True)):
                        w_date[old] = prev_w

        weights.loc[date] = w_date
        held = list(w_date[w_date > 0].index)  # 实际持仓（受约束后）

    # 非调仓日沿用最近一次调仓目标；首次调仓前为空仓
    weights = weights.ffill().fillna(0.0)
    return weights


def monthly_momentum_weights(prices: pd.DataFrame, lookback: int = 120, top_n: int = 10,
                             volume: pd.DataFrame | None = None,
                             min_volume: float = 0.0) -> pd.DataFrame:
    """原始动量策略入口，保留向后兼容。新代码请直接用 factor_weights。"""
    from . import factors  # 局部导入，避免循环依赖

    score = factors.momentum(prices, lookback)
    if volume is not None and min_volume > 0:
        avg_vol = volume.rolling(lookback).mean()
        score = score.where(avg_vol >= min_volume)
    return factor_weights(score, top_n=top_n, freq="M")
