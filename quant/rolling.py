"""稳健性检验：分年度绩效分解 + 滚动 / walk-forward 样本外回测。

核心回答一个问题：策略漂亮的净值曲线，是靠「某一两段行情」撑起来的，
还是「不管从哪一年开始都能赚钱」？

- annual_breakdown：把全样本净值按自然年切开，看每年独立表现。
- run_rolling     ：把样本切成若干不重叠窗口（walk-forward），每段独立回测，
                    拼接成一条「持续投资」的净值，并给出每段绩效。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import run, cost_kwargs
from .strategy import factor_weights, weights_from_args


def annual_breakdown(equity: pd.Series, detail: pd.DataFrame,
                     periods_per_year: int = 252) -> pd.DataFrame:
    """按自然年切分，返回每年一行的绩效表。

    列：year, trading_days, return, sharpe, max_drawdown, annual_volatility
    return 为该年首末净值之比（含当年所有交易日）。
    """
    rows = []
    years = equity.index.year
    for year in sorted(set(years)):
        mask = years == year
        sub_eq = equity[mask]
        sub_det = detail[mask]
        if len(sub_eq) < 2:
            continue
        ret = float(sub_eq.iloc[-1] / sub_eq.iloc[0] - 1)
        port = sub_det["portfolio_return"]
        vol = port.std()
        sharpe = float(port.mean() / vol * np.sqrt(periods_per_year)) if vol and not np.isnan(vol) else 0.0
        max_dd = float(sub_det["drawdown"].min())
        rows.append({
            "year": int(year),
            "trading_days": int(len(port)),
            "return": ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "annual_volatility": float(vol * np.sqrt(periods_per_year)) if vol else 0.0,
        })
    return pd.DataFrame(rows)


def rolling_windows(n: int, window: int = 504, step: int | None = None,
                   min_periods: int = 252):
    """生成不重叠（或步进）窗口的 [start, end) 行索引区间列表。

    window: 每段长度（交易日）
    step  : 步进；默认 = window（不重叠 walk-forward）。
            若想做成滑动重叠窗口，可传更小值（如 window=504, step=252）。
    min_periods: 段长不足此值则停止。
    """
    if step is None:
        step = window
    windows = []
    start = 0
    while start + min_periods <= n:
        end = min(start + window, n)
        windows.append((start, end))
        start += step
    return windows


def run_rolling(score, prices, open_prices, can_buy, can_sell, args, windows,
                weight_cap=None):
    """对每段窗口独立回测，返回 [(start_date, end_date, equity, metrics), ...]。

    每段净值从上一窗口末值衔接，模拟「一直按策略投资」的连续曲线；
    段内收益 / 成本独立计算，互不串扰。
    """
    results = []
    prev_end = 1.0
    for (s, e) in windows:
        sc = score.iloc[s:e]
        pr = prices.iloc[s:e]
        op = open_prices.iloc[s:e] if open_prices is not None else None
        cb = can_buy.iloc[s:e] if can_buy is not None else None
        cs = can_sell.iloc[s:e] if can_sell is not None else None
        w = weights_from_args(sc, args, can_buy=cb, can_sell=cs, prices=pr,
                              weight_cap=(weight_cap.iloc[s:e] if weight_cap is not None else None))
        eq, m, _det = run(pr, w, open_prices=op, **cost_kwargs(args))
        eq = eq * prev_end
        prev_end = float(eq.iloc[-1])
        results.append((eq.index[0], eq.index[-1], eq, m))
    return results


def concat_equity(results):
    """把各段净值拼成一条连续曲线（已按 prev_end 衔接，直接 concat 即可）。"""
    return pd.concat([r[2] for r in results])
