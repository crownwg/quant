"""风险归因：把策略收益按「市场状态」拆分，看策略到底靠什么行情吃饭。

为什么要做这件事
----------------
全样本「总收益」「夏普」是高度聚合的数字，容易被某一段大行情撑起来，
掩盖「其余时间都在亏」的真相（阶段5 的滚动回测已经暴露过这个问题）。
风险归因换一个切面：不问「从哪一年开始」，而问「市场在牛 / 熊 / 震荡时，
策略分别表现如何」。

市场状态怎么定
--------------
用基准指数的「中期趋势」划分每个交易日：
  - 牛 (bull)   ：基准过去 window 日累计收益 >= +band
  - 熊 (bear)   ：基准过去 window 日累计收益 <= -band
  - 震荡 (side) ：介于两者之间
默认 window=60（≈3 个月）、band=±5%。这是行业里最朴素的「趋势状态」代理，
不依赖任何未来信息，纯靠已经走出来的中期涨跌幅。

归因口径
--------
对每种状态，独立统计（只看「市场处于该状态时」的那部分交易日）：
  - 天数 / 占样本交易日比例
  - 策略累计收益：该状态下每日 portfolio_return 复利
  - 基准累计收益：同期买入持有基准的复利
  - 超额收益 = 策略 - 基准
  - 夏普：状态内日收益年化（periods_per_year=252）
  - 日胜率：状态内正收益交易日占比
  - 收益贡献度：用「对数分解」让各状态贡献之和恰好 = 100%
    （复利不可简单相加，ln(1+总收益)=Σ ln(1+各状态收益)，故按此比例拆分）
"""

from __future__ import annotations

import math

import pandas as pd

REGIME_LABELS = {"bull": "牛市", "bear": "熊市", "side": "震荡"}
REGIME_ORDER = ["bull", "bear", "side"]


def market_regime(bench_close: pd.Series, window: int = 60, band: float = 0.05) -> pd.Series:
    """返回与 bench_close 同索引的状态 Series（'bull' / 'bear' / 'side'）。

    bench_close : 基准指数收盘价（按日期索引，可含预热期以让早期判定更准确）。
    window      : 判定窗口（交易日），默认 60。
    band        : 牛熊阈值，默认 0.05（±5%）。
    """
    ret = bench_close.pct_change().fillna(0.0)
    # 用 min_periods=1 让最前面几天也有判定（用可得历史），避免大段误判为震荡
    trailing = ret.rolling(window, min_periods=1).sum()
    regime = pd.Series("side", index=bench_close.index, dtype=object)
    regime.loc[trailing >= band] = "bull"
    regime.loc[trailing <= -band] = "bear"
    return regime


def regime_attribution(equity: pd.Series, detail: pd.DataFrame,
                       benchmark_equity: pd.Series, regime: pd.Series,
                       periods_per_year: int = 252) -> pd.DataFrame:
    """核心归因：返回每状态一行的绩效表。

    参数
    ----
    equity           : 策略净值（日，按日期索引）
    detail           : backtest.run 返回的明细，需含 'portfolio_return'
    benchmark_equity : 基准净值（日，与 equity 同索引对齐）
    regime           : market_regime 产出的状态序列（与 equity 同索引对齐）
    """
    port = detail["portfolio_return"]
    bench_ret = benchmark_equity.pct_change().fillna(0.0)

    total_ret = float(equity.iloc[-1] - 1)
    # 对数分解的分母；收益接近 0 或 <= -100% 时退化成按天数占比分配
    total_log = math.log1p(max(total_ret, -0.9999))
    n = len(port)
    if n == 0:
        return pd.DataFrame()

    rows = []
    for state in REGIME_ORDER:
        mask = regime == state
        sub_port = port[mask]
        days = int(len(sub_port))
        if days == 0:
            continue
        strat_ret = float((1.0 + sub_port).prod() - 1.0)
        bench_cum = float((1.0 + bench_ret[mask]).prod() - 1.0)
        excess = strat_ret - bench_cum
        vol = sub_port.std()
        sharpe = (float(sub_port.mean() / vol * math.sqrt(periods_per_year))
                  if (vol and vol == vol and vol > 0) else 0.0)
        win = float((sub_port > 0).mean())
        avg_daily = float(sub_port.mean())
        if abs(total_log) > 1e-9 and strat_ret > -0.9999:
            contrib = math.log1p(strat_ret) / total_log * 100.0
        else:
            contrib = days / n * 100.0
        rows.append({
            "regime": state,
            "regime_label": REGIME_LABELS[state],
            "days": days,
            "day_ratio": days / n,
            "strategy_return": strat_ret,
            "benchmark_return": bench_cum,
            "excess_return": excess,
            "sharpe": sharpe,
            "win_rate": win,
            "avg_daily_return": avg_daily,
            "contribution_pct": contrib,
        })
    return pd.DataFrame(rows)
