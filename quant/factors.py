"""因子库：把价格 / 成交量面板转换成可横截面比较的「打分表」。

约定
----
每个因子函数返回 date × code 的 DataFrame，**数值越大代表越看好**（方向已内置，
例如反转因子内部已经取负、低波动因子内部已经取负）。

返回 NaN 表示该股票当日「不具备打分资格」，选股时会被自动剔除：
  - 历史不足（上市初期的滚动窗口未填满）
  - 不满足前提条件（如均线趋势要求价格站上长均线）

这样设计的好处：因子只管「打分」，不用关心选几只、什么时候调仓、
能不能买——那些是 strategy.py 的事。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 单因子

def momentum(prices: pd.DataFrame, lookback: int = 120, skip_recent: int = 0) -> pd.DataFrame:
    """动量：过去 lookback 个交易日的涨幅。涨得越多分越高（追涨）。

    skip_recent: 跳过最近 N 个交易日再回看，即「12-1 动量」里的那个 -1。
    学术上用它规避最近一个月的短期反转噪声，实盘常见取 skip_recent=20。
    """
    base = prices.shift(skip_recent) if skip_recent else prices
    return base.pct_change(lookback)


def reversal(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """反转：过去 lookback 日涨幅取负。跌得越惨分越高（赌均值回归）。

    适合震荡市，与动量因子常常互补——可以两个一起放进多因子里对冲。
    """
    return -prices.pct_change(lookback)


def ma_trend(prices: pd.DataFrame, short: int = 20, long: int = 60) -> pd.DataFrame:
    """均线趋势：短均线相对长均线的偏离度。

    额外要求「收盘价站上长均线」，不满足的直接置 NaN 失去入选资格——
    只做多头趋势，不做超跌反弹，避免在下跌通道里反复抄底。
    """
    if short >= long:
        raise ValueError(f"short({short}) 必须小于 long({long})")
    ma_s = prices.rolling(short).mean()
    ma_l = prices.rolling(long).mean()
    trend = ma_s / ma_l - 1
    return trend.where(prices > ma_l)


def ma_breakout(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """均线突破：收盘价相对 N 日均线的偏离度，等价于「距离均线的距离」。

    和 ma_trend 的区别：它衡量单条均线的乖离，趋势确认更温和。
    """
    ma = prices.rolling(window).mean()
    return prices / ma - 1


def low_volatility(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """低波动：过去 lookback 日日收益标准差取负。波动越小分越高。

    低波动异象：长期看低波动股票的风险调整收益往往优于高波动股票。
    """
    return -prices.pct_change().rolling(lookback).std()


def volume_trend(volume: pd.DataFrame, short: int = 5, long: int = 60) -> pd.DataFrame:
    """量能：近 short 日均量 / 近 long 日均量。温和放量得分高。

    注意这是「相对量能」而非绝对成交量——它剔除了个股规模差异，
    可以横向比较；绝对流动性门槛请交给 filters 的 min_volume。
    """
    long_avg = volume.rolling(long).mean().replace(0, np.nan)
    return volume.rolling(short).mean() / long_avg


# ------------------------------------------------------- 横截面标准化

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """横截面 z-score：每个交易日内部 (x - 均值) / 标准差。

    让不同量纲的因子可以直接相加。标准差为 0（当日所有股票因子值相同）
    时整行置 NaN，避免除零产生 inf。
    """
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def rank_pct(df: pd.DataFrame) -> pd.DataFrame:
    """横截面分位排名：把因子值转成 0~1 的名次百分比。

    相比 z-score 更抗极端值（一个涨停不会把整个横截面拉歪），
    实践中多因子合成常用它。
    """
    return df.rank(axis=1, pct=True)


STANDARDIZERS = {"zscore": zscore, "rank": rank_pct}


# --------------------------------------------------------- 多因子合成

def combine(factors: dict[str, pd.DataFrame], weights: dict[str, float],
            method: str = "zscore") -> pd.DataFrame:
    """把多个因子按权重合成为一张总打分表。

    参数
    ----
    factors : {因子名: 分数面板}
    weights : {因子名: 权重}，权重为正表示「该因子值越大越看好」，
              为负表示反向使用。例如 {"momentum": 1.0, "low_volatility": 0.5}
    method  : "zscore"（对量纲敏感，保留分布信息）或 "rank"（抗极端值）

    缺失处理：单个因子缺失按 0（中性）处理，不会污染其他因子；
    但所有因子都缺失的行仍为 NaN，即不参与选股。
    """
    if method not in STANDARDIZERS:
        raise ValueError(f"method 必须是 {sorted(STANDARDIZERS)} 之一，收到 {method!r}")
    if not weights:
        raise ValueError("weights 不能为空")

    standardize = STANDARDIZERS[method]
    total: pd.DataFrame | None = None
    valid: pd.DataFrame | None = None

    for name, w in weights.items():
        if name not in factors:
            raise KeyError(f"因子 {name} 不在因子库中，可选: {sorted(factors)}")
        raw = factors[name]
        scored = standardize(raw) * w
        total = scored if total is None else total.add(scored, fill_value=0.0)
        notna = raw.notna()
        valid = notna if valid is None else (valid | notna)

    if total is None:
        raise ValueError("未能合成任何因子")
    # 所有因子都缺失的行 → 该日该股没有分数 → 不参与选股
    return total.where(valid.any(axis=1), axis=0)
