"""组合构建：把「打分表」变成「每日目标权重矩阵」。

核心职责四件事：
  1. 决定什么时候调仓（月度 / 周度 / 季度）
  2. 调仓日按分数选前 top_n 名
  3. 按指定方案分配权重（等权 / 波动率倒数 / 分数加权 / 排名加权）
  4. 施加约束：单票权重上限、流动性容量上限、涨跌停停牌可执行性

关于权重矩阵的构造，有一个极易踩的坑，这里显式处理了：
    weights 必须用 NaN 初始化，调仓日整行显式写 0（非 NaN），再覆盖入选股票。
    原因见 factor_weights 内的注释。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------ 权重分配方案

WEIGHTING_SCHEMES = ("equal", "inv_vol", "score", "rank")


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


def _allocate(picks: list[str], scores: pd.Series, vol_row: pd.Series | None,
              scheme: str) -> pd.Series:
    """在入选股票之间分配权重（未施加约束，权重之和为 1）。

    scheme:
      equal   —— 等权。最稳、最不容易过拟合，但忽略了个股风险差异。
      inv_vol —— 波动率倒数加权，即「风险平价」在忽略相关性时的实用近似：
                 低波动票拿更多权重，组合的事前波动更均衡。
                 学术上严格的风险平价需要协方差矩阵，等相关的假设下就退化成
                 波动率倒数——对月度调仓的低频组合，这个近似足够。
      score   —— 按因子分数（横截面 z-score 后平移到正）加权，
                 分数越高拿越多，保留了分数的**幅度**信息。
      rank    —— 按排名线性递减加权（第 1 名 n 份、第 2 名 n-1 份……），
                 只用**次序**信息，抗极端分数。
    """
    n = len(picks)
    if n == 0:
        return pd.Series(dtype=float)
    if scheme == "equal" or scheme not in WEIGHTING_SCHEMES:
        return pd.Series(1.0 / n, index=picks)

    if scheme == "inv_vol":
        if vol_row is None:
            return pd.Series(1.0 / n, index=picks)
        iv = vol_row.reindex(picks).astype(float)
        iv = iv.where(iv > 0)
        if iv.notna().sum() == 0:
            return pd.Series(1.0 / n, index=picks)   # 全缺波动率 → 退回等权
        iv = iv.fillna(iv.median())                  # 个别缺失用中位数补，避免丢票
        # ⚠️ 必须取倒数！写成 iv/iv.sum() 会变成「波动率越大权重越大」，
        # 与风险平价的方向完全相反（真实踩过：高波动票拿到 57% 仓位）。
        iv = 1.0 / iv
        return iv / iv.sum()

    s = scores.reindex(picks).astype(float)

    if scheme == "rank":
        # 分数越高名次越靠前 → 权重越大
        order = s.sort_values(ascending=False).index
        w = pd.Series(np.arange(n, 0, -1, dtype=float), index=order)
        return w / w.sum()

    # scheme == "score"
    if s.isna().all() or s.std() == 0:
        return pd.Series(1.0 / n, index=picks)
    z = (s - s.mean()) / s.std()
    shifted = z - z.min() + 1.0        # 平移到 [1, ...]，保证全正
    return shifted / shifted.sum()


def _apply_weight_caps(w: pd.Series, caps: pd.Series) -> pd.Series:
    """迭代施加单票权重上限：超限部分按比例再分配给**未触顶**的标的。

    迭代是必需的——把超额部分分给别人之后，别人可能因此也触顶，
    需要再来一轮。若所有标的都触顶，剩余权重就留作现金（不强行满仓）。

    等价于「注水法」：这是带上限约束下最接近原始权重的可行解。

    语义约定：caps 里的 NaN 表示**无约束**（数据缺失，不该因此剔除标的），
    0 表示**不可持有**（容量为零），负值按 0 处理。
    """
    w = w.astype(float).copy()
    caps = caps.reindex(w.index).astype(float).clip(lower=0.0)
    for _ in range(50):
        over = caps.notna() & (w > caps + 1e-12)
        if not over.any():
            break
        excess = float((w[over] - caps[over]).sum())
        w[over] = caps[over]
        free = w.index[~over]
        free_sum = float(w[free].sum())
        if len(free) == 0 or free_sum <= 1e-12:
            break                               # 全员触顶 → 剩余留现金
        w[free] = w[free] + excess * w[free] / free_sum
    # 兜底：极端情况下（上限之和 < 1 且无标的可吸）仍可能超限，最后硬截一次
    w = w.clip(upper=caps.fillna(np.inf))
    return w


def factor_weights(score: pd.DataFrame, top_n: int = 10, freq: str = "M",
                   min_names: int = 1, buffer: int = 0,
                   can_buy: pd.DataFrame | None = None,
                   can_sell: pd.DataFrame | None = None,
                   exec_shift: int = 0,
                   weighting: str = "equal",
                   vol: pd.DataFrame | None = None,
                   max_weight: float = 0.0,
                   weight_cap: pd.DataFrame | None = None) -> pd.DataFrame:
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
    weighting : 权重分配方案，见 _allocate 的说明。
    vol      : date × code 波动率面板，仅 weighting="inv_vol" 时需要。
    max_weight : 单票权重上限（如 0.2 表示任何一只不超过 20%）。0 表示不限。
                 集中度风险的主要来源就是等权下的「选 5 只 = 每只 20%」，
                 以及纯分数加权下的「第一名拿 40%」。
    weight_cap : date × code 的容量上限面板（来自流动性约束，见 filters.capacity_cap）。
                 与 max_weight 取小生效。

    返回
    ----
    date × code 的权重矩阵。**只在调仓日发生变化**，非调仓日 ffill 沿用；
    每行之和通常等于 1，受约束时可能小于 1（买不进的票 / 触顶部分持有现金）。

    关键设计（修复历史 bug）
    ------------------------
    可行性约束（涨跌停/停牌）只在「调仓日当天」生效，不做逐日迭代：
      - 调仓日想买但买不进（涨停/停牌）→ 该票当月权重置 0（现金），下月再评估；
      - 调仓日想清仓但卖不出（跌停）→ 保留上期权重继续持有。
    这样权重矩阵在非调仓日恒定，调仓次数不会被「一次调仓拆散到多天」而虚高。
    （旧版 apply_tradability 逐日 min/max 迭代会把单次调仓拆成多次，已废弃。）
    """
    if weighting not in WEIGHTING_SCHEMES:
        raise ValueError(f"weighting 必须是 {WEIGHTING_SCHEMES} 之一，收到 {weighting!r}")

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

        if len(picks) >= min_names:
            vol_row = vol.loc[date] if vol is not None else None
            # ⚠️ 必须 reindex 到完整列再 fillna(0)：
            # _allocate 只返回入选股票的权重，若直接赋值给 weights 面板，
            # 未入选的列会变成 NaN；而末尾的 ffill 会把 NaN 理解成「沿用上期持仓」，
            # 结果就是永远不卖出（真实踩过：某次回测 13 次调仓只有买入、0 笔卖出）。
            w_date = _allocate(picks, row, vol_row, weighting).reindex(score.columns).fillna(0.0)
        else:
            w_date = pd.Series(0.0, index=score.columns)

        # ---- 约束 1：单票上限 + 容量上限（取小）----
        caps: pd.Series | None = None
        if max_weight and max_weight > 0:
            caps = pd.Series(max_weight, index=w_date.index)
        if weight_cap is not None:
            row_cap = weight_cap.loc[date].reindex(w_date.index)
            caps = row_cap if caps is None else pd.concat([caps, row_cap], axis=1).min(axis=1)
        if caps is not None and len(w_date):
            w_date = _apply_weight_caps(w_date, caps)

        # ---- 约束 2：可执行性（仅本调仓日生效，不跨日迭代）----
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


def realized_vol(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """已实现波动率面板（日收益标准差），供 weighting="inv_vol" 使用。"""
    return prices.pct_change().rolling(lookback).std()


def _cfg_get(cfg, key, default):
    """同时兼容 argparse 命名空间与 dict 配置（daily.py 用的是 JSON dict）。"""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def weights_from_args(score: pd.DataFrame, cfg, can_buy=None, can_sell=None,
                      prices: pd.DataFrame | None = None, vol: pd.DataFrame | None = None,
                      weight_cap: pd.DataFrame | None = None, **overrides) -> pd.DataFrame:
    """按统一的配置口径构造权重矩阵 —— 所有入口都应该走这里。

    为什么要抽这一层：main / grid / compare / combine / rolling / daily 六个入口
    各自拼一遍 factor_weights 的参数，历史上已经造成过口径分裂
    （典型：主回测接上了 exec_shift，滚动回测忘了接）。
    新增的 weighting / max_weight / weight_cap 如果各自手接一遍，同样的坑会再犯一次。

    cfg 可以是 argparse 命名空间，也可以是 daily.py 的 JSON dict。
    overrides 用于网格搜索这类需要临时改参的场景。
    """
    weighting = overrides.get("weighting", _cfg_get(cfg, "weighting", "equal"))
    if weighting == "inv_vol" and vol is None and prices is not None:
        vol = realized_vol(prices, _cfg_get(cfg, "vol_lookback", 60))

    kw = dict(
        top_n=_cfg_get(cfg, "top_n", 10),
        freq=_cfg_get(cfg, "rebalance", "M"),
        min_names=_cfg_get(cfg, "min_names", 1),
        buffer=_cfg_get(cfg, "buffer", 0),
        can_buy=can_buy,
        can_sell=can_sell,
        exec_shift=1 if _cfg_get(cfg, "use_open", False) else 0,
        weighting=weighting,
        vol=vol,
        max_weight=_cfg_get(cfg, "max_weight", 0.0),
        weight_cap=weight_cap,
    )
    kw.update(overrides)
    return factor_weights(score, **kw)



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
