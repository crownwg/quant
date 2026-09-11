"""交易可行性过滤：停牌、涨跌停、流动性不足。

回测最容易自欺的地方，就是假设「想买就能按收盘价买到」。
真实 A 股里至少有四种情况让你买不进 / 卖不出：

  1. 停牌      —— 完全不能交易
  2. 涨停      —— 买单排不上队，基本买不进（但能卖）
  3. 跌停      —— 卖单排不上队，基本卖不出（但能买）
  4. 流动性差  —— 挂单量太小，冲击成本高，实盘难以成交

本模块把上述情况统一成两张布尔面板：can_buy / can_sell。
True = 该股当日该方向可以交易。
"""

from __future__ import annotations

import pandas as pd

# 各板块涨跌停幅度（ST 股为 5%，代码上看不出来，需单独指定）
_CRE_LIMIT = 0.20      # 创业板 300/301、科创板 688
_BJ_LIMIT = 0.30       # 北交所 4/8 开头
_DEFAULT_LIMIT = 0.10  # 主板、中小板
_ST_LIMIT = 0.05       # ST / *ST


def limit_pct_by_code(codes, st_codes=()) -> pd.Series:
    """按股票代码推断涨跌停幅度，返回 index=code 的 Series。"""
    st = {str(c).zfill(6) for c in st_codes}
    out = {}
    for raw in codes:
        code = str(raw).zfill(6)
        if code in st:
            out[code] = _ST_LIMIT
        elif code.startswith(("300", "301", "688")):
            out[code] = _CRE_LIMIT
        elif code.startswith(("4", "8")):
            out[code] = _BJ_LIMIT
        else:
            out[code] = _DEFAULT_LIMIT
    return pd.Series(out)


def suspension_mask(volume: pd.DataFrame) -> pd.DataFrame:
    """停牌 / 未上市：成交量为 0 或缺失。

    依赖 data.load_panel 把缺失的成交量填成 0 而不是 ffill——
    否则停牌日的量会被前一天的成交量顶替，永远检测不到。
    """
    return volume.fillna(0.0) <= 0


def limit_masks(close: pd.DataFrame, limit_pct, tolerance: float = 0.005):
    """涨跌停判定，返回 (涨停面板, 跌停面板)。

    tolerance: 容差。前复权价会让实际涨跌幅偏离整数关口
    （例如 10.05% 或 9.97%），不留容差会漏判。

    limit_pct 可以传 float（全市场统一），也可以传 index=code 的 Series
    （按板块区分，配合 limit_pct_by_code 使用）。
    """
    ret = close.pct_change()
    if isinstance(limit_pct, pd.Series):
        # 按列广播：每只股票用自己的涨跌停幅度
        limit_pct = limit_pct.reindex(close.columns)
        up = ret.ge(limit_pct - tolerance, axis=1)
        down = ret.le(-(limit_pct - tolerance), axis=1)
    else:
        up = ret >= limit_pct - tolerance
        down = ret <= -(limit_pct - tolerance)
    return up.fillna(False), down.fillna(False)


def illiquid_mask(volume: pd.DataFrame, min_volume: float, window: int = 20,
                  min_periods: int = 5) -> pd.DataFrame | None:
    """流动性不足掩码：近 window 日均成交量低于 min_volume 的标记为 True。

    停牌日（成交量为 0）先剔除再算均值，否则一次长期停牌会把均值拉低，
    导致复牌后很长一段时间被误判为流动性不足。
    """
    if min_volume <= 0:
        return None
    # 停牌日（成交量 0）先剔除再算均值，否则一次长期停牌会把均值拉低，
    # 导致复牌后很长一段时间被误判为流动性不足。
    # 注意不能用 replace(0, pd.NA)：pd.NA 会把浮点列变成 object dtype，
    # 后面的 rolling 直接报 "Cannot aggregate non-numeric type"。用 where 保持 float。
    clean = volume.astype(float).where(volume.astype(float) > 0)
    avg = clean.rolling(window, min_periods=min_periods).mean()
    return (avg < min_volume).fillna(True)  # 历史不足的也视为不可交易


def tradability(close: pd.DataFrame, volume: pd.DataFrame,
                limit_pct=0.10, illiquid: pd.DataFrame | None = None,
                enable_limit: bool = True, enable_suspend: bool = True):
    """生成 (can_buy, can_sell) 两张可执行性面板。

    规则：
      停牌      → 买卖都禁
      涨停      → 禁买（能卖）
      跌停      → 禁卖（能买）
      流动性差  → 买卖都禁
    """
    idx, cols = close.index, close.columns
    blocked_buy = pd.DataFrame(False, index=idx, columns=cols)
    blocked_sell = pd.DataFrame(False, index=idx, columns=cols)

    if enable_suspend:
        susp = suspension_mask(volume).reindex(index=idx, columns=cols).fillna(True)
        blocked_buy |= susp
        blocked_sell |= susp

    if enable_limit:
        up, down = limit_masks(close, limit_pct)
        blocked_buy |= up.reindex(index=idx, columns=cols).fillna(False)
        blocked_sell |= down.reindex(index=idx, columns=cols).fillna(False)

    if illiquid is not None:
        il = illiquid.reindex(index=idx, columns=cols).fillna(True)
        blocked_buy |= il
        blocked_sell |= il

    return ~blocked_buy, ~blocked_sell
