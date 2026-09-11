"""交易可行性过滤：停牌、涨跌停、流动性不足、次新股（上市未满）。

回测最容易自欺的地方，就是假设「想买就能按目标价买到」。
真实 A 股里至少有五种情况让你买不进 / 卖不出：

  1. 停牌      —— 完全不能交易
  2. 涨停      —— 买单排不上队，基本买不进（但能卖）
  3. 跌停      —— 卖单排不上队，基本卖不出（但能买）
  4. 流动性差  —— 挂单量太小，冲击成本高，实盘难以成交
  5. 次新股    —— 上市初期连续涨停 / 无涨跌停约束期，价格不可信

本模块把上述情况统一成两张布尔面板：can_buy / can_sell。
True = 该股当日该方向可以交易。

涨跌停判定为什么必须用「价格」而不是「涨跌幅」
--------------------------------------------
旧实现用 `close.pct_change()` 与 9.5% 阈值比较，有两个漏判：
  - 前复权会让实际涨跌幅偏离整数关口（9.97% / 10.05%），阈值法要么漏要么误；
  - 无法区分「一字板」与「盘中触板」——前者一定买不进，后者往往买得进。
现在改为：由前收盘价按板块幅度推出涨跌停**价**，再用 open/high/low/close
与这个价格比较，从而精确识别：
  - 一字涨停（open 与 low 均贴在涨停价）→ 全天买不进
  - 收盘封涨停（close 贴在涨停价）      → 收盘时刻买不进（保守假设）
停牌日价量全为 0 / 缺失，不会误判。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 各板块涨跌停幅度（ST 股为 5%，代码上看不出来，需单独指定）
_CRE_LIMIT = 0.20      # 创业板 300/301、科创板 688/689
_BJ_LIMIT = 0.30       # 北交所 920xxx / 43xxxx / 83xxxx / 87xxxx / 88xxxx
_DEFAULT_LIMIT = 0.10  # 主板、中小板
_ST_LIMIT = 0.05       # ST / *ST

# 涨跌停价的默认比对容差（相对前收盘价）。
# 复权后的价格带浮点误差，且交易所按分位四舍五入，留 0.2% 容差避免漏判。
_DEFAULT_TOLERANCE = 0.002


def limit_pct_by_code(codes, st_codes=()) -> pd.Series:
    """按股票代码推断涨跌停幅度，返回 index=code 的 Series。"""
    st = {str(c).zfill(6) for c in st_codes}
    out = {}
    for raw in codes:
        code = str(raw).zfill(6)
        if code in st:
            out[code] = _ST_LIMIT
        elif code.startswith(("300", "301", "688", "689")):
            out[code] = _CRE_LIMIT
        elif code.startswith("920") or code.startswith(("4", "8")):
            out[code] = _BJ_LIMIT
        else:
            out[code] = _DEFAULT_LIMIT
    return pd.Series(out)


def _broadcast_pct(limit_pct, columns) -> pd.DataFrame | float:
    """把 limit_pct 统一成可对 close 面板逐列广播的形式。"""
    if isinstance(limit_pct, pd.Series):
        return limit_pct.reindex(columns)
    return float(limit_pct)


def _round_tick(x: pd.DataFrame) -> pd.DataFrame:
    """按 A 股最小报价单位 0.01 元四舍五入（交易所用的就是四舍五入）。"""
    return np.floor(x * 100 + 0.5) / 100


def limit_prices(close: pd.DataFrame, limit_pct=0.10):
    """由前收盘价推涨停价 / 跌停价，返回 (涨停价面板, 跌停价面板)。

    limit_pct 可以是 float（全市场统一）或 index=code 的 Series
    （配合 limit_pct_by_code 按板块区分）。
    首行没有前收盘价，涨跌停价为空（NaN）。
    """
    prev = close.shift(1)
    pct = _broadcast_pct(limit_pct, close.columns) if isinstance(limit_pct, pd.Series) else float(limit_pct)
    up = _round_tick(prev * (1 + pct))
    down = _round_tick(prev * (1 - pct))
    return up, down


def suspension_mask(volume: pd.DataFrame) -> pd.DataFrame:
    """停牌 / 未上市：成交量为 0 或缺失。

    依赖 data.load_panel 把缺失的成交量填成 0 而不是 ffill——
    否则停牌日的量会被前一天的成交量顶替，永远检测不到。
    """
    return volume.fillna(0.0) <= 0


def limit_masks(close: pd.DataFrame, limit_pct=0.10, tolerance: float = _DEFAULT_TOLERANCE,
                open_: pd.DataFrame | None = None, high: pd.DataFrame | None = None,
                low: pd.DataFrame | None = None):
    """涨跌停判定，返回四个布尔面板：

      (一字涨停, 收盘封涨停, 一字跌停, 收盘封跌停)

    含义：
      - 一字涨停：开盘即封死，全天最低价都没跌破涨停价 → 全天没有可成交的卖单，**绝对买不进**
      - 收盘封涨停：收盘价贴在涨停价 → 收盘时点买不进（保守起见默认也禁买）
      - 一字跌停：开盘即封死，全天最高价都没涨破跌停价 → **绝对卖不出**
      - 收盘封跌停：收盘价贴在跌停价 → 收盘时点卖不出

    注意「一字」的判据是「全天不曾离开限价」，因此涨停看 low（不能跌破）、
    跌停看 high（不能涨破）。这两个方向不能用同一个字段，否则会把
    「开盘跌停后拉起」误判成一字跌停。

    open_/high/low 缺省时退化为「只用收盘价」的保守判定：
    此时一字板无法识别，会与「收盘封板」合并计为封板。

    tolerance 为相对前收盘价的容差，用于吸收复权浮点误差与分位取整误差。
    """
    up, down = limit_prices(close, limit_pct)
    tol = (close.shift(1).abs() * tolerance).fillna(0.0)

    close_up = (close >= up - tol).fillna(False)
    close_down = (close <= down + tol).fillna(False)

    if open_ is not None and high is not None and low is not None:
        open_ = open_.reindex(index=close.index, columns=close.columns)
        high = high.reindex(index=close.index, columns=close.columns)
        low = low.reindex(index=close.index, columns=close.columns)
        # 一字涨停：开盘即封死，且全天最低价都没跌破涨停价（=既没开板也没回落）
        one_word_up = ((open_ >= up - tol) & (low >= up - tol)).fillna(False)
        # 一字跌停：开盘即封死，且全天最高价都没涨破跌停价。
        # ⚠️ 这里必须用 high 而不是 low：用 low 会把「开盘跌停后拉起」的日子
        #    （如 2020-02-04 的 002157，开盘 10.57 跌停、最高 11.68、收 11.48）
        #    误判成一字跌停，进而错误地禁止卖出。
        one_word_down = ((open_ <= down + tol) & (high <= down + tol)).fillna(False)
    else:
        one_word_up = pd.DataFrame(False, index=close.index, columns=close.columns)
        one_word_down = pd.DataFrame(False, index=close.index, columns=close.columns)

    return one_word_up, close_up, one_word_down, close_down


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


def listing_age_mask(prices: pd.DataFrame, min_days: int = 0) -> pd.DataFrame | None:
    """次新股掩码：上市未满 min_days 个自然日的标的标记为 False（不可选）。

    为什么要剔除次新股：A 股新股上市初期往往连续一字涨停、且前 5 个交易日不设
    涨跌停限制，价格严重失真；这段时间的动量因子会被极值污染，实盘也根本买不进。

    ⚠️ 依赖调用方把面板起点设得足够早（回测起点再往前推 min_days），
    否则「老股票的可用历史」会被误判成「刚上市」。main.py 已按此调整预热期。
    """
    if min_days <= 0:
        return None
    idx = prices.index
    cols = {}
    for code in prices.columns:
        first = prices[code].first_valid_index()
        if first is None:
            cols[code] = False
        else:
            cols[code] = idx >= (first + pd.Timedelta(days=min_days))
    return pd.DataFrame(cols, index=idx).fillna(False)


def tradability(close: pd.DataFrame, volume: pd.DataFrame,
                limit_pct=0.10, illiquid: pd.DataFrame | None = None,
                enable_limit: bool = True, enable_suspend: bool = True,
                open_: pd.DataFrame | None = None, high: pd.DataFrame | None = None,
                low: pd.DataFrame | None = None, exec_at_open: bool = False,
                tolerance: float = _DEFAULT_TOLERANCE,
                stats: dict | None = None):
    """生成 (can_buy, can_sell) 两张可执行性面板。

    规则：
      停牌                    → 买卖都禁
      流动性差                → 买卖都禁

      按**收盘价**成交（exec_at_open=False）时，判定同样基于收盘价：
        收盘封涨停 / 一字涨停 → 禁买（能卖）
        收盘封跌停 / 一字跌停 → 禁卖（能买）

      按**开盘价**成交（exec_at_open=True）时，判定只基于开盘价：
        开盘即涨停价 → 禁买；开盘即跌停价 → 禁卖
        （此时**不看收盘价**——收盘还没发生，用它判定就是前视偏差）

    参数
    ----
    exec_at_open : 回测是否按次日**开盘价**成交（对应 main.py 的 --use-open）。
                   传 True 时必须同时给 open_，否则退化为收盘价判定并给出提示。
    stats        : 传入 dict 时回填各类涨跌停的命中次数，便于诊断。

    关于口径一致（这是本函数最容易出错的地方）：
    判定所用的价格必须与成交价是同一个时点。修复前统一用收盘价判定，
    在 --use-open 下等于「用一个买入时还没发生的信息决定能不能下单」，
    会把「开盘一字板买不进」的日子漏掉、又把「开盘正常但收盘封板」的日子错杀。
    """
    idx, cols = close.index, close.columns
    blocked_buy = pd.DataFrame(False, index=idx, columns=cols)
    blocked_sell = pd.DataFrame(False, index=idx, columns=cols)
    suspend_buy = suspend_sell = limit_buy = limit_sell = illiquid_mask_panel = None

    if enable_suspend:
        susp = suspension_mask(volume).reindex(index=idx, columns=cols).fillna(True)
        blocked_buy |= susp
        blocked_sell |= susp
        suspend_buy = suspend_sell = susp

    if enable_limit:
        use_open = exec_at_open and open_ is not None
        if exec_at_open and open_ is None:
            print("⚠️ exec_at_open=True 但未提供开盘价面板，涨跌停判定退回收盘价口径")

        # 一字板统计两种模式下都有意义，先统一算出来
        one_up, close_up, one_down, close_down = limit_masks(
            close, limit_pct, tolerance=tolerance, open_=open_, high=high, low=low)

        if use_open:
            open_px = open_.reindex(index=idx, columns=cols)
            up, down = limit_prices(close, limit_pct)
            tol = (close.shift(1).abs() * tolerance).fillna(0.0)
            lim_buy = (open_px >= up - tol).fillna(False)
            lim_sell = (open_px <= down + tol).fillna(False)
        else:
            lim_buy = close_up
            lim_sell = close_down

        # 分开累计，便于诊断时区分「涨跌停挡的」和「停牌/流动性挡的」
        limit_buy, limit_sell = lim_buy, lim_sell
        blocked_buy |= lim_buy
        blocked_sell |= lim_sell

        if stats is not None:
            stats["held_price"] = "open" if use_open else "close"
            stats["limit_up_oneword"] = int(one_up.sum().sum())
            stats["limit_up_close"] = int(close_up.sum().sum())
            stats["limit_down_oneword"] = int(one_down.sum().sum())
            stats["limit_down_close"] = int(close_down.sum().sum())
            stats["limit_blocked_buy"] = int(lim_buy.sum().sum())
            stats["limit_blocked_sell"] = int(lim_sell.sum().sum())

    if illiquid is not None:
        il = illiquid.reindex(index=idx, columns=cols).fillna(True)
        blocked_buy |= il
        blocked_sell |= il
        illiquid_mask_panel = il

    if stats is not None:
        stats["blocked_buy"] = int(blocked_buy.sum().sum())
        stats["blocked_sell"] = int(blocked_sell.sum().sum())
        if suspend_buy is not None:
            stats["suspend_blocked"] = int(suspend_buy.sum().sum())
        if illiquid_mask_panel is not None:
            stats["illiquid_blocked"] = int(illiquid_mask_panel.sum().sum())

    return ~blocked_buy, ~blocked_sell
