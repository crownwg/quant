"""strategy.py 逐行中文注解版（逻辑与原文完全一致，只加注释，方便阅读）

读法建议：
1. 先记住 prices 长什么样：行 = 交易日，列 = 股票代码，格子 = 收盘价。
   例：
              600031  002557  600519
   2024-01-02   13.2    35.1   1680
   2024-01-03   13.5    35.4   1695
2. 本文件里所有函数调用几乎都是「整张表一起算」，不是一只股票一只股票地算，
   这是 pandas 的核心思维（向量化），也是这段代码看起来抽象的原因。
"""

import numpy as np
import pandas as pd


def monthly_momentum_weights(prices: pd.DataFrame, lookback: int = 120, top_n: int = 10,
                             volume: pd.DataFrame | None = None, min_volume: float = 0.0) -> pd.DataFrame:
    """月度动量策略：每月选出过去 lookback 天涨得最猛的 top_n 只股票，等权持有。

    参数
    ----
    prices    : 收盘价表（行=日期，列=股票）
    lookback  : 回看天数，120 约等于半年
    top_n     : 每月持仓数量
    volume    : 成交量表，传了才会做流动性过滤
    min_volume: 日均成交量下限，低于它的股票被剔除

    返回
    ----
    weights   : 权重表，形状与 prices 一致，每行（每天）持仓权重之和约为 1
    """

    # ---------- 第 2 步：算动量信号 ----------
    # pct_change(120)：今天价格相比 120 天前涨了百分之多少 → (今价 - 120天前价) / 120天前价
    # 前 120 天算不出来，结果是 NaN（空值），后面 dropna() 会丢掉它们
    # .shift(1)：把整列往下挪一天。原因很重要——
    #   当天收盘价算出的信号，当天收盘才知道，所以只能用「昨天的信号」决定「今天的持仓」，
    #   否则就是拿未来数据作弊（前视偏差 / look-ahead bias）
    signal = prices.pct_change(lookback).shift(1)

    # ---------- 第 3 步：找出每月第一个交易日 ----------
    # to_period("M")：把 2024-01-02、2024-01-03 都变成 2024-01 这个「月」
    month = prices.index.to_period("M")

    # month != month.shift(1)：今天的月份 ≠ 昨天的月份 → 说明今天是本月第一个交易日
    #   结果形如 [True, False, False, ..., True, False, ...]
    # 注意：PeriodIndex 之间比较返回的是 numpy 数组，不是 Series，
    #   所以要手动包一层 pd.Series(...) 才能用 .fillna() 和布尔索引
    #   fillna(True)：第一天没有「昨天」可比，结果是 NaN，需要当成 True（第一天就建仓）
    rebalance = pd.Series(month != month.shift(1), index=prices.index).fillna(True)

    # ---------- 第 5 步之准备：先造一张全是空值的权重表 ----------
    # 关键：用 np.nan（空值）初始化，不是用 0 初始化
    #   - 如果初始化成 0 → 0 不是空值，ffill 不会去覆盖它 → 非调仓日永远是 0，等于没持仓
    #   - 初始化成 NaN → ffill 才会把最近一次调仓日的权重「搬」到非调仓日
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    # ---------- 第 4 步：遍历每个调仓日，选股写权重 ----------
    # prices.index[rebalance]：用 True/False 数组挑出所有调仓日的日期（布尔索引）
    for date in prices.index[rebalance]:

        # 取这一天的动量信号（一整行，每只股票一个分数）
        # dropna()：丢掉还没攒够 120 天数据的新股
        scores = signal.loc[date].dropna()

        # 可选：流动性过滤
        if volume is not None and min_volume > 0:
            # volume.loc[:date]      → 从开头到今天的所有成交量
            # .tail(lookback)        → 只取最近 lookback 天
            # .mean()                → 每只股票求日均成交量（对「列」求平均，得到一行）
            avg_vol = volume.loc[:date].tail(lookback).mean()

            # avg_vol[avg_vol >= min_volume].index → 挑出日均量达标的股票代码
            liquid = avg_vol[avg_vol >= min_volume].index

            # intersection 取交集：既要信号有效，又要流动性达标
            # reindex 保证剩下的顺序和原来一致
            scores = scores.reindex(scores.index.intersection(liquid))

        # nlargest(top_n)：取分数最大的 top_n 只（即涨得最猛的 10 只）
        scores = scores.nlargest(top_n)

        # 关键三步：
        # 1. weights 用 NaN 初始化 → 非调仓日的空值才能被 ffill 沿用上次目标权重
        # 2. 调仓日整行先写 0（不是 NaN）→ 显式清掉上一期权重
        #    必须写 0 而不是留 NaN：否则 ffill 会跨过这个调仓日，把旧权重填回来，
        #    导致已清仓的股票「复活」
        # 3. 入选股票写 1/n → 等权，比如 10 只就是每只 10%
        weights.loc[date, :] = 0.0
        if len(scores):
            weights.loc[date, scores.index] = 1.0 / len(scores)

    # ---------- 第 5 步：填充成每日权重 ----------
    # ffill()      = forward fill，用上面最近一个非空值往下填 → 非调仓日沿用上次持仓
    # fillna(0.0)  = 剩下最开头那几天（还没到第一次调仓）填 0，表示空仓
    return weights.ffill().fillna(0.0)


# ============================================================
# 需要提前补的 Python / pandas 知识点（按优先级排序）
# ============================================================
# 1. DataFrame 的基本形状：index（行标签）、columns（列名）、.loc[行, 列] 取值
# 2. NaN 是什么：pandas 里的「空值」，ffill / fillna / dropna 都围着它转
# 3. 布尔索引：用一串 True/False 挑行或挑列，是 pandas 最常用也最烧脑的写法
# 4. 向量化思维：别想「for 循环遍历每只股票」，要想「对整张表做一次运算」
# 5. shift / pct_change / mean / nlargest：这 4 个函数吃透，本文件就通了
#
# 推荐练习（在 Python 里跑一遍，比看书快 10 倍）：
#   import pandas as pd
#   df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
#   print(df.pct_change(2))      # 看 NaN 是怎么出现的
#   print(df.shift(1))           # 看整体下移一行
#   print(df["A"] > 2)           # 看布尔索引长什么样
#   print(df[df["A"] > 2])       # 看用布尔索引挑行
