"""回测引擎：把「目标权重」变成「净值曲线 + 评价指标 + 成本明细」。

成交价假设
----------
- 不传 open_prices：close-to-close，即默认按当日收盘价成交。
  这个假设偏乐观——你其实是收盘那一刻才知道信号，却假装用收盘价成交了。
- 传入 open_prices：open-to-open，模拟「收盘算信号，次日开盘价成交」，
  消除前视偏差，更接近实盘。**推荐始终开启**。

成本模型
--------
只算佣金和印花税会系统性低估摩擦成本（月频换手 10 只，实测滑点通常吃掉
1~3%/年）。完整的成本项：

  1. 佣金     ：双边，费率可调；**受单笔最低佣金约束**（A 股常见 5 元/笔）。
                低本金时这一项远比费率本身重要。
  2. 印花税   ：仅卖出单边收。
  3. 滑点     ：双边，按成交额计。买在卖一之上、卖在买一之下，
                这是最不可忽略的一项。
  4. 冲击成本 ：可选，平方根模型 impact = coef × sqrt(成交量 / 日均成交额)。
                仓位越重、标的越不流动，冲击越大。需要传 adv 面板。

关于最低佣金为什么必须按「笔数」算
----------------------------------
佣金 = max(成交额 × 费率, 最低佣金)，是对**每一笔委托**成立的，不是对当日
汇总成立。权重尺度下「单笔最低佣金」= min_commission / capital，
所以当日的佣金下限 = 笔数 × min_commission / capital。
用 100 万本金买 10 只票，每笔 5 元下限合计 50 元，折合 0.005%；
但若本金只有 10 万，同样的 50 元就是 0.05%——比万三费率高一倍。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cost_kwargs(args, adv: pd.DataFrame | None = None) -> dict:
    """从 argparse 命名空间（或 dict 配置）提取成本模型参数，保证各入口口径一致。

    所有需要调用 `run()` 的模块（main / grid / compare / combine / rolling / daily）
    都应通过本函数取参，避免出现「主回测算了滑点、网格搜索没算」这种口径分裂。

    adv 的传递：main.py 在加载面板后会算一次 ADV 并挂到 args.adv_panel 上，
    这里优先用显式传入的 adv，否则回退到 cfg.adv_panel。ADV 在 run() 内部
    按权重矩阵对齐，所以无需预先切片。
    """

    def g(key, default):
        if isinstance(args, dict):
            return args.get(key, default)
        return getattr(args, key, default)

    if adv is None:
        adv = g("adv_panel", None)
    impact = g("impact_coef", 0.0)
    return dict(
        fee=g("fee", 0.0003),
        stamp_tax=g("stamp_tax", 0.0005),
        slippage=g("slippage", 0.0),
        min_commission=g("min_commission", 0.0),
        capital=g("capital", 0.0),
        impact_coef=impact,
        adv=adv if impact > 0 else None,
    )


def run(prices: pd.DataFrame, target_weights: pd.DataFrame,
        open_prices: pd.DataFrame | None = None,
        fee: float = 0.0003, stamp_tax: float = 0.0005,
        slippage: float = 0.0, min_commission: float = 0.0,
        capital: float = 0.0, impact_coef: float = 0.0,
        adv: pd.DataFrame | None = None,
        risk_free: float = 0.0, periods_per_year: int = 252,
        delay: int | None = None):
    """执行回测。

    参数
    ----
    delay : 决策到生效之间滞后的 K 线根数，None 表示自动推断：
        - 收盘价成交 → 1：信号在 close[t] 算出，按 close[t] 成交，
          赚 close[t]→close[t+1]，对应 weights.shift(1)。
        - 开盘价成交 → 2：信号在 close[t] 算出，最早只能 open[t+1] 买入，
          赚 open[t+1]→open[t+2]，必须滞后两根。
          只滞后的 1 根的话，等于用 close[t] 的信息去赚 open[t]→open[t+1]，
          而 open[t] 在 close[t] 之前——这是半天前视偏差。
    slippage : 单边滑点率，按成交额计。0.0005 = 5bp。
    min_commission : 单笔最低佣金（元）。需同时给 capital 才能生效。
    capital  : 组合本金（元）。用于把「单笔最低佣金」「冲击成本」换算到权重尺度。
    impact_coef : 平方根冲击成本系数。0 表示不启用。需同时给 adv 与 capital。
        impact_rate = impact_coef × sqrt(该票成交额 / 该票日均成交额)
    adv : 日均成交额面板（date×code，单位与 capital 一致，即元）。

    返回 (equity, metrics, detail)。
    """
    base = open_prices if open_prices is not None else prices
    returns = base.pct_change().fillna(0.0)

    if delay is None:
        delay = 2 if open_prices is not None else 1
    # 目标权重滞后 delay 根：今天的决策要等 delay 根之后才真正生效
    weights = target_weights.shift(delay).fillna(0.0)

    # ---- 换手 ----
    delta = weights.diff()
    # 首行没有前一期权重（diff 出全 NaN）→ 整仓算作新建仓
    first_row = bool(delta.iloc[0].isna().all()) if len(delta) else False
    # 首次建仓 diff 为 NaN，此时整个仓位都是「买入」
    buy = delta.clip(lower=0).sum(axis=1).fillna(weights.abs().sum(axis=1))
    sell = (-delta).clip(lower=0).sum(axis=1).fillna(0.0)
    turnover = buy + sell                      # 双边换手

    # ---- 当日下单笔数（用于最低佣金）----
    traded = delta.abs() > 1e-12
    n_orders = traded.sum(axis=1).astype(float)
    if first_row:
        # 首行：整仓都是新建，笔数 = 非零权重个数
        n_orders.iloc[0] = float((weights.iloc[0].abs() > 1e-12).sum())

    # ---- 成本明细 ----
    # 佣金：先按费率算，再对「单笔最低佣金」取大；两者都是权重尺度
    fee_cost = turnover * fee
    if min_commission > 0 and capital > 0:
        floor_cost = n_orders * (min_commission / capital)
        fee_cost = pd.concat([fee_cost, floor_cost], axis=1).max(axis=1)

    tax_cost = sell * stamp_tax                # 印花税：仅卖出
    slippage_cost = turnover * slippage        # 滑点：双边

    impact_cost = pd.Series(0.0, index=weights.index)
    if impact_coef > 0 and adv is not None and capital > 0:
        a = adv.reindex(index=weights.index, columns=weights.columns).astype(float)
        a = a.where(a > 0)                      # 成交额为 0（停牌/未上市）→ 无 ADV
        notional = delta.abs() * capital        # 每只票的成交金额（元）
        # 参与率：该票成交额 / 其日均成交额。上限截到 1.0，避免 ADV 缺失时爆出 inf
        participation = (notional / a).clip(upper=1.0)
        impact_rate = impact_coef * np.sqrt(participation.fillna(0.0))
        impact_cost = (delta.abs() * impact_rate).sum(axis=1)

    cost = fee_cost + tax_cost + slippage_cost + impact_cost

    portfolio = (weights * returns).sum(axis=1) - cost
    equity = (1 + portfolio).cumprod()
    drawdown = equity / equity.cummax() - 1

    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total_return = float(equity.iloc[-1] - 1)

    # ---- 风险指标 ----
    # 超额收益：扣掉日频无风险利率后再算夏普 / 索提诺
    excess = portfolio - risk_free / periods_per_year
    std = excess.std()
    downside = excess[excess < 0].std()
    # 年化收益用几何法；若期末净值为负（理论上不会）退化为算术平均
    annual_return = float((equity.iloc[-1] ** (1 / years) - 1)) if equity.iloc[-1] > 0 else -1.0
    max_dd = float(drawdown.min())
    sharpe = float(excess.mean() / std * (periods_per_year ** 0.5)) if std else 0.0
    sortino = float(excess.mean() / downside * (periods_per_year ** 0.5)) if downside else 0.0
    calmar = float(annual_return / abs(max_dd)) if max_dd else 0.0

    # 最长回撤持续天数：从历史最高点回落到重新创新高的最长一段
    under_water = drawdown < -1e-12
    dd_days = 0
    cur = 0
    for flag in under_water:
        cur = cur + 1 if flag else 0
        dd_days = max(dd_days, cur)

    win = portfolio[portfolio > 0]
    loss = portfolio[portfolio < 0]

    metrics = {
        # 收益
        "total_return": total_return,
        "annual_return": annual_return,
        "final_equity": float(equity.iloc[-1]),
        # 风险
        "max_drawdown": max_dd,
        "max_drawdown_days": int(dd_days),
        "annual_volatility": float(std * (periods_per_year ** 0.5)),
        # 风险调整收益
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        # 交易特征
        "win_rate": float(len(win) / len(portfolio)) if len(portfolio) else 0.0,
        "profit_loss_ratio": float(win.mean() / abs(loss.mean())) if len(loss) and loss.mean() else 0.0,
        "average_daily_turnover": float(turnover.mean()),
        "annual_turnover": float(turnover.mean() * periods_per_year),
        # 成本
        "total_commission": float(fee_cost.sum()),
        "total_stamp_tax": float(tax_cost.sum()),
        "total_slippage": float(slippage_cost.sum()),
        "total_impact": float(impact_cost.sum()),
        "total_cost": float(cost.sum()),
        "cost_drag_annual": float(cost.sum() / years),
        # 样本信息
        "trading_days": int(len(portfolio)),
        "years": float(years),
    }

    detail = pd.DataFrame({
        "portfolio_return": portfolio,
        "gross_return": (weights * returns).sum(axis=1),
        "buy_turnover": buy,
        "sell_turnover": sell,
        "turnover": turnover,
        "orders": n_orders,
        "commission": fee_cost,
        "stamp_tax": tax_cost,
        "slippage": slippage_cost,
        "impact": impact_cost,
        "cost": cost,
        "equity": equity,
        "drawdown": drawdown,
    })

    return equity, metrics, detail
