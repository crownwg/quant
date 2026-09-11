"""回测引擎：把「目标权重」变成「净值曲线 + 评价指标 + 成本明细」。

成交价假设
----------
- 不传 open_prices：close-to-close，即默认按当日收盘价成交。
  这个假设偏乐观——你其实是收盘那一刻才知道信号，却假装用收盘价成交了。
- 传入 open_prices：open-to-open，模拟「收盘算信号，次日开盘价成交」，
  消除前视偏差，更接近实盘。**推荐始终开启**。

成本模型
--------
- 佣金：买卖双边收，默认万三
- 印花税：仅卖出单边收，默认千一（A 股现行标准的一半量级，可按需调）
"""

from __future__ import annotations

import pandas as pd


def run(prices: pd.DataFrame, target_weights: pd.DataFrame,
        open_prices: pd.DataFrame | None = None,
        fee: float = 0.0003, stamp_tax: float = 0.0005,
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

    返回 (equity, metrics, detail)。
    """
    base = open_prices if open_prices is not None else prices
    returns = base.pct_change().fillna(0.0)

    if delay is None:
        delay = 2 if open_prices is not None else 1
    # 目标权重滞后 delay 根：今天的决策要等 delay 根之后才真正生效
    weights = target_weights.shift(delay).fillna(0.0)

    # ---- 换手与成本 ----
    delta = weights.diff()
    # 首次建仓 diff 为 NaN，此时整个仓位都是「买入」
    buy = delta.clip(lower=0).sum(axis=1).fillna(weights.abs().sum(axis=1))
    sell = (-delta).clip(lower=0).sum(axis=1).fillna(0.0)
    turnover = buy + sell                      # 双边换手
    fee_cost = turnover * fee                  # 佣金：双边
    tax_cost = sell * stamp_tax                # 印花税：仅卖出
    cost = fee_cost + tax_cost

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
        "commission": fee_cost,
        "stamp_tax": tax_cost,
        "cost": cost,
        "equity": equity,
        "drawdown": drawdown,
    })

    return equity, metrics, detail
