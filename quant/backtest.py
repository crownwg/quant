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


def brake_exposure_from_equity(equity: pd.Series, threshold: float,
                               action: float, resume: float,
                               shadow: pd.Series | None = None) -> pd.Series:
    """按净值回撤生成每日敞口：跌破阈值降仓，回撤收窄到 resume 以内才解除。

    为什么必须带滞回带（resume < threshold）：没有它，净值在阈值附近抖动会
    导致「今天清仓、明天满仓、后天再清仓」，换手和成本反而把好处全吃回去。
    这正是择时模块里 band 参数存在的同一个理由。

    为什么恢复要看 shadow（不熔断的净值）而不是自己的净值
    ------------------------------------------------------
    第一版踩的坑：触发后如果降到 0 仓，组合净值就**不再变动**，回撤被
    冻结在触发点，恢复条件永远不成立——一旦熔断就永久空仓。
    实测工程机械池「跌 10% 清仓」1623 天里有 1606 天在熔断，收益 -13%。

    为什么恢复判据是「从低点反弹」而不是「距历史高点还有多远」
    ------------------------------------------------------------
    第二版改成看影子净值（不熔断会走成什么样）的回撤，结果还是 99% 空仓：
    这个池子 6 年跌了 67%，影子净值相对**历史高点**的回撤长期在 -60% 以下，
    永远收窄不到恢复线。长期下跌的市场里，「回到高点」是个等不到的条件。
    所以第三版改为：熔断期间跟踪影子净值的最低点，从那个低点**反弹 resume**
    就重新入场。下跌途中也会有像样的反弹，这个条件才够得着。

    resume 的语义：从熔断后低点反弹多少才重新入场。默认取阈值的一半。

    因果性（与择时模块同一条铁律）：**t 日的敞口只看 t-1 日及更早的净值**。
    t 日收盘才知道 t 日跌了多少，用它去决定 t 日当天的仓位是未来函数。
    所以下面用 equity[i] 的判断去写 exposure[i+1]。
    第一天的敞口恒为 1（还没有任何历史可判断）。
    """
    n = len(equity)
    exp = np.ones(n, dtype=float)
    if n == 0:
        return pd.Series(exp, index=equity.index)

    eq = equity.to_numpy(dtype=float)
    sh = shadow.to_numpy(dtype=float) if shadow is not None else eq

    braked = False
    peak_eq = float(eq[0])
    sh_low: float | None = None      # 熔断期间影子净值的最低点
    for i in range(n):
        v = eq[i]
        if v > peak_eq:
            peak_eq = v
        # peak 为 0 或负（理论上不会）时不触发，避免除零造出莫名其妙的信号
        dd = (v / peak_eq - 1.0) if peak_eq > 0 else 0.0

        s = sh[i]

        if braked:
            # 熔断中：只回答「能不能回来」。
            # 绝不能再去看 dd——清仓后自己的净值是不动的，dd 会一直停在触发点，
            # 那样恢复分支永远走不到，等于一熔断就永久空仓（第一、二版都栽在这）。
            if sh_low is None or s < sh_low:
                sh_low = s
            if sh_low > 0 and (s / sh_low - 1.0) >= resume:
                braked = False
                sh_low = None
                # 重新起算高点：清仓期间自己的净值是冻结的，不重置的话
                # 「相对历史高点回撤」永远还是 -threshold，恢复完第二天立刻再触发，
                # 恢复只维持一天。每次熔断后都当作新的一段来观察。
                peak_eq = v
        # 容差不是洁癖：1.0 → 0.8 浮点算出来是 -0.19999999999999996，
        # 严格比较会「刚好差一点点没触发」，而用户填的就是 20%。
        elif dd <= -threshold + 1e-12:
            braked = True
            sh_low = s
        # 不满足恢复条件时保持熔断——这就是滞回带，避免反复买卖
        if i + 1 < n:
            exp[i + 1] = action if braked else 1.0
    return pd.Series(exp, index=equity.index)


# 熔断是「仓位影响净值、净值又决定仓位」的定点问题，迭代求解。
# 10 次足够：每轮只会让敞口更少地触发，实践中 2~3 轮就稳定。
_BRAKE_MAX_ITER = 10


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


def brake_kwargs(args) -> dict:
    """提取回撤熔断参数。与 cost_kwargs 分开只是语义不同，用法一致：
    所有调用 run() 的入口都通过它取参，避免「主回测有熔断、网格搜索没有」
    这种口径分裂——那会让网格挑出来的参数在实盘上完全不是一回事。
    """
    def g(key, default):
        if isinstance(args, dict):
            return args.get(key, default)
        return getattr(args, key, default)

    return dict(
        dd_brake=float(g("dd_brake", 0.0) or 0.0),
        dd_brake_action=float(g("dd_brake_action", 0.0) or 0.0),
        dd_brake_resume=(None if g("dd_brake_resume", None) is None
                         else float(g("dd_brake_resume", 0.0))),
    )


def run(prices: pd.DataFrame, target_weights: pd.DataFrame,
        open_prices: pd.DataFrame | None = None,
        fee: float = 0.0003, stamp_tax: float = 0.0005,
        slippage: float = 0.0, min_commission: float = 0.0,
        capital: float = 0.0, impact_coef: float = 0.0,
        adv: pd.DataFrame | None = None,
        risk_free: float = 0.0, periods_per_year: int = 252,
        delay: int | None = None,
        dd_brake: float = 0.0, dd_brake_action: float = 0.0,
        dd_brake_resume: float | None = None):
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
    dd_brake : 回撤熔断阈值，0 表示关闭。0.15 = 组合从历史高点跌 15% 就降仓。
        与择时不同：择时判断「市场好不好」，熔断判断「我自己亏了多少」，
        所以熔断亏到一定程度一定会触发，不需要预测能力。
    dd_brake_action : 触发后把敞口降到多少。0.0 = 清仓，0.5 = 半仓。
    dd_brake_resume : 回撤收窄到多少才解除（滞回带）。默认取阈值的一半——
        没有滞回带的话，净值在阈值附近抖动会反复买卖，交易成本能吃掉全部好处。

    返回 (equity, metrics, detail)。
    """
    base = open_prices if open_prices is not None else prices
    returns = base.pct_change().fillna(0.0)

    if delay is None:
        delay = 2 if open_prices is not None else 1
    # 目标权重滞后 delay 根：今天的决策要等 delay 根之后才真正生效
    weights = target_weights.shift(delay).fillna(0.0)

    # 抽成内部函数是因为回撤熔断要迭代调用它：降仓会改变净值，
    # 净值又决定要不要继续降仓，得反复算到自洽为止。
    def _simulate(w: pd.DataFrame) -> dict:
        # ---- 换手 ----
        delta = w.diff()
        # 首行没有前一期权重（diff 出全 NaN）→ 整仓算作新建仓
        first_row = bool(delta.iloc[0].isna().all()) if len(delta) else False
        # 首次建仓 diff 为 NaN，此时整个仓位都是「买入」
        buy = delta.clip(lower=0).sum(axis=1).fillna(w.abs().sum(axis=1))
        sell = (-delta).clip(lower=0).sum(axis=1).fillna(0.0)
        turnover = buy + sell                      # 双边换手

        # ---- 当日下单笔数（用于最低佣金）----
        traded = delta.abs() > 1e-12
        n_orders = traded.sum(axis=1).astype(float)
        if first_row:
            # 首行：整仓都是新建，笔数 = 非零权重个数
            n_orders.iloc[0] = float((w.iloc[0].abs() > 1e-12).sum())

        # ---- 成本明细 ----
        # 佣金：先按费率算，再对「单笔最低佣金」取大；两者都是权重尺度
        fee_cost = turnover * fee
        if min_commission > 0 and capital > 0:
            floor_cost = n_orders * (min_commission / capital)
            fee_cost = pd.concat([fee_cost, floor_cost], axis=1).max(axis=1)

        tax_cost = sell * stamp_tax                # 印花税：仅卖出
        slippage_cost = turnover * slippage        # 滑点：双边

        impact_cost = pd.Series(0.0, index=w.index)
        if impact_coef > 0 and adv is not None and capital > 0:
            a = adv.reindex(index=w.index, columns=w.columns).astype(float)
            a = a.where(a > 0)                      # 成交额为 0（停牌/未上市）→ 无 ADV
            notional = delta.abs() * capital        # 每只票的成交金额（元）
            # 参与率：该票成交额 / 其日均成交额。上限截到 1.0，避免 ADV 缺失时爆出 inf
            participation = (notional / a).clip(upper=1.0)
            impact_rate = impact_coef * np.sqrt(participation.fillna(0.0))
            impact_cost = (delta.abs() * impact_rate).sum(axis=1)

        cost = fee_cost + tax_cost + slippage_cost + impact_cost

        portfolio = (w * returns).sum(axis=1) - cost
        equity = (1 + portfolio).cumprod()
        return {
            "portfolio": portfolio, "equity": equity, "cost": cost,
            "buy": buy, "sell": sell, "turnover": turnover, "orders": n_orders,
            "commission": fee_cost, "stamp_tax": tax_cost,
            "slippage": slippage_cost, "impact": impact_cost,
        }

    # ---- 回撤熔断：按净值回撤缩放敞口 ----
    # 与择时敞口是相乘关系：择时看「市场好不好」，熔断看「我自己亏了多少」，
    # 两者回答不同的问题，不该互相替代。
    exposure = pd.Series(1.0, index=weights.index)
    brake_converged = True
    if dd_brake and dd_brake > 0:
        resume = dd_brake * 0.5 if dd_brake_resume is None else float(dd_brake_resume)
        # 解除线不能比触发线更宽松，否则刚触发就解除，等于没有滞回带
        resume = min(resume, float(dd_brake))
        # 影子净值：不熔断的话会走成什么样。它不随迭代变化，只算一次，
        # 作用是回答「市场涨回来了没有」——自己空仓时净值是不会动的。
        shadow = _simulate(weights)["equity"]
        for _ in range(_BRAKE_MAX_ITER):
            eq = _simulate(weights.mul(exposure, axis=0))["equity"]
            new_exp = brake_exposure_from_equity(eq, float(dd_brake),
                                                 float(dd_brake_action), resume,
                                                 shadow=shadow)
            if new_exp.equals(exposure):
                break
            exposure = new_exp
        else:
            # 转满还没稳定：多半是阈值卡在净值来回穿越的位置上。
            # 不报错，但把这件事记进 metrics——不该默默给出一个抖动的解。
            brake_converged = False

    w_eff = weights.mul(exposure, axis=0)      # 目标权重 × 熔断敞口 = 实际持仓
    sim = _simulate(w_eff)
    portfolio = sim["portfolio"]
    equity = sim["equity"]
    cost = sim["cost"]
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
        "average_daily_turnover": float(sim["turnover"].mean()),
        "annual_turnover": float(sim["turnover"].mean() * periods_per_year),
        # 成本
        "total_commission": float(sim["commission"].sum()),
        "total_stamp_tax": float(sim["stamp_tax"].sum()),
        "total_slippage": float(sim["slippage"].sum()),
        "total_impact": float(sim["impact"].sum()),
        "total_cost": float(cost.sum()),
        "cost_drag_annual": float(cost.sum() / years),
        # 样本信息
        "trading_days": int(len(portfolio)),
        "years": float(years),
        # 回撤熔断（不开时全为默认值，方便对比表直接对齐列）
        "dd_brake_on": bool(dd_brake and dd_brake > 0),
        "dd_brake_days": int((exposure < 1.0 - 1e-12).sum()) if dd_brake else 0,
        "dd_brake_converged": bool(brake_converged),
    }

    detail = pd.DataFrame({
        "portfolio_return": portfolio,
        "gross_return": (w_eff * returns).sum(axis=1),
        "buy_turnover": sim["buy"],
        "sell_turnover": sim["sell"],
        "turnover": sim["turnover"],
        "orders": sim["orders"],
        "commission": sim["commission"],
        "stamp_tax": sim["stamp_tax"],
        "slippage": sim["slippage"],
        "impact": sim["impact"],
        "cost": cost,
        "equity": equity,
        "drawdown": drawdown,
        # 熔断后的实际敞口，画出来能直接看出「哪几段是空仓躲过去的」
        "exposure": exposure,
    })

    return equity, metrics, detail
