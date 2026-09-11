"""调仓清单：把「权重变化」翻译成「买卖多少股」。

回测输出的是权重，但实盘下单需要的是股数。本模块负责这层翻译，
并且按 A 股规则处理两件事：
  - 100 股整数倍（1 手），不足 1 手的部分不下单
  - 买卖方向分别计算佣金与印花税（印花税仅卖出收）

生成的清单可以直接对着券商软件下单，也可以用来核对回测是否自洽。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_plan(weights: pd.DataFrame, prices: pd.DataFrame,
               capital: float = 1_000_000.0,
               fee: float = 0.0003, stamp_tax: float = 0.0005,
               share_unit: int = 100,
               can_buy: pd.DataFrame | None = None,
               can_sell: pd.DataFrame | None = None,
               exec_shift: int = 0) -> pd.DataFrame:
    """生成调仓清单。

    参数
    ----
    weights : 每日目标权重（已经过可执行性约束）
    prices  : 用于成交的价格面板（通常传开盘价，与回测一致）
    capital : 组合总本金，用于把权重换算成金额
    share_unit : 最小交易单位，A 股为 100 股
    can_buy / can_sell : 若提供，被禁止的方向会标注为「冻结」而不是直接跳过，
                         方便你看清每次调仓有多少是因为买不进 / 卖不出而没做成
    exec_shift : 决策到下单之间延迟的交易日数，必须与回测保持一致：
        - 收盘价成交（回测 delay=1，信号日收盘下单）→ 0
        - 开盘价成交（回测 delay=2，次日开盘下单）→ 1

    返回
    ----
    DataFrame，一行为一次调仓动作：
      date, code, action, prev_weight, target_weight, delta_weight,
      price, shares, amount, commission, stamp_tax, cash_flow
    """
    if share_unit <= 0:
        raise ValueError("share_unit 必须为正整数")

    prev = weights.shift(1).fillna(0.0)
    delta = weights - prev
    if exec_shift:
        # 决策在 t 日收盘做出，实际在 t+exec_shift 日下单，
        # 因此把调仓指令和当天的可交易状态一起往后挪
        delta = delta.shift(exec_shift)
        prev = prev.shift(exec_shift)
        weights = weights.shift(exec_shift)
        # shift 会把布尔列变成 object 并引入 NaN，这里补回 True（无信息即视为可交易）
        if can_buy is not None:
            can_buy = can_buy.shift(exec_shift).fillna(True).astype(bool)
        if can_sell is not None:
            can_sell = can_sell.shift(exec_shift).fillna(True).astype(bool)

    rows = []
    for date in weights.index:
        day_delta = delta.loc[date]
        day_delta = day_delta[day_delta.abs() > 1e-12]
        if day_delta.empty:
            continue

        row_price = prices.loc[date]
        row_prev = prev.loc[date]
        row_target = weights.loc[date]
        cb = None if can_buy is None else can_buy.loc[date]
        cs = None if can_sell is None else can_sell.loc[date]

        for code, dw in day_delta.items():
            price = row_price.get(code, np.nan)
            if pd.isna(price) or price <= 0:
                continue  # 无价格（停牌/未上市）无法下单

            side = "buy" if dw > 0 else "sell"
            allowed = True
            if side == "buy" and cb is not None and not bool(cb.get(code, True)):
                allowed = False
            if side == "sell" and cs is not None and not bool(cs.get(code, True)):
                allowed = False

            target_amount = abs(dw) * capital
            # 整手取整：向下取整到 100 股的整数倍
            shares = int(target_amount / price / share_unit) * share_unit
            amount = shares * price
            commission = amount * fee
            tax = amount * stamp_tax if side == "sell" else 0.0

            if side == "buy":
                cash_flow = -(amount + commission + tax)
            else:
                cash_flow = amount - commission - tax

            p_prev = float(row_prev.get(code, 0.0))
            p_target = float(row_target.get(code, 0.0))
            if p_prev <= 1e-12:
                action = "建仓"
            elif p_target <= 1e-12:
                action = "清仓"
            elif side == "buy":
                action = "加仓"
            else:
                action = "减仓"
            if not allowed:
                action = "冻结·" + action

            rows.append({
                "date": date.date(),
                "code": code,
                "action": action,
                "prev_weight": round(p_prev, 6),
                "target_weight": round(p_target, 6),
                "delta_weight": round(float(dw), 6),
                "price": round(float(price), 4),
                "shares": int(shares),
                "amount": round(amount, 2),
                "commission": round(commission, 2),
                "stamp_tax": round(tax, 2),
                "cash_flow": round(cash_flow, 2),
            })

    plan = pd.DataFrame(rows)
    if not plan.empty:
        plan = plan.sort_values(["date", "action", "code"]).reset_index(drop=True)
    return plan


def summarize(plan: pd.DataFrame) -> dict:
    """对调仓清单做汇总统计，回答「这套策略到底有多能折腾」。"""
    if plan.empty:
        return {"rebalance_count": 0}

    traded = plan[~plan["action"].str.startswith("冻结")]
    buys = traded[traded["delta_weight"] > 0]
    sells = traded[traded["delta_weight"] < 0]
    frozen = plan[plan["action"].str.startswith("冻结")]

    return {
        "rebalance_count": int(traded["date"].nunique()),
        "order_count": int(len(traded)),
        "buy_count": int(len(buys)),
        "sell_count": int(len(sells)),
        "frozen_count": int(len(frozen)),
        "total_buy_amount": float(buys["amount"].sum()),
        "total_sell_amount": float(sells["amount"].sum()),
        "total_commission": float(traded["commission"].sum()),
        "total_stamp_tax": float(traded["stamp_tax"].sum()),
    }
