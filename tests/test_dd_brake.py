"""回撤熔断：状态机的正确性。

这里的每个用例都对应一个实际踩过的坑。熔断这类「状态机 + 路径依赖」的逻辑，
写错了不会报错，只会表现为「怎么净值一直平的」或者「怎么没效果」，
从结果上极难反推，所以必须把每条边界钉死。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest import brake_exposure_from_equity, brake_kwargs, run


def _s(*vals) -> pd.Series:
    return pd.Series(list(vals), dtype=float)


# ----------- 基本行为 -----------

def test_first_day_always_full():
    """第一天没有任何历史可判断，必须满仓，不能凭空降仓。"""
    eq = _s(1.0, 0.5, 0.4)
    exp = brake_exposure_from_equity(eq, 0.2, 0.0, 0.1)
    assert exp.iloc[0] == 1.0


def test_triggers_after_threshold_breach():
    eq = _s(1.0, 0.9, 0.79, 0.79)          # 第三天跌破 -20%
    exp = brake_exposure_from_equity(eq, 0.2, 0.0, 0.1)
    assert exp.iloc[1] == 1.0              # 第一天判断用的是 1.0，未触发
    assert exp.iloc[2] == 1.0              # 用的是 0.9，回撤 -10%，未触发
    assert exp.iloc[3] == 0.0              # 用的是 0.79，回撤 -21%，触发


def test_causality_exposure_only_uses_past():
    """铁律：t 日敞口只能由 t 日及之前的净值决定，不能用未来的。

    做法：把序列后面一段改掉，前面几天的敞口必须一字不差。
    """
    head = [1.0, 0.95, 0.85, 0.80]
    a = brake_exposure_from_equity(_s(*head, 0.70, 0.60), 0.2, 0.0, 0.1)
    b = brake_exposure_from_equity(_s(*head, 1.50, 2.00), 0.2, 0.0, 0.1)
    assert list(a.iloc[:len(head)]) == list(b.iloc[:len(head)])


def test_action_half_keeps_half():
    eq = _s(1.0, 0.79, 0.79)
    exp = brake_exposure_from_equity(eq, 0.2, 0.5, 0.1)
    assert exp.iloc[2] == 0.5


# ----------- 三个实际踩过的坑 -----------

def test_cleared_position_can_recover():
    """坑 1（最严重）：清仓后净值不动，回撤被冻结 → 一度永久空仓。

    工程机械池实测「跌 10% 清仓」1623 天里熔断 1606 天、收益 -13%。
    恢复必须看影子净值（不熔断会走成什么样），不能看自己那个冻结的净值。
    """
    eq = _s(1.0, 0.78, 0.78, 0.78, 0.78)   # 清仓后完全不动
    shadow = _s(1.0, 0.78, 0.70, 0.85, 0.90)  # 不熔断的话先跌到 0.70 再反弹
    exp = brake_exposure_from_equity(eq, 0.2, 0.0, 0.10, shadow=shadow)
    assert exp.iloc[2] == 0.0              # 触发后清仓
    assert exp.iloc[4] == 1.0, "影子净值已从低点反弹，必须恢复满仓"


def test_recovery_needs_rebound_not_return_to_peak():
    """坑 2：恢复判据不能是「距历史高点还有多远」。

    这个池子 6 年跌 67%，影子净值相对历史高点永远回不来，
    用那个条件等于永远不恢复。要的是「从低点反弹了多少」。
    """
    # 影子净值一直远低于历史高点，但从低点 0.50 反弹到 0.60（+20%）。
    # 多给一天：恢复发生在 i=3，要 i=4 才看得到仓位回到 1。
    eq = _s(1.0, 0.70, 0.70, 0.70, 0.70)
    shadow = _s(1.0, 0.70, 0.50, 0.60, 0.62)
    exp = brake_exposure_from_equity(eq, 0.2, 0.0, 0.15, shadow=shadow)
    assert exp.iloc[3] == 0.0, "反弹前还在熔断"
    assert exp.iloc[4] == 1.0, "从低点反弹 20% > 15%，应恢复"


def test_no_immediate_retrigger_after_recovery():
    """坑 3：恢复后不能立刻再次触发。

    清仓期间净值冻结在 0.8，回撤一直是 -20%；不重置高点基准的话，
    恢复完第二天马上又被打回熔断，恢复只维持一天。
    """
    eq = _s(1.0, 0.78, 0.78, 0.78, 0.78, 0.78)
    shadow = _s(1.0, 0.78, 0.70, 0.85, 0.86, 0.87)
    exp = brake_exposure_from_equity(eq, 0.2, 0.0, 0.10, shadow=shadow)
    assert exp.iloc[4] == 1.0, "刚恢复的这天不该又熔断"
    assert exp.iloc[5] == 1.0, "恢复后净值没再跌，不该反复触发"


def test_hysteresis_holds_state_in_between():
    """滞回带：反弹不够时保持熔断，不来回切换。"""
    eq = _s(1.0, 0.79, 0.79, 0.79)
    shadow = _s(1.0, 0.79, 0.75, 0.78)     # 从 0.75 反弹 4% < resume 10%
    exp = brake_exposure_from_equity(eq, 0.2, 0.0, 0.10, shadow=shadow)
    assert exp.iloc[3] == 0.0, "反弹不够，应继续熔断"


def test_never_triggers_when_no_drawdown():
    eq = _s(1.0, 1.1, 1.2, 1.3)
    exp = brake_exposure_from_equity(eq, 0.2, 0.0, 0.1)
    assert (exp == 1.0).all()


def test_empty_series_safe():
    exp = brake_exposure_from_equity(_s(), 0.2, 0.0, 0.1)
    assert len(exp) == 0


# ----------- 与回测引擎的集成 -----------

def _flat_market(n: int = 60, codes=("AAA", "BBB")) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(1.0, index=idx, columns=list(codes))


def test_run_without_brake_unchanged():
    """不开熔断时必须与改动前逐位一致——这是最重要的回归保证。"""
    prices = _flat_market()
    w = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
    eq1, m1, d1 = run(prices, w, fee=0.0, stamp_tax=0.0, slippage=0.0)
    eq2, m2, d2 = run(prices, w, fee=0.0, stamp_tax=0.0, slippage=0.0,
                      dd_brake=0.0)
    pd.testing.assert_series_equal(eq1, eq2)
    # 逐个比而不是 m1 == m2：指标里有 NaN（无下行波动时 sortino），
    # 而 NaN != NaN 会让整字典比较直接失败
    assert set(m1) == set(m2)
    for k in m1:
        v1, v2 = m1[k], m2[k]
        if isinstance(v1, float) and np.isnan(v1):
            assert np.isnan(v2), k
        else:
            assert v1 == v2, f"{k}: {v1} != {v2}"
    assert not m1["dd_brake_on"]


def test_run_with_brake_reduces_drawdown():
    """构造一段先涨后暴跌的行情：熔断必须真的把回撤压下来。"""
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    codes = ["AAA", "BBB"]
    # 前 30 天一路上涨，之后腰斩
    up = np.linspace(1.0, 2.0, 30)
    down = np.linspace(2.0, 0.9, n - 30)
    path = np.concatenate([up, down])
    prices = pd.DataFrame({c: path for c in codes}, index=idx)
    w = pd.DataFrame(0.5, index=idx, columns=codes)

    _, m_off, _ = run(prices, w, fee=0.0, stamp_tax=0.0, slippage=0.0)
    _, m_on, d_on = run(prices, w, fee=0.0, stamp_tax=0.0, slippage=0.0,
                        dd_brake=0.15, dd_brake_action=0.0)

    assert m_on["dd_brake_on"] is True
    assert m_on["dd_brake_days"] > 0, "应该确实熔断过"
    assert m_on["max_drawdown"] > m_off["max_drawdown"], (
        f"熔断后回撤应更小：{m_on['max_drawdown']} vs {m_off['max_drawdown']}")
    assert "exposure" in d_on.columns


def test_run_brake_converges():
    """迭代必须收敛；不收敛时要在 metrics 里标出来，不能默默给个抖动的解。"""
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    codes = ["AAA", "BBB"]
    path = np.concatenate([np.linspace(1.0, 2.0, 30), np.linspace(2.0, 0.9, n - 30)])
    prices = pd.DataFrame({c: path for c in codes}, index=idx)
    w = pd.DataFrame(0.5, index=idx, columns=codes)
    _, m, _ = run(prices, w, fee=0.0, stamp_tax=0.0, slippage=0.0,
                  dd_brake=0.15, dd_brake_action=0.0)
    assert m["dd_brake_converged"] is True


# ----------- 参数提取 -----------

def test_brake_kwargs_defaults():
    kw = brake_kwargs({})
    assert kw == {"dd_brake": 0.0, "dd_brake_action": 0.0, "dd_brake_resume": None}


def test_brake_kwargs_from_namespace():
    class A:
        dd_brake = 0.25
        dd_brake_action = 0.5
        dd_brake_resume = 0.1
    assert brake_kwargs(A()) == {"dd_brake": 0.25, "dd_brake_action": 0.5,
                                 "dd_brake_resume": 0.1}


def test_brake_kwargs_tolerates_none():
    """没传的参数（argparse 默认 None）必须归零，不能让 run() 拿到 None 崩掉。"""
    kw = brake_kwargs({"dd_brake": None, "dd_brake_action": None})
    assert kw["dd_brake"] == 0.0 and kw["dd_brake_action"] == 0.0
