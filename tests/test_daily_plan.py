"""调仓清单（实盘化）：列名约定、手数取整、清仓判定。

**为什么专门建这个文件**：`main.py` 打印清单时写的是 `r['shares']`，
但 `_build_today_plan` 产出的列叫 `delta_shares` —— 一执行就 KeyError，
进程以退出码 1 结束。CLI 下只表现为「清单没打印出来」，Web 下直接报失败，
而 **CSV 其实早就正确写盘了**，所以从现象反推原因很困难。

教训：这里必须锁住「列名约定」，而不只是测数值算得对不对。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import main as qmain


class _Args:
    """模拟 argparse.Namespace，只带 _build_today_plan 用到的字段。"""

    def __init__(self, **kw):
        self.today = ""
        self.holdings = ""
        self.capital = 1_000_000.0
        for k, v in kw.items():
            setattr(self, k, v)


def _market():
    """一天的行情 + 目标权重。002557 的目标权重是 0（意味着该清掉）。"""
    idx = pd.DatetimeIndex([pd.Timestamp("2024-09-09")])
    prices = pd.DataFrame(
        {"600519": [1000.0], "000858": [100.0], "002557": [50.0]}, index=idx)
    weights = pd.DataFrame(
        {"600519": [0.5], "000858": [0.5], "002557": [0.0]}, index=idx)
    return prices, weights


def _empty_plan() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "code", "action", "delta_weight",
                                 "price", "shares", "amount"])


def _write_holdings(tmp_path, text: str) -> str:
    p = tmp_path / "h.csv"
    p.write_text(text, encoding="utf-8")
    return str(p)


PLAN_COLUMNS = ["code", "action", "current_shares", "target_shares",
                "delta_shares", "price", "amount"]


# --------------------------------------------------- 列名约定（本次 bug 的正主）

def test_holdings_mode_column_names(tmp_path):
    """变动股数必须叫 delta_shares，且不得出现裸的 shares 列。"""
    prices, weights = _market()
    args = _Args(holdings=_write_holdings(tmp_path, "code,shares\n600519,100\n"))
    out = qmain._build_today_plan(_empty_plan(), prices, weights, args)
    assert list(out.columns) == PLAN_COLUMNS
    assert "shares" not in out.columns, "打印代码读的就是这列，改名会让它 KeyError"


def test_today_mode_column_names(tmp_path):
    """mode 1（只给 --today）要把 plan 里的 shares 重命名成 delta_shares。"""
    prices, weights = _market()
    plan = pd.DataFrame([{
        "date": pd.Timestamp("2024-09-09").date(), "code": "600519",
        "action": "建仓", "delta_weight": 0.5, "price": 1000.0,
        "shares": 100, "amount": 100_000.0,
    }])
    args = _Args(today="20240909")
    out = qmain._build_today_plan(plan, prices, weights, args)
    # mode 1 会额外保留 delta_weight，所以只校验「该有的都在、且没有裸 shares」
    assert set(PLAN_COLUMNS) <= set(out.columns)
    assert "shares" not in out.columns
    assert int(out["delta_shares"].iloc[0]) == 100


def test_printed_columns_exist():
    """打印用到的三列必须都在约定列名里（直接对应当年那个 KeyError）。"""
    for col in ("action", "delta_shares", "price", "amount"):
        assert col in PLAN_COLUMNS


# --------------------------------------------------- 数值语义

def test_holdings_mode_computes_delta(tmp_path):
    """1000 股茅台 + 2000 股五粮液 + 300 股 002557，各自该怎么动。"""
    prices, weights = _market()
    args = _Args(holdings=_write_holdings(
        tmp_path, "code,shares\n600519,100\n000858,2000\n002557,300\n"))
    out = qmain._build_today_plan(_empty_plan(), prices, weights, args)
    got = {r["code"]: r for r in out.to_dict("records")}

    # 600519：目标 100 股、当前 100 股 → 不动，不进清单
    assert "600519" not in got
    # 000858：目标 1500、当前 2000 → 减仓 500
    assert got["000858"]["action"] == "减仓"
    assert int(got["000858"]["delta_shares"]) == -500
    assert got["000858"]["amount"] == pytest.approx(500 * 100.0)
    # 002557：目标 0、当前 300 → 清仓
    assert got["002557"]["action"] == "清仓"
    assert int(got["002557"]["delta_shares"]) == -300
    assert int(got["002557"]["target_shares"]) == 0


def test_holdings_mode_rounds_to_lots(tmp_path):
    """目标股数必须取整到 100 股（1 手）。"""
    idx = pd.DatetimeIndex([pd.Timestamp("2024-09-09")])
    prices = pd.DataFrame({"600519": [1000.0]}, index=idx)
    weights = pd.DataFrame({"600519": [0.3333]}, index=idx)
    args = _Args(holdings=_write_holdings(tmp_path, "code,shares\n600519,0\n"))
    out = qmain._build_today_plan(_empty_plan(), prices, weights, args)
    assert int(out["target_shares"].iloc[0]) % 100 == 0


def test_small_delta_is_dropped(tmp_path):
    """变动不足 1 手的不进清单（避免为了 30 股来回交易）。"""
    prices, weights = _market()
    # 600519 目标 100 股，当前 50 股 → delta 50，不足 1 手
    args = _Args(holdings=_write_holdings(tmp_path, "code,shares\n600519,50\n"))
    out = qmain._build_today_plan(_empty_plan(), prices, weights, args)
    assert "600519" not in set(out.get("code", []))


def test_unknown_holding_is_dropped_from_frame(tmp_path):
    """池外股票（没有价格）不进 frame —— 它的告警在 main() 里打印。

    这里至少锁住「不会因为缺价格而抛异常」。
    """
    prices, weights = _market()
    args = _Args(holdings=_write_holdings(
        tmp_path, "code,shares\n600519,100\n000001,5000\n"))
    out = qmain._build_today_plan(_empty_plan(), prices, weights, args)
    assert "000001" not in set(out.get("code", []))


def test_non_rebalance_day_returns_empty():
    """mode 1 指定的日期不是调仓日 → 返回空表而不是报错。"""
    prices, weights = _market()
    args = _Args(today="20240909")
    out = qmain._build_today_plan(_empty_plan(), prices, weights, args)
    assert out.empty


def test_holdings_must_have_required_columns(tmp_path):
    prices, weights = _market()
    args = _Args(holdings=_write_holdings(tmp_path, "code,qty\n600519,100\n"))
    with pytest.raises(ValueError):
        qmain._build_today_plan(_empty_plan(), prices, weights, args)


def test_capital_falls_back_when_no_priced_holdings(tmp_path):
    """持仓全都没有价格时退回 --capital，而不是拿 0 去除权。"""
    prices, weights = _market()
    args = _Args(holdings=_write_holdings(tmp_path, "code,shares\n000001,5000\n"),
                 capital=1_000_000.0)
    out = qmain._build_today_plan(_empty_plan(), prices, weights, args)
    # 目标权重还在，所以仍应给出建仓建议（按 capital 而非 0）
    assert not out.empty
    assert set(out["action"]) <= {"建仓", "加仓", "减仓", "清仓"}
