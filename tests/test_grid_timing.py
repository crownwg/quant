"""把择时参数纳入网格 / 样本外搜索的回归测试（quant/grid.py）。

这一层锁的是**「择时到底该不该用」这个问题有没有被正确地提出来**。
它比单个择时函数更容易出错，因为错法是安静的：

  - 搜索空间里漏掉 off → walk-forward 只能在给定的几组择时参数里挑，
    永远得不出「不如不择时」的结论（典型的选择偏差放大器）；
  - off 跟着 timing_lookback × timing_band 一起展开 → 同一个策略被重复
    计入 N 次，把「按择时模式分组的平均指标」搅浑；
  - 按 combo 现算敞口时忘了按 end_pos 切片 → 训练窗用到了测试窗的价格，
    样本外结果直接失真，而且看起来会「特别稳」。

运行：.venv/Scripts/python.exe -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import grid, timing


# ----------------------------------------------------------------- 造数据

def _prices(n_days: int = 900, n_codes: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, size=(n_days, n_codes)), axis=0)
    return pd.DataFrame(data, index=idx, columns=[f"{600000 + i:06d}" for i in range(n_codes)])


def _trending_proxy(n: int = 900, seed: int = 7) -> pd.Series:
    """先涨后跌再涨 —— 让趋势择时真的有 0/1 两种状态，避免测试在恒定敞口上假通过。

    必须带上噪声：完全线性的斜坡日收益恒定 → 滚动波动率为 0 → 波动率目标
    会被 `fillna(cap)` 兜成全仓，测试就测不到连续缩放那一段了。
    """
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    third = n // 3
    drift = np.concatenate([np.full(third, 0.0022),
                            np.full(third, -0.0020),
                            np.full(n - 2 * third, 0.0024)])
    ret = drift + rng.normal(0.0, 0.012, n)
    return pd.Series(100.0 * np.cumprod(1 + ret), index=idx)


class _Args:
    """最小可用的 argparse 替身（覆盖 grid / strategy / backtest 会读到的字段）。"""

    def __init__(self, **kw):
        self.top_n = 2
        self.rebalance = "M"
        self.min_names = 1
        self.buffer = 0
        self.use_open = False
        self.weighting = "equal"
        self.max_weight = 0.0
        self.vol_lookback = 60
        self.lookback = 120
        self.skip_recent = 0
        self.fee = 0.0003
        self.stamp_tax = 0.0005
        self.slippage = 0.0
        self.min_commission = 0.0
        self.capital = 1_000_000.0
        self.impact_coef = 0.0
        # 择时
        self.timing = "off"
        self.timing_lookback = 120
        self.timing_band = 0.0
        self.timing_ma_slope = 0
        self.timing_min_exposure = 0.0
        self.timing_max_exposure = 1.0
        self.timing_smooth = 0
        self.timing_update = "rebalance"
        self.vol_target = 0.0
        self.vol_target_lookback = 60
        self.vol_floor = 0.2
        self.vol_cap = 1.0
        for k, v in kw.items():
            setattr(self, k, v)


# ===================================================== 1. 搜索空间构造

def test_build_combos_without_timing_only_off():
    """不扫择时：组合数就是选股网格大小，全部标成 off。"""
    combos = grid.build_combos([60, 120], [10, 15], [0])
    assert len(combos) == 4
    assert {c["timing"] for c in combos} == {"off"}
    assert all(c["timing_lookback"] == 0 and c["timing_band"] == 0.0 for c in combos)
    assert sorted((c["lookback"], c["top_n"], c["buffer"]) for c in combos) == [
        (60, 10, 0), (60, 15, 0), (120, 10, 0), (120, 15, 0)]


def test_build_combos_forces_off_into_search_space():
    """搜索空间必须自带「不择时」对照组 —— 否则永远问不出「该不该用择时」。

    只给 ["ma"] 也必须把 off 补进去。
    """
    combos = grid.build_combos([120], [10], [0], timing_modes=["ma"],
                              timing_lookbacks=[60], timing_bands=[0.0])
    assert {c["timing"] for c in combos} == {"off", "ma"}
    assert len(combos) == 2


def test_build_combos_does_not_expand_off_across_window_and_band():
    """off 只有一个：窗口/滞回带对「不择时」没有任何影响，跟着展开会把统计搅浑。"""
    combos = grid.build_combos([120], [10], [0],
                               timing_modes=["off", "ma"],
                               timing_lookbacks=[60, 120], timing_bands=[0.0, 0.02])
    offs = [c for c in combos if c["timing"] == "off"]
    mas = [c for c in combos if c["timing"] == "ma"]
    assert len(offs) == 1
    assert len(mas) == 4                       # 2 窗口 × 2 带
    assert len(combos) == 5


def test_build_combos_expands_over_stock_grid():
    combos = grid.build_combos([60, 120], [10, 15], [0, 2],
                               timing_modes=["ma", "momentum"],
                               timing_lookbacks=[60], timing_bands=[0.02])
    # off 1 份 + ma/momentum 各 1 份（1 窗口 × 1 带）= 3 份，每份 2×2×2=8 个选股组合
    assert len(combos) == 24
    assert len({(c["lookback"], c["top_n"], c["buffer"],
                 c["timing"], c["timing_lookback"], c["timing_band"]) for c in combos}) == 24


# ===================================================== 2. 组合标签

def test_combo_label_formats_timing():
    off = grid.combo_label({"lookback": 120, "top_n": 15, "buffer": 0, "timing": "off"})
    assert "lb=120 n=15 buf=0" in off and "不择时" in off

    ma = grid.combo_label({"lookback": 60, "top_n": 10, "buffer": 2, "timing": "ma",
                           "timing_lookback": 120, "timing_band": 0.02})
    assert "ma(120日,带2.0%)" in ma


def test_combo_label_tolerates_missing_timing_keys():
    """旧结果 DataFrame（没有 timing 列）也要能渲染，不能抛 KeyError。"""
    assert "不择时" in grid.combo_label({"lookback": 60, "top_n": 10, "buffer": 0})


# ===================================================== 3. 敞口按 combo 构造

def test_exposure_getter_uses_fallback_when_not_searching():
    """不扫择时时，网格沿用调用方给的那一条敞口（--timing ma 的旧路径）。"""
    proxy = _trending_proxy()
    fallback = timing.trend_exposure(proxy, lookback=60, mode="ma")
    get = grid._exposure_getter(proxy, None, _Args(), fallback=fallback, search_timing=False)
    got = get({"timing": "off"}, 500)
    pd.testing.assert_series_equal(got, fallback.iloc[:500])


def test_exposure_getter_returns_none_for_off_combo():
    """扫择时时，off 组合 = 不择时 = 没有任何敞口缩放。"""
    proxy = _trending_proxy()
    get = grid._exposure_getter(proxy, None, _Args(), fallback=None, search_timing=True)
    assert get({"timing": "off", "timing_lookback": 0, "timing_band": 0.0}, 500) is None


def test_exposure_getter_builds_trend_exposure_per_combo():
    proxy = _trending_proxy()
    get = grid._exposure_getter(proxy, None, _Args(), search_timing=True)
    ex = get({"timing": "ma", "timing_lookback": 30, "timing_band": 0.0}, None)
    assert ex is not None
    assert set(np.unique(ex.values)) <= {0.0, 1.0}
    assert ex.max() == 1.0 and ex.min() == 0.0, "先涨后跌的走势必须产生 0 和 1 两种状态"


def test_exposure_getter_respects_combo_window():
    """不同窗口必须产出不同敞口 —— 否则「扫窗口」是假的。"""
    proxy = _trending_proxy()
    get = grid._exposure_getter(proxy, None, _Args(), search_timing=True)
    a = get({"timing": "ma", "timing_lookback": 20, "timing_band": 0.0}, None)
    b = get({"timing": "ma", "timing_lookback": 150, "timing_band": 0.0}, None)
    assert not a.equals(b)


def test_exposure_getter_caches_per_combo(monkeypatch):
    """同一组择时参数只算一次：训练窗选参会反复取同一条敞口，重算是纯浪费。"""
    proxy = _trending_proxy()
    calls = {"n": 0}
    real = timing.build_exposure

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(timing, "build_exposure", counting)
    get = grid._exposure_getter(proxy, None, _Args(), search_timing=True)
    combo = {"timing": "ma", "timing_lookback": 60, "timing_band": 0.0}
    for _ in range(3):
        get(combo, 500)
    get({"timing": "ma", "timing_lookback": 60, "timing_band": 0.02}, 500)
    assert calls["n"] == 2


def test_exposure_getter_is_causal_under_grid():
    """网格版敞口同样不得偷看未来。

    篡改 end_pos 之后的价格，前 end_pos 个点的敞口必须一字不变。
    """
    base = _trending_proxy()
    tampered = base.copy()
    tampered.iloc[400:] = tampered.iloc[400:] * 5.0
    get_a = grid._exposure_getter(base, None, _Args(), search_timing=True)
    get_b = grid._exposure_getter(tampered, None, _Args(), search_timing=True)
    combo = {"timing": "ma", "timing_lookback": 60, "timing_band": 0.0}
    a = get_a(combo, 400)
    b = get_b(combo, 400)
    pd.testing.assert_series_equal(a, b)


def test_exposure_getter_vol_target_only_when_off():
    """off + 波动率目标 → 敞口来自波动率，不是 None（--vol-target 是全局开关）。"""
    proxy = _trending_proxy()
    cfg = _Args(vol_target=0.15, vol_floor=0.2)
    get = grid._exposure_getter(proxy, None, cfg, search_timing=True)
    ex = get({"timing": "off", "timing_lookback": 0, "timing_band": 0.0}, None)
    assert ex is not None
    assert ex.between(0.0, 1.0).all()
    assert ex.nunique() > 2, "波动率目标是连续缩放，不应只有两档"


# ===================================================== 4. 网格搜索端到端

def test_grid_search_reports_timing_columns():
    p = _prices()
    args = _Args()
    res = grid.grid_search(p, p, None, None, args, [60, 120], [2], [0],
                           start_ts=p.index[300],
                           timing_modes=["off", "ma"],
                           timing_lookbacks=[30], timing_bands=[0.0],
                           proxy=_trending_proxy())
    assert len(res) == 4
    assert set(res["timing"]) == {"off", "ma"}
    assert {"timing_lookback", "timing_band"} <= set(res.columns)


def test_grid_search_off_combo_matches_no_timing_baseline():
    """off 组合的结果必须与「完全不传敞口」的回测逐位一致。

    这条防的是「off 悄悄带上了别的敞口」——那样 off 就不是对照组了，
    「择时有没有用」的结论会整体偏掉。
    """
    p = _prices()
    args = _Args()
    start = p.index[300]
    res = grid.grid_search(p, p, None, None, args, [60], [2], [0], start_ts=start,
                           timing_modes=["off", "ma"], timing_lookbacks=[30],
                           timing_bands=[0.0], proxy=_trending_proxy())
    base = grid.grid_search(p, p, None, None, args, [60], [2], [0], start_ts=start,
                            exposure=None)
    off_row = res[res["timing"] == "off"].iloc[0]
    assert off_row["total_return"] == pytest.approx(base.iloc[0]["total_return"], abs=1e-12)
    assert off_row["sharpe"] == pytest.approx(base.iloc[0]["sharpe"], abs=1e-12)


# ===================================================== 5. 样本外验证

def _wf(args, p, **kw):
    return grid.walk_forward_search(
        p, p, None, None, args, [60, 120], [2], [0], p.index[0],
        train_years=0.5, test_years=0.5, proxy=_trending_proxy(n=len(p)), **kw)


def test_walk_forward_search_space_includes_off():
    """样本外把择时参数纳入搜索时，不择时必须也在候选里。"""
    p = _prices()
    res = _wf(_Args(), p, timing_modes=["ma"], timing_lookbacks=[30], timing_bands=[0.0])
    assert res["search_timing"] is True
    # off(1) + ma(1) = 2 个择时配置 × 2 个 lookback × 1 top_n × 1 buffer = 4 组
    assert res["n_combos"] == 4


def test_walk_forward_records_chosen_timing_and_note():
    p = _prices()
    res = _wf(_Args(), p, timing_modes=["off", "ma"], timing_lookbacks=[30],
              timing_bands=[0.0])
    folds = res["folds"]
    assert "timing" in folds.columns and "chosen" in folds.columns
    assert folds["timing"].isin({"off", "ma"}).all()
    assert res["timing_note"], "扫了择时就要报告各折选中了什么"
    assert "不择时" in res["default_label"]


def test_walk_forward_without_timing_keeps_legacy_shape():
    """不扫择时：行为与改造前一致（combo 只有选股部分，无 timing_note）。"""
    p = _prices()
    res = _wf(_Args(), p)
    assert res["search_timing"] is False
    assert res["n_combos"] == 2
    assert res["timing_note"] == ""
    assert all(t == "off" for t in res["folds"]["timing"])


def test_walk_forward_default_combo_carries_args_timing():
    """默认对照组 = 用户在命令行给的参数（含 --timing），不是硬编码的 off。"""
    p = _prices()
    res = _wf(_Args(timing="ma", timing_lookback=45, timing_band=0.01), p)
    dc = res["default_combo"]
    assert dc["timing"] == "ma" and dc["timing_lookback"] == 45
    assert res["default_label"].startswith("lb=120 n=2 buf=0")
