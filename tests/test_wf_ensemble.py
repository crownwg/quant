"""参数集成 / 邻域平滑选参的回归测试（quant/grid.py 的第二层反过拟合）。

背景：把择时参数纳入 walk-forward 之后，实测发现**「训练窗取 argmax」挑到的
大概率是噪声**——5 折里滞回带 0 与 2% 各占一半，这一维根本没有信息。
本模块加了两条不做单点择优的对照：

  1. 邻域平滑选参：先在参数曲面上做均值滤波，再取 argmax（选「最好的一片」）
  2. 参数集成：不选，把候选的权重等权平均后一起持有；并且只在**平滑分前 K 组**
     内平均（无差别平均全部会把已知最差的区域也等权持有了）

这一层最容易出的错：
  - 邻域定义跨了择时模式（把「换了方法」当成「相邻参数」，平滑就失去了意义）；
  - 集成把「平均各条净值曲线」当成「平均权重」（前者要 N 份资金各付一遍成本，
    既不可实现也把成本重复计算了）；
  - 集成臂的权重用了超过测试窗终点的数据（前视，且会让样本外虚高）；
  - 集成度 K 被事后挑成「最好的那个 K」——那不过是把单点择优从参数挪到了 K 上，
    K 必须以多条臂并列摆出（`ensemble_k_curve` / `k_note`），不能只报最优。

运行：.venv/Scripts/python.exe -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import grid


def _prices(n_days: int = 900, n_codes: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, size=(n_days, n_codes)), axis=0)
    return pd.DataFrame(data, index=idx, columns=[f"{600000 + i:06d}" for i in range(n_codes)])


class _Args:
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


# ===================================================== 1. 邻域定义

def test_neighbor_map_self_and_stock_dims():
    """一维选股网格上，相邻 lookback 互为邻居，且自身总在邻域里。"""
    combos = grid.build_combos([60, 90, 120], [2], [0])
    nbrs = grid._neighbor_map(combos, [60, 90, 120], [2], [0], [], [])
    for i, g in enumerate(nbrs):
        assert i in g, "自身必须在邻域内（否则平滑会丢掉自己的分数）"
    # 60 与 90 相邻 → 互为邻居
    assert 1 in nbrs[0] and 0 in nbrs[1]
    # 60 与 120 相隔 2 档 → 不是邻居
    assert 2 not in nbrs[0]


def test_neighbor_map_two_dim_step_is_neighbor():
    """斜对角（两个维度各差 1 档）也算邻居——邻域是立方体，不是十字。"""
    combos = grid.build_combos([60, 90], [2, 4], [0])
    nbrs = grid._neighbor_map(combos, [60, 90], [2, 4], [0], [], [])
    idx = {(c["lookback"], c["top_n"]): i for i, c in enumerate(combos)}
    i00 = idx[(60, 2)]
    i11 = idx[(90, 4)]
    assert i11 in nbrs[i00]


def test_neighbor_map_does_not_cross_timing_mode():
    """择时模式不同 = 换了方法，不是「相邻参数」，不得互为邻居。"""
    combos = grid.build_combos([120], [2], [0], timing_modes=["off", "ma"],
                              timing_lookbacks=[60], timing_bands=[0.0])
    nbrs = grid._neighbor_map(combos, [120], [2], [0], [60], [0.0])
    for i, c in enumerate(combos):
        for j in nbrs[i]:
            assert combos[j]["timing"] == c["timing"], "邻域跨了择时模式"


def test_neighbor_map_timing_window_step():
    """同一模式内，窗口差 1 档是邻居、差 2 档不是。"""
    combos = grid.build_combos([120], [2], [0], timing_modes=["ma"],
                              timing_lookbacks=[40, 60, 120], timing_bands=[0.0])
    nbrs = grid._neighbor_map(combos, [120], [2], [0], [40, 60, 120], [0.0])
    idx = {c["timing_lookback"]: i for i, c in enumerate(combos)}
    assert idx[60] in nbrs[idx[40]]
    assert idx[120] not in nbrs[idx[40]]


def test_neighbor_map_every_combo_is_self_consistent():
    """邻域关系必须对称：j 在 i 的邻域里 ⇔ i 在 j 的邻域里。"""
    combos = grid.build_combos([60, 90], [2, 4], [0, 2],
                               timing_modes=["off", "ma", "momentum"],
                               timing_lookbacks=[40, 60], timing_bands=[0.0, 0.02])
    nbrs = grid._neighbor_map(combos, [60, 90], [2, 4], [0, 2], [40, 60], [0.0, 0.02])
    for i, g in enumerate(nbrs):
        for j in g:
            assert i in nbrs[j], f"邻域不对称：{i}→{j} 但 {j}↛{i}"


# ===================================================== 2. 邻域平滑

def test_smooth_scores_prefers_plateau_over_spike():
    """轻度孤立尖峰会被平滑拉下来，输给「整片都不错」的高原。

    这是「训练窗 argmax 会踩噪声」的最小复现，也是参数选择里最常见的形态：
    某组参数样本内 0.30、左右邻居只有 0.05；另一片参数彼此都 0.22。
    argmax 会挑尖峰，邻域平滑会挑高原。

    注意幅度：邻域窗口每维 3 档、自身占 1/3 权重，只能抹平这种量级的抖动。
    相差 50 倍的极端点属于异常值，均值滤波拉不下来（那是数据/指标问题）。
    """
    combos = grid.build_combos([40, 60, 90, 120], [2], [0])
    nbrs = grid._neighbor_map(combos, [40, 60, 90, 120], [2], [0], [], [])
    spikes = [0.05, 0.30, 0.05, 0.05]        # 60 日是孤立尖峰
    plateau = [0.22, 0.22, 0.22, 0.22]       # 整片都不错，没有尖峰

    assert int(np.argmax(spikes)) == 1, "原始 argmax 会挑到尖峰"
    sm_spike = grid.smooth_scores(spikes, nbrs)
    assert int(np.argmax(sm_spike)) != 1, "平滑后不应再挑到尖峰"
    assert max(grid.smooth_scores(plateau, nbrs)) > max(sm_spike), \
        "邻域高原平滑后的峰值必须高于孤立尖峰"


def test_smooth_scores_preserves_length_and_self():
    """长度不变；邻域只含自身时，平滑值 == 原值。"""
    s = [1.0, 2.0, 3.0]
    nbrs = [[0], [1], [2]]
    assert grid.smooth_scores(s, nbrs) == pytest.approx(s)


# ===================================================== 3. 集成权重

def test_ensemble_weights_is_mean_of_weights():
    idx = pd.bdate_range("2020-01-01", periods=5)
    a = pd.DataFrame([[0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                     index=idx, columns=["600000", "600001"])
    b = pd.DataFrame([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [0.5, 0.5]],
                     index=idx, columns=["600000", "600001"])
    ens = grid.ensemble_weights([a, b])
    pd.testing.assert_frame_equal(ens, (a + b) / 2)


def test_ensemble_weights_single_input_is_identity():
    idx = pd.bdate_range("2020-01-01", periods=3)
    a = pd.DataFrame([[1.0, 0.0]] * 3, index=idx, columns=["600000", "600001"])
    pd.testing.assert_frame_equal(grid.ensemble_weights([a]), a)


def test_ensemble_weights_does_not_leverage_up():
    """等权平均不会把总仓位放大：每行和 ≤ 各输入行和的最大值。"""
    idx = pd.bdate_range("2020-01-01", periods=4)
    a = pd.DataFrame([[0.6, 0.4]] * 4, index=idx, columns=["600000", "600001"])
    b = pd.DataFrame([[0.3, 0.3]] * 4, index=idx, columns=["600000", "600001"])
    ens = grid.ensemble_weights([a, b])
    assert (ens.sum(axis=1) <= max(a.sum(axis=1).max(), b.sum(axis=1).max()) + 1e-12).all()


def test_ensemble_weights_commutes_with_slicing():
    """切片与求均值可交换 → 集成臂不需要看到 end_pos 之后的数据。

    这条是集成臂的**因果性**保证：先截断再平均 == 先平均再截断。
    """
    idx = pd.bdate_range("2020-01-01", periods=6)
    a = pd.DataFrame(np.linspace(1, 6, 12).reshape(6, 2) / 10, index=idx,
                     columns=["600000", "600001"])
    b = pd.DataFrame(np.linspace(6, 1, 12).reshape(6, 2) / 10, index=idx,
                     columns=["600000", "600001"])
    cut = 4
    sliced = grid.ensemble_weights([a.iloc[:cut], b.iloc[:cut]])
    pd.testing.assert_frame_equal(sliced, grid.ensemble_weights([a, b]).iloc[:cut])


# ===================================================== 4. walk-forward 全部臂

def _wf(args, p, **kw):
    n = len(p)
    return grid.walk_forward_search(
        p, p, None, None, args, [60, 120], [2], [0], p.index[0],
        train_years=0.5, test_years=0.5,
        proxy=pd.Series(np.linspace(100, 150, n), index=p.index), **kw)


def test_walk_forward_summary_has_all_arms():
    p = _prices()
    res = _wf(_Args(), p)
    keys = list(res["summary"]["key"])
    assert keys[:2] == ["walk_forward", "smooth"]
    assert keys[-2:] == ["full_sample_best", "default"]
    assert keys[2:-2] == ["ens3", "ens5", "ens10", "ensall"]
    # 每条曲线都要有每折记录，长度一致
    lens = {k: len(v) for k, v in res["curves"].items()}
    assert len(set(lens.values())) == 1, lens


def test_walk_forward_folds_record_all_selection_modes():
    p = _prices()
    res = _wf(_Args(), p)
    f = res["folds"]
    for col in ("chosen", "smooth_chosen", "test_return", "test_return_smooth",
                "test_return_ens3", "test_return_ens5", "test_return_ens10",
                "test_return_ensall"):
        assert col in f.columns, col
    # 两种选参方式看的是同一段测试窗，收益不应完全一样（同参数时才会一样）
    assert not np.allclose(f["test_return"], f["test_return_smooth"])


def test_walk_forward_ensemble_note_reports_delta():
    p = _prices()
    res = _wf(_Args(), p)
    note = res["ensemble_note"]
    assert "邻域平滑" in note and "集成 K=" in note
    # 差值必须与汇总表自洽
    sm = res["summary"].set_index("key")
    d = float(sm.loc["smooth", "total_return"]) - float(sm.loc["walk_forward", "total_return"])
    assert f"{d:+.1%}" in note


def test_walk_forward_report_renders_all_arms(tmp_path):
    """报告生成必须能吃下全部臂（旧版硬编码三条，容易漏改）。"""
    import json
    import re
    p = _prices()
    res = _wf(_Args(timing="ma", timing_lookback=30), p,
              timing_modes=["off", "ma"], timing_lookbacks=[30], timing_bands=[0.0])
    out = tmp_path / "wf.html"
    grid.build_wf_report(res, str(out))
    html = out.read_text(encoding="utf-8")
    payload = json.loads(re.search(r"const DATA = (\{.*?\});", html, re.S).group(1))
    keys = set(res["summary"]["key"])
    assert len(payload["summary"]) == len(keys) == 8
    assert set(payload["eq"].keys()) == keys
    assert payload["ensemble_note"]
    assert len(payload["k_curve"]) == 4
    assert "smooth_chosen" in payload["folds"][0]
    assert "test_return_ens3" in payload["folds"][0]


def test_walk_forward_default_lookback_outside_grid(tmp_path):
    """默认对照组的 lookback 不在网格候选里时不得崩。

    这是一个真实存在的隐藏 bug：打分缓存原先只按网格候选预建，
    而默认对照组用的是 args.lookback（默认 120）。
    于是 `--grid-lookbacks 60,90` 会让 walk-forward 直接 KeyError ——
    但把 120 写进候选就永远不会暴露。
    """
    p = _prices()
    args = _Args(lookback=120)
    res = grid.walk_forward_search(
        p, p, None, None, args, [60, 90], [2], [0], p.index[0],
        train_years=0.5, test_years=0.5,
        proxy=pd.Series(np.linspace(100, 150, len(p)), index=p.index))
    assert res["default_combo"]["lookback"] == 120
    assert res["n_combos"] == 2
    # 默认对照组必须真的跑出结果（不是 NaN）
    sm = res["summary"].set_index("key")
    assert np.isfinite(sm.loc["default", "total_return"])


def test_walk_forward_single_combo_still_works():
    """候选只剩 1 组时（等于没得选），全部自适应臂必须一致而不是崩掉。"""
    p = _prices()
    res = grid.walk_forward_search(
        p, p, None, None, _Args(), [60], [2], [0], p.index[0],
        train_years=0.5, test_years=0.5,
        proxy=pd.Series(np.linspace(100, 150, len(p)), index=p.index))
    assert res["n_combos"] == 1
    sm = res["summary"].set_index("key")
    for k in ("smooth", "ens3", "ens5", "ens10", "ensall"):
        assert sm.loc[k, "total_return"] == pytest.approx(
            sm.loc["walk_forward", "total_return"], abs=1e-9)


# ===================================================== 4b. 集成度 K（只在好区域内平均）

def test_parse_ensemble_ks_maps_zero_to_all():
    assert grid.parse_ensemble_ks("3,5,10,0") == (3, 5, 10, None)
    assert grid.parse_ensemble_ks("1") == (1,)
    assert grid.parse_ensemble_ks("-2") == (None,)
    assert grid.parse_ensemble_ks("5,5,3") == (5, 3)          # 去重且保序
    assert grid.parse_ensemble_ks(None) == (3, 5, 10, None)   # 默认口径
    assert grid.parse_ensemble_ks("") == (3, 5, 10, None)
    assert grid.parse_ensemble_ks([3, 0]) == (3, None)


def test_parse_ensemble_ks_is_idempotent():
    """必须能吃下自己的输出 —— main.py 先解析一次，walk_forward_search 内部再解析一次。

    真实 bug 回归：早期版本对已解析的 `(3, 5, 10, None)` 会 `int(None)` 抛 TypeError。
    单测原来只喂字符串，所以 `--walk-forward --wf-ensemble-k 3,5,10,0` 一跑就崩。
    """
    once = grid.parse_ensemble_ks("3,5,10,0")
    assert grid.parse_ensemble_ks(once) == once
    assert grid.parse_ensemble_ks((3, 5, 10, None)) == (3, 5, 10, None)
    assert grid.parse_ensemble_ks((None,)) == (None,)


def test_parse_ensemble_ks_rejects_garbage():
    with pytest.raises(ValueError):
        grid.parse_ensemble_ks("3,abc")
    with pytest.raises(ValueError):
        grid.parse_ensemble_ks("3,2.5")


def test_ensemble_arm_keys_and_labels():
    ks = (3, 10, None)
    assert grid.ensemble_arm_keys(ks) == ["ens3", "ens10", "ensall"]
    assert "3" in grid.ensemble_arm_label(3)
    assert "全部" in grid.ensemble_arm_label(None, 31)
    assert "31" in grid.ensemble_arm_label(None, 31)
    # 臂名必须能用 K 反查回来（报告里靠它配对曲线与 K）
    assert len(set(grid.ensemble_arm_keys(grid.parse_ensemble_ks("3,10,0")))) == 3


def test_ensemble_k_is_capped_by_candidate_count():
    """候选少于 K 时，该臂退化为「全部候选」；K=候选数 与 ensall 必须逐折一致。

    这是集成度这一维唯一可以**确定性**断言的关系：picked 集合相同时结果必然相同。
    """
    p = _prices()
    n = len(p)
    res = grid.walk_forward_search(
        p, p, None, None, _Args(), [60, 90, 120, 200, 250], [2, 3], [0], p.index[0],
        train_years=0.5, test_years=0.5, ensemble_ks=(3, 5, 10, 0),
        proxy=pd.Series(np.linspace(100, 150, n), index=p.index))
    assert res["n_combos"] == 10
    # 有效 K：3 / 5 / 10(取满) / 10(取满)
    assert [r["k"] for r in res["ensemble_k_curve"]] == [3, 5, 10, 10]
    f = res["folds"]
    assert np.allclose(f["test_return_ens10"], f["test_return_ensall"]), (
        f["test_return_ens10"].tolist(), f["test_return_ensall"].tolist())
    sm = res["summary"].set_index("key")
    assert sm.loc["ens10", "total_return"] == pytest.approx(
        sm.loc["ensall", "total_return"], abs=1e-9)


def test_ensemble_k_curve_is_self_consistent():
    p = _prices()
    res = _wf(_Args(), p)
    curve = res["ensemble_k_curve"]
    assert [r["key"] for r in curve] == ["ens3", "ens5", "ens10", "ensall"]
    assert all(r["k"] == 2 for r in curve)            # 2 组候选 → 有效 K 全是 2
    sm = res["summary"].set_index("key")
    for r in curve:
        assert r["total_return"] == pytest.approx(float(sm.loc[r["key"], "total_return"]))
        assert r["vs_argmax"] == pytest.approx(
            r["total_return"] - float(sm.loc["walk_forward", "total_return"]))
    assert res["k_note"]


def test_ensemble_k1_reproduces_smooth_selection():
    """K=1 时，集成臂必须**逐折**等于邻域平滑臂 —— 这一条同时钉住两件事：

    1. 前 K 组的排序依据是**平滑分**（若用原始分，K=1 会等于 argmax 臂）；
    2. 集成权重在 K=1 时退化为那一组的权重，没有额外变换。

    这是一个可观测的等价关系，比断言「两者都是浮点数」强得多。
    """
    p = _prices()
    res = _wf(_Args(), p, ensemble_ks=(1,))
    keys = list(res["summary"]["key"])
    assert "ens1" in keys and "ens3" not in keys
    f = res["folds"]
    assert np.allclose(f["test_return_ens1"], f["test_return_smooth"]), (
        f["test_return_ens1"].tolist(), f["test_return_smooth"].tolist())
    # 同时 ens1 与 argmax 臂不必然相同 —— 否则上面的等价关系退化成「网格只有一组」
    assert "test_return" in f.columns


def test_ensemble_ks_controls_which_arms_are_produced():
    """K 列表直接决定报告里出现哪些集成臂（接线正确性）。"""
    p = _prices()
    res = _wf(_Args(), p, ensemble_ks=(1, 2, 0))
    keys = list(res["summary"]["key"])
    assert keys == ["walk_forward", "smooth", "ens1", "ens2", "ensall",
                    "full_sample_best", "default"]
    assert [r["key"] for r in res["ensemble_k_curve"]] == ["ens1", "ens2", "ensall"]
    # 逐折表里集成列必须与臂一一对应
    for key in ("ens1", "ens2", "ensall"):
        assert f"test_return_{key}" in res["folds"].columns
    assert "test_return_ens3" not in res["folds"].columns
    # 每条集成臂都有自己的净值曲线记录
    for key in ("ens1", "ens2", "ensall"):
        assert key in res["curves"] and len(res["curves"][key]) == len(res["folds"])


def test_walk_forward_accepts_already_parsed_k_tuple():
    """main.py 传的是已解析的元组（可能含 None），必须与传字符串等价。"""
    p = _prices()
    a = _wf(_Args(), p, ensemble_ks=(3, 5, 10, 0))
    b = _wf(_Args(), p, ensemble_ks=(3, 5, 10, None))
    assert list(a["summary"]["key"]) == list(b["summary"]["key"])
    np.testing.assert_allclose(
        a["summary"]["total_return"].to_numpy(), b["summary"]["total_return"].to_numpy())
    # 默认（不传 ensemble_ks）也必须与显式默认口径一致
    c = _wf(_Args(), p)
    np.testing.assert_allclose(
        a["summary"]["total_return"].to_numpy(), c["summary"]["total_return"].to_numpy())


def test_ensemble_top_picks_recorded_and_consistent():
    """逐折必须记录「最窄集成臂持有了哪几组」——这是判断集成是否只押一个模式的依据。"""
    p = _prices()
    res = _wf(_Args(timing="ma", timing_lookback=30), p, ensemble_ks=(2, 0),
              timing_modes=["off", "ma"], timing_lookbacks=[30], timing_bands=[0.0])
    f = res["folds"]
    assert "ens_top" in f.columns and "ens_top_modes" in f.columns
    for _, r in f.iterrows():
        # 最窄臂 K=2 → 恰好两组
        assert len(r["ens_top"].split(" / ")) == 2
        assert len(r["ens_top_modes"].split("+")) == 2
        # 每一组都必须能在候选标签里找到
        labels = {grid.combo_label(c) for c in grid.build_combos(
            [60, 120], [2], [0], ["off", "ma"], [30], [0.0])}
        for lb in r["ens_top"].split(" / "):
            assert lb in labels, (lb, labels)
    assert res["ens_note"], "含择时搜索时 ens_note 不该为空"


def test_ensemble_note_absent_without_timing_search():
    """没有搜索择时时不编造「持有哪些模式」的结论。"""
    p = _prices()
    res = _wf(_Args(), p)          # 不传 timing_modes → 只搜参数、不搜择时
    assert res["ens_note"] == ""
    assert res["timing_note"] == ""
