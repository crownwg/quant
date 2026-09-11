"""P1 修复的回归测试：成本模型 / 权重方案 / 容量约束 / 中性化 / 因子有效性 / 样本外验证。

这些测试大多锁的是**真实踩过的坑**，不是假想的边界情况：
  - 权重矩阵出现 NaN 会被 ffill 当成「继续持有」→ 回测里永远不卖出
  - 最低佣金若按当日汇总额算就不会生效（必须按笔数算）
  - 一字板的判据方向必须按涨/跌分开（涨停看 low、跌停看 high）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import factor_eval as fe
from quant import filters
from quant import grid
from quant import neutralize as nz
from quant import strategy
from quant.backtest import cost_kwargs, run


# ----------------------------------------------------------------- 造数据

def _prices(n_days: int = 500, n_codes: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, size=(n_days, n_codes)), axis=0)
    cols = [f"{600000 + i:06d}" for i in range(n_codes)]
    return pd.DataFrame(data, index=idx, columns=cols)


class _Args:
    """最小可用的 argparse 替身。"""

    def __init__(self, **kw):
        self.top_n = 3
        self.rebalance = "M"
        self.min_names = 1
        self.buffer = 0
        self.use_open = False
        self.weighting = "equal"
        self.max_weight = 0.0
        self.vol_lookback = 60
        self.lookback = 120
        self.fee = 0.0003
        self.stamp_tax = 0.0005
        self.slippage = 0.0
        self.min_commission = 0.0
        self.capital = 1_000_000.0
        self.impact_coef = 0.0
        for k, v in kw.items():
            setattr(self, k, v)


# ===================================================== 1. 成本模型

def test_min_commission_only_binds_for_small_capital():
    """最低佣金是「每笔」的下限：本金越小，它相对费率越重要。"""
    p = _prices()
    w = strategy.factor_weights(_prices(seed=1).rank(axis=1), top_n=4, freq="M")

    _, m_big, _ = run(p, w, fee=0.0003, min_commission=5.0, capital=1e9)
    _, m_small, _ = run(p, w, fee=0.0003, min_commission=5.0, capital=1e4)

    # 大本金下最低佣金形同虚设，两者应几乎相等
    _, m_big_no_floor, _ = run(p, w, fee=0.0003, min_commission=0.0, capital=1e9)
    assert m_big["total_commission"] == pytest.approx(m_big_no_floor["total_commission"], rel=1e-9)
    # 小本金下最低佣金必须显式抬高成本
    assert m_small["total_commission"] > m_big["total_commission"] * 3


def test_min_commission_zero_is_equivalent_to_old_behaviour():
    p = _prices()
    w = strategy.factor_weights(_prices(seed=1).rank(axis=1), top_n=4, freq="M")
    _, m0, _ = run(p, w, fee=0.0003, stamp_tax=0.0005)
    _, m1, _ = run(p, w, fee=0.0003, stamp_tax=0.0005, slippage=0.0,
                   min_commission=0.0, impact_coef=0.0)
    assert m0["total_cost"] == pytest.approx(m1["total_cost"])


def test_slippage_monotonically_increases_cost():
    p = _prices()
    w = strategy.factor_weights(_prices(seed=1).rank(axis=1), top_n=4, freq="M")
    _, base, _ = run(p, w, slippage=0.0)
    _, mid, _ = run(p, w, slippage=0.001)
    _, high, _ = run(p, w, slippage=0.003)

    assert base["total_slippage"] == 0.0
    assert mid["total_slippage"] > 0
    assert high["total_slippage"] > mid["total_slippage"]
    assert high["total_return"] < base["total_return"]


def test_impact_cost_grows_as_adv_shrinks():
    """冲击成本 = coef × sqrt(成交额 / ADV)：ADV 越小冲击越大。"""
    p = _prices()
    w = strategy.factor_weights(_prices(seed=1).rank(axis=1), top_n=4, freq="M")
    idx, cols = w.index, w.columns

    adv_rich = pd.DataFrame(1e12, index=idx, columns=cols)
    adv_poor = pd.DataFrame(1e5, index=idx, columns=cols)

    _, m_rich, _ = run(p, w, impact_coef=0.1, capital=1e7, adv=adv_rich)
    _, m_poor, _ = run(p, w, impact_coef=0.1, capital=1e7, adv=adv_poor)

    assert m_poor["total_impact"] > m_rich["total_impact"]
    # 参与率被截在 1.0，所以冲击率上限是 coef
    assert m_poor["total_impact"] > 0


def test_impact_requires_adv_and_capital():
    p = _prices()
    w = strategy.factor_weights(_prices(seed=1).rank(axis=1), top_n=4, freq="M")
    _, m, _ = run(p, w, impact_coef=0.1, capital=1e7, adv=None)
    assert m["total_impact"] == 0.0


def test_cost_kwargs_supports_namespace_and_dict():
    ns = _Args(slippage=0.001, min_commission=5.0)
    kw = cost_kwargs(ns)
    assert kw["slippage"] == 0.001 and kw["min_commission"] == 5.0
    kw2 = cost_kwargs({"fee": 0.0002, "slippage": 0.002})
    assert kw2["fee"] == 0.0002 and kw2["slippage"] == 0.002


def test_cost_kwargs_drops_adv_when_impact_disabled():
    """冲击成本关闭时不该把 ADV 传进去（否则白白做一次大矩阵运算）。"""
    kw = cost_kwargs(_Args(impact_coef=0.0), adv=pd.DataFrame({"A": [1.0]}))
    assert kw["adv"] is None


# ===================================================== 2. 权重方案

def _score_panel(n_days=250, n_codes=8, seed=2):
    p = _prices(n_days, n_codes, seed)
    return p.pct_change(20)


@pytest.mark.parametrize("scheme", ["equal", "inv_vol", "score", "rank"])
def test_weighting_schemes_sum_to_one(scheme):
    p = _prices()
    score = _score_panel()
    vol = strategy.realized_vol(p, 60)
    w = strategy.factor_weights(score, top_n=4, freq="M", weighting=scheme, vol=vol)

    rows = w[w.sum(axis=1) > 0]
    assert not rows.empty
    assert np.allclose(rows.sum(axis=1), 1.0, atol=1e-9)


def test_inv_vol_gives_more_weight_to_lower_volatility():
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = ["600000", "600001", "600002"]
    # 三只票分数相同，只有波动率不同
    score = pd.DataFrame(1.0, index=idx, columns=cols)
    vol = pd.DataFrame({"600000": [0.01] * 5, "600001": [0.02] * 5, "600002": [0.04] * 5},
                       index=idx)
    w = strategy.factor_weights(score, top_n=3, freq="M", weighting="inv_vol", vol=vol)
    row = w.iloc[0]
    assert row["600000"] > row["600001"] > row["600002"]
    assert row.sum() == pytest.approx(1.0)


def test_ties_in_inv_vol_fall_back_to_equal():
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = ["600000", "600001"]
    score = pd.DataFrame(1.0, index=idx, columns=cols)
    vol = pd.DataFrame(0.02, index=idx, columns=cols)   # 波动率完全相同
    w = strategy.factor_weights(score, top_n=2, freq="M", weighting="inv_vol", vol=vol)
    assert w.iloc[0]["600000"] == pytest.approx(0.5)


def test_rank_weighting_is_decreasing_by_score():
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = ["600000", "600001", "600002", "600003"]
    score = pd.DataFrame([[4.0, 3.0, 2.0, 1.0]] * 5, index=idx, columns=cols)
    w = strategy.factor_weights(score, top_n=4, freq="M", weighting="rank")
    row = w.iloc[0]
    assert row["600000"] > row["600001"] > row["600002"] > row["600003"]


# ===================================================== 3. 单票上限

def test_max_weight_cap_truncates_and_redistributes():
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = [f"60000{i}" for i in range(5)]
    score = pd.DataFrame([[5.0, 4.0, 3.0, 2.0, 1.0]] * 5, index=idx, columns=cols)
    # rank 加权下第一名本来会拿到 5/15 = 33.3%，上限 30% 逼它把超额吐出来
    w = strategy.factor_weights(score, top_n=5, freq="M", weighting="rank",
                                max_weight=0.30)
    row = w.iloc[0]
    assert (row <= 0.30 + 1e-12).all()
    # 5×30% = 150% > 100%，所以超额能被未触顶的标的完全吸收，权重仍满仓
    assert row.sum() == pytest.approx(1.0)
    # 且第一名被削到上限附近，不再是原始的 33.3%
    assert row["600000"] < 5 / 15


def test_all_capped_leaves_cash_instead_of_forcing_full_investment():
    """所有标的都触顶时，剩余权重留现金——不该为了满仓去违反上限。"""
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = [f"60000{i}" for i in range(5)]
    score = pd.DataFrame(1.0, index=idx, columns=cols)
    # 5 只票等权 = 每只 20%，上限 10% → 最多用掉 50%
    w = strategy.factor_weights(score, top_n=5, freq="M", max_weight=0.10)
    row = w.iloc[0]
    assert row.sum() == pytest.approx(0.5, abs=1e-9)
    assert (row <= 0.10 + 1e-12).all()


def test_weight_cap_nan_means_unconstrained():
    w = pd.Series({"A": 0.5, "B": 0.5})
    caps = pd.Series({"A": np.nan, "B": 0.2})
    out = strategy._apply_weight_caps(w, caps)
    assert out["A"] == pytest.approx(0.8)
    assert out["B"] == pytest.approx(0.2)


def test_weight_cap_zero_blocks_holding():
    w = pd.Series({"A": 0.5, "B": 0.5})
    caps = pd.Series({"A": 0.0, "B": 1.0})
    out = strategy._apply_weight_caps(w, caps)
    assert out["A"] == pytest.approx(0.0)
    assert out["B"] == pytest.approx(1.0)


# ===================================================== 4. 权重矩阵完整性（真实 bug 回归）

def test_weights_matrix_has_no_nan_after_first_rebalance():
    """回归：权重矩阵若出现 NaN，末尾的 ffill 会把它当成「沿用上期持仓」，
    导致组合永远不卖出（真实踩过：13 次调仓 382 笔订单里 0 笔卖出）。"""
    p = _prices(n_days=600, n_codes=10, seed=3)
    score = p.rank(axis=1)          # 排名每天变化 → 应该持续换仓
    w = strategy.factor_weights(score, top_n=4, freq="M")
    assert not w.isna().any().any()
    assert (w.sum(axis=1) > 0).all()


def test_rebalancing_actually_sells():
    p = _prices(n_days=600, n_codes=10, seed=4)
    score = p.rank(axis=1)
    w = strategy.factor_weights(score, top_n=4, freq="M")
    _, m, detail = run(p, w)
    assert detail["sell_turnover"].sum() > 0
    assert m["total_return"] != 0


# ===================================================== 5. 容量约束

def test_capacity_cap_formula():
    idx = pd.bdate_range("2020-01-01", periods=3)
    adv = pd.DataFrame({"A": [1e8] * 3}, index=idx)
    cap = filters.capacity_cap(adv, capital=1e7, max_participation=0.1)
    # 0.1 × 1e8 / 1e7 = 1.0
    assert cap.iloc[0]["A"] == pytest.approx(1.0)


def test_capacity_cap_nan_when_adv_missing():
    """ADV 为 0（停牌/数据缺失）时必须返回 NaN（无约束），
    返回 0 会把该标的的仓位直接清零。"""
    idx = pd.bdate_range("2020-01-01", periods=3)
    adv = pd.DataFrame({"A": [np.nan, 0.0, 1e8]}, index=idx)
    cap = filters.capacity_cap(adv, capital=1e7, max_participation=0.1)
    assert np.isnan(cap.iloc[0]["A"])
    assert np.isnan(cap.iloc[1]["A"])
    assert cap.iloc[2]["A"] > 0


def test_capacity_cap_disabled_returns_none():
    idx = pd.bdate_range("2020-01-01", periods=3)
    adv = pd.DataFrame({"A": [1e8] * 3}, index=idx)
    assert filters.capacity_cap(adv, 1e7, 0.0) is None
    assert filters.capacity_cap(None, 1e7, 0.1) is None


def test_adv_notional_ignores_suspension_zeros():
    """停牌日成交额为 0，必须剔除而非计入均值——否则一次停牌会把 ADV 拉低一个量级。"""
    idx = pd.bdate_range("2020-01-01", periods=30)
    amount = pd.DataFrame({"A": [1e8] * 29 + [0.0]}, index=idx)
    adv = filters.adv_notional(amount=amount, window=20, min_periods=5)
    assert adv.iloc[-1]["A"] == pytest.approx(1e8)   # 0 被剔除，均值仍是 1e8
    bad = amount.rolling(20, min_periods=5).mean()   # 反面：不剔除会掉到 19/20
    assert bad.iloc[-1]["A"] == pytest.approx(1e8 * 19 / 20)


def test_illiquid_mask_by_amount_uses_money_not_shares():
    idx = pd.bdate_range("2020-01-01", periods=30)
    amount = pd.DataFrame({"CHEAP": [1e7] * 30, "RICH": [1e9] * 30}, index=idx)
    m = filters.illiquid_mask_by_amount(amount, min_amount=1e8)
    assert bool(m.iloc[-1]["CHEAP"]) is True     # 成交额不足 → 不可交易
    assert bool(m.iloc[-1]["RICH"]) is False


# ===================================================== 6. 中性化

def test_industry_demean_makes_group_means_zero():
    idx = pd.bdate_range("2020-01-01", periods=3)
    cols = ["A1", "A2", "A3", "B1", "B2", "B3"]
    f = pd.DataFrame([[10.0, 12.0, 14.0, 100.0, 102.0, 104.0]] * 3,
                     index=idx, columns=cols)
    industry = pd.Series({"A1": "A", "A2": "A", "A3": "A",
                          "B1": "B", "B2": "B", "B3": "B"})
    out = nz.neutralize(f, industry=industry)
    assert out.loc[idx[0], ["A1", "A2", "A3"]].mean() == pytest.approx(0.0, abs=1e-12)
    assert out.loc[idx[0], ["B1", "B2", "B3"]].mean() == pytest.approx(0.0, abs=1e-12)
    # 行业内排序必须保持不变（去均值只是平移）
    assert out.loc[idx[0], "A1"] < out.loc[idx[0], "A3"]


def test_single_industry_pool_ranking_unchanged():
    """池子只有一个行业时，行业中性化只是全体平移，选股结果不该变。"""
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = ["X1", "X2", "X3"]
    f = pd.DataFrame([[3.0, 1.0, 2.0]] * 5, index=idx, columns=cols)
    industry = pd.Series({c: "同一行业" for c in cols})
    out = nz.neutralize(f, industry=industry)
    # 各列之间的相对大小关系必须一致
    assert (out.iloc[0].rank().tolist() == f.iloc[0].rank().tolist())


def test_unmapped_stocks_pass_through_unchanged():
    idx = pd.bdate_range("2020-01-01", periods=3)
    cols = ["A1", "A2", "Z9"]
    f = pd.DataFrame([[1.0, 2.0, 99.0]] * 3, index=idx, columns=cols)
    industry = pd.Series({"A1": "A", "A2": "A"})     # Z9 没有行业
    out = nz.neutralize(f, industry=industry)
    assert out.iloc[0]["Z9"] == pytest.approx(99.0)


def test_size_bucket_demean_removes_size_effect():
    """构造一个「因子值 = 市值」的面板，市值中性化后因子应该被抹平。"""
    idx = pd.bdate_range("2020-01-01", periods=3)
    cols = [f"C{i}" for i in range(10)]
    caps = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    f = pd.DataFrame([caps] * 3, index=idx, columns=cols)
    mktcap = pd.DataFrame([caps * 1e9] * 3, index=idx, columns=cols)
    out = nz.neutralize(f, mktcap=mktcap, size_buckets_n=5)
    # 每个桶内去均值后，桶内均值应为 0
    rank = mktcap.rank(axis=1, pct=True)
    bucket = np.ceil(rank * 5)
    for b in range(1, 6):
        m = (bucket == b).iloc[0]
        vals = out.iloc[0][m.values]
        assert vals.mean() == pytest.approx(0.0, abs=1e-9)


def test_parse_neutralize_spec():
    assert nz.parse_spec("") == set()
    assert nz.parse_spec("industry") == {"industry"}
    assert nz.parse_spec("industry,size") == {"industry", "size"}
    with pytest.raises(ValueError):
        nz.parse_spec("market")


def test_mktcap_uses_raw_vwap_not_adjusted_price():
    """市值必须用真实成交均价（成交额/成交量），不能用前复权收盘价。
    这里令复权价是真实价的 1/10，市值应基于真实价计算。"""
    idx = pd.bdate_range("2020-01-01", periods=2)
    close = pd.DataFrame({"A": [10.0, 11.0]}, index=idx)        # 前复权价
    volume = pd.DataFrame({"A": [1000.0, 1000.0]}, index=idx)
    amount = pd.DataFrame({"A": [100000.0, 100000.0]}, index=idx)  # 真实均价 100
    share = pd.DataFrame({"A": [1e6, 1e6]}, index=idx)
    mc = nz.mktcap_panel(close, amount=amount, volume=volume, outstanding_share=share)
    assert mc.iloc[0]["A"] == pytest.approx(100.0 * 1e6)


# ===================================================== 7. 因子有效性

def test_forward_returns_are_shifted_negative():
    idx = pd.bdate_range("2020-01-01", periods=5)
    p = pd.DataFrame({"A": [1.0, 2.0, 4.0, 8.0, 16.0]}, index=idx)
    fwd = fe.forward_returns(p, horizon=1)
    assert fwd.iloc[0]["A"] == pytest.approx(1.0)      # 2/1 - 1
    assert pd.isna(fwd.iloc[-1]["A"])                  # 最后一行没有未来


def test_ic_is_one_for_perfectly_rank_correlated_factor():
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = [f"C{i}" for i in range(6)]
    prices = pd.DataFrame([100 + np.arange(6), 110 + 2 * np.arange(6),
                           120 + 3 * np.arange(6), 130 + 4 * np.arange(6),
                           140 + 5 * np.arange(6)], index=idx, columns=cols).astype(float)
    factor = prices.copy()                              # 因子 = 当期价格
    ic = fe.ic_series(factor, fe.forward_returns(prices, 1), min_pairs=3)
    # 价格单调递增 → 未来收益也单调 → 秩相关应为 1
    assert ic.dropna().iloc[0] == pytest.approx(1.0)


def test_ic_is_negative_when_factor_inverted():
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = [f"C{i}" for i in range(6)]
    prices = pd.DataFrame([100 + np.arange(6), 110 + 2 * np.arange(6),
                           120 + 3 * np.arange(6), 130 + 4 * np.arange(6),
                           140 + 5 * np.arange(6)], index=idx, columns=cols).astype(float)
    ic = fe.ic_series(-prices, fe.forward_returns(prices, 1), min_pairs=3)
    assert ic.dropna().iloc[0] == pytest.approx(-1.0)


def test_ic_summary_math():
    s = pd.Series([0.1, 0.2, 0.3, 0.4])
    out = fe.ic_summary(s)
    assert out["n"] == 4
    assert out["mean_ic"] == pytest.approx(0.25)
    assert out["ir"] == pytest.approx(0.25 / s.std())
    assert out["t_stat"] == pytest.approx(out["ir"] * 2)
    assert out["positive_ratio"] == pytest.approx(1.0)


def test_ic_summary_handles_empty():
    out = fe.ic_summary(pd.Series([], dtype=float))
    assert out["n"] == 0 and out["verdict"] == "样本不足"


def test_quantile_returns_are_monotonic_for_a_real_factor():
    p = _prices(n_days=500, n_codes=30, seed=7)
    # 用未来收益本身当因子 → 分组收益必然单调递增（构造性验证）
    factor = fe.forward_returns(p, 20)
    q = fe.quantile_returns(factor, p, n_groups=5, horizon=20)
    assert len(q) == 5
    assert q["mean_return"].is_monotonic_increasing
    assert fe.long_short_spread(q) > 0


def test_ic_decay_returns_one_row_per_horizon():
    p = _prices(n_days=300, n_codes=15, seed=8)
    factor = p.pct_change(60)
    d = fe.ic_decay(factor, p, horizons=(1, 5, 20))
    assert d["horizon"].tolist() == [1, 5, 20]
    assert d["n"].gt(0).all()


def test_min_pairs_filters_thin_cross_sections():
    idx = pd.bdate_range("2020-01-01", periods=3)
    cols = [f"C{i}" for i in range(20)]
    factor = pd.DataFrame(np.nan, index=idx, columns=cols)
    factor.iloc[0, :3] = [1.0, 2.0, 3.0]                 # 只有 3 个有效样本
    factor.iloc[1, :] = np.arange(20, dtype=float)       # 20 个有效样本
    fwd = pd.DataFrame(np.tile(np.arange(20, dtype=float), (3, 1)), index=idx, columns=cols)
    ic = fe.ic_series(factor, fwd, min_pairs=10)
    assert pd.isna(ic.iloc[0])
    assert not pd.isna(ic.iloc[1])


# ===================================================== 8. 样本外验证

def test_walk_forward_folds_do_not_overlap_on_test_window():
    folds = grid.walk_forward_folds(n=2000, start_pos=0, train_n=500, test_n=125)
    for (s1, se1, te1), (s2, se2, te2) in zip(folds, folds[1:]):
        assert te1 <= se2        # 测试窗之间不重叠
        assert se1 == s1 + 500   # 训练窗长度固定
        assert te1 - se1 == 125  # 测试窗长度固定


def test_walk_forward_folds_train_always_precedes_test():
    for s, se, te in grid.walk_forward_folds(n=1500, start_pos=0, train_n=400, test_n=100):
        assert s < se < te


def test_walk_forward_folds_empty_when_sample_too_short():
    assert grid.walk_forward_folds(n=300, start_pos=0, train_n=500, test_n=125) == []


def test_walk_forward_rejects_too_short_windows():
    p = _prices(n_days=800, n_codes=6)
    args = _Args(use_open=False)
    with pytest.raises(ValueError):
        grid.walk_forward_search(p, p, None, None, args, [60], [3], [0],
                                 pd.Timestamp(p.index[0]), train_years=0.1, test_years=0.02)


def test_walk_forward_runs_and_reports_all_arms():
    """walk-forward 必须给出全部对照组：两条选参方式 + 多个集成度 + 事后选参 + 不调参。

    为什么是这么多条而不是最初 3 条：实测「训练窗取 argmax」挑到的多半是噪声
    （滞回带 0 与 2% 各折各半），所以补了「邻域平滑」与「参数集成」两条不做
    单点择优的对照——没有它们就分不清「参数有效」和「选择本身带来的虚假优势」。
    集成又按 K（只在平滑分前 K 组内平均）展开成多条臂，是为了不把「K 取多少」
    变成新一轮事后择优：K 本身也必须作为一维诚实地摆出来。
    """
    p = _prices(n_days=1500, n_codes=10, seed=11)
    args = _Args(use_open=False)
    res = grid.walk_forward_search(p, p, None, None, args, [60, 120], [3, 5], [0],
                                   pd.Timestamp(p.index[0]), train_years=2, test_years=0.5)
    assert len(res["folds"]) >= 2
    keys = list(res["summary"]["key"])
    assert keys[:2] == ["walk_forward", "smooth"]
    assert keys[-2:] == ["full_sample_best", "default"]
    ens = keys[2:-2]
    assert ens, "至少要有一条集成臂"
    # 集成臂名形如 ens3 / ens10 / ensall，且顺序与 K 列表一致
    assert all(k.startswith("ens") for k in ens), ens
    assert ens == grid.ensemble_arm_keys(grid.parse_ensemble_ks(None))
    assert set(res["curves"]) == set(keys)
    # 每一折都必须给出训练期与测试期两套指标
    for _, r in res["folds"].iterrows():
        assert "train_sharpe" in r and "test_sharpe" in r
        assert r["train_start"] < r["test_start"]


# ===================================================== 9. 统一配置入口

def test_weights_from_args_accepts_dict_and_namespace():
    score = _score_panel(n_days=200, n_codes=6, seed=12)
    w_ns = strategy.weights_from_args(score, _Args(), prices=_prices(200, 6, 12))
    w_dict = strategy.weights_from_args(score, {"top_n": 3, "rebalance": "M",
                                                "use_open": False, "weighting": "equal"},
                                        prices=_prices(200, 6, 12))
    assert w_ns.shape == score.shape and w_dict.shape == score.shape


def test_weights_from_args_applies_exec_shift():
    score = _score_panel(n_days=200, n_codes=6, seed=13)
    cb = pd.DataFrame(True, index=score.index, columns=score.columns)
    w0 = strategy.weights_from_args(score, _Args(use_open=False), can_buy=cb)
    w1 = strategy.weights_from_args(score, _Args(use_open=True), can_buy=cb)
    assert w0.shape == w1.shape


def test_weights_from_args_overrides_take_precedence():
    score = _score_panel(n_days=200, n_codes=8, seed=14)
    p = _prices(200, 8, 14)
    w = strategy.weights_from_args(score, _Args(top_n=3), prices=p, top_n=2)
    # 每行最多 2 个非零权重
    assert (w.gt(0).sum(axis=1) <= 2).all()
