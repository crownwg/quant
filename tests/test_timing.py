"""择时 / 仓位管理（quant/timing.py）的回归测试。

重点锁三件事：
  1. **因果性**——改动「未来」的价格，不得影响「过去」的敞口。
     这是择时模块唯一会静默毁掉整个回测的错法：一旦偷看未来，
     净值会好看得离谱，而且没有任何别的检查能发现。
  2. 滞回带的语义——进场门槛与出场门槛必须不对称，中间地带维持原状态。
  3. 敞口缩放的语义——敞口 0 必须是真空仓、敞口 0.5 必须刚好一半资金，
     且缩放发生在单票上限之后（不能把上限约束绕过）。

运行：.venv/Scripts/python.exe -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import strategy, timing


# ----------------------------------------------------------------- 造数据

def _series(n: int = 400, pattern: str = "updown") -> pd.Series:
    """构造一段确定性走势：先涨后跌（updown）/ 单调上涨（up）/ 横盘（flat）。"""
    idx = pd.bdate_range("2020-01-01", periods=n)
    if pattern == "up":
        v = np.linspace(100.0, 200.0, n)
    elif pattern == "flat":
        v = np.full(n, 100.0)
    else:
        half = n // 2
        v = np.concatenate([np.linspace(100.0, 200.0, half),
                            np.linspace(200.0, 120.0, n - half)])
    return pd.Series(v, index=idx)


def _prices(n_days: int = 300, n_codes: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, size=(n_days, n_codes)), axis=0)
    return pd.DataFrame(data, index=idx, columns=[f"{600000 + i:06d}" for i in range(n_codes)])


class _Cfg:
    """最小可用的 argparse 替身（只带择时相关字段）。"""

    def __init__(self, **kw):
        self.timing = "off"
        self.timing_lookback = 60
        self.timing_band = 0.0
        self.timing_ma_slope = 0
        self.timing_min_exposure = 0.0
        self.timing_max_exposure = 1.0
        self.timing_smooth = 0
        self.vol_target = 0.0
        self.vol_target_lookback = 60
        self.vol_floor = 0.2
        self.vol_cap = 1.0
        self.weighting = "equal"
        self.top_n = 2
        self.rebalance = "M"
        self.min_names = 1
        self.buffer = 0
        self.max_weight = 0.0
        self.use_open = False
        for k, v in kw.items():
            setattr(self, k, v)


# -------------------------------------------------------- 1. 因果性（最关键）

@pytest.mark.parametrize("mode", ["ma", "momentum", "dual"])
def test_trend_exposure_is_causal(mode):
    """篡改未来价格，不得改变过去任一日的敞口。

    做法：把后半段价格整体乘 10（制造一段不存在的暴涨），
    逐点比较两版敞口在中点之前的取值——必须完全相同。
    """
    base = _series(400, "updown")
    cut = 250
    tampered = base.copy()
    tampered.iloc[cut:] = tampered.iloc[cut:] * 10.0

    ex_base = timing.trend_exposure(base, lookback=60, mode=mode, band=0.01)
    ex_tamp = timing.trend_exposure(tampered, lookback=60, mode=mode, band=0.01)

    past = ex_base.index[:cut]
    pd.testing.assert_series_equal(ex_base.loc[past], ex_tamp.loc[past])


def test_trend_exposure_reacts_to_recent_history():
    """反向验证：改**近期**历史必须改变敞口。

    这条是为了防「因果性测试假通过」——如果信号压根没生效，
    改未来不改过去也会通过。横盘时永不在场，把最近 50 天（落在 60 日窗口内）
    拉高后必须转为在场。注意：更早于窗口的历史**本来就不应该**影响当前敞口，
    那是正确的滚动窗口语义，不是 bug。
    """
    base = _series(400, "flat")
    ex_a = timing.trend_exposure(base, lookback=60, mode="ma")
    assert ex_a.sum() == 0.0, "横盘（价格=均线）不应在场"

    changed = base.copy()
    changed.iloc[350:] = 130.0
    ex_b = timing.trend_exposure(changed, lookback=60, mode="ma")
    assert ex_b.iloc[-1] == 1.0, "近期拉高应转为在场"


def test_vol_target_exposure_is_causal():
    """波动率目标同样不得偷看未来：篡改未来收益不影响过去的敞口。"""
    proxy = _series(300, "updown")
    tampered = proxy.copy()
    tampered.iloc[200:] = tampered.iloc[200:] * 3.0
    a = timing.vol_target_exposure(proxy, target_vol=0.15, lookback=60)
    b = timing.vol_target_exposure(tampered, target_vol=0.15, lookback=60)
    pd.testing.assert_series_equal(a.loc[a.index[:200]], b.loc[b.index[:200]])


# -------------------------------------------------------- 2. 趋势择时的语义

def test_trend_exposure_is_binary_and_tracks_direction():
    """先涨后跌：前半段在场、后半段落袋离场。"""
    ex = timing.trend_exposure(_series(400, "updown"), lookback=60, mode="ma")
    assert set(np.unique(ex.values)) <= {0.0, 1.0}
    assert ex.iloc[180] == 1.0, "上涨段应当在场"
    assert ex.iloc[-1] == 0.0, "跌破均线后应当离场"


def test_trend_exposure_flat_market_stays_out():
    """横盘（价格恰好等于均线）→ 信号恒为 0，不应触发进场。"""
    ex = timing.trend_exposure(_series(200, "flat"), lookback=60, mode="ma", band=0.0)
    assert ex.sum() == 0.0


def test_band_creates_hysteresis():
    """滞回带必须让进场/出场门槛不对称：中间地带维持原状态。

    构造：价格长期 100 → 跳到 105 并维持足够久（让均线收敛到 105）→ 回落到 103.5。
    回落幅度 -1.4%：无缓冲时价格已跌破均线（离场）；有 2% 缓冲时应继续持有。
    """
    n = 340
    idx = pd.bdate_range("2020-01-01", periods=n)
    v = np.full(n, 100.0)
    v[120:280] = 105.0      # 足够长，让 60 日均线收敛到 105
    v[280:] = 103.5         # 回落 1.4%
    proxy = pd.Series(v, index=idx)

    wide = timing.trend_exposure(proxy, lookback=60, mode="ma", band=0.02)
    tight = timing.trend_exposure(proxy, lookback=60, mode="ma", band=0.0)

    assert tight.iloc[279] == 1.0 and wide.iloc[279] == 1.0, "高点时都应在场"
    assert wide.iloc[-1] == 1.0, "有 2% 缓冲时，1.4% 的回落不应触发离场"
    assert tight.iloc[-1] == 0.0, "无缓冲时价格跌破均线即离场"
    # 带缓冲的版本切换次数不应更多
    assert int((wide.diff().abs() > 1e-9).sum()) <= int((tight.diff().abs() > 1e-9).sum())


def test_min_max_exposure_bounds():
    """最低仓位约束：看空也应保留指定仓位；上限不得超过。"""
    ex = timing.trend_exposure(_series(400, "updown"), lookback=60, mode="ma",
                              min_exposure=0.3, max_exposure=0.8)
    assert ex.min() == pytest.approx(0.3)
    assert ex.max() == pytest.approx(0.8)


def test_dual_mode_is_stricter_than_ma():
    """dual（均线+斜率双确认）的在场时间不应多于单纯 ma。"""
    proxy = _series(400, "updown")
    ma = timing.trend_exposure(proxy, lookback=60, mode="ma")
    dual = timing.trend_exposure(proxy, lookback=60, mode="dual")
    assert dual.sum() <= ma.sum()


# -------------------------------------------------------- 3. 波动率目标

def test_vol_target_方向性_高波动低仓位():
    """波动越大仓位越低——这是波动率目标的定义，写反了就成了放大器。"""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=400)
    low_vol = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.005, 400)), index=idx)
    high_vol = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.045, 400)), index=idx)

    ex_low = timing.vol_target_exposure(low_vol, target_vol=0.15, lookback=60)
    ex_high = timing.vol_target_exposure(high_vol, target_vol=0.15, lookback=60)
    assert ex_high.mean() < ex_low.mean()


def test_vol_target_respects_floor_and_cap():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=300)
    wild = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.08, 300)), index=idx)
    ex = timing.vol_target_exposure(wild, target_vol=0.15, lookback=60,
                                    floor=0.25, cap=0.9)
    assert ex.max() <= 0.9 + 1e-12
    assert ex.min() >= 0.25 - 1e-12


# -------------------------------------------------------- 4. 池子等权组合

def test_pool_index_equals_equal_weight_portfolio():
    """等权组合净值应等于各票日收益等权平均后的复利。"""
    p = _prices(60, 3)
    idx_series = timing.pool_index(p)
    expected = (1 + p.pct_change().mean(axis=1).fillna(0.0)).cumprod()
    pd.testing.assert_series_equal(idx_series, expected, check_names=False)


def test_pool_index_handles_missing_names():
    """池里某只票尚未上市（全 NaN 在前面）时不应把组合打成 NaN。"""
    idx = pd.bdate_range("2020-01-01", periods=20)
    a = pd.Series(np.linspace(10, 12, 20), index=idx)
    b = pd.Series([np.nan] * 10 + list(np.linspace(20, 22, 10)), index=idx)
    p = pd.DataFrame({"A": a, "B": b})
    out = timing.pool_index(p)
    assert out.notna().all()
    assert out.iloc[-1] > 0


# -------------------------------------------------------- 5. 敞口缩放

def test_apply_exposure_scales_weights():
    """敞口 0.5 → 权重整体减半（剩余是现金，不是 bug）；敞口 0 → 真空仓。"""
    n = 30
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = ["600000", "600001"]
    w = pd.DataFrame(0.5, index=idx, columns=cols)
    ex = pd.Series([1.0] * 10 + [0.5] * 10 + [0.0] * 10, index=idx)

    out = strategy.apply_exposure(w, ex)
    assert out.iloc[5].sum() == pytest.approx(1.0)
    assert out.iloc[15].sum() == pytest.approx(0.5)
    assert out.iloc[25].sum() == pytest.approx(0.0)


def test_apply_exposure_none_is_identity():
    idx = pd.bdate_range("2020-01-01", periods=5)
    w = pd.DataFrame(0.5, index=idx, columns=["600000", "600001"])
    pd.testing.assert_frame_equal(strategy.apply_exposure(w, None), w)


def test_weights_from_args_applies_exposure_after_caps():
    """单票上限 20% + 3 只票 + 敞口 0.5 的联合约束。

    等权 3 只本应各 1/3，被 20% 上限截住后三只全部触顶 → 只能用到 60%，
    剩余 40% 是现金（这是 _apply_weight_caps 的既定语义：全员触顶就留现金）。
    再乘敞口 0.5 → 总仓位 30%。

    这个用例锁的是**顺序**：缩放必须发生在上限之后。若反过来先缩放再施加上限，
    上限会变成「缩放前权重的上限」，约束就失效了。
    """
    cfg = _Cfg(max_weight=0.2, top_n=3)
    n = 40
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = ["600000", "600001", "600002"]
    score = pd.DataFrame([[3.0, 2.0, 1.0]] * n, index=idx, columns=cols)
    ex = pd.Series(0.5, index=idx)

    w = strategy.weights_from_args(score, cfg, exposure=ex)
    row = w.iloc[-1]
    assert (row <= 0.2 + 1e-12).all(), "单票上限必须仍然成立"
    assert row.sum() == pytest.approx(0.3), "60% 上限用量 × 0.5 敞口 = 30%"
    assert (row <= 0.2 * 0.5 + 1e-12).all()


def test_weights_from_args_exposure_full_investment_when_unconstrained():
    """上限不构成约束时，敞口 0.5 必须刚好是一半资金。"""
    cfg = _Cfg(max_weight=0.3, top_n=4)
    n = 40
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = ["600000", "600001", "600002", "600003"]
    score = pd.DataFrame([[4.0, 3.0, 2.0, 1.0]] * n, index=idx, columns=cols)
    ex = pd.Series(0.5, index=idx)

    w = strategy.weights_from_args(score, cfg, exposure=ex)
    assert w.iloc[-1].sum() == pytest.approx(0.5)


def test_weights_from_args_exposure_none_matches_plain():
    """不传敞口时行为与改动前一致（向后兼容）。"""
    cfg = _Cfg()
    n = 40
    idx = pd.bdate_range("2020-01-01", periods=n)
    score = pd.DataFrame([[3.0, 2.0, 1.0]] * n, index=idx,
                         columns=["600000", "600001", "600002"])
    a = strategy.weights_from_args(score, cfg)
    b = strategy.weights_from_args(score, cfg, exposure=None)
    pd.testing.assert_frame_equal(a, b)
    assert a.iloc[-1].sum() == pytest.approx(1.0)


# -------------------------------------------------------- 6. 组装与开关

def test_build_exposure_returns_none_when_disabled():
    assert timing.build_exposure(_Cfg(), prices=_prices(100)) is None


def test_build_exposure_combines_trend_and_vol():
    """趋势 × 波动率：两者叠加时，敞口不得超过任一单独作用的结果。"""
    cfg = _Cfg(timing="ma", vol_target=0.15)
    p = _prices(400, 6, seed=3)
    proxy = timing.pool_index(p)
    trend = timing.trend_exposure(proxy, lookback=60)
    vol = timing.vol_target_exposure(proxy, target_vol=0.15, lookback=60)
    combo = timing.build_exposure(cfg, proxy=proxy, verbose=False)
    assert combo is not None
    assert (combo <= trend + 1e-9).all()
    assert (combo <= vol + 1e-9).all()


def test_exposure_summary_counts_switches():
    idx = pd.bdate_range("2020-01-01", periods=10)
    ex = pd.Series([1, 1, 1, 0, 0, 1, 1, 1, 1, 1], index=idx, dtype=float)
    s = timing.exposure_summary(ex)
    assert s["switches"] == 2, "1→0、0→1 各算一次"
    assert s["avg_exposure"] == pytest.approx(0.8)
    assert s["in_market_ratio"] == pytest.approx(0.8)


def test_smooth_exposure_reduces_total_variation():
    """平滑的意义是降低仓位自身的换手幅度（总变差），而不是让每天的 diff 变成 0。

    对 0/1 交替的信号做移动平均，结果仍是每天变化的小数——
    所以用「总变差」衡量，不用「变化天数」。
    """
    idx = pd.bdate_range("2020-01-01", periods=100)
    ex = pd.Series([1, 0] * 50, index=idx, dtype=float)
    sm = timing.smooth_exposure(ex, 5)
    assert sm.diff().abs().sum() < ex.diff().abs().sum()
    assert sm.min() >= 0.0 and sm.max() <= 1.0


def test_smooth_exposure_reduces_switches_for_long_blocks():
    """长段位的 0/1 信号加平滑后，小幅噪声不再触发仓位翻转。"""
    idx = pd.bdate_range("2020-01-01", periods=200)
    raw = np.ones(200)
    raw[50:53] = 0.0        # 三天的小回撤（噪声）
    raw[120:135] = 0.0      # 真正的趋势离场
    ex = pd.Series(raw, index=idx)
    sm = timing.smooth_exposure(ex, 10)
    # 平滑后中间那段噪声不再把仓位打到 0（仍保持明显过半）
    assert sm.iloc[52] > 0.5
    assert sm.iloc[130] < 0.5


def test_norm_index_symbol():
    assert timing.norm_index_symbol("000932") == "sh000932"
    assert timing.norm_index_symbol("399006") == "sz399006"
    assert timing.norm_index_symbol("sh000300") == "sh000300"
    assert timing.norm_index_symbol("") == ""


# -------------------------------------------------------- 7. 敞口更新频率

def test_align_exposure_only_updates_on_rebalance_days():
    """默认只在调仓日更新敞口——月内保持不变。

    这是修掉「波动率目标把月频策略变成日频对倒」的关键机制：
    逐日敞口直接乘到权重上时，实测跑出 1125 次调仓 / 17041 笔订单，
    成本把收益全部吃光，且现实中不可能执行。
    """
    idx = pd.bdate_range("2020-01-01", periods=90)
    daily_ex = pd.Series(np.linspace(0.1, 0.9, 90), index=idx)   # 逐日变化

    aligned = strategy.align_exposure(daily_ex, idx, "M", mode="rebalance")

    # 同一自然月内敞口必须恒定
    for _, grp in aligned.groupby(aligned.index.to_period("M")):
        assert grp.nunique() == 1
    # 每个调仓日取原序列当日的值
    flags = strategy.rebalance_flags(idx, "M")
    for d in idx[flags]:
        assert aligned.loc[d] == pytest.approx(daily_ex.loc[d])
    # 未对齐时几乎每天变，对齐后只在月初变
    assert aligned.diff().abs().gt(1e-9).sum() <= 4


def test_align_exposure_daily_mode_is_identity():
    """mode='daily' 保留逐日更新，供刻意观察高换手代价时使用。"""
    idx = pd.bdate_range("2020-01-01", periods=30)
    ex = pd.Series(np.linspace(0.1, 0.9, 30), index=idx)
    out = strategy.align_exposure(ex, idx, "M", mode="daily")
    pd.testing.assert_series_equal(out, ex)


def test_weights_from_args_default_rebalance_timing_keeps_turnover_monthly():
    """端到端锁一次：逐日敞口 + 默认配置，权重矩阵的变化次数应回到月频量级。"""
    n = 120
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = ["600000", "600001"]
    score = pd.DataFrame([[2.0, 1.0]] * n, index=idx, columns=cols)
    daily_ex = pd.Series(np.linspace(0.2, 1.0, n), index=idx)
    cfg = _Cfg(top_n=2)

    w = strategy.weights_from_args(score, cfg, exposure=daily_ex)
    changed = (w.diff().abs().sum(axis=1) > 1e-12).sum()
    assert changed <= 6, f"月频调仓下权重变化次数应≈月数，实际 {changed}"

    # 逐日模式下会明显更多（对照）
    cfg.adv_panel = None
    cfg2 = _Cfg(top_n=2, timing_update="daily")
    w2 = strategy.weights_from_args(score, cfg2, exposure=daily_ex)
    assert (w2.diff().abs().sum(axis=1) > 1e-12).sum() > changed
