"""P0 三项修复的回归测试。

覆盖：
  1. 涨跌停判定改用「涨跌停价」+ 一字板识别（filters.limit_prices / limit_masks）
  2. 判定口径与成交价对齐（tradability 的 exec_at_open）
  3. 幸存者偏差相关：成分股时点快照（universe）、全市场池、次新股掩码（filters.listing_age_mask）
  4. 连带修复：缓存头部回补与覆盖写（data._ensure_cached / _append_cache）

运行：.venv/Scripts/python.exe -m pytest tests -q
"""
from __future__ import annotations

import pandas as pd
import pytest

from quant import data as data_mod
from quant import filters, strategy, universe


# --------------------------------------------------------------- 测试工具

def _panel(rows: dict, cols=("600000",)):
    """构造 date×code 价格面板。rows = {日期: {代码: 价格}}。"""
    df = pd.DataFrame(rows).T
    df.index = pd.to_datetime(df.index)
    return df.reindex(columns=list(cols)).astype(float)


def _ohlc(dates, o, h, l, c, code="600000"):
    idx = pd.to_datetime(dates)
    return (
        pd.DataFrame({code: o}, index=idx, dtype=float),
        pd.DataFrame({code: h}, index=idx, dtype=float),
        pd.DataFrame({code: l}, index=idx, dtype=float),
        pd.DataFrame({code: c}, index=idx, dtype=float),
    )


# ------------------------------------------------- 1. 涨跌停价与一字板判定

def test_limit_prices_rounding():
    """涨跌停价按 0.01 元四舍五入，非整关口价格会体现出这点。"""
    close = _panel({"2024-01-02": {"600000": 10.00},
                    "2024-01-03": {"600000": 10.00}})
    up, down = filters.limit_prices(close, 0.10)
    assert up.iloc[1]["600000"] == pytest.approx(11.00)
    assert down.iloc[1]["600000"] == pytest.approx(9.00)

    # 3.03 * 1.1 = 3.333 → 3.33；3.03 * 0.9 = 2.727 → 2.73
    close2 = _panel({"2024-01-02": {"600000": 3.03},
                     "2024-01-03": {"600000": 3.03}})
    up2, down2 = filters.limit_prices(close2, 0.10)
    assert up2.iloc[1]["600000"] == pytest.approx(3.33)
    assert down2.iloc[1]["600000"] == pytest.approx(2.73)


def test_limit_pct_by_code_boards():
    pct = filters.limit_pct_by_code(["600000", "300750", "688111", "830799", "000001"],
                                    st_codes=["000001"])
    assert pct["600000"] == 0.10
    assert pct["300750"] == 0.20
    assert pct["688111"] == 0.20
    assert pct["830799"] == 0.30
    assert pct["000001"] == 0.05  # ST


def test_one_word_limit_up_blocks_buy():
    """一字涨停：开=高=低=收 全在涨停价 → 买卖判定中禁买、仍可卖。"""
    dates = ["2024-01-02", "2024-01-03"]
    o, h, l, c = _ohlc(dates, [10.0, 11.0], [10.0, 11.0], [10.0, 11.0], [10.0, 11.0])
    vol = pd.DataFrame({"600000": [1e6, 1e6]}, index=pd.to_datetime(dates))

    one_up, close_up, _, _ = filters.limit_masks(c, 0.10, open_=o, high=h, low=l)
    assert bool(one_up.iloc[1]["600000"]) is True
    assert bool(close_up.iloc[1]["600000"]) is True

    can_buy, can_sell = filters.tradability(c, vol, limit_pct=0.10, open_=o, high=h, low=l)
    assert bool(can_buy.iloc[1]["600000"]) is False   # 买不进
    assert bool(can_sell.iloc[1]["600000"]) is True   # 卖得出


def test_open_board_limit_up_is_not_one_word():
    """盘中开板：开盘封板但最低价回落 → 不是一字板（但收盘仍封板）。"""
    dates = ["2024-01-02", "2024-01-03"]
    o, h, l, c = _ohlc(dates, [10.0, 11.0], [10.0, 11.0], [10.0, 10.50], [10.0, 11.0])
    one_up, close_up, _, _ = filters.limit_masks(c, 0.10, open_=o, high=h, low=l)
    assert bool(one_up.iloc[1]["600000"]) is False
    assert bool(close_up.iloc[1]["600000"]) is True


def test_open_at_down_limit_then_rebound_is_not_one_word():
    """开盘跌停后拉起：**不是**一字跌停，收盘也未封板 → 可正常卖出。

    真实案例：002157 于 2020-02-04 开盘 10.57（=跌停价）、最高 11.68、收 11.48。
    若一字跌停的判据误用 low，这天会被错判成卖不出。
    """
    dates = ["2024-01-02", "2024-01-03"]
    # day2: prev=10.00 → 跌停价 9.00；开 9.00，最高 10.00，收 9.80
    o, h, l, c = _ohlc(dates, [10.0, 9.00], [10.0, 10.00], [10.0, 9.00], [10.0, 9.80])
    vol = pd.DataFrame({"600000": [1e6, 1e6]}, index=pd.to_datetime(dates))

    _, _, one_down, close_down = filters.limit_masks(c, 0.10, open_=o, high=h, low=l)
    assert bool(one_down.iloc[1]["600000"]) is False
    assert bool(close_down.iloc[1]["600000"]) is False

    can_buy, can_sell = filters.tradability(c, vol, limit_pct=0.10, open_=o, high=h, low=l)
    assert bool(can_sell.iloc[1]["600000"]) is True


def test_true_one_word_down_blocks_sell():
    """真正的一字跌停：开=高=低=收 都在跌停价 → 卖不出。"""
    dates = ["2024-01-02", "2024-01-03"]
    o, h, l, c = _ohlc(dates, [10.0, 9.0], [10.0, 9.0], [10.0, 9.0], [10.0, 9.0])
    vol = pd.DataFrame({"600000": [1e6, 1e6]}, index=pd.to_datetime(dates))
    _, _, one_down, _ = filters.limit_masks(c, 0.10, open_=o, high=h, low=l)
    assert bool(one_down.iloc[1]["600000"]) is True
    _, can_sell = filters.tradability(c, vol, limit_pct=0.10, open_=o, high=h, low=l)
    assert bool(can_sell.iloc[1]["600000"]) is False


def test_open_execution_ignores_close_seal():
    """按开盘价成交时，收盘才封板**不应**影响当下能否买入（否则就是前视）。"""
    dates = ["2024-01-02", "2024-01-03"]
    # day2: 开盘 10.20（远离涨停价 11.00），收盘 11.00 封板
    o, h, l, c = _ohlc(dates, [10.0, 10.20], [10.0, 11.00], [10.0, 10.10], [10.0, 11.00])
    vol = pd.DataFrame({"600000": [1e6, 1e6]}, index=pd.to_datetime(dates))

    buy_open, _ = filters.tradability(c, vol, limit_pct=0.10, open_=o, high=h, low=l,
                                      exec_at_open=True)
    assert bool(buy_open.iloc[1]["600000"]) is True

    # 收盘价成交口径则会被封板挡住
    buy_close, _ = filters.tradability(c, vol, limit_pct=0.10, open_=o, high=h, low=l,
                                       exec_at_open=False)
    assert bool(buy_close.iloc[1]["600000"]) is False


def test_down_limit_blocks_sell_but_allows_buy():
    dates = ["2024-01-02", "2024-01-03"]
    o, h, l, c = _ohlc(dates, [10.0, 9.0], [10.0, 9.0], [10.0, 9.0], [10.0, 9.0])
    vol = pd.DataFrame({"600000": [1e6, 1e6]}, index=pd.to_datetime(dates))
    can_buy, can_sell = filters.tradability(c, vol, limit_pct=0.10, open_=o, high=h, low=l)
    assert bool(can_buy.iloc[1]["600000"]) is True
    assert bool(can_sell.iloc[1]["600000"]) is False


def test_suspension_blocks_both_directions():
    dates = ["2024-01-02", "2024-01-03"]
    c = _panel({"2024-01-02": {"600000": 10.0}, "2024-01-03": {"600000": 10.0}})
    vol = pd.DataFrame({"600000": [1e6, 0.0]}, index=pd.to_datetime(dates))
    can_buy, can_sell = filters.tradability(c, vol, limit_pct=0.10)
    assert bool(can_buy.iloc[1]["600000"]) is False
    assert bool(can_sell.iloc[1]["600000"]) is False


# ------------------------------------------- 2. 判定口径与成交价一致

def test_exec_at_open_uses_open_price_not_close():
    """开盘即涨停价、但当天回落收阴：按开盘价成交就买不到，按收盘价成交则不受限。

    这正是修复前的口径错位——用收盘信息决定开盘能不能下单。
    """
    dates = ["2024-01-02", "2024-01-03"]
    # day2 开盘 11.00（涨停价），最低 10.30，收盘 10.50（未封板）
    o, h, l, c = _ohlc(dates, [10.0, 11.0], [10.0, 11.0], [10.0, 10.30], [10.0, 10.50])
    vol = pd.DataFrame({"600000": [1e6, 1e6]}, index=pd.to_datetime(dates))

    # 收盘价成交口径：收盘没封板 → 可买
    buy_close, _ = filters.tradability(c, vol, limit_pct=0.10, open_=o, high=h, low=l,
                                       exec_at_open=False)
    assert bool(buy_close.iloc[1]["600000"]) is True

    # 开盘价成交口径：开盘就贴在涨停价 → 买不到
    buy_open, _ = filters.tradability(c, vol, limit_pct=0.10, open_=o, high=h, low=l,
                                      exec_at_open=True)
    assert bool(buy_open.iloc[1]["600000"]) is False


def test_exec_at_open_does_not_block_normal_day():
    """普通交易日（开盘价远离涨跌停价）不应被新增约束误伤。"""
    dates = ["2024-01-02", "2024-01-03"]
    o, h, l, c = _ohlc(dates, [10.0, 10.10], [10.0, 10.40], [10.0, 9.95], [10.0, 10.30])
    vol = pd.DataFrame({"600000": [1e6, 1e6]}, index=pd.to_datetime(dates))
    buy, sell = filters.tradability(c, vol, limit_pct=0.10, open_=o, high=h, low=l,
                                    exec_at_open=True)
    assert bool(buy.iloc[1]["600000"]) is True
    assert bool(sell.iloc[1]["600000"]) is True


def test_limit_stats_reported():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    o, h, l, c = _ohlc(dates, [10.0, 11.0, 9.9], [10.0, 11.0, 9.9],
                       [10.0, 11.0, 9.9], [10.0, 11.0, 9.9])
    vol = pd.DataFrame({"600000": [1e6, 1e6, 1e6]}, index=pd.to_datetime(dates))
    stats: dict = {}
    filters.tradability(c, vol, limit_pct=0.10, open_=o, high=h, low=l, stats=stats)
    assert stats["limit_up_oneword"] == 1
    assert stats["limit_down_oneword"] == 1
    assert stats["limit_blocked_buy"] == 1   # 只数涨跌停挡的
    assert stats["limit_blocked_sell"] == 1


def test_stats_separate_limit_from_suspension():
    """诊断口径必须分开：停牌不应被算进「涨跌停致禁买」。"""
    dates = ["2024-01-02", "2024-01-03"]
    o, h, l, c = _ohlc(dates, [10.0, 10.2], [10.0, 10.3], [10.0, 10.1], [10.0, 10.2])
    vol = pd.DataFrame({"600000": [1e6, 0.0]}, index=pd.to_datetime(dates))  # 次日停牌
    stats: dict = {}
    filters.tradability(c, vol, limit_pct=0.10, open_=o, high=h, low=l, stats=stats)
    assert stats["limit_blocked_buy"] == 0     # 涨跌停没挡任何东西
    assert stats["suspend_blocked"] == 1       # 停牌挡了 1
    assert stats["blocked_buy"] == 1           # 合计 1


# ------------------------------------------- 3. 幸存者偏差相关

def test_listing_age_mask_excludes_new_listings():
    """上市天数按「面板内首个有效观测」起算，未满 min_days 的不可选。"""
    idx = pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01",
                          "2021-06-01", "2022-01-01", "2022-06-01"])
    prices = pd.DataFrame(
        {"OLD": [1.0] * 6, "NEW": [None, None, None, 1.0, 1.0, 1.0]}, index=idx)
    mask = filters.listing_age_mask(prices, 250)

    # OLD 从面板首日起算，满 250 自然日（2020-09-07）后才可用
    assert bool(mask.loc["2020-01-01", "OLD"]) is False
    assert bool(mask.loc["2021-01-01", "OLD"]) is True
    # NEW 2021-06-01 才上市 → 2022-01-01 仍未满 250 日，2022-06-01 才可用
    assert bool(mask.loc["2022-01-01", "NEW"]) is False
    assert bool(mask.loc["2022-06-01", "NEW"]) is True


def test_listing_age_mask_needs_long_enough_warmup():
    """预热期足够长时，老股票在评估起点一定是可用的（不会误伤）。"""
    idx = pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"])
    prices = pd.DataFrame({"OLD": [1.0, 1.0, 1.0]}, index=idx)
    mask = filters.listing_age_mask(prices, 250)
    # 评估起点 2021-01-01 距面板首日已 2 年 > 250 日
    assert bool(mask.loc["2021-01-01", "OLD"]) is True


def test_listing_age_mask_disabled_returns_none():
    prices = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2020-01-01"]))
    assert filters.listing_age_mask(prices, 0) is None


def test_snapshot_asof_picks_latest_not_after(tmp_path):
    d = str(tmp_path)
    universe.save_snapshot("000300", ["600000", "600001"], as_of="2022-01-31", data_dir=d)
    universe.save_snapshot("000300", ["600000", "600002"], as_of="2023-01-31", data_dir=d)

    codes, meta = universe.get_index_constituents_asof("000300", as_of="2022-06-01", data_dir=d)
    assert codes == ["600000", "600001"]
    assert meta["pit"] is True
    assert meta["snapshot_date"] == pd.Timestamp("2022-01-31")
    assert meta["message"] == ""


def test_snapshot_asof_falls_back_with_warning(tmp_path, monkeypatch):
    """请求日期早于所有快照 → 必须回退且明确告警，不能静默返回有偏数据。"""
    d = str(tmp_path)
    universe.save_snapshot("000300", ["600000"], as_of="2022-01-31", data_dir=d)
    monkeypatch.setattr(universe, "get_index_constituents",
                        lambda *a, **k: ["600000", "600999"])

    codes, meta = universe.get_index_constituents_asof("000300", as_of="2019-01-01", data_dir=d)
    assert codes == ["600000", "600999"]
    assert meta["pit"] is False
    assert "幸存者偏差" in meta["message"]


def test_snapshot_asof_today_is_fine(tmp_path, monkeypatch):
    """请求当下时点 → 当前成分股本身就是正确时点名单，不应告警。"""
    d = str(tmp_path)
    monkeypatch.setattr(universe, "get_index_constituents", lambda *a, **k: ["600000"])
    codes, meta = universe.get_index_constituents_asof(
        "000300", as_of=pd.Timestamp.now().normalize(), data_dir=d)
    assert codes == ["600000"]
    assert meta["pit"] is True
    assert meta["message"] == ""


def test_build_universe_report_marks_bias(tmp_path, monkeypatch):
    d = str(tmp_path)
    universe.save_snapshot("000300", ["600000"], as_of="2023-01-31", data_dir=d)
    codes, report = universe.build_universe_report("hs300", data_dir=d, as_of="2020-01-01")
    # 无 2020 年的快照 → 回退当前成分股并标记 pit=False
    assert report and report[0]["pit"] is False
    assert "幸存者偏差" in report[0]["message"]


def test_all_market_token_recognised():
    for tok in ("all", "ALL", "全市场", "a股"):
        assert universe.is_all_market(tok) is True
    assert universe.is_all_market("hs300") is False


def test_build_universe_all_uses_all_market(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "get_all_market_codes",
                        lambda *a, **k: ["600000", "000001", "600000"])
    codes, report = universe.build_universe_report("all", data_dir=str(tmp_path))
    assert codes == ["600000", "000001"]        # 去重保序
    assert report[0]["pit"] is True


# ------------------------------------------- 5. 决策日 vs 成交日对齐

def _score_setup():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04",
                          "2024-02-01", "2024-02-02"])
    # A 分低、B 分高 → top_n=1 时优先选 B
    score = pd.DataFrame({"A": [1.0] * 5, "B": [2.0] * 5}, index=idx)
    return score


def test_exec_shift_uses_next_day_tradability():
    """开盘成交时，可交易性必须取**成交当日**（t+1），而不是决策日（t）。"""
    score = _score_setup()
    idx = score.index
    # B 只在 2024-01-03（= 1/02 调仓日的次日）不可买
    can_buy = pd.DataFrame(True, index=idx, columns=["A", "B"])
    can_buy.loc["2024-01-03", "B"] = False

    w0 = strategy.factor_weights(score, top_n=1, freq="M", can_buy=can_buy, exec_shift=0)
    assert w0.loc["2024-01-02", "B"] == pytest.approx(1.0)   # 决策日 B 可买 → 选中

    w1 = strategy.factor_weights(score, top_n=1, freq="M", can_buy=can_buy, exec_shift=1)
    assert w1.loc["2024-01-02", "B"] == pytest.approx(0.0)   # 成交日 B 买不进 → 空仓


def test_exec_shift_keeps_position_when_cannot_sell():
    """想清仓但成交日卖不出 → 保留上期权重继续持有。"""
    score = _score_setup()
    idx = score.index
    can_buy = pd.DataFrame(True, index=idx, columns=["A", "B"])
    # 1/02 建仓 B；2/01 决策清仓，但 2/02（成交日）B 跌停卖不出
    can_sell = pd.DataFrame(True, index=idx, columns=["A", "B"])
    can_sell.loc["2024-02-02", "B"] = False

    w = strategy.factor_weights(score, top_n=1, freq="M",
                                can_buy=can_buy, can_sell=can_sell, exec_shift=1)
    assert w.loc["2024-01-02", "B"] == pytest.approx(1.0)
    assert w.loc["2024-02-01", "B"] == pytest.approx(1.0)   # 卖不出，继续持有


def test_exec_shift_zero_is_backward_compatible():
    """不传 exec_shift 时行为与旧版一致（默认 0）。"""
    score = _score_setup()
    idx = score.index
    can_buy = pd.DataFrame(True, index=idx, columns=["A", "B"])
    a = strategy.factor_weights(score, top_n=1, freq="M", can_buy=can_buy)
    b = strategy.factor_weights(score, top_n=1, freq="M", can_buy=can_buy, exec_shift=0)
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------- 4. 缓存头部回补与覆盖写

def _write_cache(path, dates, base=10.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": base, "high": base, "low": base, "close": base, "volume": 1e6,
    }).to_csv(path, index=False)


def test_head_backfill_pulls_full_history(tmp_path):
    """请求区间早于本地缓存 → 必须回补头部，而不是静默返回截断数据。"""
    calls = {"all": 0}

    def fetch_full(code, start, end, retries, backoff):
        raise AssertionError("头部缺口场景不应走 fetch_full")

    def fetch_all(code, retries, backoff):
        calls["all"] += 1
        return pd.DataFrame({
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2021-01-08"]),
            "open": 9.0, "high": 9.0, "low": 9.0, "close": 9.0, "volume": 1e6,
        })

    _write_cache(tmp_path / "600000.csv", ["2021-01-04", "2021-01-08"])
    out = data_mod._ensure_cached("600000", "2020-01-01", "2021-01-08", str(tmp_path),
                                  fetch_full=fetch_full, fetch_all=fetch_all)

    assert calls["all"] == 1
    assert out["date"].min() == pd.Timestamp("2020-01-02")   # 头部已回补
    # 覆盖写：旧缓存的价格（10.0）应被新数据（9.0）整体替换，不产生拼接台阶
    on_disk = pd.read_csv(tmp_path / "600000.csv")
    assert set(on_disk["close"].round(2)) == {9.0}


def test_head_backfill_idempotent_after_refetch(tmp_path):
    """回补后 cache_min 落到真实上市日，下次同样请求不应再触发全量拉取。"""
    calls = {"all": 0}

    def fetch_all(code, retries, backoff):
        calls["all"] += 1
        return pd.DataFrame({
            "date": pd.to_datetime(["2020-01-02", "2021-01-08"]),
            "open": 9.0, "high": 9.0, "low": 9.0, "close": 9.0, "volume": 1e6,
        })

    _write_cache(tmp_path / "600000.csv", ["2021-01-04", "2021-01-08"])
    data_mod._ensure_cached("600000", "2020-01-01", "2021-01-08", str(tmp_path),
                            fetch_all=fetch_all)
    data_mod._ensure_cached("600000", "2020-01-01", "2021-01-08", str(tmp_path),
                            fetch_all=fetch_all)
    assert calls["all"] == 1


def test_tail_append_still_incremental(tmp_path):
    """往后追加的场景保持增量，不应该退化成全量重拉。"""
    calls = {"range": 0, "all": 0}

    def fetch_in_range(code, s, e, retries, backoff):
        calls["range"] += 1
        return pd.DataFrame({
            "date": pd.to_datetime(["2021-01-11"]),
            "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0, "volume": 1e6,
        })

    def fetch_all(code, retries, backoff):
        calls["all"] += 1
        raise AssertionError("尾部追加不应触发全量回补")

    _write_cache(tmp_path / "600000.csv", ["2021-01-04", "2021-01-08"])
    out = data_mod._ensure_cached("600000", "2021-01-04", "2021-01-11", str(tmp_path),
                                  fetch_in_range=fetch_in_range, fetch_all=fetch_all)
    assert calls["range"] == 1 and calls["all"] == 0
    assert out["date"].max() == pd.Timestamp("2021-01-11")


def test_covered_cache_read_directly(tmp_path):
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("已完整覆盖时不应发起任何网络拉取")

    _write_cache(tmp_path / "600000.csv", ["2021-01-04", "2021-01-05", "2021-01-08"])
    out = data_mod._ensure_cached("600000", "2021-01-04", "2021-01-08", str(tmp_path),
                                  fetch_in_range=boom, fetch_full=boom, fetch_all=boom)
    assert calls["n"] == 0
    assert len(out) == 3
