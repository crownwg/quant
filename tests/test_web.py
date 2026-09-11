"""Web 控制台：表单参数 → 命令行参数的翻译，以及结果解析。

为什么这些值得锁：页面上的错误很少以「报错」形式出现，而是表现为
「结果看着不对」——参数名拼错、择时没接上、NaN 没法序列化，
在浏览器里都只表现为一个奇怪的数字或一个空图。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import web


def _p(**kw) -> web.RunParams:
    base = dict(pool="消费", strategy="momentum", lookback=120, top_n=15)
    base.update(kw)
    return web.RunParams(**base)


def _get(cmd: list, flag: str):
    """取 flag 后面跟的值；flag 不存在时返回 None。"""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# --------------------------------------------------- 命令行拼装

def test_basic_cmd_uses_pool():
    cmd = web._build_cmd(_p(), "px_")
    assert cmd[:3] == [str(web.PYTHON), "-m", "quant.main"]
    assert _get(cmd, "--pool") == "消费"
    assert "--codes" not in cmd
    assert _get(cmd, "--out-prefix") == "px_"


def test_codes_used_when_pool_empty():
    cmd = web._build_cmd(_p(pool="", codes="600031,002557"), "px_")
    assert _get(cmd, "--codes") == "600031,002557"
    assert "--pool" not in cmd


def test_use_open_flag_toggles():
    assert "--use-open" in web._build_cmd(_p(use_open=True), "px_")
    assert "--use-open" not in web._build_cmd(_p(use_open=False), "px_")


def test_timing_off_adds_nothing():
    """不择时必须一个择时参数都不传，否则等于偷偷改了基线。"""
    cmd = web._build_cmd(_p(timing="off"), "px_")
    for flag in ("--timing", "--timing-lookback", "--timing-band",
                 "--timing-min-exposure", "--timing-max-exposure"):
        assert flag not in cmd, flag


def test_timing_ma_adds_full_block():
    cmd = web._build_cmd(
        _p(timing="ma", timing_lookback=60, timing_band=0.02,
           timing_min_exposure=0.3, timing_max_exposure=0.8), "px_")
    assert _get(cmd, "--timing") == "ma"
    assert _get(cmd, "--timing-lookback") == "60"
    assert _get(cmd, "--timing-band") == "0.02"
    assert _get(cmd, "--timing-min-exposure") == "0.3"
    assert _get(cmd, "--timing-max-exposure") == "0.8"


def test_timing_mode_is_normalized():
    assert _get(web._build_cmd(_p(timing=" MA "), "px_"), "--timing") == "ma"


def test_vol_target_only_when_positive():
    assert "--vol-target" not in web._build_cmd(_p(vol_target=0.0), "px_")
    assert _get(web._build_cmd(_p(vol_target=0.15), "px_"), "--vol-target") == "0.15"


def test_timing_proxy_auto_is_omitted():
    assert "--timing-proxy" not in web._build_cmd(_p(timing_proxy="auto"), "px_")
    assert "--timing-proxy" not in web._build_cmd(_p(timing_proxy=""), "px_")
    assert _get(web._build_cmd(_p(timing_proxy="000932"), "px_"),
                "--timing-proxy") == "000932"


def test_walk_forward_pins_stock_grid():
    """walk-forward 必须把选股网格钉死。

    不传 grid-lookbacks/topn/buffers 时 main.py 会套用 CLI 默认的
    4×3×2=24 组选股网格，再乘择时组合能膨胀到几百组，一次要跑几小时。
    """
    p = _p(walk_forward=True, lookback=120, top_n=15, buffer=0, grid_timing="off,ma")
    cmd = web._build_cmd(p, "px_")
    assert "--walk-forward" in cmd
    assert _get(cmd, "--grid-lookbacks") == "120"
    assert _get(cmd, "--grid-topn") == "15"
    assert _get(cmd, "--grid-buffers") == "0"
    assert _get(cmd, "--grid-timing") == "off,ma"


def test_walk_forward_default_timing_grid():
    """留空时补上完整模式列表，否则只有一个候选、样本外没有意义。"""
    cmd = web._build_cmd(_p(walk_forward=True, grid_timing=""), "px_")
    assert _get(cmd, "--grid-timing") == "off,ma,momentum,dual"


def test_walk_forward_keeps_timing_block():
    """样本外验证也要带上用户填的择时参数（作为 off 之外那条固定敞口）。"""
    cmd = web._build_cmd(_p(walk_forward=True, timing="ma", timing_lookback=90), "px_")
    assert _get(cmd, "--timing") == "ma"
    assert _get(cmd, "--timing-lookback") == "90"


def test_single_run_has_no_walk_forward_flags():
    cmd = web._build_cmd(_p(walk_forward=False), "px_")
    assert "--walk-forward" not in cmd
    assert "--grid-timing" not in cmd


# --------------------------------------------------- NaN → null

def test_records_turns_nan_into_none():
    """NaN 必须变成 null。

    DataFrame.to_dict('records') 会保留 NaN，FastAPI 序列化时直接抛
    「Out of range float values are not JSON compliant」，整页报错。
    """
    df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", None]})
    recs = web._records(df)
    assert recs[0]["a"] == 1.0
    assert recs[1]["a"] is None
    assert recs[1]["b"] is None


def test_records_is_json_serializable():
    import json
    df = pd.DataFrame({"x": [np.nan, np.inf, -np.inf, 1.5]})
    recs = web._records(df)
    json.dumps(recs)          # 不抛异常即通过


def test_records_preserves_zero_padded_codes():
    """股票代码必须原样保留 "000858"。

    这里踩过两次：先用 to_dict 遇到 NaN 报错，改用 to_json 后它又把
    「看起来像数字的字符串」列转成了数字 —— 000858 → 858、002714 → 2714，
    页面上代码全错。修法必须同时满足「NaN 安全」和「字符串不动」两条。
    """
    df = pd.DataFrame({"code": ["000858", "600519", "002714"],
                       "v": [1.0, np.nan, 2.0]})
    recs = web._records(df)
    assert [r["code"] for r in recs] == ["000858", "600519", "002714"]
    assert recs[1]["v"] is None


def test_records_keeps_mixed_types():
    df = pd.DataFrame({"s": ["000001"], "i": [7], "f": [1.5],
                       "b": [True], "n": [np.nan]})
    rec = web._records(df)[0]
    assert rec["s"] == "000001"
    assert rec["i"] == 7 and isinstance(rec["i"], int)
    assert rec["f"] == 1.5
    assert rec["b"] is True
    assert rec["n"] is None


def test_records_empty():
    assert web._records(pd.DataFrame()) == []
    assert web._records(pd.DataFrame({"a": []})) == []


def test_records_keeps_rows_and_order():
    df = pd.DataFrame({"k": ["a", "b", "c"], "v": [1, 2, 3]})
    assert [r["k"] for r in web._records(df)] == ["a", "b", "c"]


def test_read_csv_codes_restores_leading_zeros(tmp_path):
    """从 CSV 读 code 时必须补回前导零。

    pandas 会把 "000858" 推断成整数 858、"000001" 变成 1，页面上代码就少几位。
    这条锁的是**读入端** —— 只在输出端（_json_safe）做类型处理是修不掉的，
    因为进到 _records 时数据已经是错的。这正是本轮踩过的坑。
    """
    p = tmp_path / "plan.csv"
    p.write_text("code,action,amount\n000858,清仓,104870\n000001,清仓,100\n",
                 encoding="utf-8")
    df = web._read_csv_codes(p)
    assert list(df["code"]) == ["000858", "000001"]


def test_read_csv_codes_tolerates_missing_code_column(tmp_path):
    p = tmp_path / "other.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    df = web._read_csv_codes(p)
    assert list(df.columns) == ["a", "b"]


# --------------------------------------------------- 文件访问安全

def test_safe_file_allows_plain_name(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "ROOT", tmp_path)
    (tmp_path / "ok_report.html").write_text("<html>hi</html>", encoding="utf-8")
    assert web._safe_file("ok_report.html", (".html",)).name == "ok_report.html"


def test_safe_file_blocks_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "ROOT", tmp_path)
    (tmp_path / "ok_report.html").write_text("x", encoding="utf-8")
    for bad in ("../secret.html", "sub/ok_report.html", "ok_report.html/../x"):
        with pytest.raises(Exception):
            web._safe_file(bad, (".html",))


def test_safe_file_rejects_wrong_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "ROOT", tmp_path)
    (tmp_path / "data.csv").write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(Exception):
        web._safe_file("data.csv", (".html",))


def test_safe_file_raises_for_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "ROOT", tmp_path)
    with pytest.raises(Exception):
        web._safe_file("nope.html", (".html",))


# --------------------------------------------------- 持仓解析 & 调仓清单

def test_materialize_holdings_parses_common_formats(tmp_path, monkeypatch):
    """用户是直接粘过来的，格式会很乱：逗号/空格/分号混合、带注释、代码没补零。"""
    monkeypatch.setattr(web, "ROOT", tmp_path)
    text = "\n".join([
        "# 我的持仓",
        "600519,100",
        "000858 1000",       # 空格分隔
        "002557;500",        # 分号分隔
        "1,200",             # 代码不足 6 位 -> 补零
        "",                  # 空行
        "乱七八糟",           # 无效行
        "600000,-5",         # 负数 -> 跳过
    ])
    path, src = web._materialize_holdings(text, "px_")
    assert path is not None
    df = pd.read_csv(path, dtype={"code": str})
    assert list(df["code"]) == ["600519", "000858", "002557", "000001"]
    assert list(df["shares"]) == [100, 1000, 500, 200]
    assert "4" in src


def test_materialize_holdings_falls_back_to_default_file(tmp_path, monkeypatch):
    """文本框留空时应回落到项目自带的 my_holdings.csv。"""
    monkeypatch.setattr(web, "ROOT", tmp_path)
    (tmp_path / "my_holdings.csv").write_text("code,shares\n600519,100\n", encoding="utf-8")
    path, src = web._materialize_holdings("", "px_")
    assert path == str(tmp_path / "my_holdings.csv")
    assert "my_holdings" in src


def test_materialize_holdings_none_when_unusable(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "ROOT", tmp_path)
    assert web._materialize_holdings("没有数字", "px_") == (None, "")
    assert web._materialize_holdings("", "px_") == (None, "")   # 也没有默认文件


def test_plan_cmd_adds_holdings_flag():
    cmd = web._build_cmd(_p(), "px_", plan_only=True, holdings_path="/tmp/h.csv")
    assert _get(cmd, "--holdings") == "/tmp/h.csv"


def test_plan_cmd_adds_today_only_when_given():
    a = web._build_cmd(_p(plan_date="20240909"), "px_",
                       plan_only=True, holdings_path="/tmp/h.csv")
    assert _get(a, "--today") == "20240909"
    b = web._build_cmd(_p(plan_date=""), "px_",
                       plan_only=True, holdings_path="/tmp/h.csv")
    assert "--today" not in b


def test_plan_flags_absent_from_normal_run():
    """普通回测绝不能带 --holdings。

    main.py 的 holdings 分支是独立 return 的，一旦带上就不会产出净值曲线，
    页面上的图全空——这种错很难从现象反推原因。
    """
    cmd = web._build_cmd(_p(holdings_text="600519,100"), "px_")
    assert "--holdings" not in cmd
    assert "--today" not in cmd


# --------------------------------------------------- 指标计算

def test_drawdown_series_zero_at_new_high():
    eq = pd.Series([1.0, 1.2, 1.1, 1.3])
    dd = web._drawdown_series(eq)
    assert dd[0] == 0.0 and dd[1] == 0.0
    assert dd[2] == pytest.approx(1.1 / 1.2 - 1)
    assert dd[3] == 0.0
    assert min(dd) <= 0


def test_compute_monthly_shape():
    idx = pd.date_range("2023-01-01", periods=90, freq="D")
    df = pd.DataFrame({"date": idx, "equity": np.linspace(1.0, 1.5, 90)})
    out = web._compute_monthly(df)
    assert out["years"] == [2023]
    assert {1, 2, 3} <= set(out["months"])
    assert len(out["values"]) == 1
    assert len(out["values"][0]) == len(out["months"])
