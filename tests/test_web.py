"""Web 控制台：表单参数 → 命令行参数的翻译，以及结果解析。

为什么这些值得锁：页面上的错误很少以「报错」形式出现，而是表现为
「结果看着不对」——参数名拼错、择时没接上、NaN 没法序列化，
在浏览器里都只表现为一个奇怪的数字或一个空图。
"""
from __future__ import annotations

import asyncio
import pathlib
import re

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


# --------------------------------------------------- 数据体检 / 数据更新

def test_pool_and_codes_are_additive():
    """--pool 与 --codes 必须能同时传。

    main.py 的语义是两者取并集，页面早期写成二选一，导致
    「消费池 + 002557 洽洽食品」这种用法完全做不到——而洽洽食品
    不在任何指数的成分股里，只能靠 --codes 补进来。
    """
    cmd = web._build_cmd(_p(pool="消费", codes="002557"), "px_")
    assert _get(cmd, "--pool") == "消费"
    assert _get(cmd, "--codes") == "002557"


def test_codes_only_still_works():
    cmd = web._build_cmd(_p(pool="", codes="600031"), "px_")
    assert "--pool" not in cmd
    assert _get(cmd, "--codes") == "600031"


def test_refresh_cmd_has_flag_and_nothing_else():
    """数据更新命令只该带 --refresh-data，不该掺择时 / 样本外的参数。

    掺进去不会报错，但会让人以为「更新数据」也顺带跑了回测，
    或者反之以为回测已经跑过。
    """
    cmd = web._build_cmd(_p(pool="消费"), "px_", refresh_only=True)
    assert "--refresh-data" in cmd
    assert "--walk-forward" not in cmd
    assert "--timing" not in cmd
    assert "--vol-target" not in cmd
    assert "--holdings" not in cmd


def test_normal_run_never_has_refresh_flag():
    assert "--refresh-data" not in web._build_cmd(_p(), "px_")
    assert "--refresh-data" not in web._build_cmd(
        _p(walk_forward=True), "px_")
    assert "--refresh-data" not in web._build_cmd(
        _p(), "px_", plan_only=True, holdings_path="h.csv")


def test_freshness_flags_historical_replay(monkeypatch):
    """结束日远早于数据最新日 —— 最危险的情况（默认 20240909 就在这里）。"""
    monkeypatch.setattr(web, "_resolve_codes", lambda p: (["600519"], None))
    monkeypatch.setattr(web, "cache_freshness",
                        lambda cs: [{"code": "600519", "last": "2026-09-10", "rows": 1}])
    d = asyncio.run(web.data_freshness(_p(end="20240909")))
    assert d["latest"] == "2026-09-10"
    assert d["ahead_of_data"] is True
    assert d["usable"] is False
    assert "历史回放" in d["message"]


def test_freshness_ok_when_end_matches_latest(monkeypatch):
    monkeypatch.setattr(web, "_resolve_codes", lambda p: (["600519"], None))
    monkeypatch.setattr(web, "cache_freshness",
                        lambda cs: [{"code": "600519", "last": "2026-09-10", "rows": 1}])
    d = asyncio.run(web.data_freshness(_p(end="20260910")))
    assert d["usable"] is True
    assert d["ahead_of_data"] is False and d["behind_end"] is False


def test_freshness_flags_end_beyond_data(monkeypatch):
    monkeypatch.setattr(web, "_resolve_codes", lambda p: (["600519"], None))
    monkeypatch.setattr(web, "cache_freshness",
                        lambda cs: [{"code": "600519", "last": "2026-09-10", "rows": 1}])
    d = asyncio.run(web.data_freshness(_p(end="20261231")))
    assert d["behind_end"] is True and d["usable"] is False
    assert "更新数据" in d["message"]


def test_freshness_tolerates_weekend_gap(monkeypatch):
    """差 1~5 天是周末 / 长假 / 停牌的常态，不该天天报警。"""
    monkeypatch.setattr(web, "_resolve_codes", lambda p: (["600519"], None))
    monkeypatch.setattr(web, "cache_freshness",
                        lambda cs: [{"code": "600519", "last": "2026-09-09", "rows": 1}])
    d = asyncio.run(web.data_freshness(_p(end="20260910")))
    assert d["usable"] is True


def test_freshness_empty_cache_has_same_keys(monkeypatch):
    """早退分支也要给齐字段：前端只有一套渲染逻辑，缺键会渲染成 undefined。"""
    monkeypatch.setattr(web, "_resolve_codes", lambda p: (["999999"], None))
    monkeypatch.setattr(web, "cache_freshness",
                        lambda cs: [{"code": "999999", "last": "", "rows": 0}])
    d = asyncio.run(web.data_freshness(_p(end="20260910")))
    normal = {"ok", "n_codes", "with_data", "latest", "end", "usable",
              "behind_end", "ahead_of_data", "stale", "n_stale",
              "empty", "n_empty", "message"}
    assert normal <= set(d)
    assert d["with_data"] == 0 and d["usable"] is False
    assert d["n_empty"] == 1


def test_freshness_reports_stale_and_empty(monkeypatch):
    monkeypatch.setattr(web, "_resolve_codes", lambda p: (["600519", "002557", "999999"], None))
    monkeypatch.setattr(web, "cache_freshness", lambda cs: [
        {"code": "600519", "last": "2026-09-10", "rows": 1},
        {"code": "002557", "last": "2024-09-09", "rows": 1},   # 停更两年
        {"code": "999999", "last": "", "rows": 0},             # 完全没数据
    ])
    d = asyncio.run(web.data_freshness(_p(end="20260910")))
    assert [r["code"] for r in d["stale"]] == ["002557"]
    assert d["empty"] == ["999999"]
    assert d["n_stale"] == 1 and d["n_empty"] == 1
    assert "明显落后" in d["message"] and "没有缓存" in d["message"]


def test_freshness_pool_error_is_not_exception(monkeypatch):
    """池名写错要给可读提示，而不是 500。"""
    monkeypatch.setattr(web, "_resolve_codes", lambda p: ([], "股票池解析失败: 未知股票池"))
    d = asyncio.run(web.data_freshness(_p(pool="乱写")))
    assert d["ok"] is False and "未知股票池" in d["error"]


# --------------------------------------------------- _resolve_codes 并集语义

def test_resolve_codes_merges_pool_and_codes(monkeypatch):
    from quant import universe
    monkeypatch.setattr(universe, "build_universe_report",
                        lambda spec, **kw: (["600519", "000858"], []))
    codes, err = web._resolve_codes(_p(pool="消费", codes="002557, 858"))
    assert err is None
    # 池内两只 + 新加一只；858 与已有的 000858 去重，不重复出现
    assert codes == ["600519", "000858", "002557"]


def test_resolve_codes_empty_is_error():
    codes, err = web._resolve_codes(_p(pool="", codes=""))
    assert codes == [] and err


# --------------------------------------------------- 前端 DOM 契约（静态检查）

# 这几项检查的是「HTML 与 JS 对不上」这类错误。它们在浏览器里不报错，
# 只是表现为「点了没反应」或「结果不对」，是前端最难查的一类问题。
# 真实案例：预设按钮里的 $('lookback') 对应的 input 只有 name 没有 id，
# 取到 null 后赋值抛 TypeError，整个 fillPreset 在第 4 行就中断 ——
# 池子切了、因子变了，看起来「有反应」，所以一直没被发现。

def _index_html() -> str:
    return (web.TEMPLATE).read_text(encoding="utf-8")


def _inline_js(html: str) -> str:
    blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
    assert blocks, "index.html 里找不到内联脚本"
    return max(blocks, key=len)


def test_js_references_existing_element_ids():
    html = _index_html()
    js = _inline_js(html)
    used = set(re.findall(r"\$\('([A-Za-z0-9_\-]+)'\)", js))
    defined = set(re.findall(r'\bid="([A-Za-z0-9_\-]+)"', html))
    assert used - defined == set(), f"JS 引用了不存在的 id: {sorted(used - defined)}"


def test_js_reads_existing_form_fields():
    html = _index_html()
    js = _inline_js(html)
    form = re.search(r'<form id="runForm".*?</form>', html, re.S)
    assert form, "找不到 runForm"
    names = set(re.findall(r'\bname="([A-Za-z0-9_]+)"', form.group(0)))
    used = set(re.findall(r"\bf\.([A-Za-z_][A-Za-z0-9_]*)\.", js))
    used |= set(re.findall(r"num\('([A-Za-z0-9_]+)'", js))
    used |= set(re.findall(r"_setField\(\s*\w+\s*,\s*'([A-Za-z0-9_]+)'", js))
    used.discard("")
    assert used - names == set(), f"JS 读了表单里没有的字段: {sorted(used - names)}"


def test_onclick_handlers_exist():
    html = _index_html()
    js = _inline_js(html)
    called = set(re.findall(r'onclick="([A-Za-z_][A-Za-z0-9_]*)\(', html))
    called |= set(re.findall(r'onsubmit="return ([A-Za-z_][A-Za-z0-9_]*)\(', html))
    funcs = set(re.findall(r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", js))
    assert called - funcs == set(), f"onclick 指向未定义的函数: {sorted(called - funcs)}"


def test_frontend_only_calls_registered_api():
    """前端 fetch 的 /api/xxx 必须在 web.py 里有对应路由。

    路径打错时页面表现为「请求失败」，看不出是 404 还是服务挂了。
    """
    html = _index_html()
    js = _inline_js(html)
    called = set(re.findall(r"fetch\('(/api/[^'`]*)'", js))
    src = (pathlib.Path(web.__file__)).read_text(encoding="utf-8")
    routes = set(re.findall(r'@app\.(?:get|post)\("(/api/[^"]*)"', src))
    assert called - routes == set(), f"前端调了未注册的接口: {sorted(called - routes)}"


def test_freshness_ui_elements_present():
    """数据体检那几个元素必须在：按钮、文案位、结果区横幅样式。"""
    html = _index_html()
    for el in ('id="freshbar"', 'id="freshIcon"', 'id="freshText"',
               'id="latestBtn"', 'id="refreshBtn"'):
        assert el in html, f"缺少 {el}"
    assert "freshnessBanner" in _inline_js(html)
