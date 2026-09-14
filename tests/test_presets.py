"""自选组合预设：存/取/删 的正确性，以及 Web 接口的错误处理。

值得锁的点：预设是「用户数据的唯一持久化出口」，写坏了不会报错——
下次打开下拉里少一个组合，用户只会以为自己没存过。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from quant import presets, web


@pytest.fixture
def tmp_custom(tmp_path, monkeypatch):
    """把自定义预设的落盘位置指到临时目录，免得跑测试弄脏仓库。"""
    monkeypatch.setattr(presets, "CUSTOM_PATH", tmp_path / "my_presets.json")
    return presets.CUSTOM_PATH


# ----------- 代码归一化 -----------

def test_normalize_codes_splits_and_pads():
    assert presets.normalize_codes("600031, 000157 002557；603338") == [
        "600031", "000157", "002557", "603338"]


def test_normalize_codes_pads_short_and_dedups():
    assert presets.normalize_codes(["858", "000858", "858"]) == ["000858"]


def test_normalize_codes_drops_non_numeric():
    """用户粘「600031 三一重工」过来不该整条存不进去。"""
    assert presets.normalize_codes("600031 三一重工, 000157") == ["600031", "000157"]


def test_normalize_codes_empty():
    assert presets.normalize_codes("") == []


# ----------- 名称校验 -----------

def test_validate_name_rejects_empty_and_too_long():
    assert presets.validate_name("")
    assert presets.validate_name("x" * (presets.MAX_NAME + 1))


def test_validate_name_rejects_path_chars():
    for bad in ("a/b", "a\\b", "a:b", "a*b", "a?b", 'a"b', "a<b", "a>b", "a|b"):
        assert presets.validate_name(bad), bad


def test_validate_name_rejects_builtin_name():
    name = next(iter(presets.BUILTIN))
    assert presets.validate_name(name)


def test_validate_name_accepts_normal():
    assert presets.validate_name("老登组合") is None


# ----------- 存 / 读 / 删 -----------

def test_save_then_load_roundtrip(tmp_custom):
    data, err = presets.save_custom("我的组合", "600031,000157")
    assert err is None and data["我的组合"] == ["600031", "000157"]
    assert presets.load_custom() == {"我的组合": ["600031", "000157"]}
    assert json.loads(tmp_custom.read_text(encoding="utf-8"))["我的组合"] == [
        "600031", "000157"]


def test_save_rejects_no_codes(tmp_custom):
    _, err = presets.save_custom("空组合", "")
    assert err and "没有识别到" in err
    assert presets.load_custom() == {}


def test_save_rejects_builtin_name(tmp_custom):
    name = next(iter(presets.BUILTIN))
    _, err = presets.save_custom(name, "600031")
    assert err and "内置" in err


def test_save_overwrites_same_name(tmp_custom):
    presets.save_custom("组合", "600031")
    _, err = presets.save_custom("组合", "000157,000425")
    assert err is None
    assert presets.load_custom()["组合"] == ["000157", "000425"]


def test_save_rejects_too_many_codes(tmp_custom):
    many = ",".join(f"{600000 + i}" for i in range(presets.MAX_CODES + 1))
    _, err = presets.save_custom("太多", many)
    assert err and "最多" in err


def test_delete_custom(tmp_custom):
    presets.save_custom("临时", "600031")
    data, err = presets.delete_custom("临时")
    assert err is None and "临时" not in data
    assert presets.load_custom() == {}


def test_delete_missing_reports_error(tmp_custom):
    _, err = presets.delete_custom("不存在")
    assert err and "没有" in err


def test_delete_builtin_not_allowed(tmp_custom):
    name = next(iter(presets.BUILTIN))
    _, err = presets.delete_custom(name)
    assert err


def test_load_custom_survives_corrupt_file(tmp_custom):
    tmp_custom.write_text("{ 这不是 json", encoding="utf-8")
    assert presets.load_custom() == {}


def test_load_custom_ignores_bad_entries(tmp_custom):
    """坏条目不能让整个文件读不出来；能救回来的部分照救。

    123 会被宽容地补成 000123（数字串本来就该这么处理），
    xyz 这种非数字才真的丢掉。
    """
    tmp_custom.write_text(json.dumps({"好的": ["600031"], "坏的": ["xyz", 123]}),
                          encoding="utf-8")
    data = presets.load_custom()
    assert data["好的"] == ["600031"]
    assert data["坏的"] == ["000123"]


# ----------- 下拉数据 -----------

def test_all_presets_shape(tmp_custom):
    presets.save_custom("我的", "600031,000157")
    d = presets.all_presets()
    assert [x["name"] for x in d["builtin"]] == list(presets.BUILTIN)
    assert d["custom"][0] == {"name": "我的", "codes": ["600031", "000157"], "n": 2}


def test_builtin_codes_all_six_digits():
    for name, codes in presets.BUILTIN.items():
        assert codes, f"内置预设 {name} 是空的"
        for c in codes:
            assert len(c) == 6 and c.isdigit(), f"{name} 里 {c} 不是 6 位数字"


# ----------- Web 接口 -----------

def test_list_presets_route(tmp_custom):
    d = asyncio.run(web.list_presets())
    assert d["builtin"] and isinstance(d["custom"], list)


def test_save_preset_route(tmp_custom):
    d = asyncio.run(web.save_preset(web.PresetIn(name="测试", codes="600031,000157")))
    assert d["ok"] and d["custom"][0]["n"] == 2


def test_save_preset_route_invalid_name(tmp_custom):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(web.save_preset(web.PresetIn(name="", codes="600031")))
    assert ei.value.status_code == 400


def test_delete_preset_route(tmp_custom):
    asyncio.run(web.save_preset(web.PresetIn(name="待删", codes="600031")))
    d = asyncio.run(web.delete_preset("待删"))
    assert d["ok"] and d["custom"] == []


def test_delete_preset_route_missing(tmp_custom):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(web.delete_preset("不存在"))
    assert ei.value.status_code == 404


# ----------- 前端契约 -----------

def test_preset_ui_elements_present():
    html = open(web.TEMPLATE, encoding="utf-8").read()
    for el in ('id="presetSelect"', 'id="presetUseBtn"', 'id="presetSaveBtn"',
               'id="presetDelBtn"', 'id="presetHint"'):
        assert el in html, f"缺少 {el}"


def test_preset_handlers_defined():
    html = open(web.TEMPLATE, encoding="utf-8").read()
    for fn in ("async function loadPresets", "function onUsePreset",
               "async function onSavePreset", "async function onDeletePreset"):
        assert fn in html, f"缺少 {fn}"
    assert "loadPresets();" in html, "页面加载时没有拉取预设"
