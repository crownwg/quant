"""自选组合预设：给一组常用股票列表起个名字，下次下拉直接选。

存在的理由：用户每次想跑自己的持仓组合都要重新敲一长串代码，
敲错一位（000858 打成 000885）回测照样跑通，只是结果完全对不上——
预设把「这一篮子股票到底是什么」固定下来，只维护一次。

内置预设只读；用户自己存的写在项目根目录 `my_presets.json`（已 gitignore，
属于个人数据，不该进仓库）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUSTOM_PATH = ROOT / "my_presets.json"

# 内置预设：只放「确实常用、且代码不容易记错」的组合。
BUILTIN: dict[str, list[str]] = {
    "老登组合（消费+蓝筹）": [
        "002557", "600519", "000858", "600887", "603288",
        "600031", "000651", "000333", "600036", "601318", "600276", "002594",
    ],
    "消费白马": ["002557", "600519", "000858", "600887", "603288"],
    "工程机械": ["600031", "000157", "000425", "601100", "000528", "603338"],
    "白酒": ["600519", "000858", "000568", "002304", "600809", "603369"],
    "金融蓝筹": ["600036", "601318", "601166", "000001", "600030"],
}

MAX_NAME = 20
MAX_CODES = 100
# 禁掉路径分隔符和 Windows 文件名非法字符：预设名只用于展示，
# 但防一手总比以后被人塞个路径进来强。
_BAD_NAME = re.compile(r'[/\\:*?"<>|\r\n\t]')


def normalize_codes(raw: str | list[str]) -> list[str]:
    """把「600031, 000157」或列表都整理成去重后的 6 位代码列表。

    只保留纯数字段：用户很可能直接粘「600031 三一重工」过来，
    汉字部分丢掉即可，不该让整条预设存不进去。
    """
    if isinstance(raw, list):
        parts = [str(x) for x in raw]
    else:
        # 中英文逗号/分号/顿号/空白都算分隔符：用户常从 Excel 或微信里
        # 复制一整串过来，分隔符号是什么全看当时用的输入法。
        parts = re.split(r"[,\s;；，、\t]+", str(raw or ""))
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if p.isdigit():
            c6 = p.zfill(6)
            if c6 not in out:
                out.append(c6)
    return out


def validate_name(name: str) -> str | None:
    """返回错误信息；合法则返回 None。"""
    name = (name or "").strip()
    if not name:
        return "预设名不能为空"
    if len(name) > MAX_NAME:
        return f"预设名最长 {MAX_NAME} 个字"
    if _BAD_NAME.search(name):
        return '预设名不能包含 / \\ : * ? " < > | 这些字符'
    if name in BUILTIN:
        return f"「{name}」是内置预设，不能改；换一个名字"
    return None


def load_custom() -> dict[str, list[str]]:
    if not CUSTOM_PATH.exists():
        return {}
    try:
        data = json.loads(CUSTOM_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  手改坏了不该让整个页面打不开
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        codes = normalize_codes(v) if isinstance(v, (list, str)) else []
        if codes:
            out[str(k)] = codes
    return out


def _write_custom(data: dict[str, list[str]]) -> None:
    CUSTOM_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_custom(name: str, codes: str | list[str]) -> tuple[dict[str, list[str]], str | None]:
    """保存（同名覆盖）。返回 (全部自定义预设, 错误信息)。"""
    err = validate_name(name)
    if err:
        return load_custom(), err
    name = name.strip()
    cs = normalize_codes(codes)
    if not cs:
        return load_custom(), "没有识别到股票代码（每行或逗号分隔的 6 位数字）"
    if len(cs) > MAX_CODES:
        return load_custom(), f"一次最多存 {MAX_CODES} 只"

    data = load_custom()
    data[name] = cs
    try:
        _write_custom(data)
    except Exception as exc:  # noqa: BLE001
        return load_custom(), f"写入失败: {exc}"
    return data, None


def delete_custom(name: str) -> tuple[dict[str, list[str]], str | None]:
    data = load_custom()
    if name not in data:
        return data, f"没有名为「{name}」的自定义预设"
    del data[name]
    try:
        _write_custom(data)
    except Exception as exc:  # noqa: BLE001
        return data, f"删除失败: {exc}"
    return data, None


def all_presets() -> dict[str, dict]:
    """给前端下拉用：区分内置 / 自定义，并带上代码数量。"""
    return {
        "builtin": [{"name": k, "codes": v, "n": len(v)} for k, v in BUILTIN.items()],
        "custom": [{"name": k, "codes": v, "n": len(v)}
                   for k, v in load_custom().items()],
    }
