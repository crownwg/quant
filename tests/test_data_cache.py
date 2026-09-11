"""数据缓存体检（quant.data.cache_freshness）。

这块容易被忽略，但它是「能不能拿到今天的调仓清单」的前置条件：
缓存更新只覆盖本次用到的标的，池里没被选中的股票会一直停在旧日期。
抽样发现 30 只里有 15 只停在几个月前——正是这种沉默的停滞最难察觉。
"""
from __future__ import annotations

import pandas as pd
import pytest

from quant.data import cache_freshness


def _write(data_dir, code, dates):
    pd.DataFrame({
        "date": pd.to_datetime(dates),
        "close": [1.0] * len(dates),
    }).to_csv(data_dir / f"{code}.csv", index=False)


def test_reports_last_date_per_code(tmp_path):
    _write(tmp_path, "600519", ["2026-09-09", "2026-09-10"])
    _write(tmp_path, "000858", ["2024-09-09"])
    rows = {r["code"]: r for r in cache_freshness(["600519", "000858"], str(tmp_path))}
    assert rows["600519"]["last"] == "2026-09-10"
    assert rows["600519"]["rows"] == 2
    assert rows["000858"]["last"] == "2024-09-09"


def test_missing_file_is_reported_not_skipped(tmp_path):
    """完全没有缓存文件的代码也要出现在结果里。

    静默跳过会让「池里 36 只」和「体检报告 30 只」对不上，
    用户没法发现少掉的 6 只。
    """
    _write(tmp_path, "600519", ["2026-09-10"])
    rows = cache_freshness(["600519", "999999"], str(tmp_path))
    assert len(rows) == 2
    miss = [r for r in rows if r["code"] == "999999"][0]
    assert miss["last"] == "" and miss["rows"] == 0


def test_broken_cache_does_not_raise(tmp_path):
    """一个缓存文件损坏不该让整池体检失败。"""
    _write(tmp_path, "600519", ["2026-09-10"])
    (tmp_path / "000858.csv").write_text("这不是 CSV", encoding="utf-8")
    rows = {r["code"]: r for r in cache_freshness(["600519", "000858"], str(tmp_path))}
    assert rows["600519"]["last"] == "2026-09-10"
    assert rows["000858"]["last"] == ""


def test_codes_are_normalized_to_six_digits(tmp_path):
    """传进来的代码可能不带前导零（用户手打 858），要能对上 000858.csv。"""
    _write(tmp_path, "000858", ["2026-09-10"])
    rows = cache_freshness(["858", "002557"], str(tmp_path))
    assert [r["code"] for r in rows] == ["000858", "002557"]
    assert rows[0]["last"] == "2026-09-10"


def test_reads_only_date_column(tmp_path):
    """只读 date 列：缓存文件有几百 MB，多读的列纯属浪费。"""
    pd.DataFrame({
        "date": pd.to_datetime(["2026-09-10"]),
        "close": [1.0],
        "别的不认识的列": ["x"],
    }).to_csv(tmp_path / "600519.csv", index=False)
    rows = cache_freshness(["600519"], str(tmp_path))
    assert rows[0]["last"] == "2026-09-10"


def test_empty_input(tmp_path):
    assert cache_freshness([], str(tmp_path)) == []
