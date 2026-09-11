"""股票池自动化：从指数成分股 / 概念板块自动构建标的列表，免去手敲 --codes。

用法
----
- 预置指数池名：hs300 / zz500 / zz1000 / sz50 / cyb / 消费 / 白酒 / 医药 / 蓝筹 ...
- 任意 6 位指数代码：直接传 000300、399006 等
- 概念板块：concept:白酒、concept:人工智能 等（东方财富概念板块）
- 多池合并：--pool "hs300,消费" 取并集并去重

成分股列表带本地缓存（默认 7 天），避免每次回测都去拉一次。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# 池名 -> 指数代码（传给 akshare index_stock_cons）
INDEX_POOLS = {
    "hs300": "000300",      # 沪深300（大盘蓝筹代表）
    "zz500": "000905",      # 中证500（中盘）
    "zz1000": "000852",     # 中证1000（小盘）
    "sz50": "000016",       # 上证50（超大盘）
    "cyb": "399006",        # 创业板指
    "kcb": "000688",        # 科创50
    "消费": "000932",       # 中证主要消费
    "可选消费": "000989",
    "医药": "000933",
    "白酒": "399997",
    "蓝筹": "000300",       # 蓝筹 ≈ 沪深300
}


def resolve_pool(token: str) -> str:
    """池名 -> 指数代码。纯 6 位数字直接当作指数代码。"""
    token = token.strip()
    if token.isdigit() and len(token) == 6:
        return token
    if token in INDEX_POOLS:
        return INDEX_POOLS[token]
    raise ValueError(f"未知股票池 {token!r}，可选: {sorted(INDEX_POOLS)} 或任意 6 位指数代码")


def get_index_constituents(symbol: str, data_dir: str = "data", cache_days: int = 7) -> list[str]:
    """获取指数成分股代码列表（6 位字符串），带本地缓存。"""
    import akshare as ak

    path = Path(data_dir) / f"cons_{symbol}.csv"

    fresh = False
    if path.exists():
        mtime = pd.to_datetime(path.stat().st_mtime, unit="s")
        if (pd.Timestamp.now() - mtime).days < cache_days:
            fresh = True

    if fresh:
        codes = pd.read_csv(path)["code"].astype(str).str.zfill(6).tolist()
    else:
        df = ak.index_stock_cons(symbol=symbol)
        codes = df["品种代码"].astype(str).str.zfill(6).tolist()
        if not codes:
            raise ValueError(f"指数 {symbol} 未返回任何成分股")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"code": codes}).to_csv(path, index=False)
    return codes


def get_concept_constituents(name: str, data_dir: str = "data") -> list[str]:
    """获取东方财富概念板块成分股代码列表。"""
    import akshare as ak

    # 概念板块成分接口走东方财富，可能受网络影响；失败直接抛出由上层处理
    df = ak.stock_board_concept_cons_em(symbol=name)
    codes = df["代码"].astype(str).str.zfill(6).tolist()
    if not codes:
        raise ValueError(f"概念板块 {name!r} 未返回任何成分股")
    return codes


def build_universe(pool_spec: str, data_dir: str = "data") -> list[str]:
    """解析 --pool 规格，返回去重后的代码列表（保持出现顺序）。

    pool_spec 逗号分隔，每项可以是：
      - 池名（hs300 / 消费 / 蓝筹 ...）
      - 6 位指数代码（000300）
      - concept:白酒 形式的概念板块
    """
    tokens = [t for t in pool_spec.split(",") if t.strip()]
    if not tokens:
        return []

    collected: list[str] = []
    for tok in tokens:
        tok = tok.strip()
        if tok.startswith("concept:"):
            name = tok.split(":", 1)[1]
            print(f"▶ 拉取概念板块「{name}」成分股 ...", flush=True)
            codes = get_concept_constituents(name, data_dir)
        else:
            sym = resolve_pool(tok)
            print(f"▶ 拉取指数 {sym}（{tok}）成分股 ...", flush=True)
            codes = get_index_constituents(sym, data_dir)
        print(f"  获得 {len(codes)} 只", flush=True)
        collected.extend(codes)

    # 去重保序
    seen: set[str] = set()
    uniq: list[str] = []
    for c in collected:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq
