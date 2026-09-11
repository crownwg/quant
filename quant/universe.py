"""股票池自动化：从指数成分股 / 概念板块 / 全市场构建标的列表。

用法
----
- 预置指数池名：hs300 / zz500 / zz1000 / sz50 / cyb / 消费 / 白酒 / 医药 / 蓝筹 ...
- 任意 6 位指数代码：直接传 000300、399006 等
- 概念板块：concept:白酒、concept:人工智能 等（东方财富概念板块）
- 全市场：all（含已退市个股，用于彻底规避幸存者偏差）
- 多池合并：--pool "hs300,消费" 取并集并去重

幸存者偏差与「时点快照」
------------------------
指数成分股接口（akshare 的 index_stock_cons）返回的是**当前**成分股。
若用它回溯 2018 年的行情，池子里装的却是 2026 年仍留在指数里的公司——
那些当年在指数里、后来被剔除或退市并暴跌的股票压根没进池子，
结果就是「只在活下来的赢家里选股」，回测收益被系统性高估。

本模块用「时点快照」来消除它：
  1. 每次联网拉取成分股，都会自动归档一份快照到
     `data/cons_snapshots/cons_<指数>_<YYYYMMDD>.csv`，日积月累形成
     可回溯的时点历史；
  2. 查询时传 `as_of=<日期>`，会挑选「不晚于该日期的最新快照」作为
     当时的真实成分股；
  3. 若该日期早于所有已有快照（历史尚未积累），会**显式告警**并回退到
     当前成分股，绝不静默使用有偏数据。

彻底方案是 `--pool all`：用全市场（含退市）标的建池，让「当时是否上市」
完全由行情数据本身决定，不依赖成分股名单。
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

# 全市场模式的池名别名
ALL_MARKET_TOKENS = {"all", "全部", "全市场", "a股", "a"}

SNAPSHOT_SUBDIR = "cons_snapshots"


def resolve_pool(token: str) -> str:
    """池名 -> 指数代码。纯 6 位数字直接当作指数代码。"""
    token = token.strip()
    if token.isdigit() and len(token) == 6:
        return token
    if token in INDEX_POOLS:
        return INDEX_POOLS[token]
    raise ValueError(
        f"未知股票池 {token!r}，可选: {sorted(INDEX_POOLS)} + 'all'，或任意 6 位指数代码")


def is_all_market(token: str) -> bool:
    return token.strip().lower() in ALL_MARKET_TOKENS


# --------------------------------------------------------------- 时点快照

def _snapshot_dir(data_dir: str) -> Path:
    return Path(data_dir) / SNAPSHOT_SUBDIR


def snapshot_path(symbol: str, as_of, data_dir: str = "data") -> Path:
    """快照文件名：cons_<指数代码>_<YYYYMMDD>.csv"""
    day = pd.Timestamp(as_of).strftime("%Y%m%d")
    return _snapshot_dir(data_dir) / f"cons_{symbol}_{day}.csv"


def save_snapshot(symbol: str, codes, as_of=None, data_dir: str = "data") -> Path:
    """把一份成分股名单按指定日期归档为快照。已存在则不覆盖。"""
    as_of = pd.Timestamp.now() if as_of is None else pd.Timestamp(as_of)
    path = snapshot_path(symbol, as_of, data_dir)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"code": [str(c).zfill(6) for c in codes]}).to_csv(path, index=False)
    return path


def list_snapshots(symbol: str, data_dir: str = "data") -> list[tuple[pd.Timestamp, Path]]:
    """列出某指数的全部快照，按日期升序。返回 [(日期, 路径)]。"""
    d = _snapshot_dir(data_dir)
    if not d.exists():
        return []
    out = []
    for p in d.glob(f"cons_{symbol}_*.csv"):
        stamp = p.stem.rsplit("_", 1)[-1]
        try:
            out.append((pd.Timestamp(stamp), p))
        except Exception:  # noqa: BLE001  文件名不符合规范就忽略
            continue
    return sorted(out, key=lambda t: t[0])


def _read_codes(path: Path) -> list[str]:
    return pd.read_csv(path)["code"].astype(str).str.zfill(6).tolist()


# ------------------------------------------------------------- 成分股获取

def get_index_constituents(symbol: str, data_dir: str = "data", cache_days: int = 7,
                           archive_snapshot: bool = True) -> list[str]:
    """获取指数**当前**成分股（6 位字符串），带本地缓存。

    archive_snapshot=True 时，每次真正联网拉取后都会自动归档一份当日快照，
    用于逐步积累可回溯的时点历史。
    """
    import akshare as ak

    path = Path(data_dir) / f"cons_{symbol}.csv"

    fresh = False
    if path.exists():
        mtime = pd.to_datetime(path.stat().st_mtime, unit="s")
        if (pd.Timestamp.now() - mtime).days < cache_days:
            fresh = True

    if fresh:
        codes = _read_codes(path)
    else:
        df = ak.index_stock_cons(symbol=symbol)
        codes = df["品种代码"].astype(str).str.zfill(6).tolist()
        if not codes:
            raise ValueError(f"指数 {symbol} 未返回任何成分股")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"code": codes}).to_csv(path, index=False)

    if archive_snapshot:
        # 无论本次是走缓存还是联网，都补一份「今天」的快照（已存在则跳过）。
        # 否则在缓存新鲜期内永远不会归档，时点历史就积累不起来。
        save_snapshot(symbol, codes, data_dir=data_dir)
    return codes


def get_index_constituents_asof(symbol: str, as_of=None, data_dir: str = "data",
                                cache_days: int = 7):
    """按**时点**取成分股。返回 (codes, meta)。

    meta = {
      "symbol": 指数代码,
      "as_of": 请求时点,
      "snapshot_date": 实际命中的快照日期（回退时为 None）,
      "pit": 是否真正做到了时点正确,
      "message": 需要提示给用户的话（无问题时为空串）,
    }

    选取规则：
      - as_of 为空 或 不早于今天 → 当前成分股即可，视为时点正确；
      - as_of 早于今天 → 取「不晚于 as_of 的最新快照」；
      - 没有任何可用快照 → 回退当前成分股，pit=False 并给出告警文案。
    """
    now = pd.Timestamp.now().normalize()
    as_of_ts = now if as_of is None else pd.Timestamp(as_of)

    meta = {"symbol": symbol, "as_of": as_of_ts, "snapshot_date": None,
            "pit": True, "message": ""}

    # 请求的就是当下（或未来）→ 当前名单本身就是正确的时点名单
    if as_of_ts >= now:
        meta["snapshot_date"] = now
        return get_index_constituents(symbol, data_dir, cache_days), meta

    snaps = list_snapshots(symbol, data_dir)
    eligible = [s for s in snaps if s[0] <= as_of_ts]
    if eligible:
        day, path = eligible[-1]
        meta["snapshot_date"] = day
        codes = _read_codes(path)
        if not codes:
            raise ValueError(f"快照 {path} 为空")
        return codes, meta

    # 没有历史快照可用 → 明确告知偏差来源，而不是静默返回
    codes = get_index_constituents(symbol, data_dir, cache_days)
    meta["pit"] = False
    meta["message"] = (
        f"指数 {symbol} 缺少 {as_of_ts.date()} 及之前的成分股快照，"
        f"已回退使用**当前**成分股 → 该池存在幸存者偏差，历史收益被高估。\n"
        f"    缓解办法：① 改用 --pool all（全市场含退市，无此偏差）；"
        f"② 从现在起定期运行本工具积累快照（每次拉取会自动归档）；"
        f"③ 用 --min-listed-days 至少剔除次新股干扰。"
    )
    return codes, meta


def get_concept_constituents(name: str, data_dir: str = "data") -> list[str]:
    """获取东方财富概念板块成分股代码列表。"""
    import akshare as ak

    # 概念板块成分接口走东方财富，可能受网络影响；失败直接抛出由上层处理
    df = ak.stock_board_concept_cons_em(symbol=name)
    codes = df["代码"].astype(str).str.zfill(6).tolist()
    if not codes:
        raise ValueError(f"概念板块 {name!r} 未返回任何成分股")
    return codes


def _norm_code(raw) -> str | None:
    """把各种来源的代码统一成 6 位字符串，非法值返回 None。"""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "-"):
        return None
    try:
        return str(int(float(s))).zfill(6)
    except (TypeError, ValueError):
        return None


def get_all_market_codes(data_dir: str = "data", include_delisted: bool = True,
                         cache_days: int = 7) -> list[str]:
    """全部 A 股代码（含已退市），带本地缓存。

    这是规避幸存者偏差最彻底的做法：池子不再依赖指数名单，
    「某只股票在当时是否可交易」由行情数据本身（有无当日 K 线）决定。
    已退市个股必须纳入——它们正是幸存者偏差里被丢掉的那一批。

    数据源做了多级降级：不同接口在不同网络环境下可用性差别很大
    （例如某些环境下 www.bse.cn 会被代理拒绝），因此逐个尝试、谁通用谁，
    只要拿到足够数量的代码就继续，不让单一数据源故障拖垮整个建池。
    """
    import akshare as ak

    path = Path(data_dir) / "cons_all.csv"
    fresh = False
    if path.exists():
        mtime = pd.to_datetime(path.stat().st_mtime, unit="s")
        if (pd.Timestamp.now() - mtime).days < cache_days:
            fresh = True
    if fresh:
        return _read_codes(path)

    codes: set[str] = set()
    used: list[str] = []

    # ① 在市的全部 A 股：先试聚合接口，失败再走各交易所分接口
    listing_sources = [
        ("stock_info_a_code_name", None),
        ("stock_info_sh_name_code", "证券代码"),
        ("stock_info_sz_name_code", "A股代码"),
        ("stock_info_bj_name_code", "证券代码"),
    ]
    for fn_name, code_col in listing_sources:
        fn = getattr(ak, fn_name, None)
        if fn is None:
            continue
        try:
            df = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ 在市名单 {fn_name} 不可用: {exc}", flush=True)
            continue
        col = code_col or next(
            (c for c in ("code", "代码", "证券代码", "A股代码") if c in df.columns), None)
        if col is None:
            continue
        got = {c for c in (_norm_code(v) for v in df[col]) if c}
        if got:
            codes |= got
            used.append(f"{fn_name}({len(got)})")

    # ② 已退市：它们是幸存者偏差里被丢掉的那一批，必须补回来
    if include_delisted:
        for fn_name, code_col in (("stock_info_sh_delist", "公司代码"),
                                  ("stock_info_sz_delist", "证券代码")):
            fn = getattr(ak, fn_name, None)
            if fn is None:
                continue
            try:
                df = fn()
            except Exception as exc:  # noqa: BLE001  退市名单拿不到不应阻断主流程
                print(f"⚠️ 退市名单 {fn_name} 获取失败（少数退市股会漏掉）: {exc}", flush=True)
                continue
            if code_col not in df.columns:
                continue
            got = {c for c in (_norm_code(v) for v in df[code_col]) if c}
            if got:
                codes |= got
                used.append(f"{fn_name}({len(got)})")

    if not codes:
        raise ValueError("未能从任何数据源获取 A 股代码（请检查网络/代理）")
    out = sorted(codes)
    print(f"  全市场池来源: {', '.join(used)} → 去重后 {len(out)} 只", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"code": out}).to_csv(path, index=False)
    return out


# ------------------------------------------------------------- 池子组装

def build_universe_report(pool_spec: str, data_dir: str = "data", as_of=None):
    """解析 --pool 规格，返回 (去重代码列表, 每个子池的元信息行)。

    每行 meta 含 pool / symbol / n_codes / pit / message，
    便于 main.py 在终端集中打印幸存者偏差告警。
    """
    tokens = [t for t in pool_spec.split(",") if t.strip()]
    if not tokens:
        return [], []

    collected: list[str] = []
    report: list[dict] = []

    for tok in tokens:
        tok = tok.strip()
        if tok.startswith("concept:"):
            name = tok.split(":", 1)[1]
            print(f"▶ 拉取概念板块「{name}」成分股 ...", flush=True)
            codes = get_concept_constituents(name, data_dir)
            report.append({"pool": tok, "symbol": "-", "n_codes": len(codes),
                           "pit": True, "message": ""})
        elif is_all_market(tok):
            print("▶ 构建全市场股票池（含已退市，用于规避幸存者偏差）...", flush=True)
            codes = get_all_market_codes(data_dir)
            report.append({"pool": tok, "symbol": "all", "n_codes": len(codes),
                           "pit": True, "message": ""})
        else:
            sym = resolve_pool(tok)
            dates = f"（时点 {pd.Timestamp(as_of).date()}）" if as_of is not None else ""
            print(f"▶ 拉取指数 {sym}（{tok}）成分股{dates} ...", flush=True)
            codes, meta = get_index_constituents_asof(sym, as_of=as_of, data_dir=data_dir)
            report.append({"pool": tok, "symbol": sym, "n_codes": len(codes),
                           "pit": meta["pit"], "message": meta["message"]})
        print(f"  获得 {len(codes)} 只", flush=True)
        collected.extend(codes)

    # 去重保序
    seen: set[str] = set()
    uniq: list[str] = []
    for c in collected:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq, report


def build_universe(pool_spec: str, data_dir: str = "data", as_of=None) -> list[str]:
    """build_universe_report 的简化入口，只返回代码列表（向后兼容）。"""
    codes, _ = build_universe_report(pool_spec, data_dir=data_dir, as_of=as_of)
    return codes
