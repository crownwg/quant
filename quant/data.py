from pathlib import Path
import json
import time
import pandas as pd


_EASTMONEY_RENAME = {"日期": "date", "收盘": "close", "开盘": "open", "最高": "high", "最低": "low", "成交量": "volume"}

# 缓存覆盖基线：记录每只代码「已被验证服务过的最早起始日」。
# 没有它就无法区分「该股票当时确实还没上市」和「只是当初没往回拉」，
# 结果要么静默返回被截断的数据，要么每次运行都做一次全量回补。
_COVERAGE_FILE = "_coverage.json"


def _load_coverage(data_dir: str) -> dict[str, pd.Timestamp]:
    path = Path(data_dir) / _COVERAGE_FILE
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  覆盖基线损坏就当作不存在，最多多做一次回补
        return {}
    out = {}
    for k, v in raw.items():
        try:
            out[str(k).zfill(6)] = pd.Timestamp(v)
        except Exception:  # noqa: BLE001
            continue
    return out


def _save_coverage(data_dir: str, coverage: dict[str, pd.Timestamp]) -> None:
    path = Path(data_dir) / _COVERAGE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.strftime("%Y-%m-%d") for k, v in sorted(coverage.items())}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _sina_symbol(code: str) -> str:
    """6 位代码 -> 新浪接口需要的带交易所前缀 symbol。"""
    code = str(code).zfill(6)  # 补齐前导零：858 -> 000858
    # 北交所需先判：920xxx 是北交所新代码段，不能落进下面 9 开头（沪B）的分支
    if code.startswith("920") or code.startswith(("4", "8")):
        return f"bj{code}"
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("0", "2", "3")):
        return f"sz{code}"
    raise ValueError(f"无法识别交易所前缀: {code}")


def _fetch_eastmoney(code, start, end, retries, backoff):
    """东财接口，带重试退避，专门对付 RemoteDisconnected 之类的瞬时断连。"""
    import akshare as ak
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start, end_date=end, adjust="qfq",
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff * attempt)  # 2s, 4s, 6s, 8s ... 指数退避
    raise last_exc


def _fetch_sina(code: str, start: str, end: str):
    """新浪接口，返回统一英文列名的日线 DataFrame。"""
    import akshare as ak
    df = ak.stock_zh_a_daily(symbol=_sina_symbol(code), adjust="qfq")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
    if df.empty:
        raise ValueError(f"新浪接口在 {start}~{end} 内无数据（可能停牌/退市）")
    return df


def _fetch(code: str, start: str, end: str, retries: int = 5, backoff: float = 2.0):
    """主用新浪、失败降级东财。返回统一英文列名的日线 DataFrame。

    顺序说明：实测当前网络直连东财 push2his.eastmoney.com 会被 RST 断开
    （RemoteDisconnected），而新浪稳定，故新浪优先。若新浪也不通再回退东财。
    """
    import akshare as ak
    errors = []

    # 1) 新浪（主）
    try:
        return _fetch_sina(code, start, end)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sina: {exc}")

    # 2) 东方财富（兜底）
    try:
        df = _fetch_eastmoney(code, start, end, retries, backoff)
        return df.rename(columns=_EASTMONEY_RENAME)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"eastmoney: {exc}")

    raise RuntimeError(" | ".join(errors))


def _fetch_in_range(code: str, start: str, end: str, retries: int = 5, backoff: float = 2.0):
    """按区间拉取（仅增量追加场景使用），优先东财（支持 start_date），失败再新浪全量过滤。

    与 _fetch 的区别：本函数不抛"区间无数据"——增量追加时偶尔目标区间是节假日是正常的。
    """
    errors = []
    # 1) 东财：start_date/end_date 直接拉区间
    try:
        df = _fetch_eastmoney(code, start, end, retries, backoff)
        return df.rename(columns=_EASTMONEY_RENAME)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"eastmoney: {exc}")
    # 2) 新浪兜底：拿全量再裁区间
    try:
        return _fetch_sina(code, start, end)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sina: {exc}")
    raise RuntimeError(" | ".join(errors))


def _fetch_all(code: str, retries: int = 5, backoff: float = 2.0):
    """拉取标的的**全部历史**（不设下界），用于缓存头部回补。

    为什么要单独一个函数：_fetch_sina / _fetch 都会按 [start, end] 裁剪，
    一旦本地缓存当初是在较晚的 start 上建立的，早于该 start 的历史就永久缺失，
    而增量逻辑无法判断"是股票那时还没上市"还是"只是当初没拉"。
    直接拉全量历史可以让 cache_min 落到真实上市日，一次回补、永久受益。
    """
    errors = []
    try:
        import akshare as ak
        df = ak.stock_zh_a_daily(symbol=_sina_symbol(code), adjust="qfq")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sina: {exc}")
    try:
        df = _fetch_eastmoney(code, "19900101", "20991231", retries, backoff)
        return df.rename(columns=_EASTMONEY_RENAME)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"eastmoney: {exc}")
    raise RuntimeError(" | ".join(errors))


def _append_cache(new_df: pd.DataFrame, path: Path, overwrite: bool = False) -> pd.DataFrame:
    """把 new_df 追加到 path，按 date 去重 + 排序后回写。返回回写后的全量 DataFrame。

    overwrite=True 时丢弃旧缓存直接以 new_df 为准：
    前复权（qfq）价格依赖复权基准日，跨越不同区间拉回来的两段数据基点可能不一致，
    拼接会产生价格台阶。凡是"重新全量拉取"的场景都必须用 overwrite。
    """
    new_df = new_df.copy()
    new_df["date"] = pd.to_datetime(new_df["date"])
    if path.exists() and not overwrite:
        try:
            old = pd.read_csv(path)
            old["date"] = pd.to_datetime(old["date"])
            merged = pd.concat([old, new_df], ignore_index=True)
        except Exception:  # noqa: BLE001  旧缓存损坏就当不存在处理
            merged = new_df
    else:
        merged = new_df
    merged = (
        merged.drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


def _ensure_cached(code: str, start: str, end: str, data_dir: str,
                   retries: int = 5, backoff: float = 2.0,
                   fetch_in_range=None, fetch_full=None,
                   fetch_all=None) -> pd.DataFrame:
    """确保本地缓存至少覆盖 [start, end]，返回缓存 DataFrame（未做区间切片）。

    增量策略（按收益/风险权衡）：
      ① 缓存完全覆盖 [start, end]         → 直读，最快
      ② cache_max < end（往后追加）         → 拉增量，append，最常见
      ③ 需要比缓存更早的历史               → 拉全量历史后**覆盖**写，
                                             使 cache_min 落到真实上市日，一次到位
      ④ 双侧缺失 / 无缓存                   → 全量重拉

    ③ 的触发条件依赖 data/_coverage.json 里记录的「已验证基线」：
    只看 cache_min 无法区分"该股当时没上市"与"当初没往回拉"，会导致要么
    静默返回截断数据、要么每次运行都全量回补一次。基线把首次服务的起始日
    记下来，之后同样或更晚的请求就不会再触发回补。

    参数 fetch_in_range / fetch_full / fetch_all 用于测试时注入假拉取函数；
    缺省走 _fetch_in_range / _fetch / _fetch_all。
    """
    if fetch_in_range is None:
        fetch_in_range = _fetch_in_range
    if fetch_full is None:
        fetch_full = _fetch
    if fetch_all is None:
        fetch_all = _fetch_all

    code = str(code).zfill(6)
    path = Path(data_dir) / f"{code}.csv"
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)

    cached = None
    if path.exists():
        try:
            cached = pd.read_csv(path)
            cached["date"] = pd.to_datetime(cached["date"])
        except Exception:  # noqa: BLE001
            cached = None

    if cached is None or cached.empty:
        df = fetch_full(code, start, end, retries, backoff)
        return _append_cache(df, path)

    cache_max = cached["date"].max()
    cache_min = cached["date"].min()

    # 判断 [start, end] 区间在缓存里是否有数据；用户给的 start 早于股票上市日是合法的（本来就没数据），不应触发全量重拉
    in_range = cached[(cached["date"] >= start_ts) & (cached["date"] <= end_ts)]
    covered_tail = (not in_range.empty) and cache_max >= end_ts

    # 头部缺口判定：优先用「已验证基线」，没有基线时才退回 cache_min。
    # 只看 cache_min 会导致「请求起点早于真实上市日」时每次都回补一次。
    coverage = _load_coverage(data_dir)
    probed = coverage.get(code)
    missing_head = (cache_min > start_ts) if probed is None else (start_ts < probed)

    # 记录本次服务的起始日，下次同样或更晚的请求就不会重复回补
    new_floor = start_ts if probed is None else min(probed, start_ts)
    if probed is None or new_floor < probed:
        coverage[code] = new_floor
        _save_coverage(data_dir, coverage)

    if covered_tail and not missing_head:
        return cached  # 已完整覆盖

    if covered_tail and missing_head:
        # 请求区间比本地缓存更早：拉全量历史后覆盖（qfq 基点会变，不能拼接）。
        # 若该股真的直到 cache_min 才上市，全量结果与旧缓存一致，覆盖无害且下次不再触发。
        df = fetch_all(code, retries, backoff)
        return _append_cache(df, path, overwrite=True)

    if not in_range.empty and cache_max < end_ts:
        # 仅往后追加
        s = (cache_max + pd.Timedelta(days=1)).strftime("%Y%m%d")
        e = end_ts.strftime("%Y%m%d")
        df_new = fetch_in_range(code, s, e, retries, backoff)
        if df_new is None or df_new.empty:
            return cached
        return _append_cache(df_new, path)

    # 缓存里 [start, end] 完全没数据（start 晚于 cache_max，即整段都在缓存之后）：放弃增量，全量重拉
    df = fetch_full(code, start, end, retries, backoff)
    return _append_cache(df, path)


def load_daily(code: str, start: str, end: str, data_dir: str = "data") -> pd.DataFrame:
    """加载单只股票日线。

    增量缓存：本地有缓存时只拉缺口段（绝大多数情况下是"往后追加最新数据"），
    完整覆盖请求区间则直读缓存。详见 _ensure_cached 的策略说明。
    """
    code = str(code).zfill(6)  # 统一 6 位，防止 858 这类丢前导零的输入
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)

    try:
        cached = _ensure_cached(code, start, end, data_dir)
    except Exception as exc:
        raise RuntimeError(f"无法获取 {code} 数据，请检查网络或准备 data/{code}.csv。原始错误: {exc}") from exc

    df = cached[(cached["date"] >= start_ts) & (cached["date"] <= end_ts)].copy()
    if df.empty:
        raise ValueError(f"{code} 在 {start}~{end} 内无数据（可能停牌/退市/未上市）")

    rename = {"日期": "date", "收盘": "close", "开盘": "open", "最高": "high", "最低": "low", "成交量": "volume"}
    df = df.rename(columns=rename)
    required = {"date", "close", "open"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{code} 缺少字段: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    # high/low 用于涨跌停的精确判定（一字板识别），缓存里本来就存了，这里一并暴露出去。
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df.sort_values("date").set_index("date")[cols].astype(float)


def load_panel(codes, start, end, data_dir="data", sleep: float = 1.0, on_error: str = "raise"):
    """返回 {'open','high','low','close','volume'} 五个 date×code 面板。

    high/low 供 filters 做涨跌停价判定；缺失的标的该面板不会有对应列。

    on_error='raise'（默认）：单只拉取失败则整体报错，保持原有严格行为。
    on_error='skip'：单只拉取失败时打印警告并跳过，适合成分股大池（个别退市票不应拖垮整体）。
    """
    fields = ["open", "high", "low", "close", "volume"]
    panels = {f: {} for f in fields}
    for code in codes:
        code = str(code).zfill(6)  # 统一 6 位，保证 dict key 与缓存文件名一致
        try:
            df = load_daily(code, start, end, data_dir)
        except Exception as exc:  # noqa: BLE001
            if on_error == "raise":
                raise
            print(f"⚠️ 跳过 {code}（数据获取失败，已忽略）: {exc}", flush=True)
            time.sleep(sleep)
            continue
        for field in fields:
            panels[field][code] = df[field] if field in df.columns else pd.Series(dtype=float)
        time.sleep(sleep)  # 避免高频请求触发数据源限流断连

    out = {}
    for field, v in panels.items():
        df = pd.DataFrame(v).dropna(how="all")
        # volume 缺失代表停牌 / 未上市，必须填 0 而不是 ffill——
        # 用 ffill 会把停牌日的成交量顶替成前值，导致停牌永远检测不到。
        out[field] = df.fillna(0.0) if field == "volume" else df.ffill()
    return out


def load_universe(codes, start, end, data_dir="data", sleep: float = 1.0):
    """返回 close 面板（date×code），等价于 load_panel(...)['close']，向后兼容。"""
    return load_panel(codes, start, end, data_dir, sleep)["close"]


def load_index(symbol: str, start: str, end: str, data_dir: str = "data") -> pd.Series:
    """加载指数日线收盘价（symbol 形如 sh000300 / sz399006）。

    增量策略：指数接口只能拉全量，所以"增量" = 全量拉一次 + 只把 cache_max 之后的
    行 append 到本地缓存。如果请求区间已被缓存覆盖则直读，避免重复拉全量。
    返回以日期为索引的收盘价 Series，已裁剪到 start~end 区间。
    """
    import akshare as ak

    path = Path(data_dir) / f"index_{symbol}.csv"
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)

    cached = None
    if path.exists():
        try:
            cached = pd.read_csv(path)
            cached["date"] = pd.to_datetime(cached["date"])
        except Exception:  # noqa: BLE001
            cached = None

    if cached is None or cached.empty:
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
        except Exception as exc:
            raise RuntimeError(f"无法获取指数 {symbol} 数据: {exc}") from exc
        df = df.rename(columns={"日期": "date", "收盘": "close"})
        cached = _append_cache(df, path)
    elif cached["date"].max() < end_ts:
        # 缓存往后不够 → 拉全量，只把增量部分 append
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
        except Exception as exc:
            raise RuntimeError(f"无法获取指数 {symbol} 数据: {exc}") from exc
        df = df.rename(columns={"日期": "date", "收盘": "close"})
        df_new = df[pd.to_datetime(df["date"]) > cached["date"].max()]
        if not df_new.empty:
            cached = _append_cache(df_new, path)
    # else: 已完全覆盖 → 直读缓存

    s = cached.set_index("date")["close"].astype(float)
    s = s[(s.index >= start_ts) & (s.index <= end_ts)]
    return s.sort_index()
