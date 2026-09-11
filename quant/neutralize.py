"""因子中性化：把「行业/市值暴露」从因子分数里剥离出来。

为什么必须做
------------
动量因子在 A 股最出名的陷阱就是它同时是一个**小市值因子**。
某段时间小盘股整体暴涨，动量策略选出来的票恰好集中在小盘，
你以为赚的是「动量」，其实赚的是「小盘 beta」——行情切换时这部分收益会
原样还回去，而且你从净值曲线里完全看不出来。
行业同理：2020 年消费动量策略的收益，很大一部分只是「消费行业 beta」。

做法（分组去均值，而非回归）
----------------------------
学术界标准做法是横截面回归 f = α + β·log(市值) + Σγ·行业 + ε，取残差 ε。
本模块用**分组去均值**近似：在每个交易日内，先把因子值减去其所属行业的
行业均值，再减去其所属市值分组的组均值。

为什么用分组去均值而不是直接跑回归：
  - 效果几乎等价（行业哑变量做完，行业内的均值本身就是回归的拟合值）；
  - 计算量小一个量级——回归要对 5000×4000 的面板逐日做 lstsq，分组只需
    几十次向量化减法，低频回测里这个差别是「几秒」和「几分钟」；
  - 不需要处理回归的共线性/奇异矩阵等边界情况。

副产物：单行业内去均值 = 当日全体减去同一个常数，**不改变组内排序**。
所以当股票池只有一个行业（如「白酒」池）时，行业中性化不会改变选股结果——
这是符合直觉的：池子里就没有行业差异可中性化。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------- 行业映射

def build_industry_map(data_dir: str = "data", sleep: float = 0.2,
                       verbose: bool = True) -> pd.Series:
    """拉取「股票代码 → 行业」映射并缓存到 data/industry_map.csv。

    数据源：新浪行业板块（约 49 个行业）。
    东财的行业接口在当前网络下被代理拦截，新浪这个是实测可用的备选。

    返回 index=code 的 Series（值是行业名）。失败时返回空 Series，
    由调用方降级处理（只做市值中性化），绝不静默返回错误分组。
    """
    import akshare as ak
    import time

    rows: list[dict] = []
    try:
        spot = ak.stock_sector_spot(indicator="新浪行业")
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"⚠️ 行业板块列表获取失败，将跳过行业中性化: {exc}", flush=True)
        return pd.Series(dtype=object)

    labels = spot[["label", "板块"]].dropna().drop_duplicates()
    ok, failed = 0, []
    for _, r in labels.iterrows():
        try:
            det = ak.stock_sector_detail(sector=r["label"])
        except Exception:  # noqa: BLE001  单个行业失败不该拖垮整体
            failed.append(r["板块"])
            time.sleep(sleep)
            continue
        if det is None or det.empty or "code" not in det.columns:
            failed.append(r["板块"])
            time.sleep(sleep)
            continue
        for code in det["code"].astype(str):
            rows.append({"code": str(code).zfill(6), "industry": r["板块"]})
        ok += 1
        time.sleep(sleep)

    if not rows:
        if verbose:
            print("⚠️ 未能获取任何行业成分股，将跳过行业中性化", flush=True)
        return pd.Series(dtype=object)

    if verbose and failed:
        print(f"  ⚠️ {len(failed)}/{len(labels)} 个行业抓取失败（{', '.join(failed[:6])}"
              f"{'...' if len(failed) > 6 else ''}）→ 这些行业的标的不会被行业中性化")

    df = pd.DataFrame(rows).drop_duplicates(subset="code", keep="first")
    path = Path(data_dir) / "industry_map.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if verbose:
        print(f"  行业映射    : 已缓存 {len(df)} 只 / {df['industry'].nunique()} 个行业"
              f"（{ok}/{len(labels)} 个行业成功）→ {path}")
    return df.set_index("code")["industry"]


def industry_map(data_dir: str = "data", cache_days: int = 30,
                 refresh: bool = False, verbose: bool = True) -> pd.Series:
    """读取行业映射（带过期刷新）。

    cache_days 内直接读 CSV；过期或 refresh=True 才联网重拉。
    行业分类变动很慢，30 天的缓存完全够用。
    """
    path = Path(data_dir) / "industry_map.csv"
    if path.exists() and not refresh:
        try:
            age = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days
            if age <= cache_days:
                df = pd.read_csv(path, dtype={"code": str})
                return df.set_index("code")["industry"]
        except Exception:  # noqa: BLE001  缓存损坏 → 重拉
            pass
    return build_industry_map(data_dir, verbose=verbose)


# --------------------------------------------------------------- 规模代理

def mktcap_panel(close: pd.DataFrame, amount: pd.DataFrame | None = None,
                 volume: pd.DataFrame | None = None,
                 outstanding_share: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """流通市值面板（元）。返回 None 表示数据不足，调用方应降级。

    ⚠️ 关键细节：缓存里的收盘价是**前复权**价，直接乘流通股本会得到错误的市值
    （复权价随时间被整体缩放，早期市值会被系统性低估）。
    正解是用**真实成交均价 = 成交额 / 成交量**（单位是元/股，未经复权）再乘股本。
    这两个字段本地缓存里都有，不需要额外联网。
    """
    if outstanding_share is None or outstanding_share.empty:
        return None
    sh = outstanding_share.reindex(index=close.index, columns=close.columns).ffill()

    if amount is not None and not amount.empty and volume is not None and not volume.empty:
        vol = volume.reindex(index=close.index, columns=close.columns).astype(float)
        amt = amount.reindex(index=close.index, columns=close.columns).astype(float)
        raw_price = amt / vol.replace(0.0, np.nan)
        raw_price = raw_price.where(raw_price > 0)
        # 停牌日无成交 → 用最近一次的真实均价
        price = raw_price.ffill().bfill()
        # 若某些票完全没有成交额数据，退回复权价（只影响这些票，且是相对量级）
        price = price.where(price.notna(), close)
    else:
        price = close

    return price.astype(float) * sh.astype(float)


def size_buckets(mktcap: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """按当日横截面市值排名分桶，返回 1..n 的桶号（NaN 表示无市值数据）。"""
    if mktcap is None or mktcap.empty:
        return pd.DataFrame(dtype=float)
    rank = mktcap.rank(axis=1, pct=True)
    bucket = np.ceil(rank * n_buckets)
    return bucket.where(rank.notna())


# --------------------------------------------------------------- 去均值

def _group_demean(F: np.ndarray, labels: np.ndarray, min_group: int = 2) -> np.ndarray:
    """静态分组去均值：F 为 (T,N)，labels 为 (N,)（-1 表示无分组，保持原值）。

    每组内样本数不足 min_group 的日子不做处理（宁可不中性化，也不要把值整成 0）。
    """
    out = F.copy()
    for g in np.unique(labels):
        if g < 0:
            continue
        m = labels == g
        if m.sum() < min_group:
            continue
        sub = F[:, m]
        notna = ~np.isnan(sub)
        cnt = notna.sum(axis=1)
        s = np.where(notna, sub, 0.0).sum(axis=1)
        mean = np.divide(s, cnt, out=np.full(cnt.shape, np.nan), where=cnt > 0)
        mean = np.where(cnt >= min_group, mean, np.nan)
        out[:, m] = sub - mean[:, None]
    return out


def _bucket_demean(F: np.ndarray, bucket: np.ndarray, n_buckets: int,
                   min_group: int = 2) -> np.ndarray:
    """逐日分桶去均值（桶号每天变化，不能当静态分组）。"""
    out = F.copy()
    for b in range(1, n_buckets + 1):
        m = bucket == b
        if not m.any():
            continue
        sel = np.where(m, F, np.nan)
        notna = ~np.isnan(sel)
        cnt = notna.sum(axis=1)
        s = np.where(notna, sel, 0.0).sum(axis=1)
        mean = np.divide(s, cnt, out=np.full(cnt.shape, np.nan), where=cnt > 0)
        mean = np.where(cnt >= min_group, mean, np.nan)
        out = np.where(m, F - mean[:, None], out)
    return out


def neutralize(factor: pd.DataFrame, industry: pd.Series | None = None,
               mktcap: pd.DataFrame | None = None, size_buckets_n: int = 5,
               min_group: int = 2) -> pd.DataFrame:
    """对因子做行业 + 市值中性化，返回残差因子（同形状）。

    顺序：先行业去均值，再市值分组去均值。两步都可单独使用。
    NaN 视为「无资格」，不会被填充，也不会参与均值计算。
    """
    if factor is None or factor.empty:
        return factor

    F = factor.to_numpy(dtype=float)
    cols = factor.columns
    T, N = F.shape

    # ---- 行业 ----
    if industry is not None and len(industry):
        ind = industry.reindex(cols)
        codes, uniq = pd.factorize(ind)
        F = _group_demean(F, np.asarray(codes, dtype=int), min_group=min_group)

    # ---- 市值 ----
    if mktcap is not None and not mktcap.empty:
        mc = mktcap.reindex(index=factor.index, columns=cols)
        b = size_buckets(mc, size_buckets_n)
        B = b.to_numpy(dtype=float)
        # 桶号 NaN → -1，避免被 np.where(m) 误纳入
        B = np.nan_to_num(B, nan=-1.0)
        F = _bucket_demean(F, B, size_buckets_n, min_group=min_group)

    return pd.DataFrame(F, index=factor.index, columns=cols)


def parse_spec(spec: str) -> set[str]:
    """解析 --neutralize "industry,size"。空字符串 → 空集合。"""
    if not spec or not spec.strip():
        return set()
    parts = {p.strip().lower() for p in spec.split(",") if p.strip()}
    unknown = parts - {"industry", "size"}
    if unknown:
        raise ValueError(f"--neutralize 只支持 industry / size，收到 {sorted(unknown)}")
    return parts


def apply_neutralize(factor: pd.DataFrame, spec: str, data_dir: str = "data",
                     mktcap: pd.DataFrame | None = None,
                     size_buckets_n: int = 5, refresh_industry: bool = False,
                     verbose: bool = True) -> pd.DataFrame:
    """按 spec（"industry" / "size" / "industry,size"）对因子做中性化。"""
    want = parse_spec(spec)
    if not want:
        return factor

    industry = None
    if "industry" in want:
        industry = industry_map(data_dir, refresh=refresh_industry, verbose=verbose)
        if industry.empty:
            print("⚠️ 行业映射不可用 → 本次跳过行业中性化"
                  "（若要强制作行业中性化，请先联网生成 data/industry_map.csv）")
        else:
            covered = factor.columns.isin(industry.index).sum()
            if verbose:
                print(f"  行业中性化  : 覆盖 {covered}/{len(factor.columns)} 只标的"
                      f"（{industry.nunique()} 个行业）")
            if covered == 0:
                industry = None

    mc = mktcap if "size" in want else None
    if "size" in want:
        if mc is None or mc.empty:
            print("⚠️ 市值面板不可用（缺流通股本/成交额）→ 本次跳过市值中性化")
        elif verbose:
            print(f"  市值中性化  : 按当日横截面分 {size_buckets_n} 桶去均值")

    return neutralize(factor, industry=industry, mktcap=mc,
                      size_buckets_n=size_buckets_n)
