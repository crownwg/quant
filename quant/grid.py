"""参数优化：在 (lookback, top_n, buffer) 网格上回测，找「稳健最优」参数。

为什么需要它
------------
阶段5/风险归因已经说明：策略收益高度依赖行情状态，单一「全样本最优」参数
往往是曲线拟合（凑巧在 2020 消费牛里最强）。本模块做网格搜索，并对每个
参数组合同时给出：
  - 全样本指标：总收益 / 年化 / 夏普 / 最大回撤 / 卡玛 / 年化换手
  - 稳健性代理：最差年度收益、正收益年度占比（年度分解）
  - 若提供基准：超额收益、信息比率
让用户既能看「哪个参数最猛」，也能看「哪个参数最不挑行情」。

设计要点
--------
- 价量面板（含预热期）只加载一次，网格内复用；只有「因子打分 / 选股 / 回测」
  随参数变化，避免重复下载。
- 因子默认用动量（主旋钮是 lookback）；top_n / buffer 控制选股范围与换手。
- 多因子 / 其它策略的网格是后续扩展，这里先把动量这一条线做扎实。
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run, cost_kwargs
from .strategy import factor_weights, weights_from_args
from . import factors, rolling, timing


def parse_ints(text: str) -> list[int]:
    """把 '60,90,120' 解析成 [60, 90, 120]。"""
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    """把 '0,0.02,0.05' 解析成 [0.0, 0.02, 0.05]。"""
    return [float(x.strip()) for x in text.split(",") if x.strip()]


# ============================================================ 参数组合
#
# 为什么 combo 是 dict 而不是 tuple
# --------------------------------
# 原来 combo = (lookback, top_n, buffer)。加入择时后维度变成 6 个，tuple 的
# 下标访问（combo[0]/[1]/[2]）在每处都要重新数一遍，稍不留神就串位。改成 dict
# 后新增维度不需要改动任何读写点，报告里也能直接按 key 渲染。
#
# 为什么「off 组合不展开 timing_lookback × timing_band」
# ----------------------------------------------------
# 不择时时窗口取 120 还是 200 完全等价，若跟着展开，同一个策略会在网格里
# 重复出现 N 次、把「平均夏普」这类统计搅浑（等价样本被当成独立样本加权）。
# 所以 off 只保留一份。

def build_combos(lookbacks: list[int], top_ns: list[int], buffers: list[int],
                 timing_modes: list[str] | None = None,
                 timing_lookbacks: list[int] | None = None,
                 timing_bands: list[float] | None = None) -> list[dict]:
    """构造参数组合列表（每个组合是 dict）。

    timing_modes 为空 → 只有一组 timing="off"，各组合沿用调用方给定的整体敞口
    （即旧的 `--timing ma --grid ...` 行为，网格只扫选股参数）。
    timing_modes 非空 → **强制包含 "off"**：搜索空间里必须有「不择时」这个选项，
    否则 walk-forward 无法回答「择时到底该不该用」——它只会在给定的几组择时参数
    里挑一个，永远得不出「不如不择时」的结论。
    """
    modes = [str(m).strip().lower() for m in (timing_modes or []) if str(m).strip()]
    search_timing = bool(modes)
    if search_timing and "off" not in modes:
        modes.insert(0, "off")
    if not modes:
        modes = ["off"]

    tlbs = [int(x) for x in (timing_lookbacks or [120])] or [120]
    tbs = [float(x) for x in (timing_bands or [0.0])] or [0.0]

    combos: list[dict] = []
    for tm in modes:
        variants = ([{}] if tm == "off"
                    else [{"timing_lookback": lb, "timing_band": b}
                          for lb in tlbs for b in tbs])
        for v in variants:
            for lb, tn, buf in itertools.product(lookbacks, top_ns, buffers):
                combo = {"lookback": int(lb), "top_n": int(tn), "buffer": int(buf),
                         "timing": tm,
                         "timing_lookback": int(v.get("timing_lookback", 0)),
                         "timing_band": float(v.get("timing_band", 0.0))}
                combos.append(combo)
    return combos


def combo_label(combo: dict) -> str:
    """把参数组合渲染成一行可读标签，报告与日志共用（避免两处格式漂移）。"""
    base = f"lb={combo.get('lookback')} n={combo.get('top_n')} buf={combo.get('buffer')}"
    tm = (combo.get("timing") or "off").strip().lower()
    if tm in ("", "off", "none"):
        return f"{base} · 不择时"
    lb = combo.get("timing_lookback") or 0
    band = float(combo.get("timing_band") or 0.0)
    return f"{base} · {tm}({lb}日,带{band:.1%})"


def _timing_cfg(combo: dict, args) -> dict:
    """把 combo 的择时维度 + args 里其余择时旋钮合成一份 cfg（timing.build_exposure 吃 dict）。"""
    return {
        "timing": combo.get("timing") or "off",
        "timing_lookback": int(combo.get("timing_lookback") or getattr(args, "timing_lookback", 120)),
        "timing_band": float(combo.get("timing_band") or getattr(args, "timing_band", 0.0) or 0.0),
        "timing_ma_slope": int(getattr(args, "timing_ma_slope", 0) or 0),
        "timing_min_exposure": float(getattr(args, "timing_min_exposure", 0.0) or 0.0),
        "timing_max_exposure": float(getattr(args, "timing_max_exposure", 1.0) or 1.0),
        "timing_smooth": int(getattr(args, "timing_smooth", 0) or 0),
        "vol_target": float(getattr(args, "vol_target", 0.0) or 0.0),
        "vol_target_lookback": int(getattr(args, "vol_target_lookback", 60) or 60),
        "vol_floor": float(getattr(args, "vol_floor", 0.2) or 0.0),
        "vol_cap": float(getattr(args, "vol_cap", 1.0) or 1.0),
    }


def _slice_series(series: pd.Series | None, end_pos: int | None) -> pd.Series | None:
    """把敞口序列截到 end_pos（不含），保证「只用 < end_pos 的信息」。"""
    if series is None:
        return None
    return series.iloc[:end_pos] if end_pos is not None else series


def _exposure_getter(proxy: pd.Series | None, prices_all: pd.DataFrame, args,
                     fallback: pd.Series | None = None,
                     search_timing: bool = False):
    """返回 `get(combo, end_pos) -> 敞口序列 | None`。

    约定
    ----
    - search_timing=False：不扫择时维度，直接沿用调用方给的 fallback 敞口
      （即 `--timing ma --grid ...` 这种「择时定死、只扫选股参数」的旧路径）。
    - search_timing=True ：按 combo 里的 (timing, 窗口, 带) 构造敞口，
      同一组参数只算一次（训练窗选参与测试窗验证共用），之后只做切片。

    为什么可以「整条算完再切片」而不是「截断后再算」
    ----------------------------------------------
    本模块所有择时函数都只依赖 close[t] 及更早（见 timing.py 的因果性说明），
    整条序列在 t 处的取值与「只用前 t 个点重算」完全一致；带上滞回状态机也一样，
    记忆只沿时间正向流动。反过来，若每次都截断重算，状态机会在每个窗口开头
    重置为空仓，训练/测试两窗的口径反而对不上。
    """
    cache: dict[tuple, pd.Series | None] = {}

    def get(combo: dict, end_pos: int | None):
        tm = (combo.get("timing") or "off").strip().lower()
        if not search_timing:
            return _slice_series(fallback, end_pos)
        key = (tm, int(combo.get("timing_lookback") or 0), float(combo.get("timing_band") or 0.0))
        if key not in cache:
            cfg = _timing_cfg(combo, args)
            cache[key] = timing.build_exposure(
                cfg, proxy=proxy, prices=prices_all, verbose=False)
        return _slice_series(cache[key], end_pos)

    return get


# ============================================================ 参数集成
#
# 为什么要做参数集成
# ----------------
# walk-forward 的「训练窗取 argmax」有个致命弱点：**argmax 永远挑到那个尖峰**，
# 而尖峰多半是噪声。实测（消费池 2026-09 那轮）5 折选中的滞回带 0 与 2% 各占一半——
# 说明这一维根本没有信息，argmax 只是在两枚硬币里挑正面的那枚。
#
# 两种不做单点择优的替代：
#   1. **邻域平滑选参**：先在参数曲面上做一次均值滤波（每个点取「自己+相邻档」的
#      训练期均值），再取 argmax。等价于「不选最高点，选最高的那一片区域」，
#      把孤立尖峰自然抹掉。选参范式不变，只是不再踩尖峰。
#   2. **全组合等权集成**：干脆不选，把所有候选的权重等权平均后一起持有。
#      这是「承认自己不知道哪个参数对」的诚实做法，代价是收益被摊薄。
#
# 两者都是纯增量：不改变原有 walk-forward 口径，只在报告里多两条对照曲线。


def _neighbor_map(combos: list[dict], lookbacks: list[int], top_ns: list[int],
                  buffers: list[int], timing_lookbacks: list[int] | None,
                  timing_bands: list[float] | None) -> list[list[int]]:
    """为每个 combo 找出「参数空间邻域」的下标列表（含自身）。

    邻域定义
    --------
    - 选股三维（lookback / top_n / buffer）：候选序列上**相差不超过 1 档**；
    - 择时模式必须相同（off 只和 off 相邻——跨模式不是「相邻」，是「换了个方法」）；
    - 择时两维（窗口 / 滞回带）同样相差不超过 1 档。

    即 3×3×3 的立方体邻域（择时维度上再乘一个 3×3）。这样邻域均值就是参数曲面上的
    一次均值滤波：孤立的尖峰会被邻居拉下来，而**成片的高原**会被保留——
    我们要的正是「哪一片参数区域整体好」，不是「哪个点最高」。
    """
    stock_dims = [lookbacks, top_ns, buffers]
    stock_keys = ["lookback", "top_n", "buffer"]
    timing_dims = [list(timing_lookbacks or []), list(timing_bands or [])]
    timing_keys = ["timing_lookback", "timing_band"]

    def _pos(dims: list, v):
        try:
            return dims.index(v)
        except ValueError:
            return -10 ** 6          # 该维度不在候选里 → 与谁都不相邻

    stock = [tuple(_pos(stock_dims[k], c[stock_keys[k]]) for k in range(3))
             for c in combos]
    is_off = [(c.get("timing") or "off").strip().lower() in ("", "off", "none")
              for c in combos]
    timing = [None if is_off[i] else
              tuple(_pos(timing_dims[k], combos[i][timing_keys[k]]) for k in range(2))
              for i in range(len(combos))]

    nbrs: list[list[int]] = []
    for i in range(len(combos)):
        grp = []
        for j in range(len(combos)):
            if any(abs(a - b) > 1 for a, b in zip(stock[i], stock[j])):
                continue
            if is_off[i] != is_off[j]:
                continue
            if not is_off[i]:
                if combos[i]["timing"] != combos[j]["timing"]:
                    continue
                if any(abs(a - b) > 1 for a, b in zip(timing[i], timing[j])):
                    continue
            grp.append(j)
        nbrs.append(grp)
    return nbrs


def smooth_scores(scores: list[float], neighbors: list[list[int]]) -> list[float]:
    """在参数曲面上对分数做邻域均值（含自身）。纯函数，便于单测。

    这是「不踩尖峰」的实现核心：argmax(smooth_scores) 选的是**局部区域均值最高**
    的那一组，而不是单点最高。若某点分数略高但四周都平庸，平滑后它会输给一片
    整体都不错的「高原」。

    ⚠️ 能抹平的幅度有限：邻域窗口每维只有 3 档，自身占 1/2~1/3 权重，
    所以只在「尖峰幅度和邻居差得不多」时生效（而这正是参数选择里最常见的情形：
    0.42 vs 0.40 这种量级的抖动）。若某组参数的样本内分数是邻居的 50 倍，
    那属于**异常值**而不是稳健性问题，均值滤波拉不下来——那种情况该查的是
    指标计算或数据，而不是选参方式。
    """
    out = []
    for i, s in enumerate(scores):
        vals = [float(scores[j]) for j in neighbors[i]] if neighbors[i] else [float(s)]
        out.append(float(np.mean(vals)) if vals else float(s))
    return out


def parse_ensemble_ks(text) -> tuple[int | None, ...]:
    """解析 --wf-ensemble-k，如 '3,5,10,0' → (3, 5, 10, None)；0 表示「全部候选」。

    为什么要有 K 这一维：无差别平均**全部**候选会把已知最差的区域也算进来
    （实测 momentum 组的平均夏普只有 0.39、半数窗口亏钱，却被等权持有），
    集成被拖累成「收益平庸、只换来回撤最小」。
    只在训练窗表现靠前的 K 组里集成，等于把「哪个区域好」这个稳健信息用上，
    同时仍然不押单点。K=1 退化为「按平滑分数选参」，K=全部就是原来的做法。

    幂等性：本函数必须能吃下**自己的输出**（`(3, 5, 10, None)`），因为调用方
    （main.py）先解析一次再把结果传给 walk_forward_search，后者内部还会再解析一次。
    早期版本在这里 `int(None)` 直接崩 —— 单测只喂过字符串，所以没暴露。
    """
    if text is None:
        return (3, 5, 10, None)
    if isinstance(text, (list, tuple)):
        items = list(text)
    else:
        items = [x.strip() for x in str(text).split(",") if x.strip()]
    ks: list[int | None] = []
    for it in items:
        if it is None:
            k: int | None = None
        else:
            try:
                k = int(it)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"--wf-ensemble-k 含非法值 {it!r}，应为整数或 0(all)") from exc
            k = None if k <= 0 else k
        if k not in ks:
            ks.append(k)
    return tuple(ks) if ks else (3, 5, 10, None)


def ensemble_arm_keys(ks: tuple[int | None, ...]) -> list[str]:
    """把 K 列表映射成报告用的臂名（稳定顺序，便于图表配色）。"""
    return [f"ens{'all' if k is None else k}" for k in ks]


def ensemble_arm_label(k: int | None, n_combos: int = 0) -> str:
    if k is None:
        return f"参数集成（全部 {n_combos} 组等权）" if n_combos else "参数集成（全部候选等权）"
    return f"参数集成（平滑分前 {k} 组等权）"


def ensemble_weights(weight_list: list[pd.DataFrame]) -> pd.DataFrame:
    """把多个参数组合的权重矩阵等权平均 —— 「同时按所有候选参数持有」。

    为什么是**平均权重**而不是「平均各条净值曲线」：前者对应真实的执行
    （一笔净额委托，只付一次成本），后者要 N 份资金各付一遍成本，
    既不可实现也会把成本重复计算。
    """
    if not weight_list:
        raise ValueError("ensemble_weights 需要非空的权重列表")
    if len(weight_list) == 1:
        return weight_list[0]
    base = weight_list[0]
    mean = pd.concat(weight_list).groupby(level=0).mean()
    # groupby 会把 DatetimeIndex 的 freq 抹掉、也不保证列序，这里对齐回原始索引/列序，
    # 让集成权重与单个组合的权重矩阵形状完全可比（下游 run() 依赖两者的索引一致）。
    return mean.reindex(index=base.index, columns=base.columns)


def grid_search(prices_all: pd.DataFrame, open_all: pd.DataFrame,
                can_buy, can_sell, args,
                lookbacks: list[int], top_ns: list[int], buffers: list[int],
                start_ts: pd.Timestamp, benchmark_curve: pd.Series | None = None,
                weight_cap: pd.DataFrame | None = None,
                exposure: pd.Series | None = None,
                timing_modes: list[str] | None = None,
                timing_lookbacks: list[int] | None = None,
                timing_bands: list[float] | None = None,
                proxy: pd.Series | None = None) -> pd.DataFrame:
    """对每组合回测，返回结果 DataFrame（一行一组合）。

    两种模式
    --------
    - 只扫选股参数（timing_modes 为空）：所有组合沿用 exposure（`--timing ma` 那条）。
    - 同时扫择时参数（timing_modes 非空，如 ["off","ma","momentum"]）：每个组合
      按自己的择时配置构造敞口，网格里会出现「不择时」的对照组。

    ⚠️ 这是**全样本网格搜索**：在整段样本上挑最优参数，等于「事后选参」，
    结果天然偏乐观。要判断参数是否真的稳健，请用 walk_forward_search。
    """
    rows = []
    combos = build_combos(lookbacks, top_ns, buffers,
                          timing_modes, timing_lookbacks, timing_bands)
    total = len(combos)
    cost = cost_kwargs(args)
    get_exposure = _exposure_getter(proxy, prices_all, args, fallback=exposure,
                                   search_timing=bool(timing_modes))
    score_cache = {lb: factors.momentum(prices_all, lb, args.skip_recent)
                   for lb in dict.fromkeys(c["lookback"] for c in combos)}
    for i, combo in enumerate(combos, 1):
        lb, tn, buf = combo["lookback"], combo["top_n"], combo["buffer"]
        score = score_cache[lb]
        weights_all = weights_from_args(score, args, can_buy=can_buy, can_sell=can_sell,
                                        prices=prices_all, weight_cap=weight_cap,
                                        exposure=get_exposure(combo, None),
                                        top_n=tn, buffer=buf)
        keep = weights_all.index >= start_ts
        prices = prices_all.loc[keep]
        weights = weights_all.loc[keep]
        op = open_all.loc[keep] if args.use_open else None
        equity, m, detail = run(prices, weights, open_prices=op, **cost)
        annual = rolling.annual_breakdown(equity, detail)
        worst_year = float(annual["return"].min()) if len(annual) else 0.0
        pos_year = float((annual["return"] > 0).mean()) if len(annual) else 0.0

        rec = {
            "lookback": lb, "top_n": tn, "buffer": buf,
            "timing": combo["timing"], "timing_lookback": combo["timing_lookback"],
            "timing_band": combo["timing_band"],
            "total_return": m["total_return"], "annual_return": m["annual_return"],
            "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
            "calmar": m["calmar"], "annual_turnover": m["annual_turnover"],
            "worst_year_return": worst_year, "positive_year_ratio": pos_year,
        }
        if benchmark_curve is not None:
            bench = benchmark_curve.reindex(equity.index).ffill().bfill()
            if len(bench) == len(equity) and bench.iloc[0]:
                bench_eq = bench / bench.iloc[0]          # 基准原始收盘价 -> 净值（首日=1）
                excess = equity / bench_eq
                rec["excess_return"] = float(excess.iloc[-1] - 1)
                ex_daily = excess.pct_change().fillna(0.0)
                rec["information_ratio"] = (float(ex_daily.mean() / ex_daily.std() * (252 ** 0.5))
                                           if ex_daily.std() else 0.0)
        rows.append(rec)
        print(f"  [{i}/{total}] {combo_label(combo)} | 收益 {m['total_return']:.1%}"
              f" 夏普 {m['sharpe']:.2f} 回撤 {m['max_drawdown']:.1%} 最差年 {worst_year:.1%}", flush=True)
    return pd.DataFrame(rows)


# ============================================================ 样本外验证
#
# 为什么必须做 walk-forward
# ------------------------
# 全样本网格搜索有两个致命问题：
#   1. 事后选参：你在整段历史上挑出最猛的那组参数，等于让参数「看见」了未来；
#   2. 多重比较：试的组合越多，「碰巧最优」的那个就越漂亮——这是纯粹的噪声。
# 表现为：训练期夏普 1.2，实盘立刻掉到 0.3。
#
# 做法：把样本切成若干「训练窗 + 紧邻测试窗」，在训练窗内挑参、在测试窗里
# **完全不改参**地跑一遍，把各测试窗的净值接起来。这条拼接曲线才是这套方法
# 真实能拿到的成绩。同时给出两个对照组：
#   - 全样本最优固定参数（事后选参）：它在测试窗上的表现 ≈ 上界幻觉
#   - 默认参数（不调参）           ：不调参反而常常赢过「精心调参」
# 如果 walk-forward 明显跑输全样本最优，说明那组「最优参数」是拟合出来的。


def _equity_metrics(equity: pd.Series, periods_per_year: int = 252) -> dict:
    """从净值曲线算核心指标（与 backtest.run 口径一致，但不需要成本明细）。"""
    if equity is None or len(equity) < 2:
        return {"total_return": 0.0, "annual_return": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "annual_volatility": 0.0}
    rets = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    ann = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
    vol = float(rets.std() * np.sqrt(periods_per_year)) if len(rets) else 0.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(periods_per_year)) if len(rets) and rets.std() else 0.0
    dd = float((equity / equity.cummax() - 1).min())
    return {"total_return": total, "annual_return": ann, "sharpe": sharpe,
            "max_drawdown": dd, "annual_volatility": vol,
            "trading_days": int(len(equity))}


def walk_forward_folds(n: int, start_pos: int, train_n: int, test_n: int) -> list[tuple[int, int, int]]:
    """生成 (训练起点, 训练终点=测试起点, 测试终点) 三元组，测试窗之间不重叠。

    滚动方式：每次向后移动一个测试窗宽度，保证每个测试窗只用它之前的数据选参。
    """
    folds = []
    s = start_pos
    while s + train_n + test_n <= n:
        folds.append((s, s + train_n, s + train_n + test_n))
        s += test_n
    return folds


def walk_forward_search(prices_all: pd.DataFrame, open_all: pd.DataFrame,
                        can_buy, can_sell, args,
                        lookbacks: list[int], top_ns: list[int], buffers: list[int],
                        start_ts: pd.Timestamp, train_years: float = 2.0,
                        test_years: float = 0.5, weight_cap: pd.DataFrame | None = None,
                        select_metric: str = "sharpe",
                        exposure: pd.Series | None = None,
                        timing_modes: list[str] | None = None,
                        timing_lookbacks: list[int] | None = None,
                        timing_bands: list[float] | None = None,
                        proxy: pd.Series | None = None,
                        ensemble_ks: tuple[int | None, ...] | None = None,
                        periods_per_year: int = 252) -> dict:
    """滚动训练/测试的样本外验证。

    返回 dict：
      folds      每折明细 DataFrame（含三种选参方式的选中参数与样本外表现）
      curves     多条测试期拼接净值：walk_forward(训练窗 argmax) / smooth(邻域平滑)
                 / ens<k>(按平滑分取前 k 组等权集成，ensall = 全部候选)
                 / full_sample_best(事后选参) / default(不调参)
      summary    五条曲线的汇总指标 DataFrame（带 key 列，报告按 key 取用）
      timing_note    各折选中的择时模式统计（搜索空间含 off 时才有意义）
      ensemble_note  平滑/集成相对 argmax 的改善——「别踩尖峰」的直接证据

    为什么要加 smooth / ensemble 两条
    ---------------------------------
    argmax 永远挑到那个尖峰，而尖峰多半是噪声（实测滞回带 0 与 2% 各折各半）。
    邻域平滑把参数曲面抹一遍再取最高（选「最好的一片」），集成干脆不选。
    这两条不需要任何额外假设，却能把「选择本身带来了多少虚假优势」量化出来。

    特别提醒：若 timing_modes 非空，搜索空间里会强制包含「不择时」。
    这样「择时参数是不是拟合出来的」才有答案——如果各折频繁选中 ma，
    说明该池子的下跌段确实能被均线识别；如果频繁选中 off，
    说明之前那组「三项全改善」的择时参数只是在全样本上凑出来的。
    """
    idx_all = prices_all.index
    n_all = len(idx_all)
    train_n = int(round(train_years * periods_per_year))
    test_n = int(round(test_years * periods_per_year))
    if train_n < 120 or test_n < 20:
        raise ValueError(f"训练窗 {train_n} 日 / 测试窗 {test_n} 日过短，无法做 walk-forward")

    start_pos = int(np.searchsorted(idx_all.values, np.datetime64(pd.Timestamp(start_ts))))
    folds = walk_forward_folds(n_all, start_pos, train_n, test_n)
    if len(folds) < 2:
        raise ValueError(
            f"样本不足以做 walk-forward：区间内仅 {len(folds)} 折。"
            f"请拉长 --start/--end，或缩小 --fw-train / --fw-test")

    search_timing = bool(timing_modes)
    combos = build_combos(lookbacks, top_ns, buffers,
                          timing_modes, timing_lookbacks, timing_bands)
    # 打分缓存按需惰性填充：不能再假定「用到的 lookback 都在网格候选里」——
    # 默认对照组用的是 args.lookback，它完全可以不在 --grid-lookbacks 里
    # （如 --grid-lookbacks 60,90 而 --lookback 默认 120），预先按网格建缓存会 KeyError。
    score_cache: dict[int, pd.DataFrame] = {}

    def score_of(lb: int) -> pd.DataFrame:
        if lb not in score_cache:
            score_cache[lb] = factors.momentum(
                prices_all, lb, getattr(args, "skip_recent", 0))
        return score_cache[lb]

    get_exposure = _exposure_getter(proxy, prices_all, args, fallback=exposure,
                                   search_timing=search_timing)
    cost = cost_kwargs(args)

    def build_weights(combo, end_pos):
        """combo 在「只用 < end_pos 的数据」下算出的目标权重（整段前程，便于切片）。"""
        sc = score_of(combo["lookback"]).iloc[:end_pos]
        return weights_from_args(
            sc, args,
            can_buy=(can_buy.iloc[:end_pos] if can_buy is not None else None),
            can_sell=(can_sell.iloc[:end_pos] if can_sell is not None else None),
            prices=prices_all.iloc[:end_pos],
            weight_cap=(weight_cap.iloc[:end_pos] if weight_cap is not None else None),
            exposure=get_exposure(combo, end_pos),
            top_n=combo["top_n"], buffer=combo["buffer"])

    def run_weights(w, lo, hi):
        p = prices_all.iloc[lo:hi]
        o = open_all.iloc[lo:hi] if getattr(args, "use_open", False) else None
        eq, m, _ = run(p, w.iloc[lo:hi], open_prices=o, **cost)
        return eq, m

    def run_combo(combo, end_pos, lo, hi):
        """在 [lo, hi) 上跑 combo，权重只用 < end_pos 的数据计算（无前视）。

        end_pos 是「权重允许看到的数据边界」，恒等于本折测试窗的终点；
        训练窗与测试窗都用它，因此两窗的选股/择时口径完全一致，
        差别只在回测区间 [lo, hi)。
        """
        return run_weights(build_weights(combo, end_pos), lo, hi)

    default_combo = {
        "lookback": int(getattr(args, "lookback", 120)),
        "top_n": int(getattr(args, "top_n", 10)),
        "buffer": int(getattr(args, "buffer", 0)),
        "timing": (getattr(args, "timing", "off") or "off"),
        "timing_lookback": int(getattr(args, "timing_lookback", 120) or 0),
        "timing_band": float(getattr(args, "timing_band", 0.0) or 0.0),
    }

    # 参数空间邻域：用于「邻域平滑选参」（不踩尖峰）
    neighbors = _neighbor_map(combos, lookbacks, top_ns, buffers,
                             timing_lookbacks, timing_bands)

    # ---- 对照组 A：全样本事后选参 ----
    print(f"  [对照组] 全样本网格 {len(combos)} 组，用于对比「事后选参」的幻觉")
    full_rows = []
    for combo in combos:
        _, m = run_combo(combo, n_all, start_pos, n_all)
        full_rows.append({"combo": combo, select_metric: m[select_metric],
                          "total_return": m["total_return"], "sharpe": m["sharpe"],
                          "max_drawdown": m["max_drawdown"]})
    full_df = pd.DataFrame(full_rows)
    full_best_combo = full_df.sort_values(select_metric, ascending=False)["combo"].iloc[0]

    # ---- 逐折：训练窗选参 → 测试窗纯验证 ----
    print(f"\n【Walk-Forward】{len(folds)} 折，训练 {train_n} 日 / 测试 {test_n} 日"
          f"（选参依据：训练期 {select_metric}）")
    fold_rows = []
    ks = parse_ensemble_ks(ensemble_ks) if ensemble_ks is not None else (3, 5, 10, None)
    ens_keys = ensemble_arm_keys(ks)
    arms = ("walk_forward", "smooth", *ens_keys, "full_sample_best", "default")
    curves: dict[str, list] = {k: [] for k in arms}
    # 最窄那条集成臂的组数（与折无关，先算好，供逐折记录与汇总说明使用）
    min_k = min([len(combos) if k is None else min(int(k), len(combos)) for k in ks])
    for fi, (s, se, te) in enumerate(folds, 1):
        # 训练窗：把所有候选都跑一遍（后面 argmax / 邻域平滑 / 集成都要用）
        train_ms = [run_combo(combo, se, s, se)[1] for combo in combos]
        scores = [float(m[select_metric]) for m in train_ms]
        smoothed = smooth_scores(scores, neighbors)

        best_i = int(np.argmax(scores))
        wf_combo, wf_train_m = combos[best_i], train_ms[best_i]
        sm_i = int(np.argmax(smoothed))
        sm_combo, sm_train_m = combos[sm_i], train_ms[sm_i]

        _, m_wf = run_combo(wf_combo, te, se, te)
        _, m_sm = run_combo(sm_combo, te, se, te)

        # 集成臂：先按**平滑分**降序（不是原始分——原始分会被尖峰带偏），
        # 再取前 k 组把权重等权平均 → 一笔净额委托（成本只算一次）。
        # 各组合的权重算一次就够，所有 K 共用（K=全部时正好是全集）。
        order = list(np.argsort(-np.asarray(smoothed, dtype=float)))
        w_cache = {i: build_weights(combos[i], te) for i in range(len(combos))}
        # 最窄那条集成臂到底持有了哪几组？这是「集成是不是只押了一个择时模式」的
        # 直接证据——若前 3 组全是 ma，那 ens3 实质接近「只做 ma」，结构结论就成立。
        top_picks = " / ".join(combo_label(combos[i]) for i in order[:min_k])
        top_modes = [combos[i].get("timing", "off") for i in order[:min_k]]
        ens_ms: dict[str, dict] = {}
        for k, key in zip(ks, ens_keys):
            picked = order if k is None else order[:min(int(k), len(order))]
            _, m_k = run_weights(ensemble_weights([w_cache[i] for i in picked]), se, te)
            ens_ms[key] = m_k
        m_fb = run_combo(full_best_combo, te, se, te)[1]
        m_df = run_combo(default_combo, te, se, te)[1]

        # 记录测试期的净值（后续按年化收益拼接成连续曲线）
        for key, m in (("walk_forward", m_wf), ("smooth", m_sm), *ens_ms.items(),
                       ("full_sample_best", m_fb), ("default", m_df)):
            curves[key].append((se, te, m["sharpe"], m["total_return"], m["max_drawdown"]))

        fold_rows.append({
            "fold": fi,
            "train_start": idx_all[s].date(), "train_end": idx_all[se - 1].date(),
            "test_start": idx_all[se].date(), "test_end": idx_all[te - 1].date(),
            "lookback": wf_combo["lookback"], "top_n": wf_combo["top_n"],
            "buffer": wf_combo["buffer"],
            "timing": wf_combo.get("timing", "off"),
            "timing_lookback": wf_combo.get("timing_lookback", 0),
            "timing_band": wf_combo.get("timing_band", 0.0),
            "chosen": combo_label(wf_combo),
            "smooth_chosen": combo_label(sm_combo),
            "ens_top": top_picks,
            "ens_top_modes": "+".join(top_modes),
            "train_sharpe": round(float(wf_train_m["sharpe"]), 3),
            "train_return": round(float(wf_train_m["total_return"]), 4),
            "test_sharpe": round(float(m_wf["sharpe"]), 3),
            "test_return": round(float(m_wf["total_return"]), 4),
            "test_max_drawdown": round(float(m_wf["max_drawdown"]), 4),
            "decay": round(float(m_wf["sharpe"] - wf_train_m["sharpe"]), 3),
            "test_return_smooth": round(float(m_sm["total_return"]), 4),
            **{f"test_return_{key}": round(float(ens_ms[key]["total_return"]), 4)
               for key in ens_keys},
        })
        ens_desc = " / ".join(f"{key} {ens_ms[key]['total_return']:.1%}" for key in ens_keys)
        print(f"  第{fi}折 {idx_all[s].date()}~{idx_all[se-1].date()} 选参"
              f" {combo_label(wf_combo)}"
              f" | 样本内夏普 {wf_train_m['sharpe']:.2f} → 样本外 {m_wf['sharpe']:.2f}"
              f"（衰减 {fold_rows[-1]['decay']:+.2f}）"
              f" | 样本外收益 argmax {m_wf['total_return']:.1%}"
              f" / 邻域 {m_sm['total_return']:.1%}"
              f" / {ens_desc}", flush=True)

    folds_df = pd.DataFrame(fold_rows)

    # ---- 择时模式在样本外被选中的频次（回答「该不该用择时」）----
    timing_note = ""
    if search_timing and len(folds_df):
        counts = folds_df["timing"].value_counts()
        timing_note = "各折选中的择时模式：" + "、".join(
            f"{k} × {v}" for k, v in counts.items())

    # ---- 最窄集成臂实际持有了哪些模式（回答「集成是不是只押了一个模式」）----
    ens_note = ""
    if len(folds_df) and "ens_top_modes" in folds_df.columns and search_timing:
        picks = folds_df["ens_top_modes"].value_counts()
        ens_note = ("最窄集成臂（前 %d 组）各折持有：" % min_k) + "、".join(
            f"{k} × {v}" for k, v in picks.items())

    # ---- 拼接测试期净值（按各段收益连乘）----
    def chain(records):
        eq = [1.0]
        for _, _, _, ret, _ in records:
            eq.append(eq[-1] * (1 + ret))
        return eq

    summary_rows = []
    label_map = [("walk_forward", "Walk-Forward 自适应参数（训练窗 argmax）"),
                 ("smooth", "Walk-Forward 邻域平滑选参（不踩尖峰）")]
    label_map += [(key, ensemble_arm_label(k, len(combos))) for k, key in zip(ks, ens_keys)]
    label_map += [("full_sample_best", "全样本最优固定参数（事后选参）"),
                  ("default", "默认参数（不调参）")]
    for key, label in label_map:
        recs = curves[key]
        chain_vals = chain(recs)
        total = float(chain_vals[-1] - 1)
        n_years = sum((idx_all[b - 1] - idx_all[a]).days for a, b, *_ in recs) / 365.25
        n_years = max(n_years, 1 / 365.25)
        ann = float((1 + total) ** (1 / n_years) - 1) if total > -1 else -1.0
        # 样本外夏普：用各折夏普的加权平均（近似，避免拼接空隙造成的失真）
        days = [len(idx_all[a:b]) for a, b, *_ in recs]
        w_sharpe = float(np.average([r[2] for r in recs], weights=days)) if days else 0.0
        worst_dd = float(min(r[4] for r in recs)) if recs else 0.0
        summary_rows.append({
            "key": key,
            "strategy": label,
            "total_return": total,
            "annual_return": ann,
            "avg_sharpe": w_sharpe,
            "worst_fold_drawdown": worst_dd,
            "positive_folds": float(np.mean([r[3] > 0 for r in recs])) if recs else 0.0,
            "folds": len(recs),
        })

    summary_df = pd.DataFrame(summary_rows)

    # ---- 集成/平滑相对 argmax 的改善（若有，就是「别踩尖峰」的直接证据）----
    ensemble_note = ""
    ensemble_k_curve: list[dict] = []
    if summary_df is not None and len(summary_df):
        by_key = summary_df.set_index("key")
        try:
            d_arg = float(by_key.loc["walk_forward", "total_return"])
            parts = []
            if "smooth" in by_key.index:
                dd = float(by_key.loc["smooth", "total_return"]) - d_arg
                parts.append(f"邻域平滑 {dd:+.1%}")
            for k, key in zip(ks, ens_keys):
                if key in by_key.index:
                    kk = len(combos) if k is None else min(int(k), len(combos))
                    ret = float(by_key.loc[key, "total_return"])
                    ensemble_k_curve.append({
                        "k": kk, "key": key,
                        "total_return": ret, "avg_sharpe": float(by_key.loc[key, "avg_sharpe"]),
                        "worst_fold_drawdown": float(by_key.loc[key, "worst_fold_drawdown"]),
                        "vs_argmax": ret - d_arg,
                    })
            if ensemble_k_curve:
                lo = min(ensemble_k_curve, key=lambda r: r["total_return"])
                hi = max(ensemble_k_curve, key=lambda r: r["total_return"])
                if len(ensemble_k_curve) > 1:
                    # 报区间而不是只报最好看的那个 K —— 单点数字会诱导「挑最优 K」。
                    parts.append(f"集成 K={hi['k']}~{lo['k']}：{hi['vs_argmax']:+.1%} ~ "
                                 f"{lo['vs_argmax']:+.1%}（极差 "
                                 f"{hi['total_return'] - lo['total_return']:.1%}）")
                else:
                    parts.append(f"集成 K={hi['k']} {hi['vs_argmax']:+.1%}")
            if parts:
                ensemble_note = ("样本外总收益相对「训练窗 argmax」的变化：" + "，".join(parts)
                                 + "（正则说明单点择优确实在踩噪声）")
        except KeyError:
            ensemble_note = ""

    # ---- K 敏感度：同一个 K 只在同一批测试窗上比较，本身也是「选参」 ----
    k_note = ""
    if len(ensemble_k_curve) >= 3:
        vals = [r["total_return"] for r in ensemble_k_curve]
        spread = max(vals) - min(vals)
        k_note = (f"集成 K 的收益极差 {spread:.1%}（最差 K={min(ensemble_k_curve, key=lambda r: r['total_return'])['k']}，"
                  f"最优 K={max(ensemble_k_curve, key=lambda r: r['total_return'])['k']}）"
                  + ("——极差不大，说明集成度这个选择不敏感，结论稳。"
                     if spread <= 0.05 else
                     "——极差较大，别事后挑最好的 K（那又是一次过拟合），"
                     "应按原理选：K 越大方差越小、期望越被摊薄。"))

    return {
        "folds": folds_df,
        "summary": summary_df,
        "curves": curves,
        "full_best_combo": full_best_combo,
        "default_combo": default_combo,
        "full_best_label": combo_label(full_best_combo),
        "default_label": combo_label(default_combo),
        "timing_note": timing_note,
        "ens_note": ens_note,
        "ensemble_note": ensemble_note,
        "ensemble_k_curve": ensemble_k_curve,
        "k_note": k_note,
        "search_timing": search_timing,
        "n_combos": len(combos),
        "train_n": train_n,
        "test_n": test_n,
        "train_years": train_years,
        "test_years": test_years,
        "select_metric": select_metric,
    }


_WF_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  :root { --bg:#fff; --surface:#fafafa; --border:#e6e6e6; --text:#1f2329; --muted:#6b7280; --accent:#2f6df0; }
  * { box-sizing:border-box; }
  body { font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; margin:0;
         background:var(--bg); color:var(--text); line-height:1.6; }
  .wrap { max-width:1080px; margin:0 auto; padding:32px 24px 64px; }
  h1 { font-size:22px; font-weight:600; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:24px; }
  .group { margin-bottom:28px; }
  .group h2 { font-size:14px; font-weight:600; color:var(--muted); margin:0 0 10px;
              border-left:3px solid var(--accent); padding-left:8px; }
  .chart { width:100%; height:400px; background:var(--surface); border:1px solid var(--border);
           border-radius:12px; margin-bottom:14px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { border:1px solid var(--border); padding:7px 9px; text-align:right; }
  th { background:var(--surface); color:var(--muted); font-weight:600; }
  td:first-child,th:first-child { text-align:left; }
  tr:nth-child(even) td { background:#fcfcfc; }
  .pos { color:#cf1322; } .neg { color:#389e0d; }
  .note { color:var(--muted); font-size:12px; margin-top:8px; }
  .card { display:inline-block; border:1px solid var(--border); border-radius:10px;
          padding:10px 16px; margin:0 10px 10px 0; background:var(--surface); }
  .card b { display:block; font-size:18px; }
  .card span { color:var(--muted); font-size:12px; }
</style></head><body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>

  <div class="group">
    <h2>核心结论</h2>
    <div id="cards"></div>
    <div class="note" id="verdict"></div>
  </div>

  <div class="group">
    <h2>样本外测试期拼接净值（各折连乘）</h2>
    <div class="chart" id="curves"></div>
    <div class="note">只看<b>测试期</b>——训练期不参与画图。
      若「全样本最优固定参数」明显高于自适应曲线，
      说明那组最优参数吃的是样本内的运气，实盘拿不到。<br/>
      自适应这一类里有三种做法：<b>argmax</b>（训练窗取最高分）、<b>邻域平滑</b>（取最高的那一片而非最高点）、
      <b>参数集成</b>（压根不选，把候选的权重等权平均后一起持有；按平滑分只在<b>前 K 组</b>内平均）。
      K 越小越靠近「选参」、K 越大越靠近「全买」——所以 K 必须以多条臂并列看，
      不能事后再挑一个最好看的 K。</div>
  </div>

  <div class="group" id="kgroup" style="display:none">
    <h2>集成度 K 的敏感度（在同一批测试窗上比较）</h2>
    <div class="chart" id="kcurve"></div>
    <div class="note" id="knote"></div>
  </div>

  <div class="group">
    <h2>过拟合诊断：每折「训练期夏普 vs 测试期夏普」</h2>
    <div class="chart" id="decay"></div>
    <div class="note">训练期夏普普遍高、测试期大幅回落（落差越大过拟合越严重）。
      落差长期为正说明选参过程本身在拟合噪声。</div>
  </div>

  <div class="group">
    <h2>汇总对比</h2>
    <table id="summary"></table>
    <div class="note">「正收益折占比」= 测试窗里赚钱的比例，衡量方法在时间上的稳定性。
      「vs argmax」列 = 该条曲线相对「训练窗取最高分」的样本外总收益差。</div>
  </div>

  <div class="group">
    <h2>逐折明细</h2>
    <table id="folds"></table>
    <div class="note">decay = 测试期夏普 − 训练期夏普。最后几列对比同一折里
      不同选参 / 集成方式的样本外收益——它们看的是<b>同一段测试窗</b>，差别只来自怎么选参数。<br/>
      <span id="ens-note" class="muted"></span>
      <span id="timing-note"></span></div>
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
const axisCommon = { axisLine:{lineStyle:{color:'#e6e6e6'}}, axisLabel:{color:'#6b7280'} };

// 卡片
const cards = document.getElementById('cards');
const pick = k => DATA.summary.find(s => s.key === k) || {total_return:0,annual_return:0,avg_sharpe:0,positive_folds:0,folds:0};
const wf = pick('walk_forward'), sm = pick('smooth');
const ensRows = DATA.summary.filter(s => s.key.indexOf('ens') === 0);
const en = ensRows.length ? ensRows.reduce((a,b) => b.total_return > a.total_return ? b : a) : sm;
const fb = pick('full_sample_best'), df = pick('default');
const card = (label, v, hint) => `<div class="card"><span>${label}</span><b>${v}</b>` +
  (hint ? `<span>${hint}</span>` : '') + '</div>';
cards.innerHTML =
  card('Walk-Forward argmax 选参', (wf.total_return*100).toFixed(1)+'%',
       '年化 '+(wf.annual_return*100).toFixed(1)+'% · 平均夏普 '+wf.avg_sharpe.toFixed(2)) +
  card('邻域平滑选参', (sm.total_return*100).toFixed(1)+'%',
       '不选最高点、选最高的那一片 · 夏普 '+sm.avg_sharpe.toFixed(2)) +
  card('参数集成（并列多条 K，此为其一）', (en.total_return*100).toFixed(1)+'%',
       en.label + ' · 夏普 '+en.avg_sharpe.toFixed(2)) +
  card('全样本最优（事后选参）', (fb.total_return*100).toFixed(1)+'%',
       DATA.full_best_label || '同期对照，高于前三条的部分是幻觉') +
  card('默认参数（不调参）', (df.total_return*100).toFixed(1)+'%',
       DATA.default_label || '不调参的基准线');

const gap = fb.total_return - wf.total_return;
const better = Math.max(sm.total_return, en.total_return) - wf.total_return;
const kmin = DATA.k_curve && DATA.k_curve.length
  ? DATA.k_curve.reduce((a,b) => b.total_return < a.total_return ? b : a) : null;
let verdict = gap > 0.05
  ? `<b>⚠️ 事后选参高估了 ${(gap*100).toFixed(1)} 个百分点</b>——全样本最优参数在样本外明显跑输自适应选参，说明调参过程在拟合噪声。`
  : `全样本最优与自适应选参差距 ${(gap*100).toFixed(1)} 个百分点，参数对样本外的影响有限。`;
if (better > 0.005) {
  verdict += `<br/><b>不踩尖峰更赚</b>：邻域平滑 / 参数集成里最好的那条比 argmax 选参高出 ${(better*100).toFixed(1)} 个百分点——`
           + `这直接说明「训练窗取最高分」挑到的大概率是噪声，而不是真实优势。`
           + `（注意：集成族最好的那条往往是某个特定的 K，别把「最好的 K」当成结论，
              那又是一次事后选择；K 的整体走势才是有信息的。）`;
}
document.getElementById('verdict').innerHTML = verdict + (DATA.ensemble_note ? '<br/>' + DATA.ensemble_note : '');

// 净值曲线（臂数随 --wf-ensemble-k 变化，这里按 summary 顺序动态渲染）
const cv = echarts.init(document.getElementById('curves'));
const ORDER = DATA.summary.map(s => s.key).filter(k => DATA.eq[k]);
// 调色板：两条主线（argmax / 邻域平滑）最醒目，集成族用蓝色系渐深，复盘对照组用灰
const FPAL = ['#cf1322','#fa8c16','#2f6df0','#5b8ff9','#69c0ff','#91caff','#8c8c8c','#bfbfbf'];
const colors = {}, widths = {};
ORDER.forEach((k, i) => {
  colors[k] = (k === 'walk_forward') ? '#cf1322'
            : (k === 'smooth') ? '#fa8c16'
            : (k === 'full_sample_best') ? '#8c8c8c'
            : (k === 'default') ? '#bfbfbf'
            : (k.indexOf('ens') === 0 ? FPAL[2 + (i % 4)] : FPAL[i % FPAL.length]);
  widths[k] = (k === 'walk_forward' || k === 'smooth') ? 2.4 : (k.indexOf('ens') === 0 ? 2.0 : 1.5);
});
const nameOf = k => { const r = DATA.summary.find(s => s.key === k); return r ? r.label : k; };
cv.setOption({
  tooltip:{ trigger:'axis' },
  legend:{ top:0, data:ORDER.map(nameOf) },
  grid:{ left:60, right:30, top:40, bottom:50 },
  xAxis:{ type:'category', data:DATA.eq_x, axisLabel:{ fontSize:10, color:'#6b7280' }, name:'交易日（仅测试窗）' },
  yAxis:{ type:'value', name:'净值', ...axisCommon },
  series: ORDER.map(k => ({
    name:nameOf(k), type:'line', showSymbol:false, smooth:true,
    lineStyle:{ width:widths[k], color:colors[k] }, itemStyle:{ color:colors[k] },
    data: DATA.eq[k]
  }))
});

// 集成度 K 的敏感度
let kc = null;
if (DATA.k_curve && DATA.k_curve.length > 1) {
  document.getElementById('kgroup').style.display = '';
  kc = echarts.init(document.getElementById('kcurve'));
  const kc_data = DATA.k_curve;
  kc.setOption({
    tooltip:{ trigger:'axis', axisPointer:{type:'shadow'},
      formatter:function(ps){ const d = kc_data[ps[0].dataIndex];
        return 'K='+d.k+'<br/>总收益 '+(d.total_return*100).toFixed(2)+'%<br/>夏普 '+d.avg_sharpe.toFixed(2)
             + '<br/>最差折回撤 '+(d.worst_fold_drawdown*100).toFixed(2)+'%'; } },
    legend:{ top:0, data:['样本外总收益','平均夏普','最差折回撤'] },
    grid:{ left:70, right:70, top:40, bottom:40 },
    xAxis:{ type:'category', data:kc_data.map(d=>'K='+d.k+' / '+DATA.n_combos),
            axisLine:axisCommon.axisLine, axisLabel:axisCommon.axisLabel,
            name:'集成组数 / 候选总数', nameLocation:'middle', nameGap:30 },
    yAxis:[
      { type:'value', name:'收益 / 回撤', axisLabel:{formatter:v=>(v*100).toFixed(0)+'%', color:'#6b7280'}, ...axisCommon },
      { type:'value', name:'夏普', position:'right', axisLabel:{color:'#6b7280'}, ...axisCommon }
    ],
    series:[
      { name:'样本外总收益', type:'bar', barMaxWidth:30, itemStyle:{ color:'#91caff' },
        data: kc_data.map(d=>d.total_return) },
      { name:'最差折回撤', type:'bar', barMaxWidth:30, itemStyle:{ color:'#ffccc7' },
        data: kc_data.map(d=>d.worst_fold_drawdown) },
      { name:'平均夏普', type:'line', yAxisIndex:1, symbolSize:9,
        lineStyle:{width:2, color:'#cf1322'}, itemStyle:{ color:'#cf1322' },
        data: kc_data.map(d=>d.avg_sharpe) }
    ]
  });
}

// 过拟合诊断
const dc = echarts.init(document.getElementById('decay'));
dc.setOption({
  tooltip:{ trigger:'axis', axisPointer:{type:'shadow'} },
  legend:{ top:0, data:['训练期夏普','测试期夏普'] },
  grid:{ left:60, right:30, top:40, bottom:50 },
  xAxis:{ type:'category', data:DATA.folds.map(f=>'第'+f.fold+'折'), ...axisCommon },
  yAxis:{ type:'value', name:'夏普', ...axisCommon },
  series:[
    { name:'训练期夏普', type:'bar', barMaxWidth:28, itemStyle:{ color:'#91caff' },
      data: DATA.folds.map(f=>f.train_sharpe) },
    { name:'测试期夏普', type:'bar', barMaxWidth:28, itemStyle:{ color:'#cf1322' },
      data: DATA.folds.map(f=>f.test_sharpe) }
  ]
});

// 汇总表
const pct = v => `<td class="${v>=0?'pos':'neg'}">${(v*100).toFixed(2)}%</td>`;
const cnote = v => v > 0.005 ? '（优于 argmax）' : (v < -0.005 ? '（差于 argmax）' : '');
const argmaxRet = (DATA.summary.find(s=>s.key==='walk_forward')||{total_return:0}).total_return;
let sh = '<tr><th>参数策略</th><th>样本外总收益</th><th>vs argmax</th><th>年化</th><th>平均夏普</th><th>最差折回撤</th><th>正收益折占比</th><th>折数</th></tr>';
DATA.summary.forEach(s => {
  const d = s.total_return - argmaxRet;
  const isCand = (s.key === 'smooth' || s.key.indexOf('ens') === 0);
  const extra = isCand
    ? `<td class="${d>=0?'pos':'neg'}">${d>=0?'+':''}${(d*100).toFixed(2)}%${cnote(d)}</td>` : '<td>—</td>';
  sh += `<tr><td>${s.label}</td>${pct(s.total_return)}${extra}${pct(s.annual_return)}
    <td>${s.avg_sharpe.toFixed(3)}</td>${pct(s.worst_fold_drawdown)}
    <td>${(s.positive_folds*100).toFixed(0)}%</td><td>${s.folds}</td></tr>`;
});
document.getElementById('summary').innerHTML = sh;

// 逐折表（集成列随 K 变化）
const ensKeys = DATA.summary.filter(s => s.key.indexOf('ens') === 0).map(s => s.key);
const lblOf = k => { const r = DATA.summary.find(s => s.key === k); return r ? r.label.replace(/参数集成（/,'').replace(/）$/,'') : k; };
let fh = '<tr><th>折</th><th>训练期</th><th>测试期</th><th>argmax 选中参数</th>' +
         '<th>训练夏普</th><th>测试夏普</th><th>衰减</th><th>测试收益</th><th>测试回撤</th>' +
         '<th>邻域平滑选中</th><th>平滑收益</th>' +
         ensKeys.map(k => '<th>集成 '+lblOf(k)+'</th>').join('') + '</tr>';
DATA.folds.forEach(f => {
  fh += `<tr><td>${f.fold}</td><td>${f.train_start}~${f.train_end}</td>
    <td>${f.test_start}~${f.test_end}</td>
    <td>${f.chosen}</td>
    <td>${f.train_sharpe.toFixed(2)}</td><td>${f.test_sharpe.toFixed(2)}</td>
    <td class="${f.decay>=0?'pos':'neg'}">${f.decay>=0?'+':''}${f.decay.toFixed(2)}</td>
    ${pct(f.test_return)}${pct(f.test_max_drawdown)}
    <td>${f.smooth_chosen || '—'}</td>${pct(f.test_return_smooth)}` +
    ensKeys.map(k => pct(f['test_return_' + k])).join('') + '</tr>';
});
document.getElementById('folds').innerHTML = fh;
document.getElementById('timing-note').textContent = DATA.timing_note || '';
document.getElementById('knote').textContent = DATA.k_note || '';
const _en = document.getElementById('ens-note');
if (_en) _en.textContent = DATA.ens_note || '';

window.addEventListener('resize', () => { cv.resize(); dc.resize(); if (kc) kc.resize(); });
</script></body></html>
"""


def build_wf_report(res: dict, out_html: str, title: str = "样本外验证（Walk-Forward）",
                    subtitle: str = "") -> None:
    """生成 walk-forward 报告：拼接净值 + 过拟合诊断 + 汇总对比 + 逐折明细。"""
    folds = res["folds"]
    curves = res["curves"]

    # 逐折明细：日期字段转成字符串，否则 json 序列化会报 TypeError
    fold_records = []
    for _, r in folds.iterrows():
        rec = {}
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                rec[k] = v.isoformat()
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v)
            else:
                rec[k] = v
        fold_records.append(rec)

    # 拼接净值的横轴：按折数展开（每折天数不同，这里用折内序号表示）
    eq = {}
    max_len = 0
    for key in curves:
        vals, cur = [], 1.0
        for _, _, _, ret, _ in curves[key]:
            cur *= (1 + ret)
            vals.append(round(cur, 4))
        eq[key] = vals
        max_len = max(max_len, len(vals))
    for key in eq:
        while len(eq[key]) < max_len:
            eq[key].append(eq[key][-1])

    summary_payload = []
    for row in res["summary"].to_dict("records"):
        summary_payload.append({
            "key": row.get("key", ""),
            "label": str(row["strategy"]),
            "total_return": float(row["total_return"]),
            "annual_return": float(row["annual_return"]),
            "avg_sharpe": float(row["avg_sharpe"]),
            "worst_fold_drawdown": float(row["worst_fold_drawdown"]),
            "positive_folds": float(row["positive_folds"]),
            "folds": int(row["folds"]),
        })

    payload = {
        "folds": fold_records,
        "summary": summary_payload,
        "eq_x": [f"第{i+1}折" for i in range(max_len)],
        "eq": eq,
        "timing_note": res.get("timing_note", ""),
        "ens_note": res.get("ens_note", ""),
        "ensemble_note": res.get("ensemble_note", ""),
        "k_curve": res.get("ensemble_k_curve", []),
        "k_note": res.get("k_note", ""),
        "n_combos": int(res.get("n_combos", 0)),
        "full_best_label": res.get("full_best_label", ""),
        "default_label": res.get("default_label", ""),
    }

    html = (_WF_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle)
            .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False)))
    Path(out_html).write_text(html, encoding="utf-8")


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  :root { --bg:#fff; --surface:#fafafa; --border:#e6e6e6; --text:#1f2329; --muted:#6b7280; --accent:#2f6df0; }
  * { box-sizing: border-box; }
  body { font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; margin:0; background:var(--bg); color:var(--text); line-height:1.6; }
  .wrap { max-width:1080px; margin:0 auto; padding:32px 24px 64px; }
  h1 { font-size:22px; font-weight:600; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:24px; }
  .group { margin-bottom:28px; }
  .group h2 { font-size:14px; font-weight:600; color:var(--muted); margin:0 0 10px; border-left:3px solid var(--accent); padding-left:8px; }
  .chart { width:100%; height:420px; background:var(--surface); border:1px solid var(--border); border-radius:12px; margin-bottom:24px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { border:1px solid var(--border); padding:7px 9px; text-align:right; }
  th { background:var(--surface); color:var(--muted); font-weight:600; position:sticky; top:0; }
  td:first-child,th:first-child { text-align:left; }
  tr:nth-child(even) td { background:#fcfcfc; }
  .pos { color:#cf1322; } .neg { color:#389e0d; }
  .note { color:var(--muted); font-size:12px; margin-top:8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="group">
    <h2>夏普热力图（按 lookback × top_n，其余维度取均值）</h2>
    <div class="chart" id="heat"></div>
    <div class="note">颜色越红夏普越高；横轴 top_n、纵轴 lookback。看清「哪片区域普遍好」，比单点最优更抗过拟合。
      同时扫了择时参数时，这里的每个格子是「该 lookback×top_n 下所有择时配置的平均」。</div>
  </div>
  <div class="group" id="timing-group" style="display:none">
    <h2>择时维度对比（各配置的组内平均指标）</h2>
    <div class="chart" id="timing"></div>
    <div class="note">同一择时模式下的平均总收益 / 夏普 / 最大回撤。若「不择时（off）」的平均夏普最高，
      说明这一轮里择时参数整体是在减分——之前某个单点配置的「三项全改善」大概率是全样本拟合。</div>
  </div>
  <div class="group">
    <h2>全部组合明细（按夏普降序）</h2>
    <div style="max-height:520px; overflow:auto;">
      <table id="tbl"></table>
    </div>
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
const axisCommon = { axisLine:{lineStyle:{color:'#e6e6e6'}}, axisLabel:{color:'#6b7280'} };

const topNs = DATA.top_ns, lookbacks = DATA.lookbacks;
const maxV = Math.max.apply(null, DATA.heat.map(function(h){return h[2];}));
const minV = Math.min.apply(null, DATA.heat.map(function(h){return h[2];}));
const heat = echarts.init(document.getElementById('heat'));
heat.setOption({
  tooltip:{ position:'top', formatter:function(p){ return 'lookback='+lookbacks[p.value[1]]+' top_n='+topNs[p.value[0]]+'<br/>夏普 '+p.value[2].toFixed(2); } },
  grid:{ left:80, right:30, top:30, bottom:70 },
  xAxis:{ type:'category', data:topNs.map(function(n){return 'n='+n;}), axisLine:axisCommon.axisLine, axisLabel:axisCommon.axisLabel, name:'top_n', nameLocation:'middle', nameGap:34 },
  yAxis:{ type:'category', data:lookbacks.map(function(l){return 'lb='+l;}), axisLine:axisCommon.axisLine, axisLabel:axisCommon.axisLabel, name:'lookback', nameLocation:'middle', nameGap:56 },
  visualMap:{ min:minV, max:maxV, calculable:true, orient:'horizontal', left:'center', bottom:10,
    inRange:{ color:['#389e0d','#fdf2e0','#d4380d'] } },
  series:[{ type:'heatmap', data:DATA.heat,
    label:{ show:true, formatter:function(p){return p.value[2].toFixed(2);}, color:'#1f2329', fontSize:11 },
    emphasis:{ itemStyle:{ shadowBlur:8, shadowColor:'rgba(0,0,0,0.3)' } } }]
});

const cols = DATA.columns;
const tbl = document.getElementById('tbl');
let html = '<tr>' + cols.map(function(c){ return '<th>'+c.label+'</th>'; }).join('') + '</tr>';
DATA.rows.forEach(function(r){
  let tds = '';
  cols.forEach(function(c){
    const v = r[c.key];
    if (c.fmt === 'pct') { const cls = v>=0?'pos':'neg'; tds += '<td class="'+cls+'">'+(v*100).toFixed(1)+'%</td>'; }
    else if (c.fmt === 'int') { tds += '<td>'+v+'</td>'; }
    else if (c.fmt === 'txt') { tds += '<td>'+(v===null||v===undefined?'-':v)+'</td>'; }
    else { tds += '<td>'+ (typeof v==='number'? v.toFixed(2): v) +'</td>'; }
  });
  html += '<tr>'+tds+'</tr>';
});
tbl.innerHTML = html;

// 择时维度对比（只在扫了择时参数时出现）
let timingChart = null;
if (DATA.timing_groups && DATA.timing_groups.length > 1) {
  document.getElementById('timing-group').style.display = '';
  timingChart = echarts.init(document.getElementById('timing'));
  const tg = DATA.timing_groups;
  timingChart.setOption({
    tooltip:{ trigger:'axis', axisPointer:{type:'shadow'} },
    legend:{ top:0, data:['平均总收益','平均夏普','平均最大回撤'] },
    grid:{ left:70, right:70, top:40, bottom:40 },
    xAxis:{ type:'category', data:tg.map(function(g){return g.name+' (n='+g.n+')';}),
            axisLine:axisCommon.axisLine, axisLabel:axisCommon.axisLabel },
    yAxis:[
      { type:'value', name:'收益 / 回撤', axisLabel:{formatter:function(v){return (v*100).toFixed(0)+'%';}, color:'#6b7280'}, ...axisCommon },
      { type:'value', name:'夏普', position:'right', axisLabel:{color:'#6b7280'}, ...axisCommon }
    ],
    series:[
      { name:'平均总收益', type:'bar', barMaxWidth:26, itemStyle:{ color:'#91caff' },
        data:tg.map(function(g){return g.avg_return;}) },
      { name:'平均最大回撤', type:'bar', barMaxWidth:26, itemStyle:{ color:'#ffccc7' },
        data:tg.map(function(g){return g.avg_dd;}) },
      { name:'平均夏普', type:'line', yAxisIndex:1, symbolSize:9,
        lineStyle:{width:2, color:'#cf1322'}, itemStyle:{ color:'#cf1322' },
        data:tg.map(function(g){return g.avg_sharpe;}) }
    ]
  });
}

window.addEventListener('resize', function(){ heat.resize(); if (timingChart) timingChart.resize(); });
</script>
</body>
</html>
"""


def build_grid_report(results: pd.DataFrame, out_html: str,
                      title: str = "参数优化网格搜索", subtitle: str = "") -> None:
    """生成自包含 HTML：夏普热力图 + 择时维度对比 + 全组合明细表。"""
    top_ns = sorted(results["top_n"].unique().tolist())
    lookbacks = sorted(results["lookback"].unique().tolist())

    piv = results.groupby(["lookback", "top_n"])["sharpe"].mean().reset_index()
    wide = piv.pivot(index="lookback", columns="top_n", values="sharpe")
    heat = []
    for yi, lb in enumerate(lookbacks):
        for xi, tn in enumerate(top_ns):
            if lb in wide.index and tn in wide.columns:
                v = wide.loc[lb, tn]
                if pd.notna(v):
                    heat.append([xi, yi, round(float(v), 3)])

    # 择时维度对比：同一择时模式下所有组合的平均表现
    timing_groups = []
    if "timing" in results.columns:
        for tm, sub in results.groupby("timing", sort=False):
            sub = sub.dropna(subset=["sharpe"])
            if not len(sub):
                continue
            timing_groups.append({
                "name": "不择时" if str(tm).lower() in ("off", "none", "") else str(tm),
                "n": int(len(sub)),
                "avg_return": round(float(sub["total_return"].mean()), 4),
                "avg_sharpe": round(float(sub["sharpe"].mean()), 3),
                "avg_dd": round(float(sub["max_drawdown"].mean()), 4),
            })

    ordered = results.sort_values("sharpe", ascending=False)
    columns = [
        {"key": "combo", "label": "参数组合", "fmt": "txt"},
        {"key": "lookback", "label": "lookback", "fmt": "int"},
        {"key": "top_n", "label": "top_n", "fmt": "int"},
        {"key": "buffer", "label": "buffer", "fmt": "int"},
        {"key": "timing", "label": "择时", "fmt": "txt"},
        {"key": "timing_lookback", "label": "择时窗口", "fmt": "int"},
        {"key": "timing_band", "label": "滞回带", "fmt": "pct"},
        {"key": "total_return", "label": "总收益", "fmt": "pct"},
        {"key": "annual_return", "label": "年化", "fmt": "pct"},
        {"key": "sharpe", "label": "夏普", "fmt": "num"},
        {"key": "max_drawdown", "label": "最大回撤", "fmt": "pct"},
        {"key": "calmar", "label": "卡玛", "fmt": "num"},
        {"key": "annual_turnover", "label": "年化换手", "fmt": "pct"},
        {"key": "worst_year_return", "label": "最差年度", "fmt": "pct"},
        {"key": "positive_year_ratio", "label": "正年占比", "fmt": "pct"},
        {"key": "excess_return", "label": "超额收益", "fmt": "pct"},
        {"key": "information_ratio", "label": "信息比率", "fmt": "num"},
    ]
    # 仅保留结果里实际存在的列（excess_return 等在没基准时会缺席）
    columns = [c for c in columns if c["key"] == "combo" or c["key"] in results.columns]
    rows = []
    for _, r in ordered.iterrows():
        rec = {c["key"]: r[c["key"]] for c in columns if c["key"] != "combo"}
        rec["combo"] = combo_label(rec)
        rows.append(rec)

    payload = {
        "top_ns": top_ns, "lookbacks": lookbacks, "heat": heat,
        "timing_groups": timing_groups, "columns": columns, "rows": rows,
    }
    html = (_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle)
            .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False)))
    Path(out_html).write_text(html, encoding="utf-8")
