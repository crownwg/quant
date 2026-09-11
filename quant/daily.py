"""日常监控脚本：一键跑出「今天该买/卖什么」清单。

设计目标
--------
实战用法：每天盘后（或收盘后几分钟）跑一次，得到今日调仓清单。
相比 `python -m quant.main --today ...` 的零散命令行参数，daily.py 把
"配置 + 执行 + 输出 + 可选推送"四步打包成一个入口。

输入方式
--------
1. JSON 配置文件（推荐）：
   python -m quant.daily --config daily.json
2. 命令行直接传（适合临时调试）：
   python -m quant.daily --pool 消费 --strategy reversal_20 ...

输出
----
- daily_today_plan.csv   精简版今日清单（可直接对着券商下单）
- daily_equity_curve.csv 历史净值（让你一眼看出策略近期表现）
- 控制台摘要（包含要执行的关键操作）

可选：--webhook URL  把清单以 JSON 形式 POST 到指定地址（适配飞书/钉钉/企业微信
的自定义机器人；也适合自建的接收端）。不需要推送就留空。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import factors, filters, timing
from .backtest import run as backtest_run, cost_kwargs
from .data import load_index, load_panel
from .rebalance import build_plan
from .strategy import factor_weights, weights_from_args
from . import universe


# ------------------------------------------------------- 配置加载

def load_config(path: str | None) -> dict:
    """读 JSON 配置；空则返回默认骨架。"""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    return json.loads(p.read_text(encoding="utf-8"))


def merge_args_config(args, cfg: dict) -> dict:
    """命令行优先；未指定则用 config 里的值；都没有则用合理默认。"""
    out = {
        "pool": args.pool or cfg.get("pool", "消费"),
        "strategy": args.strategy or cfg.get("strategy", "reversal_20"),
        "top_n": args.top_n if args.top_n != 10 else cfg.get("top_n", 15),
        "rebalance": cfg.get("rebalance", "M"),
        "lookback": cfg.get("lookback", 120),
        "reversal_lookback": cfg.get("reversal_lookback", 20),
        "skip_recent": cfg.get("skip_recent", 0),
        "vol_lookback": cfg.get("vol_lookback", 60),
        "ma_short": cfg.get("ma_short", 20),
        "ma_long": cfg.get("ma_long", 60),
        "ma_window": cfg.get("ma_window", 60),
        "buffer": cfg.get("buffer", 2),
        "use_open": cfg.get("use_open", True),
        "min_volume": cfg.get("min_volume", 0.0),
        "min_amount": cfg.get("min_amount", 0.0),
        "max_participation": cfg.get("max_participation", 0.0),
        "weighting": cfg.get("weighting", "equal"),
        "max_weight": cfg.get("max_weight", 0.0),
        "neutralize": cfg.get("neutralize", ""),
        "slippage": cfg.get("slippage", 0.0005),
        "min_commission": cfg.get("min_commission", 5.0),
        "impact_coef": cfg.get("impact_coef", 0.0),
        "min_listed_days": cfg.get("min_listed_days", 0),
        "as_of": cfg.get("as_of", ""),
        # 择时仓位（默认关闭；开启后目标权重会被敞口缩放，清单里自动体现减仓）
        "timing": cfg.get("timing", "off"),
        "timing_lookback": cfg.get("timing_lookback", 120),
        "timing_band": cfg.get("timing_band", 0.0),
        "timing_ma_slope": cfg.get("timing_ma_slope", 0),
        "timing_min_exposure": cfg.get("timing_min_exposure", 0.0),
        "timing_max_exposure": cfg.get("timing_max_exposure", 1.0),
        "timing_smooth": cfg.get("timing_smooth", 0),
        "timing_proxy": cfg.get("timing_proxy", "pool"),
        "timing_update": cfg.get("timing_update", "rebalance"),
        "vol_target": cfg.get("vol_target", 0.0),
        "vol_target_lookback": cfg.get("vol_target_lookback", 60),
        "vol_floor": cfg.get("vol_floor", 0.2),
        "vol_cap": cfg.get("vol_cap", 1.0),
        "no_limit_filter": cfg.get("no_limit_filter", False),
        "no_suspend_filter": cfg.get("no_suspend_filter", False),
        "benchmark": args.benchmark or cfg.get("benchmark", "sh000932"),
        "capital": cfg.get("capital", 1_000_000.0),
        "fee": cfg.get("fee", 0.0003),
        "stamp_tax": cfg.get("stamp_tax", 0.0005),
        "holdings": cfg.get("holdings", ""),
        "today": cfg.get("today", ""),
        "fetch_lookback_days": cfg.get("fetch_lookback_days", 365),
        "out_prefix": cfg.get("out_prefix", "daily_"),
        "webhook": cfg.get("webhook", ""),
    }
    return out


# ------------------------------------------------------- 策略路由

def build_score(cfg: dict, prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """按 cfg['strategy'] 选对应的因子；后续可加多因子合成。"""
    s = cfg["strategy"]
    if s == "momentum_120":
        return factors.momentum(prices, lookback=cfg["lookback"], skip_recent=cfg["skip_recent"])
    if s == "momentum_60":
        return factors.momentum(prices, lookback=60)
    if s == "reversal_20":
        return factors.reversal(prices, lookback=cfg["reversal_lookback"])
    if s == "reversal_5":
        return factors.reversal(prices, lookback=5)
    if s == "low_vol_60":
        return factors.low_volatility(prices, lookback=cfg["vol_lookback"])
    if s == "ma_trend":
        return factors.ma_trend(prices, short=cfg["ma_short"], long=cfg["ma_long"])
    if s == "ma_breakout":
        return factors.ma_breakout(prices, window=cfg["ma_window"])
    if s == "combine_mom_lv":
        return factors.combine(
            {"momentum": factors.momentum(prices, lookback=cfg["lookback"]),
             "low_volatility": factors.low_volatility(prices, lookback=cfg["vol_lookback"])},
            {"momentum": 1.0, "low_volatility": 0.5}, method="rank")
    raise ValueError(f"未知策略 {s!r}；支持: momentum_120/60, reversal_20/5, "
                     "low_vol_60, ma_trend, ma_breakout, combine_mom_lv")


# ------------------------------------------------------- 主流程

def run_daily(cfg: dict) -> dict:
    """执行完整流程，返回摘要 dict（用于打印与推送）。"""
    pool = cfg["pool"]
    strategy = cfg["strategy"]

    print(f"📅 {datetime.now():%Y-%m-%d %H:%M:%S} · 池子={pool} · 策略={strategy}")

    codes = universe.build_universe(pool, as_of=cfg.get("as_of") or None)
    if not codes:
        raise ValueError(f"池子 {pool!r} 无成分股")

    # 预热期：取足够长的历史覆盖 lookback 窗口
    today = pd.Timestamp(cfg["today"]) if cfg["today"] else pd.Timestamp.today().normalize()
    lookback_days = cfg["fetch_lookback_days"]
    if cfg["min_listed_days"] > 0:
        lookback_days = max(lookback_days, cfg["min_listed_days"] + 30)
    # 择时的滚动窗口（均线 / 已实现波动率）也要预热，否则评估期开头会被迫空仓
    timing_need = max(int(cfg.get("timing_lookback", 0) or 0),
                      int(cfg.get("vol_target_lookback", 0) or 0)) * 2 + 30
    lookback_days = max(lookback_days, timing_need)
    fetch_start = (today - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    start = (today - pd.Timedelta(days=lookback_days - 30)).strftime("%Y%m%d")

    print(f"  区间：{start} ~ {end}（共 {lookback_days} 天）")
    print(f"  池子规模：{len(codes)} 只")

    panel = load_panel(codes, fetch_start, end, sleep=0.3, on_error="skip")
    prices_all = panel["close"]; open_all = panel["open"]; volume_all = panel["volume"]
    high_all = panel.get("high", pd.DataFrame()); low_all = panel.get("low", pd.DataFrame())
    amount_all = panel.get("amount", pd.DataFrame())
    n_used = len(prices_all.columns)
    if prices_all.empty:
        raise RuntimeError("池子无可用数据")
    if n_used < len(codes):
        print(f"  ⚠️ {len(codes) - n_used} 只数据不可用，实际 {n_used} 只")

    adv_all = filters.adv_notional(amount=amount_all if not amount_all.empty else None,
                                   volume=volume_all, close=prices_all)

    # 可行性约束
    limit_pct = filters.limit_pct_by_code(codes, [])
    illiquid = filters.illiquid_mask(volume_all, cfg["min_volume"])
    illiquid_amt = filters.illiquid_mask_by_amount(
        amount_all if not amount_all.empty else None, cfg.get("min_amount", 0.0))
    if illiquid_amt is not None:
        illiquid = illiquid_amt if illiquid is None else (illiquid | illiquid_amt)
    can_buy, can_sell = filters.tradability(
        prices_all, volume_all, limit_pct=limit_pct, illiquid=illiquid,
        enable_limit=not cfg["no_limit_filter"],
        enable_suspend=not cfg["no_suspend_filter"],
        open_=open_all if cfg["use_open"] else None,
        high=high_all if not high_all.empty else None,
        low=low_all if not low_all.empty else None,
        exec_at_open=cfg["use_open"],
    )

    # 因子 + 权重
    score = build_score(cfg, prices_all, volume_all)
    if cfg["min_listed_days"] > 0:
        score = score.where(filters.listing_age_mask(prices_all, cfg["min_listed_days"]))
    weight_cap_all = filters.capacity_cap(
        adv_all, cfg["capital"], cfg.get("max_participation", 0.0))

    # ---- 择时仓位：决定今日该放多少资金在场 ----
    exposure = None
    if (cfg.get("timing") or "off") != "off" or float(cfg.get("vol_target") or 0.0) > 0:
        proxy, label = None, f"池子等权组合({cfg['pool']})"
        spec = (cfg.get("timing_proxy") or "pool").strip()
        sym = ""
        if spec in ("benchmark", "auto"):
            sym = cfg.get("benchmark") or ""
        elif spec != "pool":
            sym = spec
        if sym:
            try:
                proxy = load_index(timing.norm_index_symbol(sym), fetch_start, end)
                proxy.index = pd.to_datetime(proxy.index)
                label = f"指数 {sym}"
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️ 择时代理 {sym} 获取失败，回退到池子等权组合: {exc}")
                proxy = None
        exposure = timing.build_exposure(cfg, proxy=proxy, prices=prices_all,
                                         proxy_label=label, verbose=True)

    weights = weights_from_args(score, cfg, can_buy=can_buy, can_sell=can_sell,
                                prices=prices_all, weight_cap=weight_cap_all,
                                exposure=exposure)

    # 切片到「近期评估期」（默认 5 年）
    eval_start = today - pd.Timedelta(days=365 * 5)
    keep = prices_all.index >= eval_start
    prices = prices_all.loc[keep]; open_prices = open_all.loc[keep]; w_eval = weights.loc[keep]

    # 历史回测（用于观察近期表现）
    equity, metrics, _ = backtest_run(
        prices, w_eval,
        open_prices=open_prices if cfg["use_open"] else None,
        **cost_kwargs(cfg, adv=(adv_all.loc[keep] if adv_all is not None else None)),
    )

    # 基准对比
    bench = None
    if cfg["benchmark"]:
        try:
            idx_full = load_index(cfg["benchmark"], fetch_start, end)
            idx_close = idx_full[idx_full.index >= eval_start]
            bench = (idx_close / idx_close.iloc[0]).reindex(equity.index).ffill().bfill()
            if len(bench) == len(equity) and bench.iloc[0]:
                excess = equity / bench
                metrics["excess_return"] = float(excess.iloc[-1] - 1)
                ex_daily = excess.pct_change().fillna(0.0)
                metrics["information_ratio"] = (
                    float(ex_daily.mean() / ex_daily.std() * (252 ** 0.5))
                    if ex_daily.std() else 0.0
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ 基准获取失败: {exc}")

    # 调仓计划
    exec_prices = open_prices if cfg["use_open"] else prices
    plan = build_plan(w_eval, exec_prices, capital=cfg["capital"],
                      fee=cfg["fee"], stamp_tax=cfg["stamp_tax"],
                      can_buy=can_buy.loc[keep], can_sell=can_sell.loc[keep],
                      exec_shift=1 if cfg["use_open"] else 0)

    # 今日清单
    last_ts = w_eval.index[-1]
    if last_ts.date() != today.date() and not cfg["today"]:
        # 自动模式下，last_ts 可能是最近交易日
        pass
    today_plan = plan[plan["date"] == last_ts.date()].copy() if not plan.empty else plan
    today_plan = today_plan[["code", "action", "prev_weight", "target_weight",
                              "delta_weight", "price", "shares", "amount"]].rename(
        columns={"shares": "delta_shares"})

    # ---- holdings 模式：实际 delta ----
    if cfg.get("holdings"):
        try:
            holdings_df = pd.read_csv(cfg["holdings"])
            holdings_df["code"] = holdings_df["code"].astype(str).str.zfill(6)
            cur_value = 0.0
            price_today = prices.loc[last_ts] if last_ts in prices.index else exec_prices.loc[last_ts]
            for row in holdings_df.itertuples():
                p = price_today.get(row.code, float("nan"))
                if pd.notna(p) and p > 0:
                    cur_value += int(row.shares) * float(p)
            total_capital = cur_value if cur_value > 0 else cfg["capital"]
            rows = []
            target_w = weights.loc[last_ts]
            codes_set = set(target_w[target_w > 1e-6].index) | set(holdings_df["code"])
            for code in sorted(codes_set):
                w = float(target_w.get(code, 0.0))
                p = float(price_today.get(code, float("nan")))
                if pd.isna(p) or p <= 0:
                    continue
                target_shares = int(w * total_capital / p / 100) * 100
                m = holdings_df[holdings_df["code"] == code]
                cur = int(m["shares"].iloc[0]) if not m.empty else 0
                delta = target_shares - cur
                if abs(delta) < 100:
                    continue
                if delta > 0:
                    action = "建仓" if cur == 0 else "加仓"
                else:
                    action = "清仓" if target_shares == 0 else "减仓"
                rows.append({"code": code, "action": action,
                             "current_shares": cur, "target_shares": target_shares,
                             "delta_shares": delta, "price": round(p, 4),
                             "amount": round(abs(delta) * p, 2)})
            today_plan = pd.DataFrame(rows).sort_values(["action", "code"]) if rows else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ holdings 处理失败：{exc}")

    # ---- 输出 ----
    prefix = cfg["out_prefix"]
    today_plan.to_csv(f"{prefix}today_plan.csv", index=False)
    equity.rename("equity").to_csv(f"{prefix}equity_curve.csv")
    if not plan.empty:
        plan.to_csv(f"{prefix}rebalance_plan.csv", index=False)
    # 指标
    pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()]
    ).to_csv(f"{prefix}metrics.csv", index=False)

    # 控制台摘要
    n_buy = int((today_plan["delta_shares"] > 0).sum()) if not today_plan.empty else 0
    n_sell = int((today_plan["delta_shares"] < 0).sum()) if not today_plan.empty else 0
    print(f"\n  📈 近期表现: 收益 {metrics['total_return']:.1%} 夏普 {metrics['sharpe']:.2f}"
          f" 回撤 {metrics['max_drawdown']:.1%}"
          + (f" 超额 {metrics.get('excess_return', 0):.1%}" if bench is not None else ""))
    print(f"\n  📋 今日清单（{last_ts.date()}）: 买 {n_buy} 笔 / 卖 {n_sell} 笔")
    if not today_plan.empty:
        for _, r in today_plan.head(20).iterrows():
            print(f"    {r['action']:6s} {r['code']} {int(r['delta_shares']):>6} 股 "
                  f"@ {r['price']:.2f} = {r['amount']:>12,.2f}")
        if len(today_plan) > 20:
            print(f"    ... 还有 {len(today_plan) - 20} 条")

    summary = {
        "as_of": str(last_ts.date()),
        "pool": pool,
        "strategy": strategy,
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        "today_count": {"buy": n_buy, "sell": n_sell},
        "today_actions": today_plan.to_dict("records"),
    }
    return summary


def send_webhook(url: str, summary: dict) -> None:
    """POST 摘要到 webhook（适配飞书/钉钉/企业微信机器人）。失败不抛错。"""
    if not url:
        return
    payload = {
        "msg_type": "text",
        "content": {
            "text": (f"📊 量化日报 {summary['as_of']}\n"
                     f"池子: {summary['pool']} / 策略: {summary['strategy']}\n"
                     f"近期: 收益 {summary['metrics']['total_return']:.1%} "
                     f"夏普 {summary['metrics']['sharpe']:.2f}\n"
                     f"今日: 买 {summary['today_count']['buy']} 笔 / "
                     f"卖 {summary['today_count']['sell']} 笔\n"
                     f"明细见 daily_today_plan.csv")
        }
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"  📤 webhook 已推送（状态 {resp.status}）")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ webhook 推送失败: {exc}")


# ------------------------------------------------------- CLI

def main() -> None:
    p = argparse.ArgumentParser(description="日常监控脚本：一键生成今日调仓清单")
    p.add_argument("--config", default="", help="JSON 配置文件路径（推荐）")
    p.add_argument("--pool", default="", help="股票池（覆盖 config）")
    p.add_argument("--strategy", default="", help="策略（覆盖 config）")
    p.add_argument("--benchmark", default="", help="基准指数（覆盖 config）")
    p.add_argument("--top-n", type=int, default=10, help="选股数（默认 10 即用 config）")
    p.add_argument("--webhook", default="", help="推送 URL（覆盖 config）")
    args = p.parse_args()

    cfg = load_config(args.config)
    merged = merge_args_config(args, cfg)
    if args.webhook:
        merged["webhook"] = args.webhook

    summary = run_daily(merged)
    send_webhook(merged.get("webhook", ""), summary)


if __name__ == "__main__":
    main()