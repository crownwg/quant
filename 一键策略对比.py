# -*- coding: utf-8 -*-
"""
一键策略对比 —— 给任意一只（或一组）股票，自动跑完「数据 → 全样本 → 样本外 → 报告」。

用法（在项目根目录）：
    .venv/Scripts/python.exe 一键策略对比.py 600031
    .venv/Scripts/python.exe 一键策略对比.py 600031,000157,000425,601100 --top-n 2
    .venv/Scripts/python.exe 一键策略对比.py 002557,600519,000858 --top-n 2 --end 20260910

或者直接双击  一键策略对比.bat  （会提示你输入股票代码）

它做什么：
  1. 补齐数据（增量，只拉缺的）
  2. 全样本扫描：31 种择时组合，回头看哪个最好
  3. 样本外验证：切成 8 段，只用前面的段选策略、在后面没见过的段上验证
  4. 多只票时额外跑：选股因子对比（反转 / 动量 / 低波动）
  5. 生成一份 HTML 报告

为什么必须跑第 3 步：本项目三个案例实测——「回头看最优」三次都被样本外推翻。
只跑第 2 步得到的「最优参数」基本都是拟合出来的。
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"

TIMING_MODES = "off,ma,momentum,dual"
TIMING_LB = "40,60,90,120,200"
TIMING_BAND = "0,0.02"
MODE_CN = {"off": "不择时", "ma": "均线", "momentum": "动量", "dual": "双确认"}


# ---------------- 基础工具 ----------------

def norm_codes(raw: str) -> list[str]:
    """宽容解析股票代码：逗号/空格/分号/中文逗号都能分隔，不足 6 位补零。"""
    parts = re.split(r"[,\s;；，、]+", raw.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.isdigit():
            out.append(p.zfill(6))
    # 去重保序
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def run_main(args: list[str], tag: str = "") -> subprocess.CompletedProcess:
    cmd = [str(PY), "-m", "quant.main"] + args
    print(f"\n{'='*72}\n▶ {tag}\n$ {' '.join(args)}\n{'='*72}")
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    tail = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.returncode else "")
    print(tail[-4000:] if len(tail) > 4000 else tail)
    if r.returncode != 0:
        print(f"⚠️ 这一步退出码 {r.returncode}，继续...")
    return r


def detect_end(codes: list[str]) -> str:
    """从本地缓存探测这批标的共同的最新日期。

    为什么不能直接用「今天」：数据接口当天往往还没有数据，
    end 晚于已有数据会让 load_panel 去拉一段「未来」的区间然后失败，
    整个回测都跑不起来（实测 refresh 到 20260914 时新浪返回「该区间内无数据」）。
    取所有标的里最保守（最小）的那个日期，保证回测区间内大家都有数据。
    """
    import pandas as pd
    latest = []
    for c in codes:
        p = ROOT / "data" / f"{c}.csv"
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p, usecols=["date"])["date"].iloc[-1]
            latest.append(str(d).replace("-", ""))
        except Exception:
            continue
    return min(latest) if latest else datetime.now().strftime("%Y%m%d")


def ask_codes_interactive() -> tuple[list[str], int]:
    """没有命令行参数时，交互式询问（供 .bat 双击使用）。"""
    print("\n你要分析哪只（或哪几只）股票？")
    print("  例：600031          单只")
    print("      600031,000157,000425    多只（逗号分隔）")
    raw = input("\n股票代码 > ").strip()
    codes = norm_codes(raw)
    if not codes:
        print("没识别到股票代码，退出。")
        sys.exit(1)
    top_n = 1
    if len(codes) > 1:
        s = input(f"\n每次持有几只？（1~{len(codes)}，直接回车用 {min(5, len(codes))}）> ").strip()
        if s.isdigit() and 1 <= int(s) <= len(codes):
            top_n = int(s)
        else:
            top_n = min(5, len(codes))
    return codes, top_n


# ---------------- 结果解析 ----------------

def parse_grid(path: Path) -> dict:
    """全样本网格：基线 + 各择时模式分组统计。"""
    import pandas as pd
    df = pd.read_csv(path)
    base = df[df["timing"] == "off"].iloc[0]
    res = {
        "base_ret": base["total_return"] * 100,
        "base_sh": float(base["sharpe"]),
        "base_dd": base["max_drawdown"] * 100,
        "modes": {},
    }
    for m in ["ma", "momentum", "dual"]:
        g = df[df["timing"] == m]
        if g.empty:
            continue
        r = g["total_return"] * 100
        res["modes"][m] = {
            "win": int((r > res["base_ret"]).sum()), "n": len(g),
            "mean": float(r.mean()), "best": float(r.max()), "worst": float(r.min()),
            "sharpe": float(g["sharpe"].mean()), "dd": float(g["max_drawdown"].mean() * 100),
        }
    return res


def parse_wf(folds_p: Path, sum_p: Path) -> dict:
    """样本外：各折选中的模式 + 各做法成绩。"""
    import pandas as pd
    folds = pd.read_csv(folds_p)
    summ = pd.read_csv(sum_p)
    modes = folds["timing"].value_counts().to_dict()
    arms = {}
    for _, r in summ.iterrows():
        arms[r["key"]] = {
            "name": r["strategy"], "ret": r["total_return"] * 100,
            "sh": float(r["avg_sharpe"]), "wdd": r["worst_fold_drawdown"] * 100,
            "pos": r["positive_folds"] * 100,
        }
    fold_rows = []
    for _, r in folds.iterrows():
        fold_rows.append({
            "fold": int(r["fold"]), "test": f"{r['test_start']}~{r['test_end']}",
            "mode": r["timing"], "lb": int(r["timing_lookback"]),
            "band": float(r["timing_band"]) * 100,
            "tr_sh": float(r["train_sharpe"]), "te_sh": float(r["test_sharpe"]),
            "ret": float(r["test_return"]) * 100,
            "ens3": float(r["test_return_ens3"]) * 100,
        })
    return {"modes": modes, "arms": arms, "folds": fold_rows}


def parse_compare(path: Path) -> list[dict]:
    """多策略对比结果（选股因子）。"""
    import pandas as pd
    df = pd.read_csv(path)
    df = df.sort_values("sharpe", ascending=False)
    out = []
    for _, r in df.iterrows():
        out.append({
            "strategy": str(r.get("strategy", "")),
            "ret": float(r.get("total_return", 0)) * 100,
            "sharpe": float(r.get("sharpe", 0)),
            "dd": float(r.get("max_drawdown", 0)) * 100,
        })
    return out


# ---------------- 报告 ----------------

def cls(v):
    return "up" if v > 0 else ("down" if v < 0 else "flat")


def sgn(v, d=1):
    return f"{v:+.{d}f}%"


def build_html(codes, top_n, start, end, grid, wf, compare, out: Path):
    multi = len(codes) > 1
    oos = wf["arms"]
    modes = wf["modes"]

    # 结论：全样本最优 vs 样本外
    fs_best_mode = None
    if grid and grid["modes"]:
        fs_best_mode = max(grid["modes"].items(), key=lambda kv: kv[1]["mean"])[0]
    if fs_best_mode:
        hit = int(modes.get(fs_best_mode, 0))
        n_folds = len(wf["folds"])
        if hit == 0:
            verdict = (f"全样本看起来 <b>{MODE_CN[fs_best_mode]}</b> 最好，"
                       f"但样本外 {n_folds} 段里 <b class='bad'>一次都没选中它</b>。"
                       f"这个「最优」是拟合出来的，不能信。")
        elif hit <= n_folds // 3:
            verdict = (f"全样本看起来 <b>{MODE_CN[fs_best_mode]}</b> 最好，"
                       f"但样本外 {n_folds} 段里只选中 {hit} 次。"
                       f"证据很弱，别当结论用。")
        else:
            verdict = (f"全样本和样本外都指向 <b>{MODE_CN[fs_best_mode]}</b>"
                       f"（样本外 {hit}/{n_folds} 段选中），这个结论相对可信。")
    else:
        verdict = "数据不足，无法给出结论。"

    # 全样本表
    fs_html = ""
    if grid:
        fs_html = (f"<tr class='base'><td><b>不择时（一直拿着）</b></td><td class='num'>—</td>"
                   f"<td class='num {cls(grid['base_ret'])}'>{sgn(grid['base_ret'])}</td>"
                   f"<td class='num'>{grid['base_sh']:.2f}</td>"
                   f"<td class='num down'>{sgn(grid['base_dd'])}</td>"
                   f"<td class='num'>—</td></tr>")
        for m, v in grid["modes"].items():
            hl = " class='hl'" if m == fs_best_mode else ""
            fs_html += (f"<tr{hl}><td>{MODE_CN[m]}择时</td><td class='num'>{v['win']}/{v['n']}</td>"
                        f"<td class='num'>{sgn(v['mean'])}</td><td class='num'>{v['sharpe']:.2f}</td>"
                        f"<td class='num down'>{sgn(v['dd'])}</td>"
                        f"<td class='num'>{sgn(v['best'])} / {sgn(v['worst'])}</td></tr>")

    # 样本外被选中次数
    cnt_html = ""
    for m in ["ma", "momentum", "dual", "off"]:
        c = int(modes.get(m, 0))
        pct = c / max(len(wf["folds"]), 1) * 100
        zero = " class='zero'" if c == 0 else ""
        cnt_html += (f"<tr{zero}><td>{MODE_CN.get(m, m)}</td><td class='num'><b>{c}</b></td>"
                     f"<td><div class='bar'><div class='fill' style='width:{pct:.0f}%'></div></div></td></tr>")

    # 样本外成绩
    keys = ["walk_forward", "smooth", "ens3", "ens5", "ens10", "ensall",
            "full_sample_best", "default"]
    oos_html = ""
    for k in keys:
        if k not in oos:
            continue
        o = oos[k]
        hl = " class='hl'" if k == "ens3" else ""
        tag = "<span class='tag'>诚实预期</span>" if k in ("smooth", "ens3") else ""
        oos_html += (f"<tr{hl}><td>{o['name']} {tag}</td>"
                     f"<td class='num {cls(o['ret'])}'><b>{sgn(o['ret'])}</b></td>"
                     f"<td class='num'>{o['sh']:.2f}</td>"
                     f"<td class='num down'>{sgn(o['wdd'])}</td>"
                     f"<td class='num'>{o['pos']:.0f}%</td></tr>")

    # 逐折
    f_html = ""
    for r in wf["folds"]:
        f_html += (f"<tr><td class='idx'>{r['fold']}</td><td class='dt'>{r['test']}</td>"
                   f"<td><b>{MODE_CN.get(r['mode'], r['mode'])}</b>({r['lb']}日,带{r['band']:.0f}%)</td>"
                   f"<td class='num'>{r['tr_sh']:.2f}</td><td class='num'>{r['te_sh']:.2f}</td>"
                   f"<td class='num {cls(r['ret'])}'><b>{sgn(r['ret'])}</b></td>"
                   f"<td class='num {cls(r['ens3'])}'>{sgn(r['ens3'])}</td></tr>")

    # 选股因子对比
    cmp_html = ""
    if compare:
        for r in compare:
            cmp_html += (f"<tr><td><b>{r['strategy']}</b></td>"
                         f"<td class='num {cls(r['ret'])}'>{sgn(r['ret'])}</td>"
                         f"<td class='num'>{r['sharpe']:.2f}</td>"
                         f"<td class='num down'>{sgn(r['dd'])}</td></tr>")
    cmp_block = ""
    if compare:
        cmp_block = f"""
<div class="panel">
  <h2>④ 选股因子对比（多只票才有意义）</h2>
  <div class="hint">同样这批票，换「按什么标准挑股票」，横向比一比。按夏普从高到低排。</div>
  <table>
    <thead><tr><th>选股标准</th><th class="num">总收益</th><th class="num">夏普</th><th class="num">最大回撤</th></tr></thead>
    <tbody>{cmp_html}</tbody>
  </table>
  <div class="hint" style="margin:12px 0 0">注意：这只是<b>全样本</b>结果，同样没过样本外验证，仅供参考排序、别直接当最优。</div>
</div>"""

    title = "、".join(codes) if len(codes) <= 4 else f"{len(codes)} 只股票组合"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · 策略对比报告</title>
<style>
  *{{box-sizing:border-box}}
  body{{margin:0;padding:32px 24px 60px;background:#f5f6f8;color:#1a1d21;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;line-height:1.62}}
  .wrap{{max-width:1060px;margin:0 auto}}
  h1{{font-size:24px;margin:0 0 6px;font-weight:600}}
  .sub{{color:#6b7280;font-size:14px;margin-bottom:22px}}
  .up{{color:#d92b2b}} .down{{color:#0a9b53}} .flat{{color:#6b7280}}
  .ok{{color:#0a7a42;font-weight:600}} .bad{{color:#d92b2b;font-weight:600}}
  .verdict{{background:#fff;border:2px solid #d92b2b;border-radius:12px;padding:20px 24px;margin-bottom:22px;font-size:14.5px}}
  .verdict .t{{font-size:16px;font-weight:600;color:#d92b2b;margin-bottom:8px}}
  .panel{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px 20px;margin-bottom:18px}}
  .panel h2{{font-size:16px;margin:0 0 6px;font-weight:600}}
  .hint{{font-size:12.5px;color:#6b7280;margin-bottom:14px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{text-align:left;padding:9px 10px;border-bottom:2px solid #e5e7eb;color:#6b7280;font-weight:500;font-size:12px}}
  td{{padding:9px 10px;border-bottom:1px solid #f0f1f3;vertical-align:top}}
  tr:hover td{{background:#fafbfc}}
  .num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
  .idx{{color:#9ca3af;font-size:12px}} .dt{{font-size:12px;color:#6b7280;white-space:nowrap}}
  tr.base td{{background:#f0f4f8}} tr.hl td{{background:#fff5f5}} tr.zero td{{opacity:.55}}
  .tag{{background:#e8f3ec;color:#0a7a42;font-size:10.5px;padding:2px 6px;border-radius:4px;margin-left:6px}}
  .bar{{width:100%;height:16px;background:#eef0f2;border-radius:3px;overflow:hidden}}
  .fill{{height:100%;background:#4b5563}}
  .two{{display:flex;gap:18px;flex-wrap:wrap}} .two>div{{flex:1 1 400px}}
  .warn{{background:#fff8e6;border:1px solid #f5d98b;border-radius:10px;padding:16px 20px;font-size:13px;color:#6b5518}}
  .warn b{{color:#8a6d1a}} .warn ul{{margin:8px 0 0;padding-left:20px}} .warn li{{margin-bottom:6px}}
</style></head><body><div class="wrap">

<h1>{title} · 策略对比报告</h1>
<div class="sub">
  {'、'.join(codes)}{f'（每次持有 {top_n} 只）' if multi else ''} · {start} ~ {end} ·
  31 种择时组合 · 切成 {len(wf['folds'])} 段做样本外验证 · 生成于 {datetime.now():%Y-%m-%d %H:%M}
</div>

<div class="verdict">
  <div class="t">先看结论</div>
  <div>{verdict}</div>
</div>

<div class="two">
  <div><div class="panel">
    <h2>① 全样本（回头看）</h2>
    <div class="hint">用整段历史跑 31 种组合。「跑赢基线」= 超过「一直拿着」的 {sgn(grid['base_ret']) if grid else '—'}。</div>
    <table>    <thead><tr><th>策略</th><th class="num">跑赢基线</th><th class="num">平均收益</th>
    <th class="num">夏普</th><th class="num">回撤</th><th class="num">最好/最差</th></tr></thead>
    <tbody>{fs_html}</tbody></table>
    <div class="hint" style="margin:12px 0 0">这只是「回头看」，<b>不能直接当结论</b>——见右边。</div>
  </div></div>
  <div><div class="panel">
    <h2>② 样本外（事前选）</h2>
    <div class="hint">切成 {len(wf['folds'])} 段，只用前面的段选策略、在后面没见过的段上验证。</div>
    <table><thead><tr><th>策略</th><th class="num">被选中</th><th></th></tr></thead>
    <tbody>{cnt_html}</tbody></table>
    <div class="hint" style="margin:12px 0 0">这才是接近真实的「事前」结论。</div>
  </div></div>
</div>

<div class="panel">
  <h2>③ 样本外真实成绩</h2>
  <div class="hint">
    <b>带「诚实预期」标签的两行最关键</b>——它们故意不挑最好的参数（挑最好本身就是作弊），
    最接近实盘能拿到的结果。
  </div>
  <table><thead><tr><th>做法</th><th class="num">样本外总收益</th><th class="num">平均夏普</th>
  <th class="num">最差段回撤</th><th class="num">赚钱段占比</th></tr></thead>
  <tbody>{oos_html}</tbody></table>
  <div class="hint" style="margin:12px 0 0">
    <b>怎么读</b>：如果「挑最好的参数」很好看、但「诚实预期」是负的，说明那个好看的数字主要是运气。
  </div>
</div>

<div class="panel">
  <h2>④ 逐段明细</h2>
  <div class="hint">「样本内夏普」是选策略时看到的分数，「样本外夏普」是真实考分。两者差距大 = 中看不中用。</div>
  <table><thead><tr><th>段</th><th>验证区间</th><th>选中的策略</th><th class="num">样本内夏普</th>
  <th class="num">样本外夏普</th><th class="num">该段收益</th><th class="num">集成版</th></tr></thead>
  <tbody>{f_html}</tbody></table>
</div>
{cmp_block}

<div class="warn">
  <b>⚠️ 这不是预测，也不是买卖建议</b>
  <ul>
    <li>报告里所有买卖信号都是<b>用历史数据倒推出来的</b>——「假如时光倒流回那天，按这个规则会怎么操作」。
        它<b>不能</b>告诉你明天该不该买。</li>
    <li>本项目已实测三个案例：<b>「回头看最优」三次都被样本外推翻</b>。
        任何「最优参数」在过样本外这一关之前都不可信。</li>
    <li>判断某收益是真是假：看「诚实预期」（不挑尖峰）那两行还赚不赚钱。
        不赚 → 那个好看的数字主要是运气。</li>
    <li>样本只有 {len(wf['folds'])} 段，段数少时结论更不稳定。</li>
    <li>本工具只负责算，不负责下单。</li>
  </ul>
</div>

</div></body></html>"""
    out.write_text(html, encoding="utf-8")
    return out


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser(
        description="一键策略对比：给股票代码，自动跑全样本+样本外并出报告")
    ap.add_argument("codes", nargs="?", default="", help="股票代码，逗号分隔，如 600031 或 600031,000157")
    ap.add_argument("--top-n", type=int, default=0, help="每次持有几只（多只票时必填，默认取 min(5, 票数)）")
    ap.add_argument("--start", default="20200101", help="开始日期 YYYYMMDD")
    ap.add_argument("--end", default="", help="结束日期 YYYYMMDD，默认今天")
    ap.add_argument("--out", default="", help="输出 HTML 文件名")
    ap.add_argument("--no-compare", action="store_true", help="多只票时跳过选股因子对比")
    a = ap.parse_args()

    if not (ROOT / ".venv" / "Scripts" / "python.exe").exists():
        print("❌ 找不到 .venv/Scripts/python.exe，请在项目根目录运行。")
        sys.exit(1)

    if a.codes:
        codes = norm_codes(a.codes)
        top_n = a.top_n if a.top_n > 0 else (min(5, len(codes)) if len(codes) > 1 else 1)
    else:
        codes, top_n = ask_codes_interactive()

    if not codes:
        print("❌ 没识别到股票代码。")
        sys.exit(1)
    if top_n > len(codes):
        top_n = len(codes)

    # 补数据时尽力补到今天；但回测的 end 用缓存里实际有的最新日期，
    # 避免「end 晚于数据」导致拉不到当天数据、整个回测跑不起来。
    if a.end:
        refresh_end = end = a.end
    else:
        refresh_end = datetime.now().strftime("%Y%m%d")
        end = detect_end(codes)

    multi = len(codes) > 1
    codes_s = ",".join(codes)
    print(f"\n标的：{codes_s}（{len(codes)} 只，每次持有 {top_n} 只）  区间：{a.start} ~ {end}")

    # 统一前缀，便于后面找文件。
    # ⚠️ 必须包含「全部代码」的哈希，不能只用 codes[0]：
    #    单只 002557 与多只 002557,600519,... 若撞名，本次跑失败时会读到上一次的旧结果，
    #    出一份「名字是组合、内容却是单只」的报告——这类错误看不出来，最危险。
    key = hashlib.md5(codes_s.encode()).hexdigest()[:6]
    prefix = f"_auto_{codes[0]}_{key}_"

    # 跑之前先清掉同前缀的旧文件：这样本次若失败，后面找不到文件就会报错退出，
    # 而不是静默拿上一次的过期结果出报告。
    for f in ROOT.glob(f"{prefix}*"):
        try:
            f.unlink()
        except OSError:
            pass

    # ---- 1. 补数据 ----
    r = run_main(["--codes", codes_s, "--refresh-data",
                  "--start", a.start, "--end", refresh_end], "① 补齐数据")
    if r.returncode != 0:
        print("\n⚠️ 数据补齐没完全成功（常见于网络/代理抽风，东财接口尤其容易断）。")
        missing = [c for c in codes if not (ROOT / "data" / f"{c}.csv").exists()]
        if missing:
            print(f"   本地完全没有数据、本次会被跳过：{', '.join(missing)}")
        print("   有缓存的标的不受影响，会照常参与。\n")

    # 固定选股因子为 reversal_20：
    #   ① 单只票时选股因子对结果无影响（top_n=1 永远选中它自己），但固定住可保证口径可复现；
    #   ② 多只票时 reversal_20 是本项目的稳健默认（见 README「reversal_20 在 A 股常被忽视」）。
    # 想换因子可自行在 common 里改 --strategy / --reversal-lookback。
    common = ["--codes", codes_s, "--top-n", str(top_n), "--use-open",
              "--rebalance", "M", "--start", a.start, "--end", end, "--benchmark", "",
              "--strategy", "reversal", "--reversal-lookback", "20",
              "--grid-lookbacks", "20", "--grid-topn", str(top_n), "--grid-buffers", "0",
              "--grid-timing", TIMING_MODES,
              "--grid-timing-lookbacks", TIMING_LB,
              "--grid-timing-bands", TIMING_BAND]

    # ---- 2. 全样本 ----
    run_main(common + ["--grid", "--out-prefix", f"{prefix}grid_"], "② 全样本扫描（31 组）")
    if not (ROOT / f"{prefix}grid_grid_results.csv").exists():
        print(f"\n❌ 全样本扫描没产出结果（{prefix}grid_grid_results.csv），中止——"
              f"绝不拿上一次的过期结果出报告。")
        sys.exit(1)

    # ---- 3. 样本外 ----
    run_main(common + ["--walk-forward", "--fw-train", "2", "--fw-test", "0.5",
                       "--out-prefix", f"{prefix}wf_"], "③ 样本外验证（8 折）")
    if not (ROOT / f"{prefix}wf_wf_folds.csv").exists():
        print(f"\n❌ 样本外验证没产出结果（{prefix}wf_wf_folds.csv），中止。")
        sys.exit(1)

    # ---- 4. 多只票：选股因子对比 ----
    compare = None
    if multi and not a.no_compare:
        r = run_main(["--codes", codes_s, "--top-n", str(top_n), "--use-open",
                      "--rebalance", "M", "--start", a.start, "--end", end,
                      "--benchmark", "", "--compare",
                      "--out-prefix", f"{prefix}cmp_"], "④ 选股因子对比")
        p = ROOT / f"{prefix}cmp_compare_results.csv"
        if p.exists():
            try:
                compare = parse_compare(p)
            except Exception as e:
                print(f"（选股对比结果解析失败：{e}）")

    # ---- 5. 汇总出报告 ----
    import pandas as pd  # noqa: F401
    gp = ROOT / f"{prefix}grid_grid_results.csv"
    fp = ROOT / f"{prefix}wf_wf_folds.csv"
    sp = ROOT / f"{prefix}wf_wf_summary.csv"

    grid = parse_grid(gp) if gp.exists() else None
    if not fp.exists() or not sp.exists():
        print(f"\n❌ 没找到样本外结果（{fp.name} / {sp.name}），无法出报告。")
        sys.exit(1)
    wf = parse_wf(fp, sp)

    out_name = a.out or f"{codes[0]}{'_组合' if multi else ''}_策略对比.html"
    out = ROOT / out_name
    build_html(codes, top_n, a.start, end, grid, wf, compare, out)

    print(f"\n{'='*72}")
    print(f"✅ 报告已生成：{out}")
    if grid and grid["modes"]:
        fb = max(grid["modes"].items(), key=lambda kv: kv[1]["mean"])[0]
        print(f"   全样本最优：{MODE_CN[fb]}（平均 {grid['modes'][fb]['mean']:+.1f}%）")
    print(f"   样本外选中：{'、'.join(f'{MODE_CN.get(k, k)} {v}' for k, v in wf['modes'].items() if v > 0)}")
    if "ens3" in wf["arms"]:
        print(f"   诚实预期（集成K3，样本外）：{wf['arms']['ens3']['ret']:+.1f}%")
        print(f"   不调参（一直拿着，样本外）：{wf['arms']['default']['ret']:+.1f}%")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
