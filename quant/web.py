"""量化回测 Web UI: FastAPI 后端。

启动:  python -m quant.web
访问:  http://127.0.0.1:8765/
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# 项目根目录: .../量化/
ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TEMPLATE = Path(__file__).resolve().parent / "templates" / "index.html"

app = FastAPI(title="量化回测 Web", version="1.0")


# ----- 入参模型 -----
class RunParams(BaseModel):
    pool: str = Field("", description="股票池（消费/医药/hs300 等），留空则用 codes")
    codes: str = Field("", description="手动股票代码，逗号分隔")
    strategy: str = Field("momentum", description="单因子名")
    lookback: int = Field(120, description="动量 / 低波等窗口")
    top_n: int = Field(15, description="每次调仓选几只")
    buffer: int = Field(0, description="缓冲带")
    rebalance: str = Field("M", description="调仓频率 M/W/Q")
    start: str = Field("20200101")
    end: str = Field("20240909")
    use_open: bool = Field(True)
    benchmark: str = Field("sh000300")


# ----- 路由 -----
@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    if not TEMPLATE.exists():
        raise HTTPException(500, f"模板未找到: {TEMPLATE}")
    return TEMPLATE.read_text(encoding="utf-8")


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "python": str(PYTHON),
        "python_exists": PYTHON.exists(),
        "cwd": str(ROOT),
        "template_exists": TEMPLATE.exists(),
    }


@app.post("/api/run")
async def run_backtest(p: RunParams) -> dict:
    if not PYTHON.exists():
        raise HTTPException(500, f"找不到 venv 解释器: {PYTHON}")
    if not p.pool.strip() and not p.codes.strip():
        raise HTTPException(400, "必须填写「股票池」或「手动代码」其中之一")

    prefix = f"web_{uuid.uuid4().hex[:8]}_"
    cmd = [
        str(PYTHON), "-m", "quant.main",
        "--start", p.start,
        "--end", p.end,
        "--top-n", str(p.top_n),
        "--strategy", p.strategy,
        "--lookback", str(p.lookback),
        "--rebalance", p.rebalance,
        "--buffer", str(p.buffer),
        "--benchmark", p.benchmark,
        "--out-prefix", prefix,
    ]
    if p.use_open:
        cmd.append("--use-open")
    if p.pool.strip():
        cmd += ["--pool", p.pool.strip()]
    else:
        cmd += ["--codes", p.codes.strip()]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900, cwd=str(ROOT),
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"error": "回测超时（>15 分钟），请缩小池子或缩短区间。"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"启动回测失败: {exc}"}

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1200:]
        return {"error": f"回测退出码 {proc.returncode}\n{tail}"}

    try:
        return _parse_results(prefix)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"结果解析失败: {exc}"}


# ----- 结果解析 -----
def _parse_results(prefix: str) -> dict:
    eq_path = ROOT / f"{prefix}equity_curve.csv"
    if not eq_path.exists():
        raise FileNotFoundError(f"找不到 {eq_path}")

    edf = pd.read_csv(eq_path)
    edf["date"] = pd.to_datetime(edf["date"])
    edf = edf.sort_values("date").reset_index(drop=True)
    dates = [d.strftime("%Y-%m-%d") for d in edf["date"]]
    equity = edf["equity"].astype(float).tolist()
    drawdown = _drawdown_series(pd.Series(equity))

    # metrics
    metrics: dict = {}
    metrics_path = ROOT / f"{prefix}metrics.csv"
    if metrics_path.exists():
        mdf = pd.read_csv(metrics_path)
        for _, row in mdf.iterrows():
            try:
                metrics[str(row["metric"])] = {
                    "label": str(row["label"]),
                    "value": float(row["value"]),
                }
            except (ValueError, TypeError):
                pass

    # benchmark
    benchmark = None
    bench_path = ROOT / f"{prefix}benchmark_curve.csv"
    if bench_path.exists():
        bdf = pd.read_csv(bench_path)
        bdf["date"] = pd.to_datetime(bdf["date"])
        benchmark = {
            "dates": [d.strftime("%Y-%m-%d") for d in bdf["date"]],
            "values": bdf["benchmark"].astype(float).tolist(),
        }

    # annual
    annual: list[dict] = []
    annual_path = ROOT / f"{prefix}annual_metrics.csv"
    if annual_path.exists():
        annual = pd.read_csv(annual_path).to_dict("records")

    # monthly heatmap
    monthly = _compute_monthly(edf)

    return {
        "ok": True,
        "metrics": metrics,
        "dates": dates,
        "equity": equity,
        "drawdown": drawdown,
        "benchmark": benchmark,
        "annual": annual,
        "monthly": monthly,
        "prefix": prefix,
        "stdout_tail": "",  # 调试时可拼回 main.py 的 print
    }


def _drawdown_series(equity: pd.Series) -> list[float]:
    cummax = equity.cummax()
    dd = (equity / cummax - 1.0).fillna(0.0)
    return dd.astype(float).tolist()


def _compute_monthly(equity_df: pd.DataFrame) -> dict:
    """生成月度收益热力图数据: years x months 矩阵。"""
    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["ret"] = df["equity"].pct_change().fillna(0.0)
    monthly_ret = df.groupby(["year", "month"], as_index=False)["ret"].sum()
    matrix = monthly_ret.pivot(index="year", columns="month", values="ret").fillna(0.0)
    return {
        "years": matrix.index.astype(int).tolist(),
        "months": matrix.columns.astype(int).tolist(),
        "values": [[round(float(v), 4) for v in row] for row in matrix.values.tolist()],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("WEB_PORT", "8765"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")