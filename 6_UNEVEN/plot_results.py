#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED = {"N", "heavy_period", "schedule", "threads", "time_sec", "speedup"}


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns: {', '.join(sorted(missing))}")
    df["schedule"] = df["schedule"].astype(str).str.lower()
    return df


def plot_for_size(df: pd.DataFrame, outdir: Path) -> None:
    N = int(df["N"].iloc[0])

    # время
    plt.figure()
    for sched, g in df.groupby("schedule"):
        g = g.sort_values("threads")
        plt.plot(g["threads"], g["time_sec"], marker="o", label=sched)
    plt.xlabel("Число потоков")
    plt.ylabel("Время, с")
    plt.title(f"Неравномерный цикл: N={N}, время (schedule)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"uneven_N{N}_time_by_schedule.png", dpi=150)
    plt.close()

    # ускорение
    plt.figure()
    for sched, g in df.groupby("schedule"):
        g = g.sort_values("threads")
        plt.plot(g["threads"], g["speedup"], marker="o", label=sched)
    plt.xlabel("Число потоков")
    plt.ylabel("Ускорение (T_seq / T_par)")
    plt.title(f"Неравномерный цикл: N={N}, ускорение (schedule)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"uneven_N{N}_speedup_by_schedule.png", dpi=150)
    plt.close()


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print("usage: plot_results.py bench_*.csv")
        raise SystemExit(1)

    for path in argv[1:]:
        df = load_csv(path)
        outdir = Path("plots")
        outdir.mkdir(exist_ok=True)
        plot_for_size(df, outdir)


if __name__ == "__main__":
    main(sys.argv)
