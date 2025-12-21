#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED = {
    "matrix_type",
    "bandwidth",
    "nrows",
    "ncols",
    "schedule",
    "threads",
    "time_sec",
    "speedup_vs_seq",
}


def read_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"CSV {csv_path} is missing: {', '.join(sorted(missing))}")
    df["schedule"] = df["schedule"].astype(str).str.lower()
    df["matrix_type"] = df["matrix_type"].astype(str)
    return df


def plots_per_size(df: pd.DataFrame, out_dir: Path) -> None:
    for (mt, n), chunk in df.groupby(["matrix_type", "nrows"]):
        chunk = chunk.copy()

        # Время: линии по schedule
        plt.figure()
        for sched, g in chunk.groupby("schedule"):
            g = g.sort_values("threads")
            plt.plot(g["threads"], g["time_sec"], marker="o", label=sched)
        plt.xlabel("Потоки")
        plt.ylabel("Время, с")
        plt.title(f"{mt}, n={n}: время (schedule)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{mt}_n{n}_time_by_schedule.png", dpi=150)
        plt.close()

        # Ускорение
        plt.figure()
        for sched, g in chunk.groupby("schedule"):
            g = g.sort_values("threads")
            plt.plot(g["threads"], g["speedup_vs_seq"], marker="o", label=sched)
        plt.xlabel("Потоки")
        plt.ylabel("Ускорение (T_seq / T_par)")
        plt.title(f"{mt}, n={n}: ускорение (schedule)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{mt}_n{n}_speedup_by_schedule.png", dpi=150)
        plt.close()


def plots_mean(df: pd.DataFrame, out_dir: Path) -> None:
    mean_df = (
        df.groupby(["matrix_type", "schedule", "threads"])
        .agg({"time_sec": "mean", "speedup_vs_seq": "mean"})
        .reset_index()
    )

    for mt, chunk in mean_df.groupby("matrix_type"):
        chunk = chunk.copy()

        plt.figure()
        for sched, g in chunk.groupby("schedule"):
            g = g.sort_values("threads")
            plt.plot(g["threads"], g["time_sec"], marker="o", label=sched)
        plt.xlabel("Потоки")
        plt.ylabel("Среднее время, с (по всем n)")
        plt.title(f"{mt}: усреднённое время по schedule")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{mt}_mean_time_by_schedule.png", dpi=150)
        plt.close()

        plt.figure()
        for sched, g in chunk.groupby("schedule"):
            g = g.sort_values("threads")
            plt.plot(g["threads"], g["speedup_vs_seq"], marker="o", label=sched)
        plt.xlabel("Потоки")
        plt.ylabel("Среднее ускорение (по всем n)")
        plt.title(f"{mt}: усреднённое ускорение по schedule")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{mt}_mean_speedup_by_schedule.png", dpi=150)
        plt.close()


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print("usage: plot.py bench_*.csv")
        raise SystemExit(1)

    for path in argv[1:]:
        df = read_table(path)
        mt = df["matrix_type"].iloc[0]
        out_dir = Path("plots") / mt
        out_dir.mkdir(parents=True, exist_ok=True)

        plots_per_size(df, out_dir)
        plots_mean(df, out_dir)


if __name__ == "__main__":
    main(sys.argv)
