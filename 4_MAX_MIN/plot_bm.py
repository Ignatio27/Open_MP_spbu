#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"Не найден CSV: {p}")
    df = pd.read_csv(p)
    required = {"nrows", "ncols", "threads", "time_sec", "speedup_vs_seq"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"В CSV нет колонок: {', '.join(sorted(missing))}")
    df["n"] = df["nrows"]
    return df


def plot_per_size(df: pd.DataFrame, charts_dir: Path) -> None:
    for n, chunk in df.groupby("n"):
        ordered = chunk.sort_values("threads")

        plt.figure()
        plt.plot(ordered["threads"], ordered["time_sec"], marker="o")
        plt.xlabel("Потоки")
        plt.ylabel("Время, с")
        plt.title(f"n = {n}: время выполнения")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(charts_dir / f"n{n}_time.png", dpi=150)
        plt.close()

        plt.figure()
        plt.plot(ordered["threads"], ordered["speedup_vs_seq"], marker="o")
        plt.xlabel("Потоки")
        plt.ylabel("Ускорение (T_seq / T_par)")
        plt.title(f"n = {n}: ускорение")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(charts_dir / f"n{n}_speedup.png", dpi=150)
        plt.close()


def plot_mean(df: pd.DataFrame, charts_dir: Path) -> None:
    summary = (
        df.groupby("threads")
        .agg({"time_sec": "mean", "speedup_vs_seq": "mean"})
        .reset_index()
        .sort_values("threads")
    )

    plt.figure()
    plt.plot(summary["threads"], summary["time_sec"], marker="o")
    plt.xlabel("Потоки")
    plt.ylabel("Среднее время, с (по всем n)")
    plt.title("Среднее время по всем размерам матриц")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(charts_dir / "mean_time.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(summary["threads"], summary["speedup_vs_seq"], marker="o")
    plt.xlabel("Потоки")
    plt.ylabel("Среднее ускорение (по всем n)")
    plt.title("Среднее ускорение по всем размерам матриц")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(charts_dir / "mean_speedup.png", dpi=150)
    plt.close()


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print("Использование: plot_bm.py bm_table.csv")
        raise SystemExit(1)

    df = load_table(argv[1])
    charts_dir = Path("plots")
    charts_dir.mkdir(exist_ok=True)

    plot_per_size(df, charts_dir)
    plot_mean(df, charts_dir)


if __name__ == "__main__":
    main(sys.argv)
