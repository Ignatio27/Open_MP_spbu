#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise SystemExit(f"CSV-файл не найден: {csv_path}")
    df = pd.read_csv(csv_path)
    expected = {"func", "rule", "N", "threads", "time_sec", "speedup_vs_seq"}
    if not expected.issubset(df.columns):
        missing = expected - set(df.columns)
        raise SystemExit(f"В CSV отсутствуют колонки: {', '.join(sorted(missing))}")
    return df


def plot_per_problem(df: pd.DataFrame, out_dir: Path) -> None:
    for (func, rule, n), part in df.groupby(["func", "rule", "N"]):
        ordered = part.sort_values("threads")

        # время
        plt.figure()
        plt.plot(ordered["threads"], ordered["time_sec"], marker="o")
        plt.xlabel("Потоки")
        plt.ylabel("Время, с")
        plt.title(f"{func}, {rule}, N={n}: время")
        plt.grid(True)
        plt.tight_layout()
        fname = out_dir / f"{func}_{rule}_N{n}_time_sec.png"
        plt.savefig(fname, dpi=150)
        plt.close()

        # ускорение
        plt.figure()
        plt.plot(ordered["threads"], ordered["speedup_vs_seq"], marker="o")
        plt.xlabel("Потоки")
        plt.ylabel("Ускорение (T_seq / T_par)")
        plt.title(f"{func}, {rule}, N={n}: ускорение")
        plt.grid(True)
        plt.tight_layout()
        fname = out_dir / f"{func}_{rule}_N{n}_speedup_vs_seq.png"
        plt.savefig(fname, dpi=150)
        plt.close()


def plot_mean(df: pd.DataFrame, out_dir: Path) -> None:
    grouped = (
        df.groupby(["func", "rule", "threads"])
          .agg({"time_sec": "mean", "speedup_vs_seq": "mean"})
          .reset_index()
    )

    for (func, rule), part in grouped.groupby(["func", "rule"]):
        ordered = part.sort_values("threads")

        plt.figure()
        plt.plot(ordered["threads"], ordered["time_sec"], marker="o")
        plt.xlabel("Потоки")
        plt.ylabel("Среднее время, с (по всем N)")
        plt.title(f"{func}, {rule}: усреднённое время")
        plt.grid(True)
        plt.tight_layout()
        fname = out_dir / f"mean_time_{func}_{rule}.png"
        plt.savefig(fname, dpi=150)
        plt.close()

        plt.figure()
        plt.plot(ordered["threads"], ordered["speedup_vs_seq"], marker="o")
        plt.xlabel("Потоки")
        plt.ylabel("Среднее ускорение (по всем N)")
        plt.title(f"{func}, {rule}: усреднённое ускорение")
        plt.grid(True)
        plt.tight_layout()
        fname = out_dir / f"mean_speedup_{func}_{rule}.png"
        plt.savefig(fname, dpi=150)
        plt.close()


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print("Использование: plot_bm.py bm_table.csv")
        raise SystemExit(1)

    df = load_data(argv[1])
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    plot_per_problem(df, out_dir)
    plot_mean(df, out_dir)



if __name__ == "__main__":
    main(sys.argv)
