#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def read_table(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise SystemExit(f"Не найден входной CSV-файл: {csv_path}")
    # разделитель в bm_table.csv — точка с запятой
    return pd.read_csv(csv_path, sep=";")


def plot_per_size(df: pd.DataFrame) -> None:
    """Графики времени и ускорения отдельно для каждого размера задачи."""
    metrics = [
        ("time_sec", "Время, с"),
        ("speedup", "Ускорение (seq/parallel)"),
    ]

    for size, df_size in df.groupby("size"):
        for col, y_label in metrics:
            plt.figure()
            for name, df_method in df_size.groupby("scenario"):
                ordered = df_method.sort_values("workers")
                plt.plot(
                    ordered["workers"],
                    ordered[col],
                    marker="o",
                    label=name,
                )

            plt.title(f"n = {size}")
            plt.xlabel("Потоки")
            plt.ylabel(y_label)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"plots/n_{size}_{col}.png", dpi=150)
            # plt.show()
            plt.close()


def plot_aggregated(df: pd.DataFrame) -> None:
    """Графики по данным, усреднённым по всем размерам n."""
    mean_stats = (
        df.groupby(["scenario", "workers"])
        .mean(numeric_only=True)
        .reset_index()
    )

    metrics = [
        ("time_sec", "Среднее время (по всем n), с"),
        ("speedup", "Среднее ускорение (по всем n)"),
    ]

    for col, y_label in metrics:
        plt.figure()
        for name, df_method in mean_stats.groupby("scenario"):
            ordered = df_method.sort_values("workers")
            plt.plot(
                ordered["workers"],
                ordered[col],
                marker="o",
                label=name,
            )

        plt.title("Усреднённые результаты по всем размерам задач")
        plt.xlabel("Потоки")
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/mean_{col}.png", dpi=150)
        # plt.show()
        plt.close()


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print("Использование: python draw_extrema_plots.py <benchmarks.csv>")
        raise SystemExit(1)

    df = read_table(argv[1])
    Path("plots").mkdir(exist_ok=True)

    plot_per_size(df)
    plot_aggregated(df)


if __name__ == "__main__":
    main(sys.argv)
