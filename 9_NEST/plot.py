import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _infer_total_threads(row) -> int:
    if row["mode"] == "nested":
        return int(row["outer_threads"]) * int(row["inner_threads"])
    return int(row["outer_threads"])


def _load_benchmark(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    numeric_fields = [
        "nrows",
        "ncols",
        "outer_threads",
        "inner_threads",
        "outer_chunk",
        "inner_chunk",
        "time_sec",
        "speedup_vs_seq",
    ]
    for col in numeric_fields:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_threads"] = df.apply(_infer_total_threads, axis=1)
    df = df.dropna(
        subset=["nrows", "ncols", "time_sec", "speedup_vs_seq", "total_threads", "mode"]
    )
    df["size_tag"] = df["nrows"].astype(int).astype(str) + "x" + df["ncols"].astype(int).astype(str)
    return df


def _plot_time_and_speedup_by_size(df: pd.DataFrame, out_dir: Path) -> None:
    for size, sub in df.groupby("size_tag"):
        # time vs threads
        plt.figure()
        for mode, g in sub.groupby("mode"):
            stats = (
                g.groupby("total_threads", as_index=False)
                 .agg(mean_time=("time_sec", "mean"))
                 .sort_values("total_threads")
            )
            plt.plot(stats["total_threads"], stats["mean_time"], marker="o", label=mode)

        plt.xlabel("Общий бюджет потоков")
        plt.ylabel("Время, сек")
        plt.title(f"Время vs потоки ({size})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{size}_time.png", dpi=200)
        plt.close()

        # speedup vs threads
        plt.figure()
        for mode, g in sub.groupby("mode"):
            stats = (
                g.groupby("total_threads", as_index=False)
                 .agg(mean_speedup=("speedup_vs_seq", "mean"))
                 .sort_values("total_threads")
            )
            plt.plot(stats["total_threads"], stats["mean_speedup"], marker="o", label=mode)

        plt.xlabel("Общий бюджет потоков")
        plt.ylabel("Ускорение (seq/par)")
        plt.title(f"Ускорение vs потоки ({size})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{size}_speedup.png", dpi=200)
        plt.close()


def _plot_mean_time_and_speedup(df: pd.DataFrame, out_dir: Path) -> None:
    base = (
        df.groupby(["size_tag", "mode", "total_threads"], as_index=False)
          .agg(time_sec=("time_sec", "mean"), speedup=("speedup_vs_seq", "mean"))
    )
    collapsed = (
        base.groupby(["mode", "total_threads"], as_index=False)
            .agg(mean_time=("time_sec", "mean"), mean_speedup=("speedup", "mean"))
            .sort_values("total_threads")
    )

    # mean time
    plt.figure()
    for mode, g in collapsed.groupby("mode"):
        plt.plot(g["total_threads"], g["mean_time"], marker="o", label=mode)

    plt.xlabel("Общий бюджет потоков")
    plt.ylabel("Среднее время, сек")
    plt.title("Среднее время vs потоки (усреднение по размерам)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_time.png", dpi=200)
    plt.close()

    # mean speedup
    plt.figure()
    for mode, g in collapsed.groupby("mode"):
        plt.plot(g["total_threads"], g["mean_speedup"], marker="o", label=mode)

    plt.xlabel("Общий бюджет потоков")
    plt.ylabel("Среднее ускорение (seq/par)")
    plt.title("Среднее ускорение vs потоки (усреднение по размерам)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_speedup.png", dpi=200)
    plt.close()


def _add_schedule_tag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def tag(row):
        if row["mode"] == "outer":
            return f"{row['outer_sched']}"
        return f"{row['outer_sched']}|{row['inner_sched']}"

    df["schedule_tag"] = df.apply(tag, axis=1)
    return df


def _plot_schedule_comparisons(df: pd.DataFrame, out_dir: Path) -> None:
    sched_df = _add_schedule_tag(df)

    for size, df_size in sched_df.groupby("size_tag"):
        for mode, df_mode in df_size.groupby("mode"):
            plt.figure()
            for sch, df_s in df_mode.groupby("schedule_tag"):
                stats = (
                    df_s.groupby("total_threads", as_index=False)
                        .agg(mean_time=("time_sec", "mean"))
                        .sort_values("total_threads")
                )
                plt.plot(stats["total_threads"], stats["mean_time"], marker="o", label=sch)

            plt.xlabel("Общий бюджет потоков")
            plt.ylabel("Время, сек")
            plt.title(f"Schedule сравнение: {mode} ({size})")
            plt.grid(True)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / f"{size}_{mode}_schedules_time.png", dpi=200)
            plt.close()


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # Аргументы: [csv_path] [output_dir]
    csv_path = Path(argv[0]) if argv else Path("nested_bench.csv")
    out_dir = Path(argv[1]) if len(argv) > 1 else Path("plots")

    _ensure_output(out_dir)
    df = _load_benchmark(csv_path)

    _plot_time_and_speedup_by_size(df, out_dir)
    _plot_mean_time_and_speedup(df, out_dir)
    _plot_schedule_comparisons(df, out_dir)

    print(f"Графики сохранены в: {out_dir}")


if __name__ == "__main__":
    main()
