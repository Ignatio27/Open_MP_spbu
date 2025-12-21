import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

DATA = Path("reduce_results.csv")
PLOTS_DIR = Path("plots")          
PLOTS_DIR.mkdir(exist_ok=True) 

def load():
    rows = []
    with DATA.open() as f:
        r = csv.DictReader(f)
        for row in r:
            row["n"] = int(row["n"])
            row["threads"] = int(row["threads"])
            row["time_sec"] = float(row["time_sec"])
            rows.append(row)
    return rows

def group_by(xs, key):
    mp = defaultdict(list)
    for row in xs:
        mp[row[key]].append(row)
    return mp

def plot_time_per_n(rows):
    by_n = group_by(rows, "n")
    methods = ["atomic", "critical", "lock", "reduction"]
    for n, items in sorted(by_n.items()):
        by_m = group_by(items, "method")
        plt.figure()
        for m in methods:
            if m not in by_m:
                continue
            pts = sorted(by_m[m], key=lambda r: r["threads"])
            xs = [p["threads"] for p in pts]
            ys = [p["time_sec"] for p in pts]
            plt.plot(xs, ys, marker="o", label=m)
        plt.xlabel("Число потоков")
        plt.ylabel("Время, c")
        plt.title(f"n={n}: время по методам")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR/f"reduce_n{n}_time_by_method.jpg")
        plt.close()

def plot_speedup_per_n(rows):
    by_n = group_by(rows, "n")
    methods = ["atomic", "critical", "lock", "reduction"]
    for n, items in sorted(by_n.items()):
        # найдём последовательное время: threads == 1 и метод reduction
        seq_time = None
        for r in items:
            if r["threads"] == 1 and r["method"] == "reduction":
                seq_time = r["time_sec"]
                break
        if seq_time is None:
            continue

        by_m = group_by(items, "method")
        plt.figure()
        for m in methods:
            if m not in by_m:
                continue
            pts = sorted(by_m[m], key=lambda r: r["threads"])
            xs = [p["threads"] for p in pts]
            ys = [seq_time / p["time_sec"] for p in pts]
            plt.plot(xs, ys, marker="o", label=m)
        plt.xlabel("Число потоков")
        plt.ylabel("Ускорение (T_seq / T_par)")
        plt.title(f"n={n}: ускорение по методам")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR/f"reduce_n{n}_speedup_by_method.jpg")
        plt.close()

def plot_mean_time(rows):
    methods = ["atomic", "critical", "lock", "reduction"]
    by_m = group_by(rows, "method")

    plt.figure()
    for m in methods:
        if m not in by_m:
            continue
        pts = sorted(by_m[m], key=lambda r: r["threads"])
        xs = sorted(set(p["threads"] for p in pts))
        mean_y = []
        for th in xs:
            vals = [p["time_sec"] for p in pts if p["threads"] == th]
            mean_y.append(sum(vals) / len(vals))
        plt.plot(xs, mean_y, marker="o", label=m)
    plt.xlabel("Число потоков")
    plt.ylabel("Среднее время, c (по всем n)")
    plt.title("Среднее время по методам")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"reduce_mean_time_by_method.jpg")
    plt.close()

def plot_mean_speedup(rows):
    methods = ["atomic", "critical", "lock", "reduction"]
    # для каждого n считаем последовательное время reduction
    by_n = group_by(rows, "n")
    seq_times = {}
    for n, items in by_n.items():
        for r in items:
            if r["threads"] == 1 and r["method"] == "reduction":
                seq_times[n] = r["time_sec"]
                break

    by_m = group_by(rows, "method")
    plt.figure()
    for m in methods:
        if m not in by_m:
            continue
        pts = sorted(by_m[m], key=lambda r: r["threads"])
        xs = sorted(set(p["threads"] for p in pts))
        mean_s = []
        for th in xs:
            vals = []
            for p in pts:
                if p["threads"] == th:
                    n = p["n"]
                    if n in seq_times:
                        vals.append(seq_times[n] / p["time_sec"])
            if vals:
                mean_s.append(sum(vals) / len(vals))
        plt.plot(xs, mean_s, marker="o", label=m)
    plt.xlabel("Число потоков")
    plt.ylabel("Среднее ускорение (по всем n)")
    plt.title("Среднее ускорение по методам")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"reduce_mean_speedup_by_method.jpg")
    plt.close()

def main():
    rows = load()
    plot_time_per_n(rows)
    plot_speedup_per_n(rows)
    plot_mean_time(rows)
    plot_mean_speedup(rows)

if __name__ == "__main__":
    main()
