#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

using std::string;
using std::vector;

enum class Sched {
    Static,
    Dynamic,
    Guided
};

string sched_to_str(Sched s) {
    switch (s) {
        case Sched::Static:  return "static";
        case Sched::Dynamic: return "dynamic";
        case Sched::Guided:  return "guided";
    }
    return "static";
}

Sched parse_sched(const string& s) {
    if (s == "static")  return Sched::Static;
    if (s == "dynamic") return Sched::Dynamic;
    if (s == "guided")  return Sched::Guided;
    return Sched::Static;
}

// "Тяжёлая" работа на одной итерации
double heavy_task(int i, int k_heavy) {
    // часть итераций значительно дороже остальных
    int inner = (i % k_heavy == 0) ? 5000 : 50;
    double acc = 0.0;
    for (int j = 0; j < inner; ++j) {
        acc += std::sin(0.0001 * (i + j));
    }
    return acc;
}

// последовательный вариант
double run_sequential(int n, int k_heavy, double& out_time) {
    double sum = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; ++i) {
        sum += heavy_task(i, k_heavy);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    out_time = std::chrono::duration<double>(t1 - t0).count();
    return sum;
}

// параллельный вариант с заданным schedule
double run_parallel(int n, int k_heavy,
                    int threads, Sched s, int chunk,
                    double& out_time)
{
    omp_set_num_threads(threads);
    omp_sched_t kind =
        (s == Sched::Static)  ? omp_sched_static  :
        (s == Sched::Dynamic) ? omp_sched_dynamic :
                                omp_sched_guided;

    omp_set_schedule(kind, chunk > 0 ? chunk : 0);

    double total = 0.0;
    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(runtime) reduction(+ : total)
    for (int i = 0; i < n; ++i) {
        total += heavy_task(i, k_heavy);
    }

    double t1 = omp_get_wtime();
    out_time = t1 - t0;
    return total;
}

struct Args {
    bool benchmark = false;
    int n = 500000;
    int k_heavy = 10;
    int threads = 4;
    string schedule = "static";
    int chunk = 0;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        string s = argv[i];
        auto need = [&](const char* flag) {
            if (i + 1 >= argc) {
                std::cerr << "missing value after " << flag << "\n";
                std::exit(1);
            }
        };

        if (s == "--benchmark") {
            a.benchmark = true;
        } else if (s == "--n") {
            need("--n");
            a.n = std::atoi(argv[++i]);
        } else if (s == "--heavy-period") {
            need("--heavy-period");
            a.k_heavy = std::atoi(argv[++i]);
        } else if (s == "--threads") {
            need("--threads");
            a.threads = std::atoi(argv[++i]);
        } else if (s == "--schedule") {
            need("--schedule");
            a.schedule = argv[++i];
        } else if (s == "--chunk") {
            need("--chunk");
            a.chunk = std::atoi(argv[++i]);
        } else {
            std::cerr << "unknown option: " << s << "\n";
            std::exit(1);
        }
    }
    if (a.threads < 1) a.threads = 1;
    if (a.k_heavy < 1) a.k_heavy = 1;
    return a;
}

void one_run(const Args& a) {
    Sched sch = parse_sched(a.schedule);

    double tseq = 0.0;
    double ref = run_sequential(a.n, a.k_heavy, tseq);

    double tpar = 0.0;
    double val = run_parallel(a.n, a.k_heavy,
                              a.threads, sch, a.chunk, tpar);

    if (std::fabs(ref - val) > 1e-6 * std::fabs(ref)) {
        std::cerr << "mismatch: seq=" << ref << " par=" << val << "\n";
    }

    double speedup = tseq / tpar;

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6)
              << "N=" << a.n
              << ", heavy_period=" << a.k_heavy
              << ", schedule=" << sched_to_str(sch)
              << ", threads=" << a.threads
              << ", Tseq=" << tseq
              << ", Tpar=" << tpar
              << ", S=" << speedup
              << "\n";
}

void run_bench(const Args& base) {
    int n = base.n;
    int k_heavy = base.k_heavy;

    std::cout << "N,heavy_period,schedule,threads,time_sec,speedup\n";

    // набор потоков и режимов
    int hw = omp_get_max_threads();
    vector<int> thread_list = {1, 2, 4, hw, 8};
    std::sort(thread_list.begin(), thread_list.end());
    thread_list.erase(std::unique(thread_list.begin(),
                                  thread_list.end()),
                      thread_list.end());

    vector<Sched> modes = {Sched::Static, Sched::Dynamic, Sched::Guided};

    // базовое последовательное время
    double tseq = 0.0;
    double ref = run_sequential(n, k_heavy, tseq);

    for (Sched sch : modes) {
        for (int t : thread_list) {
            double tsum = 0.0;
            int R = 3;
            double last = 0.0;

            for (int r = 0; r < R; ++r) {
                double tpar = 0.0;
                last = run_parallel(n, k_heavy, t, sch, base.chunk, tpar);
                if (std::fabs(last - ref) > 1e-6 * std::fabs(ref)) {
                    std::cerr << "parallel mismatch at threads="
                              << t << " schedule=" << sched_to_str(sch) << "\n";
                    std::exit(2);
                }
                tsum += tpar;
            }

            double tavg = tsum / R;
            double S = tseq / tavg;

            std::cout.setf(std::ios::fixed);
            std::cout << std::setprecision(6)
                      << n << ","
                      << k_heavy << ","
                      << sched_to_str(sch) << ","
                      << t << ","
                      << tavg << ","
                      << S << "\n";
        }
    }
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Args a = parse_args(argc, argv);
    if (a.benchmark) {
        run_bench(a);
    } else {
        one_run(a);
    }
    return 0;
}
