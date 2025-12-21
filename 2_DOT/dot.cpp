#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

struct DotConfig {
    bool do_bench = false;
    std::size_t length = 5'000'000;
    int nthreads = std::max(1, omp_get_max_threads());
    std::string variant = "red"; // "red", "acc", "seq"
};

using Vec = std::vector<double>;

static Vec make_vector(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Vec v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        v.push_back(dist(gen));
    }
    return v;
}

static double dot_seq(const Vec &x, const Vec &y) {
    double s = 0.0;
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
}

// вариант 1: редукция OpenMP
static double dot_omp_reduction(const Vec &x, const Vec &y, int nthreads, double &time_sec) {
    double sum = 0.0;
    omp_set_num_threads(nthreads);
    const double t0 = omp_get_wtime();

#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        sum += x[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(i)];
    }

    const double t1 = omp_get_wtime();
    time_sec = t1 - t0;
    return sum;
}

// вариант 2: ручное накопление частичных сумм + critical
static double dot_omp_accumulate(const Vec &x, const Vec &y, int nthreads, double &time_sec) {
    double global_sum = 0.0;
    omp_set_num_threads(nthreads);
    const double t0 = omp_get_wtime();

#pragma omp parallel
    {
        double local_sum = 0.0;

#pragma omp for schedule(static)
        for (int i = 0; i < static_cast<int>(x.size()); ++i) {
            local_sum += x[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(i)];
        }

#pragma omp critical
        {
            global_sum += local_sum;
        }
    }

    const double t1 = omp_get_wtime();
    time_sec = t1 - t0;
    return global_sum;
}

static DotConfig parse_args(int argc, char **argv) {
    DotConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];
        auto take_next = [&](const char *flag) -> bool {
            return opt == flag && (i + 1 < argc);
        };

        if (opt == "--bench") {
            cfg.do_bench = true;
        } else if (take_next("--n")) {
            cfg.length = std::stoull(argv[++i]);
        } else if (take_next("--threads")) {
            cfg.nthreads = std::stoi(argv[++i]);
        } else if (take_next("--mode")) {
            cfg.variant = argv[++i]; // red / acc / seq
        }
    }
    return cfg;
}

static void run_single(const DotConfig &cfg) {
    Vec a = make_vector(cfg.length, 11);
    Vec b = make_vector(cfg.length, 91);

    double t_ref = 0.0;
    {
        const double t0 = omp_get_wtime();
        double ref = dot_seq(a, b);
        const double t1 = omp_get_wtime();
        t_ref = t1 - t0;
        std::cerr << "[seq]  n=" << cfg.length
                  << " time=" << std::fixed << std::setprecision(6) << t_ref << " s\n";

        double t_par = 0.0;
        double ans = 0.0;

        if (cfg.variant == "red") {
            ans = dot_omp_reduction(a, b, cfg.nthreads, t_par);
        } else if (cfg.variant == "acc") {
            ans = dot_omp_accumulate(a, b, cfg.nthreads, t_par);
        } else {
            ans = ref;
            t_par = t_ref;
        }

        if (std::fabs(ans - ref) > 1e-6) {
            std::cerr << "mismatch: ref=" << ref << " got=" << ans << "\n";
        }

        std::cout << "mode=" << cfg.variant
                  << " n=" << cfg.length
                  << " threads=" << cfg.nthreads
                  << " dot=" << ans
                  << " time=" << std::fixed << std::setprecision(6) << t_par
                  << " speedup=" << (t_ref / t_par) << "\n";
    }
}

static void run_bench() {
    std::vector<int> thread_grid;
    const int hw = omp_get_max_threads();
    int arr[] = {1, 2, 4, hw, hw + 2, 2 * hw};
    thread_grid.assign(std::begin(arr), std::end(arr));
    std::sort(thread_grid.begin(), thread_grid.end());
    thread_grid.erase(std::unique(thread_grid.begin(), thread_grid.end()), thread_grid.end());

    std::vector<std::size_t> sizes = {100'000, 1'000'000, 5'000'000, 20'000'000};

    std::cout << "variant,vec_size,threads,time_sec,speedup_vs_seq\n";
    for (std::size_t n : sizes) {
        Vec a = make_vector(n, 123);
        Vec b = make_vector(n, 321);

        const double t0 = omp_get_wtime();
        const double base = dot_seq(a, b);
        const double t1 = omp_get_wtime();
        const double t_seq = t1 - t0;

        for (const std::string mode : {"red", "acc"}) {
            for (int th : thread_grid) {
                const int repeats = 3;
                double sum_t = 0.0;
                double last_val = 0.0;

                for (int r = 0; r < repeats; ++r) {
                    double t_run = 0.0;
                    if (mode == "red") {
                        last_val = dot_omp_reduction(a, b, th, t_run);
                    } else {
                        last_val = dot_omp_accumulate(a, b, th, t_run);
                    }
                    if (std::fabs(last_val - base) > 1e-6) {
                        std::cerr << "dot mismatch at n=" << n << " threads=" << th << "\n";
                    }
                    sum_t += t_run;
                }

                const double t_avg = sum_t / repeats;
                std::cout << mode << "," << n << "," << th << ","
                          << std::fixed << std::setprecision(6) << t_avg << ","
                          << (t_seq / t_avg) << "\n";
            }
        }
    }
}

int main(int argc, char **argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    DotConfig cfg = parse_args(argc, argv);
    if (cfg.do_bench) {
        run_bench();
    } else {
        run_single(cfg);
    }
    return 0;
}
