// extrema_search.cpp
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <string>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include <omp.h>

struct Extrema {
    double min_val;
    double max_val;
};

static Extrema run_sequential(const std::vector<double> &data) {
    Extrema e;
    e.min_val = std::numeric_limits<double>::infinity();
    e.max_val = -std::numeric_limits<double>::infinity();

    for (double x : data) {
        if (x < e.min_val) e.min_val = x;
        if (x > e.max_val) e.max_val = x;
    }
    return e;
}

static Extrema run_with_reduction(const std::vector<double> &data, int workers) {
    double gmin = std::numeric_limits<double>::infinity();
    double gmax = -std::numeric_limits<double>::infinity();

    omp_set_num_threads(workers);
    double t_start = omp_get_wtime();

#pragma omp parallel for reduction(min : gmin) reduction(max : gmax) schedule(static)
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        double v = data[i];
        if (v < gmin) gmin = v;
        if (v > gmax) gmax = v;
    }

    double t_end = omp_get_wtime();
    std::fprintf(stderr, "[reduction] elapsed = %.6f s\n", t_end - t_start);

    return {gmin, gmax};
}

static Extrema run_with_locals(const std::vector<double> &data, int workers) {
    double gmin = std::numeric_limits<double>::infinity();
    double gmax = -std::numeric_limits<double>::infinity();

    omp_set_num_threads(workers);
    double t_start = omp_get_wtime();

#pragma omp parallel
    {
        double lmin = std::numeric_limits<double>::infinity();
        double lmax = -std::numeric_limits<double>::infinity();

#pragma omp for schedule(static)
        for (int i = 0; i < static_cast<int>(data.size()); ++i) {
            double v = data[i];
            if (v < lmin) lmin = v;
            if (v > lmax) lmax = v;
        }

#pragma omp critical
        {
            if (lmin < gmin) gmin = lmin;
            if (lmax > gmax) gmax = lmax;
        }
    }

    double t_end = omp_get_wtime();
    std::fprintf(stderr, "[locals] elapsed = %.6f s\n", t_end - t_start);

    return {gmin, gmax};
}

static std::vector<double> generate_input(std::size_t n, std::uint64_t seed) {
    std::vector<double> arr(n);
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0e6, 1.0e6);
    for (double &x : arr) {
        x = dist(gen);
    }
    return arr;
}

struct CmdLine {
    bool batch_mode = false;           // если true — генерируем CSV по множеству n/потоков
    std::size_t size = 5'000'000;      // размер вектора для одиночного прогона
    int workers = 0;                   // 0 => использовать max_threads
    std::string algo = "red";         // red | loc | seq
};

static void print_usage(const char *prog) {
    std::cerr
        << "Использование: " << prog
        << " [-x] [-n <size>] [-p <threads>] [-a red|loc|seq]\n"
        << "  без -x выполняется один запуск и печатается результат\n"
        << "  с  -x выполняется серия замеров и выводится CSV\n";
}

static CmdLine parse_args(int argc, char **argv) {
    CmdLine cfg;
    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];
        if (opt == "-h" || opt == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (opt == "-x") {
            cfg.batch_mode = true;
        } else if (opt == "-n" && i + 1 < argc) {
            cfg.size = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (opt == "-p" && i + 1 < argc) {
            cfg.workers = std::stoi(argv[++i]);
        } else if (opt == "-a" && i + 1 < argc) {
            cfg.algo = argv[++i];
        } else {
            std::cerr << "Неизвестный аргумент: " << opt << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    if (cfg.workers <= 0) {
        cfg.workers = std::max(1, omp_get_max_threads());
    }
    return cfg;
}

static void single_run(const CmdLine &cfg) {
    std::vector<double> data = generate_input(cfg.size, 2025);

    double t0 = omp_get_wtime();
    Extrema base = run_sequential(data);
    double t1 = omp_get_wtime();
    double t_seq = t1 - t0;
    std::fprintf(stderr, "[sequential] elapsed = %.6f s\n", t_seq);

    Extrema res;
    if (cfg.algo == "red") {
        res = run_with_reduction(data, cfg.workers);
    } else if (cfg.algo == "loc") {
        res = run_with_locals(data, cfg.workers);
    } else {
        res = base;
    }

    if (std::fabs(res.min_val - base.min_val) > 1e-9 ||
        std::fabs(res.max_val - base.max_val) > 1e-9) {
        std::cerr << "Несовпадение результата: min=" << res.min_val
                  << " max=" << res.max_val
                  << " ref_min=" << base.min_val
                  << " ref_max=" << base.max_val << "\n";
        std::exit(2);
    }

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6);
    std::cout << "n=" << cfg.size
              << "; threads=" << cfg.workers
              << "; algo=" << cfg.algo
              << "; min=" << res.min_val
              << "; max=" << res.max_val << "\n";
}

// CSV: scenario;size;workers;time;speedup
static void batch_run() {
    int hw = omp_get_max_threads();

    std::vector<int> workers_list;
    std::vector<int> base = {1, 2, 4, 8, hw, hw + 1, 2 * hw};
    std::sort(base.begin(), base.end());
    base.erase(std::unique(base.begin(), base.end()), base.end());
    for (int t : base) {
        if (t > 0) workers_list.push_back(t);
    }

    std::vector<std::size_t> sizes = {
        100'000,
        1'000'000,
        5'000'000,
        20'000'000
    };

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6);
    std::cout << "scenario;size;workers;time_sec;speedup\n";

    for (std::size_t n : sizes) {
        std::vector<double> data = generate_input(n, 777);

        double t0 = omp_get_wtime();
        Extrema ref = run_sequential(data);
        double t1 = omp_get_wtime();
        double t_seq = t1 - t0;
        (void)ref;

        for (const std::string &mode : {"red", "loc"}) {
            for (int th : workers_list) {
                // прогрев
                if (mode == "red") {
                    (void)run_with_reduction(data, th);
                } else {
                    (void)run_with_locals(data, th);
                }

                const int REPEAT = 3;
                double sum = 0.0;
                for (int r = 0; r < REPEAT; ++r) {
                    double t2 = omp_get_wtime();
                    Extrema got = (mode == "red")
                                      ? run_with_reduction(data, th)
                                      : run_with_locals(data, th);
                    double t3 = omp_get_wtime();

                    if (std::fabs(got.min_val - ref.min_val) > 1e-9 ||
                        std::fabs(got.max_val - ref.max_val) > 1e-9) {
                        std::cerr << "Ошибка проверки при n=" << n
                                  << " threads=" << th
                                  << " mode=" << mode << "\n";
                        std::exit(3);
                    }
                    sum += (t3 - t2);
                }

                double t_avg = sum / REPEAT;
                double k = t_seq / t_avg;

                std::cout << mode << ";"
                          << n << ";"
                          << th << ";"
                          << t_avg << ";"
                          << k << "\n";
            }
        }
    }
}

int main(int argc, char **argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    CmdLine cfg = parse_args(argc, argv);
    if (cfg.batch_mode) {
        batch_run();
    } else {
        single_run(cfg);
    }
    return 0;
}
