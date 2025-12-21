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

enum class Mode {
    Atomic,
    Critical,
    Lock,
    Reduction
};

Mode parse_mode(const std::string &s) {
    if (s == "atomic")    return Mode::Atomic;
    if (s == "critical")  return Mode::Critical;
    if (s == "lock")      return Mode::Lock;
    if (s == "reduction") return Mode::Reduction;
    std::cerr << "unknown mode: " << s << ", fallback = reduction\n";
    return Mode::Reduction;
}

const char* mode_to_cstr(Mode m) {
    switch (m) {
        case Mode::Atomic:    return "atomic";
        case Mode::Critical:  return "critical";
        case Mode::Lock:      return "lock";
        case Mode::Reduction: return "reduction";
    }
    return "reduction";
}

// простая "работа" в теле цикла, чтобы уменьшить влияние накладных расходов
inline double kernel(int i) {
    double x = 0.0;
    int reps = 20 + (i % 30);
    for (int k = 0; k < reps; ++k) {
        x += std::sin(0.0001 * (i + k));
        x -= std::cos(0.0002 * (i + 3 * k));
    }
    return x;
}

struct Cmd {
    long long n = 1000000;
    int threads = 4;
    Mode mode = Mode::Reduction;
};

Cmd parse_args(int argc, char **argv) {
    Cmd cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char *flag) {
            if (i + 1 >= argc) {
                std::cerr << "missing value after " << flag << "\n";
                std::exit(1);
            }
        };

        if (a == "--n") {
            need("--n");
            cfg.n = std::atoll(argv[++i]);
        } else if (a == "--threads") {
            need("--threads");
            cfg.threads = std::atoi(argv[++i]);
        } else if (a == "--mode") {
            need("--mode");
            cfg.mode = parse_mode(argv[++i]);
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            std::exit(1);
        }
    }
    if (cfg.threads < 1) cfg.threads = 1;
    if (cfg.n < 1) cfg.n = 1;
    return cfg;
}

// последовательная сумма для контроля корректности
double seq_sum(long long n) {
    double acc = 0.0;
    for (long long i = 0; i < n; ++i) {
        acc += kernel(static_cast<int>(i));
    }
    return acc;
}

double run_parallel(const Cmd &cfg, double &elapsed) {
    omp_set_num_threads(cfg.threads);

    double sum = 0.0;
    omp_lock_t lock;
    omp_init_lock(&lock);

    double t0 = omp_get_wtime();

    switch (cfg.mode) {
        case Mode::Atomic: {
            double local = 0.0;
            #pragma omp parallel private(local)
            {
                local = 0.0;
                #pragma omp for
                for (long long i = 0; i < cfg.n; ++i) {
                    local += kernel(static_cast<int>(i));
                }
                #pragma omp atomic
                sum += local;
            }
            break;
        }
        case Mode::Critical: {
            #pragma omp parallel
            {
                double local = 0.0;
                #pragma omp for
                for (long long i = 0; i < cfg.n; ++i) {
                    local += kernel(static_cast<int>(i));
                }
                #pragma omp critical
                {
                    sum += local;
                }
            }
            break;
        }
        case Mode::Lock: {
            #pragma omp parallel
            {
                double local = 0.0;
                #pragma omp for
                for (long long i = 0; i < cfg.n; ++i) {
                    local += kernel(static_cast<int>(i));
                }
                omp_set_lock(&lock);
                sum += local;
                omp_unset_lock(&lock);
            }
            break;
        }
        case Mode::Reduction: {
            #pragma omp parallel for reduction(+ : sum)
            for (long long i = 0; i < cfg.n; ++i) {
                sum += kernel(static_cast<int>(i));
            }
            break;
        }
    }

    double t1 = omp_get_wtime();
    omp_destroy_lock(&lock);

    elapsed = t1 - t0;
    return sum;
}

int main(int argc, char **argv) {
    Cmd cfg = parse_args(argc, argv);

    // последовательная проверка только один раз, чтобы не тормозить большое количество запусков
    static bool checked = false;
    static double ref = 0.0;
    if (!checked) {
        ref = seq_sum(cfg.n);
        checked = true;
    }

    double t = 0.0;
    double res = run_parallel(cfg, t);

    // допуск на погрешность
    double eps = 1e-8 * std::max(1.0, std::fabs(ref));
    if (std::fabs(ref - res) > eps) {
        std::cerr << "WARNING: mismatch, ref=" << ref << " par=" << res << "\n";
    }

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6)
              << cfg.n << ','
              << cfg.threads << ','
              << mode_to_cstr(cfg.mode) << ','
              << t << '\n';

    return 0;
}
