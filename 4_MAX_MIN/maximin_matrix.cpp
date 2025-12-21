#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

using std::string;
using std::vector;

using MatrixData = vector<double>;

double maximin_sequential(const MatrixData& a, int rows, int cols) {
    double best = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < rows; ++i) {
        double row_min = std::numeric_limits<double>::infinity();
        const int offset = i * cols;
        for (int j = 0; j < cols; ++j) {
            double v = a[offset + j];
            if (v < row_min) {
                row_min = v;
            }
        }
        if (row_min > best) {
            best = row_min;
        }
    }
    return best;
}

double maximin_parallel(const MatrixData& a,
                        int rows, int cols,
                        int threads) {
    double global_best = -std::numeric_limits<double>::infinity();
    omp_set_num_threads(threads);

    #pragma omp parallel for reduction(max : global_best) schedule(static)
    for (int i = 0; i < rows; ++i) {
        double row_min = std::numeric_limits<double>::infinity();
        const int offset = i * cols;
        for (int j = 0; j < cols; ++j) {
            double v = a[offset + j];
            if (v < row_min) {
                row_min = v;
            }
        }
        if (row_min > global_best) {
            global_best = row_min;
        }
    }
    return global_best;
}

MatrixData make_random_matrix(int rows, int cols, std::uint64_t seed) {
    MatrixData a(static_cast<std::size_t>(rows) * cols);
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0e6, 1.0e6);
    for (double& x : a) {
        x = dist(gen);
    }
    return a;
}

struct Options {
    bool bench = false;
    int rows = 1000;
    int cols = 1000;
    int threads = omp_get_max_threads();
};

Options parse_cmd(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        auto need_value = [&](const char* flag) {
            if (i + 1 >= argc) {
                std::cerr << "Ожидалось значение после " << flag << "\n";
                std::exit(1);
            }
        };
        if (arg == "--bench") {
            opt.bench = true;
        } else if (arg == "--rows") {
            need_value("--rows");
            opt.rows = std::atoi(argv[++i]);
        } else if (arg == "--cols") {
            need_value("--cols");
            opt.cols = std::atoi(argv[++i]);
        } else if (arg == "--threads") {
            need_value("--threads");
            opt.threads = std::atoi(argv[++i]);
        } else {
            std::cerr << "Неизвестный флаг: " << arg << "\n";
            std::exit(1);
        }
    }
    if (opt.threads < 1) opt.threads = 1;
    return opt;
}

void run_single(const Options& opt) {
    MatrixData a = make_random_matrix(opt.rows, opt.cols, 123);

    double t0 = omp_get_wtime();
    double ref = maximin_sequential(a, opt.rows, opt.cols);
    double t1 = omp_get_wtime();
    double t_seq = t1 - t0;

    double t2 = omp_get_wtime();
    double val = maximin_parallel(a, opt.rows, opt.cols, opt.threads);
    double t3 = omp_get_wtime();
    double t_par = t3 - t2;

    if (std::fabs(ref - val) > 1e-9) {
        std::cerr << "Mismatch: seq=" << ref << " par=" << val << "\n";
    }

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6)
              << "rows=" << opt.rows
              << ", cols=" << opt.cols
              << ", threads=" << opt.threads
              << ", y=" << val
              << ", T_seq=" << t_seq
              << ", T_par=" << t_par << "\n";
}

void run_bench() {
    std::vector<int> sizes = {1000, 10000, 15000, 20000};
    std::vector<int> threads = {1, 2, 4, 5, 8};

    std::cout << "nrows,ncols,threads,time_sec,speedup_vs_seq\n";

    for (int n : sizes) {
        int rows = n;
        int cols = n;
        MatrixData a = make_random_matrix(rows, cols, 777);

        int seq_runs = 3;
        double seq_sum = 0.0;
        double ref = 0.0;

        for (int r = 0; r < seq_runs; ++r) {
            double t0 = omp_get_wtime();
            double val = maximin_sequential(a, rows, cols);
            double t1 = omp_get_wtime();

            if (r == 0) {
                ref = val;
            } else if (std::fabs(ref - val) > 1e-9) {
                std::cerr << "Seq mismatch at n=" << n << "\n";
                std::exit(2);
            }
            seq_sum += (t1 - t0);
        }
        double t_seq = seq_sum / seq_runs;

        for (int th : threads) {
            maximin_parallel(a, rows, cols, th);

            int R = 3;
            double tsum = 0.0;
            double last = 0.0;

            for (int r = 0; r < R; ++r) {
                double s = omp_get_wtime();
                last = maximin_parallel(a, rows, cols, th);
                double e = omp_get_wtime();
                if (std::fabs(ref - last) > 1e-9) {
                    std::cerr << "Par mismatch at n=" << n
                              << ", threads=" << th << "\n";
                    std::exit(3);
                }
                tsum += (e - s);
            }

            double t_avg = tsum / R;
            double speed = t_seq / t_avg;

            std::cout.setf(std::ios::fixed);
            std::cout << std::setprecision(6)
                      << rows << "," << cols << ","
                      << th << "," << t_avg << ","
                      << speed << "\n";
        }
    }
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Options opt = parse_cmd(argc, argv);
    if (opt.bench) {
        run_bench();
    } else {
        run_single(opt);
    }
    return 0;
}
