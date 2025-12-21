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

enum class Shape {
    Dense,
    Lower,
    Upper,
    Band
};

enum class SchedPolicy {
    Static,
    Dynamic,
    Guided
};

// ---------- преобразование строковых опций ----------

Shape parse_shape(const string& s) {
    if (s == "dense")  return Shape::Dense;
    if (s == "lower")  return Shape::Lower;
    if (s == "upper")  return Shape::Upper;
    if (s == "band")   return Shape::Band;
    return Shape::Dense;
}

string shape_name(Shape sh) {
    switch (sh) {
        case Shape::Dense:  return "dense";
        case Shape::Lower:  return "lower";
        case Shape::Upper:  return "upper";
        case Shape::Band:   return "band";
    }
    return "dense";
}

SchedPolicy parse_sched(const string& s) {
    if (s == "static")  return SchedPolicy::Static;
    if (s == "dynamic") return SchedPolicy::Dynamic;
    if (s == "guided")  return SchedPolicy::Guided;
    return SchedPolicy::Static;
}

string sched_name(SchedPolicy p) {
    switch (p) {
        case SchedPolicy::Static:  return "static";
        case SchedPolicy::Dynamic: return "dynamic";
        case SchedPolicy::Guided:  return "guided";
    }
    return "static";
}

// ---------- диапазон столбцов для строки ----------

inline void cols_for_row(int i, int cols, Shape sh, int bw, int& j_first, int& j_last) {
    switch (sh) {
        case Shape::Dense:
            j_first = 0;
            j_last  = cols - 1;
            break;
        case Shape::Lower:
            j_first = 0;
            j_last  = std::min(i, cols - 1);
            break;
        case Shape::Upper:
            j_first = std::min(i, cols - 1);
            j_last  = cols - 1;
            break;
        case Shape::Band:
            j_first = std::max(0, i - bw);
            j_last  = std::min(cols - 1, i + bw);
            break;
    }
}

// ---------- генерация данных ----------

using Matrix = vector<double>;

Matrix make_random(int rows, int cols, std::uint64_t seed) {
    Matrix a(static_cast<std::size_t>(rows) * cols);
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0e6, 1.0e6);
    for (double& x : a) {
        x = dist(gen);
    }
    return a;
}

// ---------- последовательный алгоритм ----------

double seq_maximin(const Matrix& a,
                   int rows, int cols,
                   Shape sh, int bw)
{
    double best = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < rows; ++i) {
        int j0, j1;
        cols_for_row(i, cols, sh, bw, j0, j1);
        if (j0 > j1) continue;

        double row_min = std::numeric_limits<double>::infinity();
        int base = i * cols;
        for (int j = j0; j <= j1; ++j) {
            double v = a[base + j];
            if (v < row_min) row_min = v;
        }
        if (row_min > best) best = row_min;
    }

    return best;
}

// ---------- параллельный алгоритм с разными schedule ----------

double omp_maximin(const Matrix& a,
                   int rows, int cols,
                   Shape sh, int bw,
                   int threads,
                   SchedPolicy pol,
                   int chunk)
{
    double best = -std::numeric_limits<double>::infinity();

    omp_set_num_threads(threads);
    omp_sched_t kind;

    switch (pol) {
        case SchedPolicy::Static:  kind = omp_sched_static;  break;
        case SchedPolicy::Dynamic: kind = omp_sched_dynamic; break;
        case SchedPolicy::Guided:  kind = omp_sched_guided;  break;
    }

    omp_set_schedule(kind, chunk > 0 ? chunk : 0);

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(runtime) reduction(max : best)
    for (int i = 0; i < rows; ++i) {
        int j0, j1;
        cols_for_row(i, cols, sh, bw, j0, j1);
        if (j0 > j1) continue;

        double row_min = std::numeric_limits<double>::infinity();
        int base = i * cols;
        for (int j = j0; j <= j1; ++j) {
            double v = a[base + j];
            if (v < row_min) row_min = v;
        }
        if (row_min > best) best = row_min;
    }

    double t1 = omp_get_wtime();
    std::fprintf(stderr, "[omp %s] threads=%d time=%.6f s\n",
                 sched_name(pol).c_str(), threads, t1 - t0);
    return best;
}

// ---------- разбор аргументов ----------

struct Cmd {
    bool bench = false;
    int rows = 2000;
    int cols = 2000;
    int threads = omp_get_max_threads();
    string kind = "lower";   // dense / lower / upper / band
    int bw = 10;             // радиус для band
    string sched = "static"; // static / dynamic / guided
    int chunk = 0;
};

Cmd parse_cmd(int argc, char** argv) {
    Cmd c;
    for (int i = 1; i < argc; ++i) {
        string s = argv[i];
        auto need = [&](const char* flag) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << flag << "\n";
                std::exit(1);
            }
        };

        if (s == "--bench") {
            c.bench = true;
        } else if (s == "--rows") {
            need("--rows");
            c.rows = std::atoi(argv[++i]);
        } else if (s == "--cols") {
            need("--cols");
            c.cols = std::atoi(argv[++i]);
        } else if (s == "--threads") {
            need("--threads");
            c.threads = std::atoi(argv[++i]);
        } else if (s == "--kind") {
            need("--kind");
            c.kind = argv[++i];
        } else if (s == "--bw") {
            need("--bw");
            c.bw = std::atoi(argv[++i]);
        } else if (s == "--sched") {
            need("--sched");
            c.sched = argv[++i];
        } else if (s == "--chunk") {
            need("--chunk");
            c.chunk = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << s << "\n";
            std::exit(1);
        }
    }
    if (c.threads < 1) c.threads = 1;
    return c;
}

// ---------- одиночный запуск ----------

void run_single(const Cmd& cfg) {
    Shape sh = parse_shape(cfg.kind);
    SchedPolicy pol = parse_sched(cfg.sched);

    Matrix a = make_random(cfg.rows, cfg.cols, 123);

    double t0 = omp_get_wtime();
    double ref = seq_maximin(a, cfg.rows, cfg.cols, sh, cfg.bw);
    double t1 = omp_get_wtime();
    double t_seq = t1 - t0;

    double val = omp_maximin(a, cfg.rows, cfg.cols,
                             sh, cfg.bw,
                             cfg.threads, pol, cfg.chunk);
    double t2 = omp_get_wtime();
    double t_par = t2 - t1;

    if (std::fabs(ref - val) > 1e-9) {
        std::cerr << "Mismatch: seq=" << ref << " par=" << val << "\n";
    }

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6)
              << "kind=" << shape_name(sh)
              << ", rows=" << cfg.rows
              << ", cols=" << cfg.cols
              << ", sched=" << sched_name(pol)
              << ", threads=" << cfg.threads
              << ", y=" << val
              << ", Tseq=" << t_seq
              << ", Tpar=" << t_par
              << ", speedup=" << (t_par > 0 ? t_seq / t_par : 0.0)
              << "\n";
}

// ---------- режим бенчмарка: CSV ----------

void run_bench(const Cmd& cfg) {
    Shape sh = parse_shape(cfg.kind);

    std::cout << "matrix_type,bandwidth,nrows,ncols,schedule,threads,time_sec,speedup_vs_seq\n";

    int hw = omp_get_max_threads();
    vector<int> thread_list = {1, 2, 4, hw, hw + 1, 2 * hw};
    std::sort(thread_list.begin(), thread_list.end());
    thread_list.erase(std::unique(thread_list.begin(), thread_list.end()),
                      thread_list.end());

    vector<int> sizes = {1000, 10000, 15000, 20000};
    vector<SchedPolicy> policies = {
        SchedPolicy::Static,
        SchedPolicy::Dynamic,
        SchedPolicy::Guided
    };

    for (int n : sizes) {
        int rows = n;
        int cols = n;
        Matrix a = make_random(rows, cols, 555);

        // базовое последовательное время
        int Rseq = 3;
        double ref = 0.0;
        double sum_seq = 0.0;

        for (int r = 0; r < Rseq; ++r) {
            double t0 = omp_get_wtime();
            double cur = seq_maximin(a, rows, cols, sh, cfg.bw);
            double t1 = omp_get_wtime();

            if (r == 0) {
                ref = cur;
            } else if (std::fabs(ref - cur) > 1e-9) {
                std::cerr << "Seq mismatch at n=" << n << "\n";
                std::exit(2);
            }
            sum_seq += (t1 - t0);
        }
        double t_seq = sum_seq / Rseq;

        for (SchedPolicy pol : policies) {
            for (int th : thread_list) {
                // прогрев
                omp_maximin(a, rows, cols, sh, cfg.bw, th, pol, cfg.chunk);

                int R = 3;
                double tsum = 0.0;
                double last = 0.0;

                for (int r = 0; r < R; ++r) {
                    double s = omp_get_wtime();
                    last = omp_maximin(a, rows, cols, sh, cfg.bw, th, pol, cfg.chunk);
                    double e = omp_get_wtime();
                    if (std::fabs(ref - last) > 1e-9) {
                        std::cerr << "Par mismatch at n=" << n
                                  << ", threads=" << th
                                  << ", sched=" << sched_name(pol) << "\n";
                        std::exit(3);
                    }
                    tsum += (e - s);
                }

                double t_avg = tsum / R;
                double sp = t_seq / t_avg;

                std::cout.setf(std::ios::fixed);
                std::cout << std::setprecision(6)
                          << shape_name(sh) << ","
                          << cfg.bw << ","
                          << rows << ","
                          << cols << ","
                          << sched_name(pol) << ","
                          << th << ","
                          << t_avg << ","
                          << sp << "\n";
            }
        }
    }
}

// ---------- main ----------

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Cmd cfg = parse_cmd(argc, argv);
    if (cfg.bench) {
        run_bench(cfg);
    } else {
        run_single(cfg);
    }
    return 0;
}
