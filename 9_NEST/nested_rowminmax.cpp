#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <algorithm> 

#include <omp.h>

using Real = double;
using Matrix = std::vector<Real>;

namespace {

Real pos_inf()  { return  std::numeric_limits<Real>::infinity(); }
Real neg_inf()  { return -std::numeric_limits<Real>::infinity(); }

Matrix make_random_matrix(int rows, int cols, std::uint64_t seed)
{
    Matrix data(static_cast<std::size_t>(rows) * cols);
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<Real> dist(-1e6, 1e6);

    for (auto &v : data)
        v = dist(gen);

    return data;
}

inline const Real* row_ptr(const Matrix& a, int row, int ncols)
{
    return &a[static_cast<std::size_t>(row) * ncols];
}

/// Последовательная версия: max_i min_j a[i,j]
Real ref_row_minmax(const Matrix& a, int nrows, int ncols)
{
    Real best = neg_inf();

    for (int i = 0; i < nrows; ++i) {
        Real row_min = pos_inf();
        const Real* r = row_ptr(a, i, ncols);
        for (int j = 0; j < ncols; ++j)
            row_min = std::min(row_min, r[j]);
        best = std::max(best, row_min);
    }

    return best;
}

/// Параллель по строкам (outer-only)
Real outer_parallel_row_minmax(const Matrix& a,
                               int nrows, int ncols,
                               int threads,
                               omp_sched_t sched,
                               int chunk)
{
    omp_set_dynamic(0);
    omp_set_schedule(sched, chunk);
    omp_set_num_threads(threads);

    Real best = neg_inf();

#pragma omp parallel for schedule(runtime) reduction(max: best)
    for (int i = 0; i < nrows; ++i) {
        const Real* r = row_ptr(a, i, ncols);
        Real row_min = pos_inf();
        for (int j = 0; j < ncols; ++j)
            row_min = std::min(row_min, r[j]);
        best = std::max(best, row_min);
    }

    return best;
}

/// Вложенный параллелизм: строки × столбцы
Real nested_parallel_row_minmax(const Matrix& a,
                                int nrows, int ncols,
                                int outer_threads, int inner_threads,
                                omp_sched_t outer_sched, int outer_chunk,
                                omp_sched_t inner_sched, int inner_chunk)
{
    omp_set_dynamic(0);
    omp_set_max_active_levels(2);
    omp_set_nested(1);

    Real best = neg_inf();

    omp_set_schedule(outer_sched, outer_chunk);

#pragma omp parallel num_threads(outer_threads) reduction(max: best)
    {
#pragma omp for schedule(runtime)
        for (int i = 0; i < nrows; ++i) {
            const Real* r = row_ptr(a, i, ncols);
            Real row_min = pos_inf();

            omp_set_schedule(inner_sched, inner_chunk);
#pragma omp parallel for num_threads(inner_threads) schedule(runtime) reduction(min: row_min)
            for (int j = 0; j < ncols; ++j) {
                row_min = std::min(row_min, r[j]);
            }

            best = std::max(best, row_min);
        }
    }

    return best;
}

omp_sched_t to_sched(const std::string& s)
{
    if (s == "dynamic") return omp_sched_dynamic;
    if (s == "guided")  return omp_sched_guided;
    return omp_sched_static;
}

const char* from_sched(omp_sched_t s)
{
    switch (s) {
        case omp_sched_static:  return "static";
        case omp_sched_dynamic: return "dynamic";
        case omp_sched_guided:  return "guided";
        default:                return "unknown";
    }
}

struct CmdLine {
    bool   do_bench      = false;
    int    nrows         = 2000;
    int    ncols         = 2000;
    std::string mode     = "outer";   // outer | nested

    // outer-only
    int    threads       = 4;
    std::string sched    = "static";
    int    chunk         = 1;

    // nested
    int    outer_threads = 2;
    int    inner_threads = 2;
    std::string outer_sched = "static";
    int    outer_chunk      = 1;
    std::string inner_sched = "static";
    int    inner_chunk      = 64;
};

CmdLine parse_cmd(int argc, char** argv)
{
    CmdLine cfg;

    auto has_arg = [&](const std::string& flag, int i) {
        return (flag == argv[i]) && (i + 1 < argc);
    };

    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];

        if (f == "--bench") {
            cfg.do_bench = true;
        } else if (has_arg("--nrows", i)) {
            cfg.nrows = std::stoi(argv[++i]);
        } else if (has_arg("--ncols", i)) {
            cfg.ncols = std::stoi(argv[++i]);
        } else if (has_arg("--mode", i)) {
            cfg.mode = argv[++i];
        } else if (has_arg("--threads", i)) {
            cfg.threads = std::stoi(argv[++i]);
        } else if (has_arg("--sched", i)) {
            cfg.sched = argv[++i];
        } else if (has_arg("--chunk", i)) {
            cfg.chunk = std::stoi(argv[++i]);
        } else if (has_arg("--outer_threads", i)) {
            cfg.outer_threads = std::stoi(argv[++i]);
        } else if (has_arg("--inner_threads", i)) {
            cfg.inner_threads = std::stoi(argv[++i]);
        } else if (has_arg("--outer_sched", i)) {
            cfg.outer_sched = argv[++i];
        } else if (has_arg("--outer_chunk", i)) {
            cfg.outer_chunk = std::stoi(argv[++i]);
        } else if (has_arg("--inner_sched", i)) {
            cfg.inner_sched = argv[++i];
        } else if (has_arg("--inner_chunk", i)) {
            cfg.inner_chunk = std::stoi(argv[++i]);
        }
    }

    return cfg;
}

/// Один прогон (mode=outer|nested), вывод в stdout в человекочитаемом виде
int run_single(const CmdLine& cfg)
{
    Matrix A = make_random_matrix(cfg.nrows, cfg.ncols, 123);

    double t0 = omp_get_wtime();
    Real ref  = ref_row_minmax(A, cfg.nrows, cfg.ncols);
    double t1 = omp_get_wtime();
    double t_seq = t1 - t0;

    Real res;
    double t_par0, t_par1;

    if (cfg.mode == "outer") {
        omp_sched_t s = to_sched(cfg.sched);
        t_par0 = omp_get_wtime();
        res = outer_parallel_row_minmax(A, cfg.nrows, cfg.ncols,
                                        cfg.threads, s, cfg.chunk);
        t_par1 = omp_get_wtime();
    } else {
        omp_sched_t so = to_sched(cfg.outer_sched);
        omp_sched_t si = to_sched(cfg.inner_sched);
        t_par0 = omp_get_wtime();
        res = nested_parallel_row_minmax(
                A, cfg.nrows, cfg.ncols,
                cfg.outer_threads, cfg.inner_threads,
                so, cfg.outer_chunk,
                si, cfg.inner_chunk);
        t_par1 = omp_get_wtime();
    }

    double t_par = t_par1 - t_par0;

    if (std::fabs(res - ref) > 1e-9) {
        std::cerr << "Result mismatch: ref=" << ref
                  << " got=" << res << "\n";
        return 2;
    }

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6)
              << "nrows=" << cfg.nrows
              << " ncols=" << cfg.ncols
              << " seq_time=" << t_seq
              << " par_time=" << t_par
              << " speedup=" << (t_seq / t_par)
              << " mode=" << cfg.mode
              << "\n";

    return 0;
}

/// Полный бенчмарк со строками CSV (эквивалентный исходнику)
int run_benchmark()
{
    std::cout << "mode,nrows,ncols,outer_threads,inner_threads,"
              << "outer_sched,outer_chunk,inner_sched,inner_chunk,"
              << "time_sec,speedup_vs_seq\n";

    const int hw_threads = omp_get_max_threads();
    const std::vector<int> sizes{500, 1000, 2000, 4000};
    const std::vector<std::string> sched_names{"static", "dynamic", "guided"};
    const std::vector<int> totals{1, 2, 4, std::min(8, hw_threads), hw_threads};
    const int repeats = 3;

    for (int n : sizes) {
        const int rows = n;
        const int cols = n;
        Matrix M = make_random_matrix(rows, cols, 777);

        // базовое seq-время
        double seq_acc = 0.0;
        for (int r = 0; r < repeats; ++r) {
            double t0 = omp_get_wtime();
            volatile Real tmp = ref_row_minmax(M, rows, cols);
            double t1 = omp_get_wtime();
            (void)tmp;
            seq_acc += (t1 - t0);
        }
        double seq_avg = seq_acc / repeats;

        // outer-only
        for (int total : totals) {
            for (const auto& sname : sched_names) {
                omp_sched_t sch = to_sched(sname);

                // прогрев
                (void)outer_parallel_row_minmax(M, rows, cols, total, sch, 1);

                double acc = 0.0;
                for (int r = 0; r < repeats; ++r) {
                    double t0 = omp_get_wtime();
                    Real v = outer_parallel_row_minmax(M, rows, cols, total, sch, 1);
                    double t1 = omp_get_wtime();
                    (void)v;
                    acc += (t1 - t0);
                }
                double avg = acc / repeats;

                std::cout << "outer,"
                          << rows << "," << cols << ","
                          << total << "," << 1 << ","
                          << sname << "," << 1 << ","
                          << "none," << 0 << ","
                          << std::fixed << std::setprecision(6)
                          << avg << ","
                          << (seq_avg / avg) << "\n";
            }
        }

        // nested
        for (int total : totals) {
            // подбираем несколько разбиений total = outer * inner
            std::vector<std::pair<int,int>> factorizations;
            for (int o = 1; o <= total; ++o) {
                if (total % o == 0)
                    factorizations.emplace_back(o, total / o);
            }

            std::vector<std::pair<int,int>> combos;
            if (!factorizations.empty()) {
                // (1, total)
                combos.emplace_back(1, total);

                // близкое к sqrt(total)
                int best_idx = 0;
                double best_dist = std::numeric_limits<double>::infinity();
                for (int i = 0; i < (int)factorizations.size(); ++i) {
                    double d = std::fabs(std::sqrt((double)total) -
                                         factorizations[i].first);
                    if (d < best_dist) {
                        best_dist = d;
                        best_idx = i;
                    }
                }
                combos.push_back(factorizations[best_idx]);

                // (total, 1)
                combos.emplace_back(total, 1);
            }

            std::sort(combos.begin(), combos.end());
            combos.erase(std::unique(combos.begin(), combos.end()), combos.end());

            for (auto [outer_t, inner_t] : combos) {
                for (const auto& o_name : sched_names) {
                    for (const auto& i_name : sched_names) {
                        omp_sched_t os = to_sched(o_name);
                        omp_sched_t is = to_sched(i_name);

                        // прогрев
                        (void)nested_parallel_row_minmax(
                            M, rows, cols,
                            outer_t, inner_t,
                            os, 1,
                            is, 256);

                        double acc = 0.0;
                        for (int r = 0; r < repeats; ++r) {
                            double t0 = omp_get_wtime();
                            Real v = nested_parallel_row_minmax(
                                M, rows, cols,
                                outer_t, inner_t,
                                os, 1,
                                is, 256);
                            double t1 = omp_get_wtime();
                            (void)v;
                            acc += (t1 - t0);
                        }
                        double avg = acc / repeats;

                        std::cout << "nested,"
                                  << rows << "," << cols << ","
                                  << outer_t << "," << inner_t << ","
                                  << o_name << "," << 1 << ","
                                  << i_name << "," << 64 << ","
                                  << std::fixed << std::setprecision(6)
                                  << avg << ","
                                  << (seq_avg / avg) << "\n";
                    }
                }
            }
        }
    }

    return 0;
}

} 

int main(int argc, char** argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    CmdLine cfg = parse_cmd(argc, argv);
    if (!cfg.do_bench)
        return run_single(cfg);

    return run_benchmark();
}
