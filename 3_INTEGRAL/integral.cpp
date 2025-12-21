#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

using std::string;
using std::vector;

struct FunctionSpec {
    string id;
    std::function<double(double)> f;
    std::function<double(double,double)> exact;
};

FunctionSpec pick_function(const string& name) {
    if (name == "sin") {
        return {
            "sin",
            [](double x) { return std::sin(x); },
            [](double a, double b) { return -std::cos(b) + std::cos(a); }
        };
    }
    if (name == "exp") {
        return {
            "exp",
            [](double x) { return std::exp(x); },
            [](double a, double b) { return std::exp(b) - std::exp(a); }
        };
    }
    if (name == "x2") {
        return {
            "x2",
            [](double x) { return x * x; },
            [](double a, double b) { return (b*b*b - a*a*a) / 3.0; }
        };
    }
    return pick_function("sin");
}

enum class SampleMode { LEFT, MID, RIGHT };

string mode_to_string(SampleMode m) {
    switch (m) {
        case SampleMode::LEFT:  return "left";
        case SampleMode::MID:   return "mid";
        case SampleMode::RIGHT: return "right";
    }
    return "mid";
}

SampleMode parse_mode(const string& s) {
    if (s == "left")  return SampleMode::LEFT;
    if (s == "right") return SampleMode::RIGHT;
    return SampleMode::MID;
}

inline double sample_point(long long i, double a, double h, SampleMode m) {
    switch (m) {
        case SampleMode::LEFT:  return a + i * h;
        case SampleMode::MID:   return a + (i + 0.5) * h;
        case SampleMode::RIGHT: return a + (i + 1.0) * h;
    }
    return a;
}

double integrate_seq(const FunctionSpec& F,
                     double a, double b, long long N,
                     SampleMode mode)
{
    const double h = (b - a) / static_cast<double>(N);
    double acc = 0.0;
    for (long long i = 0; i < N; ++i) {
        acc += F.f(sample_point(i, a, h, mode));
    }
    return acc * h;
}

double integrate_omp_reduction(const FunctionSpec& F,
                               double a, double b, long long N,
                               int threads,
                               SampleMode mode)
{
    const double h = (b - a) / static_cast<double>(N);
    double acc = 0.0;

    omp_set_num_threads(threads);

    #pragma omp parallel for reduction(+:acc) schedule(static)
    for (long long i = 0; i < N; ++i) {
        acc += F.f(sample_point(i, a, h, mode));
    }

    return acc * h;
}

struct Cmd {
    bool bench = false;
    double a = 0.0;
    double b = M_PI;
    long long N = 5'000'000;
    int threads = omp_get_max_threads();
    string func = "sin";
    string rule = "mid";
};

Cmd parse_args(int argc, char** argv) {
    Cmd cfg;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        auto need_val = [&](const char* msg) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << msg << "\n";
                std::exit(1);
            }
        };
        if (arg == "--bench") {
            cfg.bench = true;
        } else if (arg == "--a") {
            need_val("--a");
            cfg.a = std::atof(argv[++i]);
        } else if (arg == "--b") {
            need_val("--b");
            cfg.b = std::atof(argv[++i]);
        } else if (arg == "--N") {
            need_val("--N");
            cfg.N = std::atoll(argv[++i]);
        } else if (arg == "--threads") {
            need_val("--threads");
            cfg.threads = std::atoi(argv[++i]);
        } else if (arg == "--func") {
            need_val("--func");
            cfg.func = argv[++i];
        } else if (arg == "--rule") {
            need_val("--rule");
            cfg.rule = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::exit(1);
        }
    }
    if (cfg.threads < 1) cfg.threads = 1;
    return cfg;
}

void run_single(const Cmd& cfg) {
    auto F = pick_function(cfg.func);
    auto mode = parse_mode(cfg.rule);

    double seq_val = integrate_seq(F, cfg.a, cfg.b, cfg.N, mode);
    double par_val = integrate_omp_reduction(F, cfg.a, cfg.b, cfg.N, cfg.threads, mode);

    double exact = F.exact ? F.exact(cfg.a, cfg.b) : par_val;
    double err_seq = std::fabs(seq_val - exact);
    double err_par = std::fabs(par_val - exact);

    std::cout.setf(std::ios::fixed);
    std::cout.precision(12);

    std::cout << "func=" << F.id
              << " rule=" << cfg.rule
              << " N=" << cfg.N
              << " threads=" << cfg.threads << "\n";
    std::cout << "seq=" << seq_val << " err_seq=" << err_seq << "\n";
    std::cout << "par=" << par_val << " err_par=" << err_par << "\n";
}

void run_bench(const Cmd& cfg) {
    auto F = pick_function(cfg.func);
    auto mode = parse_mode(cfg.rule);

    int hw = omp_get_max_threads();
    vector<int> thread_list = {1, 2, 4, 5, 8};
    vector<long long> sizes = {100'000, 1'000'000, 5'000'000, 20'000'000};

    std::cout << "func,rule,N,threads,time_sec,estimate,abs_err,speedup_vs_seq\n";

    for (long long N : sizes) {
        double t0 = omp_get_wtime();
        double base_val = integrate_seq(F, cfg.a, cfg.b, N, mode);
        double t1 = omp_get_wtime();
        double tseq = t1 - t0;

        for (int t : thread_list) {
            double tsum = 0.0;
            double last = 0.0;

            for (int r = 0; r < 3; ++r) {
                double s = omp_get_wtime();
                last = integrate_omp_reduction(F, cfg.a, cfg.b, N, t, mode);
                double e = omp_get_wtime();
                tsum += (e - s);
            }

            double avg = tsum / 3.0;
            double err = std::fabs(last - F.exact(cfg.a, cfg.b));
            double speed = tseq / avg;

            std::cout.setf(std::ios::fixed);
            std::cout.precision(6);
            std::cout << F.id << ','
                      << mode_to_string(mode) << ','
                      << N << ','
                      << t << ','
                      << avg << ','
                      << last << ','
                      << err << ','
                      << speed << '\n';
        }
    }
}

int main(int argc, char** argv) {
    Cmd cfg = parse_args(argc, argv);
    if (cfg.bench) {
        run_bench(cfg);
    } else {
        run_single(cfg);
    }
    return 0;
}
