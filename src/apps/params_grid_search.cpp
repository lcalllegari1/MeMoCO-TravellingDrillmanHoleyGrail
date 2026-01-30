#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <functional>
#include <cfloat>
#include <map>
#include <sstream>

// Include your library headers
#include "lib/instance.hpp"
#include "lib/held_karp_lb.hpp"
#include "lib/heuristic_utils.hpp"
#include "lib/hierarchical_tabu_search.hpp"
#include "lib/types.hpp"

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

struct GridResult {
    std::string method_name;
    int n_starts;
    std::string tenure_desc; 
    int tenure_val;
    std::string iter_desc;   
    int iter_val;
    double threshold;        
    double objective;
    double time_seconds;
    double gap_percent;
    double speedup;
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

void convert_data(const heuristic_utils::InstanceData& src, 
                  std::vector<u16>& nodes, 
                  Matrix<f64>& costs) {
    nodes.resize(src.n_nodes);
    std::iota(nodes.begin(), nodes.end(), 0); 
    costs = Matrix<f64>(src.cost_matrix, src.n_nodes);
}

// -----------------------------------------------------------------------------
// Parameter Generators
// -----------------------------------------------------------------------------

std::vector<int> get_n_starts_grid(int n) {
    std::vector<int> s = {5, 10, 15, 20, n/10};
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    return s;
}

std::vector<std::pair<std::string, int>> get_tenure_grid(int n) {
    std::vector<std::pair<std::string, int>> t;
    // Order matters for "middle" selection logic later
    if (n >= 4) t.push_back({"N/4", n / 4});
    if (n >= 8) t.push_back({"N/8", n / 8});
    if (n >= 10) t.push_back({"N/10", n / 10});
    t.push_back({"Sqrt(N)", (int)std::sqrt(n)});
    return t;
}

std::vector<std::pair<std::string, int>> get_iter_grid(int n) {
    return {
        {"5*N",  5 * n},
        {"10*N", 10 * n},
        {"20*N", 20 * n}
    };
}

std::vector<double> get_threshold_grid() {
    return {1.15, 1.20, 1.25, 1.30, 1.40};
}

// -----------------------------------------------------------------------------
// Formatting
// -----------------------------------------------------------------------------

void print_separator() {
    std::cout << std::string(132, '-') << "\n";
}

void print_header() {
    print_separator();
    std::cout << std::left 
              << std::setw(20) << "Method"
              << std::right 
              << std::setw(8)  << "Starts"
              << std::setw(10) << "Tenure"
              << std::setw(8)  << "T.Val"
              << std::setw(10) << "Iters"
              << std::setw(10) << "I.Val"
              << std::setw(8)  << "Thresh"
              << std::setw(12) << "Time(s)"
              << std::setw(12) << "Speedup" 
              << std::setw(14) << "Obj"
              << std::setw(10) << "Gap(%)" 
              << "\n";
    print_separator();
}

void print_row(const GridResult& r) {
    std::cout << std::left 
              << std::setw(20) << r.method_name
              << std::right 
              << std::setw(8)  << r.n_starts
              << std::setw(10) << r.tenure_desc
              << std::setw(8)  << r.tenure_val
              << std::setw(10) << r.iter_desc
              << std::setw(10) << r.iter_val
              << std::setw(8)  << (r.threshold > 0 ? std::to_string(r.threshold).substr(0,4) : "-")
              << std::setw(12) << std::fixed << std::setprecision(4) << r.time_seconds
              << std::setw(12) << std::fixed << std::setprecision(2) << r.speedup 
              << std::setw(14) << std::fixed << std::setprecision(4) << r.objective
              << std::setw(10) << std::fixed << std::setprecision(2) << r.gap_percent 
              << "\n";
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <dat_file> <clusters_file> [--verbose <lvl>] [--average <runs>]\n";
        return 1;
    }

    std::string dat_file = argv[1];
    std::string clusters_file = argv[2];
    int average_runs = 1;
    bool average_mode = false;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--average") {
            average_mode = true;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                try {
                    average_runs = std::stoi(argv[i + 1]);
                    if (average_runs <= 0) average_runs = 1;
                    ++i;
                } catch (...) { average_runs = 1; }
            }
        }
    }

    try {
        auto hk_data = heuristic_utils::read_dat_file(dat_file);
        Instance instance(dat_file, clusters_file); 
        std::vector<u16> nodes;
        Matrix<f64> costs;
        convert_data(hk_data, nodes, costs);

        int N = instance.n_holes();
        double hk_lb = heuristic_utils::compute_lower_bound(hk_data);

        auto starts_grid = get_n_starts_grid(N);
        auto tenure_grid = get_tenure_grid(N);
        auto iter_grid   = get_iter_grid(N);
        auto thresh_grid = get_threshold_grid();

        int runs = average_mode ? average_runs : 1;

        std::cout << "Calculating baseline speed (Seq Pruning)... ";
        
        heuristic_utils::MetaheuristicConfig base_cfg;
        // Pick median starts (e.g. 5 or 10)
        base_cfg.n_starts = starts_grid[starts_grid.size() / 2];
        // Pick Sqrt(N) tenure (usually the last one)
        base_cfg.tenure = tenure_grid.back().second; 
        // Pick 10*N iters (middle)
        base_cfg.max_iter = iter_grid[1].second;
        // Pick 1.25 threshold (middle)
        base_cfg.pruning_threshold = 1.30;

        double baseline_time = 0.0;
        {
            auto t1 = std::chrono::high_resolution_clock::now();
            // Just run 1 time for baseline estimation to be quick
            heuristic_utils::sequential_pruning_multi_start_metaheuristic(nodes, costs, base_cfg);
            auto t2 = std::chrono::high_resolution_clock::now();
            baseline_time = std::chrono::duration<double>(t2 - t1).count();
        }
        if (baseline_time < 1e-9) baseline_time = 1e-9; // Safety

        std::cout << "Done (" << std::fixed << std::setprecision(4) << baseline_time << "s)\n";
        std::cout << "Baseline Config: Starts=" << base_cfg.n_starts 
                  << ", Tenure=" << base_cfg.tenure 
                  << ", Iter=" << base_cfg.max_iter 
                  << ", Thresh=" << base_cfg.pruning_threshold << "\n\n";

        std::map<std::string, GridResult> best_results;
        int line_counter = 0;

        std::cout << "Grid Search | Instance: " << instance.instance_name() 
                  << " (N=" << N << ") | Baseline LB: " << hk_lb << "\n";
        print_header();

        auto execute_config = [&](std::string method, 
                                  heuristic_utils::MetaheuristicConfig cfg, 
                                  std::string t_desc, std::string i_desc, 
                                  std::function<Tour()> func) {
            
            double total_time = 0.0;
            double total_obj = 0.0;

            for (int r = 0; r < runs; ++r) {
                auto t1 = std::chrono::high_resolution_clock::now();
                auto tour = func();
                auto t2 = std::chrono::high_resolution_clock::now();
                
                total_time += std::chrono::duration<double>(t2 - t1).count();
                total_obj += heuristic_utils::evaluate(tour, costs);
            }

            GridResult res;
            res.method_name = method;
            res.n_starts = cfg.n_starts;
            res.tenure_desc = t_desc;
            res.tenure_val = cfg.tenure;
            res.iter_desc = i_desc;
            res.iter_val = cfg.max_iter;
            res.threshold = (method == "Flat_Seq_Pruning") ? cfg.pruning_threshold : 0.0;
            res.time_seconds = total_time / runs;
            res.objective = total_obj / runs;
            res.gap_percent = (res.objective - hk_lb) / hk_lb * 100.0;
            
            // Speedup Calculation
            if (res.time_seconds > 1e-9) {
                res.speedup = baseline_time / res.time_seconds;
            } else {
                res.speedup = 0.0;
            }

            // Print Row
            print_row(res);
            line_counter++;

            // Re-print header every 20 lines
            if (line_counter % 20 == 0) {
                print_header();
            }

            // Update Best
            if (best_results.find(method) == best_results.end() || 
                res.objective < best_results[method].objective) {
                best_results[method] = res;
            }
        };

        for (int s : starts_grid) {
            for (const auto& t : tenure_grid) {
                for (const auto& it : iter_grid) {
                    for (double th : thresh_grid) {
                        heuristic_utils::MetaheuristicConfig cfg;
                        cfg.n_starts = s;
                        cfg.tenure = t.second;
                        cfg.max_iter = it.second;
                        cfg.pruning_threshold = th;
                        
                        execute_config("Flat_Seq_Pruning", cfg, t.first, it.first, [&]() {
                            return heuristic_utils::sequential_pruning_multi_start_metaheuristic(nodes, costs, cfg);
                        });
                    }
                }
            }
        }

        for (int s : starts_grid) {
            for (const auto& t : tenure_grid) {
                for (const auto& it : iter_grid) {
                    heuristic_utils::MetaheuristicConfig cfg;
                    cfg.n_starts = s;
                    cfg.tenure = t.second;
                    cfg.max_iter = it.second;
                    
                    execute_config("Flat_Par_Funnel", cfg, t.first, it.first, [&]() {
                        return heuristic_utils::parallel_funneling_multi_start_metaheuristic(nodes, costs, cfg);
                    });
                }
            }
        }

        HierarchicalTabuSearch hts(instance);
        for (int s : starts_grid) {
            for (const auto& t : tenure_grid) {
                for (const auto& it : iter_grid) {
                    heuristic_utils::MetaheuristicConfig cfg;
                    cfg.n_starts = s;
                    cfg.tenure = t.second;
                    cfg.max_iter = it.second;

                    execute_config("Hierarchical_Seq", cfg, t.first, it.first, [&]() {
                        return hts.optimize(cfg, cfg);
                    });

                    execute_config("Hierarchical_Par", cfg, t.first, it.first, [&]() {
                        return hts.optimize_parallel(cfg, cfg);
                    });
                }
            }
        }

        std::cout << "\n\n";
        std::cout << "====================================================================================================================================\n";
        std::cout << "                                                 BEST CONFIGURATION PER METHOD                                                      \n";
        std::cout << "====================================================================================================================================\n";
        print_header();
        for (const auto& pair : best_results) {
            print_row(pair.second);
        }
        print_separator();

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}