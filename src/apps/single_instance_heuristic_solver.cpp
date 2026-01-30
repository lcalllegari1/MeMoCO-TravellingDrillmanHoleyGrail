#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <cfloat>
#include <set>

// Include your library headers
#include "lib/instance.hpp"
#include "lib/held_karp_lb.hpp"
#include "lib/heuristic_utils.hpp"
#include "lib/hierarchical_tabu_search.hpp"
#include "lib/types.hpp"

// Structure to hold results for table printing
struct MethodResult {
    std::string name;
    double objective;
    double time_seconds;
    std::vector<u16> tour;
};

// Helper: Convert Held-Karp InstanceData to Heuristic structures
void convert_data(const heuristic_utils::InstanceData& src, 
                  std::vector<u16>& nodes, 
                  Matrix<f64>& costs) {
    nodes.resize(src.n_nodes);
    std::iota(nodes.begin(), nodes.end(), 0); // Fill 0, 1, ..., N-1
    costs = Matrix<f64>(src.cost_matrix, src.n_nodes);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <dat_file> <clusters_file> [--verbose <0|1|2>] [--average <num_runs>] [--print-tours]\n";
        return 1;
    }

    std::string dat_file = argv[1];
    std::string clusters_file = argv[2];
    
    int verbose_level = 1;
    int average_runs = 1;
    bool average_mode = false;
    bool print_tours = false;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose") {
            if (i + 1 < argc) {
                try {
                    verbose_level = std::stoi(argv[i + 1]);
                    if (verbose_level < 0 || verbose_level > 2) verbose_level = 1;
                    ++i;
                } catch (...) { verbose_level = 1; }
            }
        } else if (arg == "--average") {
            average_mode = true;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                try {
                    average_runs = std::stoi(argv[i + 1]);
                    if (average_runs <= 0) average_runs = 5;
                    ++i;
                } catch (...) { average_runs = 5; }
            } else {
                average_runs = 5;
            }
        } else if (arg == "--print-tours") {
            print_tours = true;
        }
    }

    try {
        if (verbose_level >= 1) {
            std::cout << "Loading instance data from: " << dat_file << "\n";
            std::cout << "Loading cluster data from:  " << clusters_file << "\n";
        }

        // Load for Held-Karp
        auto hk_data = heuristic_utils::read_dat_file(dat_file);
        
        // Load for Heuristics
        Instance instance(dat_file, clusters_file); 

        // Prepare raw structures for heuristics
        std::vector<u16> nodes;
        Matrix<f64> costs;
        convert_data(hk_data, nodes, costs);

        double hk_lb_val = 0.0;
        double hk_time = 0.0;
        int runs = average_mode ? average_runs : 1;

        if (verbose_level >= 1) std::cout << "Computing Held-Karp Lower Bound...\n";
        {
            double total_hk_time = 0.0;
            for(int r = 0; r < runs; ++r) {
                 auto t1 = std::chrono::high_resolution_clock::now();
                 hk_lb_val = heuristic_utils::compute_lower_bound(hk_data);
                 auto t2 = std::chrono::high_resolution_clock::now();
                 total_hk_time += std::chrono::duration<double>(t2 - t1).count();
            }
            hk_time = total_hk_time / runs;
        }

        auto n_nodes = nodes.size();

        heuristic_utils::MetaheuristicConfig config;
        config.max_iter = 10 * n_nodes;
        config.tenure = (size_t)std::sqrt(n_nodes);
        config.patience = n_nodes;
        config.use_stagnation_recovery = true;
        
        std::vector<MethodResult> results;

        auto run_benchmark = [&](std::string name, std::function<Tour()> func) {
            double total_time = 0.0;
            double total_obj = 0.0;
            Tour best_tour;
            double best_obj_run = DBL_MAX;

            for (int r = 0; r < runs; ++r) {
                auto t1 = std::chrono::high_resolution_clock::now();
                auto tour = func();
                auto t2 = std::chrono::high_resolution_clock::now();
                
                double duration = std::chrono::duration<double>(t2 - t1).count();
                double obj = heuristic_utils::evaluate(tour, costs);

                total_time += duration;
                total_obj += obj;

                if (obj < best_obj_run) {
                    best_obj_run = obj;
                    best_tour = tour;
                }
            }

            MethodResult res;
            res.name = name;
            res.objective = total_obj / runs;
            res.time_seconds = total_time / runs;
            res.tour = best_tour;
            results.push_back(res);
            
            if (verbose_level >= 2) {
                std::cout << "  Finished " << name << ": " << res.objective << " in " << res.time_seconds << "s\n";
            }
        };

        if (verbose_level >= 1) std::cout << "Running heuristics (" << runs << " run(s) each)...\n";

        run_benchmark("Nearest Neighbour", [&]() {
            return heuristic_utils::nearest_neighbour(nodes, 0, costs);
        });

        run_benchmark("NN + 2-opt", [&]() {
            auto nn_tour = heuristic_utils::nearest_neighbour(nodes, 0, costs);
            return heuristic_utils::two_opt(nn_tour, costs);
        });

        const std::string baseline_name = "Flat Tabu (Seq Pruning)";
        run_benchmark(baseline_name, [&]() {
            return heuristic_utils::sequential_pruning_multi_start_metaheuristic(nodes, costs, config);
        });

        run_benchmark("Flat Tabu (Par Funnel)", [&]() {
            return heuristic_utils::parallel_funneling_multi_start_metaheuristic(nodes, costs, config);
        });

        HierarchicalTabuSearch hts(instance);
        run_benchmark("Hierarchical TS (Seq)", [&]() {
            return hts.optimize(config, config); 
        });

        run_benchmark("Hierarchical TS (Par)", [&]() {
            return hts.optimize_parallel(config, config);
        });

        double baseline_time = 0.0;
        double baseline_obj = 0.0;
        for(const auto& res : results) {
            if (res.name == baseline_name) {
                baseline_time = res.time_seconds;
                baseline_obj  = res.objective;
                break;
            }
        }

        std::cout << "\n=== Comparative Results (Baseline: Held-Karp LB) ===\n";
        std::cout << "Instance: " << instance.instance_name() << " (N=" << instance.n_holes() << ")\n\n";

        std::cout << std::left << std::setw(30) << "Method"
                  << std::right << std::setw(15) << "Objective"
                  << std::setw(15) << "Time (s)"
                  << std::setw(15) << "Gap (%)"
                  << std::setw(15) << "Speedup*"
                  << "\n";
        std::cout << std::string(90, '-') << "\n";

        // Print Held-Karp Row
        {
             double speedup = (hk_time > 1e-9) ? (baseline_time / hk_time) : 0.0;
             std::cout << std::left << std::setw(30) << "Held-Karp LB"
                  << std::right << std::setw(15) << std::fixed << std::setprecision(4) << hk_lb_val
                  << std::setw(15) << hk_time
                  << std::setw(15) << 0.0000
                  << std::setw(15) << speedup
                  << "\n";
        }

        // Print Heuristic Rows
        for (const auto& res : results) {
            double gap = (res.objective - hk_lb_val) / hk_lb_val * 100.0;
            double speedup = (res.time_seconds > 1e-9) ? (baseline_time / res.time_seconds) : 0.0;

            std::cout << std::left << std::setw(30) << res.name
                      << std::right << std::setw(15) << std::fixed << std::setprecision(4) << res.objective
                      << std::setw(15) << res.time_seconds
                      << std::setw(15) << gap
                      << std::setw(15) << speedup
                      << "\n";
        }
        std::cout << "* Speedup relative to " << baseline_name << "\n";

        std::cout << "\n=== Tabu Search Comparison (Baseline: " << baseline_name << ") ===\n\n";
        std::cout << std::left << std::setw(30) << "Method"
                  << std::right << std::setw(15) << "Objective"
                  << std::setw(15) << "Time (s)"
                  << std::setw(15) << "Gap (%)"
                  << std::setw(15) << "Speedup"
                  << "\n";
        std::cout << std::string(90, '-') << "\n";

        // Define which methods belong to the "Tabu" category
        std::set<std::string> tabu_methods = {
            "Flat Tabu (Seq Pruning)",
            "Flat Tabu (Par Funnel)",
            "Hierarchical TS (Seq)",
            "Hierarchical TS (Par)"
        };

        for (const auto& res : results) {
            // Filter: Only print if it's in our tabu list
            if (tabu_methods.find(res.name) != tabu_methods.end()) {
                
                // Gap relative to Sequential Pruning
                double gap = (res.objective - baseline_obj) / baseline_obj * 100.0;
                
                // Speedup relative to Sequential Pruning
                double speedup = (res.time_seconds > 1e-9) ? (baseline_time / res.time_seconds) : 0.0;

                std::cout << std::left << std::setw(30) << res.name
                          << std::right << std::setw(15) << std::fixed << std::setprecision(4) << res.objective
                          << std::setw(15) << res.time_seconds
                          << std::setw(15) << gap
                          << std::setw(15) << speedup
                          << "\n";
            }
        }

        if (print_tours) {
            std::cout << "\n=== Tour Sequences ===\n";
            for (const auto& res : results) {
                std::cout << res.name << ": [";
                for (size_t i = 0; i < res.tour.size(); ++i) {
                    std::cout << res.tour[i] << (i < res.tour.size() - 1 ? ", " : "");
                }
                std::cout << "]\n";
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}