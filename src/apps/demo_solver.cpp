#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cstring> // For strcmp

#include "lib/heuristic_utils.hpp"
#include "lib/instance.hpp"
#include "lib/hierarchical_tabu_search.hpp"
#include "lib/utils.hpp"

// ==========================================
// UTILITY STRUCTURES
// ==========================================

struct Result {
    std::string method;
    f64 cost;
    double time_ms;
    Tour best_tour;
};

struct HierarchicalBreakdown {
    double t_micro_ms;  // Phase 1: Intra-cluster optimization
    double t_macro_ms;  // Phase 2: Inter-cluster ordering
    double t_stitch_ms; // Phase 3a: Stitching
    double t_polish_ms; // Phase 3b: Global Polishing
    double t_total_ms;
};

// ==========================================
// HELPERS
// ==========================================

// Explicit return type for C++11 compliance
std::chrono::high_resolution_clock::time_point now() { 
    return std::chrono::high_resolution_clock::now(); 
}

double diff_ms(std::chrono::high_resolution_clock::time_point start, 
               std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void write_tour_to_file(const std::string& filename, const Tour& tour) {
    std::ofstream outfile(filename);
    if (!outfile.good()) {
        std::cerr << "Error: Could not write tour to " << filename << std::endl;
        return;
    }
    for (size_t i = 0; i < tour.size(); ++i) {
        outfile << tour[i] << (i == tour.size() - 1 ? "" : " ");
    }
    outfile << " " << tour[0] << "\n";
}

template <typename Func>
Result run_benchmark(const std::string& name, const Instance& instance, Func solver) {
    auto start = now();
    Tour t = solver();
    auto end = now();
    
    double ms = diff_ms(start, end);
    f64 cost = heuristic_utils::evaluate(t, instance.hole_costs());
    
    return Result{name, cost, ms, t};
}

// ==========================================
// MAIN
// ==========================================

int main(int argc, char** argv) {
    // Usage: ./solver <matrix.dat> <clusters.dat> <output_tour.txt>
    std::string path_coords = (argc > 1) ? argv[1] : "";
    std::string path_clusters = (argc > 2) ? argv[2] : "";
    std::string path_output = (argc > 3) ? argv[3] : "";
    
    // Optional flag for debugging text output
    bool print_tours = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--print-tours") == 0) print_tours = true;
    }

    if (path_coords.empty() || path_clusters.empty()) {
        std::cerr << "Usage: " << argv[0] << " <matrix_path> <cluster_path> [output_tour_path]" << std::endl;
        return 1;
    }

    std::cout << "=== Loading Instance ===" << std::endl;
    std::cout << "Coords:      " << path_coords << std::endl;
    std::cout << "Clusters:    " << path_clusters << std::endl;
    if (!path_output.empty()) {
        std::cout << "Output:      " << path_output << std::endl;
    }

    // Assumes Instance class handles parsing .dat files internally
    Instance instance(path_coords, path_clusters);
    const auto& costs = instance.hole_costs();

    // Prepare node list for Flat solver
    std::vector<u16> nodes(instance.n_holes());
    std::iota(nodes.begin(), nodes.end(), 0);

    std::vector<Result> leaderboard;

    // Common Configs
    heuristic_utils::MetaheuristicConfig intra_config; 
    intra_config.n_starts = 10;
    intra_config.max_iter = 500;
    intra_config.patience = 100;

    heuristic_utils::MetaheuristicConfig inter_config; 
    inter_config.n_starts = 10;
    inter_config.max_iter = 300;
    inter_config.patience = 100;
    
    size_t K_candidates = 8; 

    std::cout << "\n[2/4] Running Hierarchical (Sequential K=" << K_candidates << ")..." << std::endl;
    leaderboard.push_back(run_benchmark("Hierarchical (Seq)", instance, [&]() -> Tour {
        HierarchicalTabuSearch solver(instance);
        return solver.optimize(inter_config, intra_config, K_candidates);
    }));
    std::cout << "      -> Done. Cost: " << std::fixed << std::setprecision(4) << leaderboard.back().cost << std::endl;

    std::cout << "\n[3/4] Running Hierarchical (Parallel K=" << K_candidates << ")..." << std::endl;
    leaderboard.push_back(run_benchmark("Hierarchical (Par)", instance, [&]() -> Tour {
        HierarchicalTabuSearch solver(instance);
        return solver.optimize_parallel(inter_config, intra_config, K_candidates);
    }));
    std::cout << "      -> Done. Cost: " << std::fixed << std::setprecision(4) << leaderboard.back().cost << std::endl;

    std::cout << "\n[4/4] Running Detailed Breakdown..." << std::endl;
    HierarchicalBreakdown stats = {0, 0, 0, 0, 0};
    
    {
        HierarchicalTabuSearch solver(instance);
        auto t_start_total = now();

        // Phase 1: Micro
        auto t1 = now();
        solver.optimize_cluster_tours(intra_config);
        stats.t_micro_ms = diff_ms(t1, now());

        // Phase 2: Macro
        auto t2 = now();
        solver.compute_top_k_cluster_orderings(inter_config, K_candidates);
        stats.t_macro_ms = diff_ms(t2, now());

        // Phase 3: Stitch & Polish
        const auto& candidates = solver.get_cluster_orderings();
        f64 breakdown_best_cost = std::numeric_limits<f64>::max();

        for (const auto& ordering : candidates) {
            auto ts_start = now();
            Tour stitched = solver.stitch_cluster_tours(ordering);
            stats.t_stitch_ms += diff_ms(ts_start, now());

            auto tp_start = now();
            Tour polished = heuristic_utils::two_opt(stitched, costs, heuristic_utils::TwoOptMode::BestImprovement);
            stats.t_polish_ms += diff_ms(tp_start, now());

            f64 c = heuristic_utils::evaluate(polished, costs);
            if (c < breakdown_best_cost) breakdown_best_cost = c;
        }
        stats.t_total_ms = diff_ms(t_start_total, now());
    }

    // ==========================================
    // FINAL REPORT
    // ==========================================
    std::sort(leaderboard.begin(), leaderboard.end(), [](const Result& a, const Result& b) {
        return a.cost < b.cost;
    });

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                                FINAL LEADERBOARD                               " << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::left << std::setw(25) << "Method" 
              << std::setw(15) << "Cost" 
              << std::setw(15) << "Time (ms)" 
              << "Gap" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    f64 best_cost = leaderboard[0].cost;
    const Tour* best_tour_ptr = &leaderboard[0].best_tour;

    for (const auto& res : leaderboard) {
        f64 gap = (res.cost - best_cost) / best_cost * 100.0;
        std::cout << std::left << std::setw(25) << res.method 
                  << std::setw(15) << std::fixed << std::setprecision(4) << res.cost
                  << std::setw(15) << std::fixed << std::setprecision(4) << res.time_ms
                  << std::fixed << std::setprecision(4) << gap << "%" << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\n=== TIMING BREAKDOWN ===" << std::endl;
    std::cout << "1. Cluster Solving (Micro):   " << std::setw(10) << stats.t_micro_ms  << " ms" << std::endl;
    std::cout << "2. Ordering Search (Macro):   " << std::setw(10) << stats.t_macro_ms  << " ms" << std::endl;
    std::cout << "3. Stitching (Geometric):     " << std::setw(10) << stats.t_stitch_ms << " ms" << std::endl;
    std::cout << "4. Global Polish (2-Opt):     " << std::setw(10) << stats.t_polish_ms << " ms" << std::endl;
    std::cout << "TOTAL:                        " << std::setw(10) << stats.t_total_ms  << " ms" << std::endl;

    // ==========================================
    // SAVE OUTPUT
    // ==========================================
    if (!path_output.empty() && best_tour_ptr) {
        write_tour_to_file(path_output, *best_tour_ptr);
        std::cout << "\n[>] Best tour saved to: " << path_output << std::endl;
    }

    if (print_tours) {
        std::cout << "\n=== TOURS ===" << std::endl;
        utils::printv(*best_tour_ptr);
    }

    return 0;
}