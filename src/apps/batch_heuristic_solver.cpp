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
#include <fstream>
#include <map>
#include <set>
#include <dirent.h>
#include <sys/types.h>

#include "lib/instance.hpp"
#include "lib/held_karp_lb.hpp"
#include "lib/heuristic_utils.hpp"
#include "lib/hierarchical_tabu_search.hpp"
#include "lib/types.hpp"

// =============================================================================
// GLOBAL CONFIGURATION
// =============================================================================

// Config for Flat Methods (Seq Pruning & Par Funneling)
heuristic_utils::MetaheuristicConfig get_flat_config(int N) {
    heuristic_utils::MetaheuristicConfig cfg;
    cfg.n_starts = 100;
    cfg.tenure = (size_t)(N/4);
    cfg.max_iter = 5 * N;
    cfg.pruning_threshold = 1.30;
    return cfg;
}

// Config for Hierarchical Methods (Seq & Par)
// Applied to both inter-cluster and intra-cluster phases for simplicity here
heuristic_utils::MetaheuristicConfig get_hierarchical_config(int N) {
    heuristic_utils::MetaheuristicConfig cfg;
    cfg.n_starts = 10;
    cfg.tenure =  (size_t)(N/4);
    cfg.max_iter = 5 * N;
    cfg.pruning_threshold = 1.30;
    return cfg;
}

// =============================================================================
// DATA STRUCTURES
// =============================================================================

struct InstanceFiles {
    std::string name; // "n50_01"
    std::string dat_path;
    std::string clu_path;
};

struct Result {
    double obj;
    double time;
    double gap;    // vs HK
    double speedup; // vs Seq Pruning
};

struct InstanceResult {
    std::string name;
    int N;
    double hk_lb;
    
    // Store results by method ID for easy access
    // 0: Flat Seq, 1: Flat Par, 2: Hier Seq, 3: Hier Par
    std::vector<Result> methods; 
};

// =============================================================================
// UTILS
// =============================================================================

// Helper: Convert Held-Karp InstanceData to Heuristic structures
void convert_data(const heuristic_utils::InstanceData& src, 
                  std::vector<u16>& nodes, 
                  Matrix<f64>& costs) {
    nodes.resize(src.n_nodes);
    std::iota(nodes.begin(), nodes.end(), 0); 
    costs = Matrix<f64>(src.cost_matrix, src.n_nodes);
}

// Check if string ends with suffix
bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Scan directory for .dat files and return map {name -> path}
std::map<std::string, std::string> scan_folder(const std::string& path, const std::string& ext) {
    std::map<std::string, std::string> files;
    DIR* dir = opendir(path.c_str());
    if (!dir) return files;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string fname = entry->d_name;
        if (fname == "." || fname == "..") continue;
        
        if (ends_with(fname, ext)) {
            // Key is filename without extension
            std::string key = fname.substr(0, fname.length() - ext.length());
            files[key] = path + "/" + fname; // Simple path concatenation
        }
    }
    closedir(dir);
    return files;
}

// =============================================================================
// MAIN EXECUTION LOGIC
// =============================================================================

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <dat_folder> <clusters_folder> [--output <file>] [--average <runs>]\n";
        return 1;
    }

    std::string dat_folder = argv[1];
    std::string clu_folder = argv[2];
    std::string output_file = "results.dat";
    int runs = 1;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output") {
            if (i + 1 < argc) output_file = argv[++i];
        } else if (arg == "--average") {
            if (i + 1 < argc) runs = std::stoi(argv[++i]);
            if (runs < 1) runs = 1;
        }
    }

    auto dat_files = scan_folder(dat_folder, ".dat");
    auto clu_files = scan_folder(clu_folder, ".clusters");
    std::vector<InstanceFiles> queue;

    for (auto const& pair : dat_files) {
        std::string name = pair.first;
        if (clu_files.count(name)) {
            queue.push_back({name, pair.second, clu_files[name]});
        } else {
            std::cerr << "Warning: No .clusters match for " << name << ". Skipping.\n";
        }
    }
    
    // Sort for consistent output order (e.g. n10, n20...)
    std::sort(queue.begin(), queue.end(), [](const InstanceFiles& a, const InstanceFiles& b){
        return a.name < b.name;
    });

    if (queue.empty()) {
        std::cerr << "Error: No matching instances found.\n";
        return 1;
    }

    std::cout << "Found " << queue.size() << " matching instances. Starting benchmark...\n\n";

    std::vector<InstanceResult> all_results;
    std::ofstream log_stream(output_file);
    if (!log_stream.is_open()) {
        std::cerr << "Error: Cannot open output file " << output_file << "\n";
        return 1;
    }

    for (const auto& inst_files : queue) {
        std::cout << "Processing " << inst_files.name << "... ";
        std::cout.flush();

        // Load
        auto hk_data = heuristic_utils::read_dat_file(inst_files.dat_path);
        Instance instance(inst_files.dat_path, inst_files.clu_path);
        std::vector<u16> nodes;
        Matrix<f64> costs;
        convert_data(hk_data, nodes, costs);

        int N = instance.n_holes();
        
        // Compute LB
        double hk_lb = heuristic_utils::compute_lower_bound(hk_data);
        
        // Define Configs for this N
        auto flat_cfg = get_flat_config(N);
        auto hier_cfg = get_hierarchical_config(N);

        InstanceResult i_res;
        i_res.name = inst_files.name;
        i_res.N = N;
        i_res.hk_lb = hk_lb;
        i_res.methods.resize(4);

        // Helper Runner
        auto run_method = [&](int idx, std::function<Tour()> func) {
            double total_time = 0.0;
            double total_obj = 0.0;
            for(int r=0; r<runs; ++r) {
                auto t1 = std::chrono::high_resolution_clock::now();
                auto tour = func();
                auto t2 = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration<double>(t2-t1).count();
                total_obj += heuristic_utils::evaluate(tour, costs);
            }
            i_res.methods[idx].time = total_time / runs;
            i_res.methods[idx].obj = total_obj / runs;
            i_res.methods[idx].gap = (i_res.methods[idx].obj - hk_lb) / hk_lb * 100.0;
        };

        run_method(0, [&](){ 
            return heuristic_utils::sequential_pruning_multi_start_metaheuristic(nodes, costs, flat_cfg); 
        });
        
        // Set Baseline Speedup
        double base_time = i_res.methods[0].time;
        if (base_time < 1e-9) base_time = 1e-9;
        i_res.methods[0].speedup = 1.0;

        run_method(1, [&](){ 
            return heuristic_utils::parallel_funneling_multi_start_metaheuristic(nodes, costs, flat_cfg); 
        });
        i_res.methods[1].speedup = base_time / (i_res.methods[1].time + 1e-10);

        HierarchicalTabuSearch hts(instance);
        run_method(2, [&](){ return hts.optimize(hier_cfg, hier_cfg); });
        i_res.methods[2].speedup = base_time / (i_res.methods[2].time + 1e-10);

        run_method(3, [&](){ return hts.optimize_parallel(hier_cfg, hier_cfg); });
        i_res.methods[3].speedup = base_time / (i_res.methods[3].time + 1e-10);

        all_results.push_back(i_res);
        std::cout << "Done.\n";

        // Write to log file immediately
        // Format: Name Time0 Obj0 Time1 Obj1 Time2 Obj2 Time3 Obj3 HK_LB
        log_stream << std::fixed << std::setprecision(5) 
                   << i_res.name << " "
                   << i_res.methods[0].time << " " << i_res.methods[0].obj << " "
                   << i_res.methods[1].time << " " << i_res.methods[1].obj << " "
                   << i_res.methods[2].time << " " << i_res.methods[2].obj << " "
                   << i_res.methods[3].time << " " << i_res.methods[3].obj << " "
                   << i_res.hk_lb << "\n";
    }
    
    log_stream.close();

    std::vector<std::string> method_names = {
        "Flat Seq Pruning (Baseline)", 
        "Flat Par Funnel", 
        "Hierarchical Seq", 
        "Hierarchical Par"
    };

    for(int m=0; m<4; ++m) {
        std::cout << "\nTable " << (m+1) << ": " << method_names[m] << "\n";
        std::cout << std::string(80, '-') << "\n";
        std::cout << std::left << std::setw(20) << "Instance"
                  << std::right << std::setw(15) << "Obj"
                  << std::setw(15) << "Time(s)"
                  << std::setw(15) << "Gap(%)"
                  << std::setw(15) << "Speedup" << "\n";
        std::cout << std::string(80, '-') << "\n";

        for(const auto& res : all_results) {
            std::cout << std::left << std::setw(20) << res.name
                      << std::right 
                      << std::fixed << std::setprecision(4)
                      << std::setw(15) << res.methods[m].obj
                      << std::setw(15) << res.methods[m].time
                      << std::setw(15) << res.methods[m].gap
                      << std::setw(15) << res.methods[m].speedup << "\n";
        }
    }

    std::cout << "\nAll done. Detailed logs saved to '" << output_file << "'.\n";
    return 0;
}