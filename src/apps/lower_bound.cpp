#include <iostream>
#include <iomanip>
#include <exception>

#include "lib/held_karp_lb.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file.dat>" << std::endl;
        return 1;
    }

    try {
        std::cout << "Reading " << argv[1] << "..." << std::endl;
        
        // Use the namespace
        heuristic_utils::InstanceData data = heuristic_utils::read_dat_file(argv[1]);
        
        std::cout << "Loaded " << data.n_nodes << " nodes." << std::endl;
        std::cout << "Computing Held-Karp Lower Bound..." << std::endl;
        
        // Compute LB
        double lb = heuristic_utils::compute_lower_bound(data, 2000);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "------------------------------" << std::endl;
        std::cout << "Lower Bound: " << lb << std::endl;
        std::cout << "------------------------------" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}