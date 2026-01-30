#ifndef HELD_KARP_H
#define HELD_KARP_H

#include <vector>
#include <string>
#include <stdexcept>

namespace heuristic_utils {

    struct InstanceData {
        int n_nodes;
        std::vector<double> cost_matrix;
    };

    inline double get_cost(const InstanceData& data, int i, int j) {
        return data.cost_matrix[i * data.n_nodes + j];
    }

    InstanceData read_dat_file(const std::string& filename);

    double get_heuristic_ub(const InstanceData& data);

    double compute_lower_bound(const InstanceData& data, int max_iter = 1000);

} // namespace heuristic_utils

#endif // HELD_KARP_H