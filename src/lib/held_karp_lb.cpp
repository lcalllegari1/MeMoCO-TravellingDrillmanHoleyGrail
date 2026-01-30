#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "lib/held_karp_lb.hpp"

namespace heuristic_utils {

    InstanceData read_dat_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open file " + filename);
        }

        InstanceData data;
        std::string line;
        
        // Read number of nodes (skipping empty lines)
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            if (ss >> data.n_nodes) break;
        }

        if (data.n_nodes <= 0) {
            throw std::runtime_error("Error: Invalid number of nodes read from file.");
        }

        // Pre-allocate memory
        data.cost_matrix.reserve(data.n_nodes * data.n_nodes);

        // Read the matrix data
        double val;
        while (file >> val) {
            data.cost_matrix.push_back(val);
        }

        if (data.cost_matrix.size() != (size_t)(data.n_nodes * data.n_nodes)) {
            throw std::runtime_error("Error: Matrix size mismatch. Expected " + 
                std::to_string(data.n_nodes * data.n_nodes) + " values.");
        }

        return data;
    }

    double get_heuristic_ub(const InstanceData& data) {
        if (data.n_nodes == 0) return 0.0;
        
        std::vector<bool> visited(data.n_nodes, false);
        int curr = 0;
        visited[0] = true;
        double total_cost = 0.0;

        for (int i = 0; i < data.n_nodes - 1; ++i) {
            int next_node = -1;
            double min_val = std::numeric_limits<double>::max();

            for (int j = 0; j < data.n_nodes; ++j) {
                if (!visited[j]) {
                    double d = get_cost(data, curr, j);
                    if (d < min_val) {
                        min_val = d;
                        next_node = j;
                    }
                }
            }
            
            // If graph is disconnected or data is invalid
            if (next_node == -1) break;

            total_cost += min_val;
            visited[next_node] = true;
            curr = next_node;
        }
        total_cost += get_cost(data, curr, 0); // Return to start
        return total_cost;
    }

    double compute_lower_bound(const InstanceData& data, int max_iter) {
        int n = data.n_nodes;
        if (n < 2) return 0.0;

        // Lagrange multipliers (penalties)
        std::vector<double> pi(n, 0.0);
        
        // We need an Upper Bound to tune the step size (Alpha)
        double ub = get_heuristic_ub(data);
        double best_lb = -std::numeric_limits<double>::infinity();
        double alpha = 2.0;

        // Structures for Prim's Algorithm (reused to avoid allocation)
        // We compute MST on nodes 1..n-1 (excluding 0)
        int mst_nodes = n - 1;
        std::vector<double> min_weight(n); 
        std::vector<int> parent(n);
        std::vector<bool> in_mst(n);
        std::vector<int> degree(n);

        for (int iter = 0; iter < max_iter; ++iter) {
            
            std::fill(in_mst.begin(), in_mst.end(), false);
            std::fill(degree.begin(), degree.end(), 0);
            
            // Initialize Prim's from node 1
            for (int i = 1; i < n; ++i) {
                min_weight[i] = std::numeric_limits<double>::max();
            }
            min_weight[1] = 0.0;
            parent[1] = -1;
            
            double mst_cost = 0.0;
            
            // Prim's loop for the subgraph (1..N-1)
            for (int i = 0; i < mst_nodes; ++i) {
                int u = -1;
                double best_w = std::numeric_limits<double>::max();

                // Find closest unvisited node
                for (int v = 1; v < n; ++v) {
                    if (!in_mst[v] && min_weight[v] < best_w) {
                        best_w = min_weight[v];
                        u = v;
                    }
                }

                if (u == -1) break; // Disconnected component

                in_mst[u] = true;
                mst_cost += best_w;
                
                // Update degree for the edge that connected u
                if (parent[u] != -1) {
                    degree[u]++;
                    degree[parent[u]]++;
                }

                // Update neighbors
                for (int v = 1; v < n; ++v) {
                    if (!in_mst[v]) {
                        // Implicit cost calculation: cost(u,v) + pi[u] + pi[v]
                        double cost = get_cost(data, u, v) + pi[u] + pi[v];
                        if (cost < min_weight[v]) {
                            min_weight[v] = cost;
                            parent[v] = u;
                        }
                    }
                }
            }

            double min1 = std::numeric_limits<double>::max();
            double min2 = std::numeric_limits<double>::max();
            int idx1 = -1, idx2 = -1;

            for (int j = 1; j < n; ++j) {
                double cost = get_cost(data, 0, j) + pi[0] + pi[j];
                if (cost < min1) {
                    min2 = min1; idx2 = idx1;
                    min1 = cost; idx1 = j;
                } else if (cost < min2) {
                    min2 = cost; idx2 = j;
                }
            }

            double one_tree_cost = mst_cost + min1 + min2;
            
            // Update degrees for node 0 and its connections
            degree[0] += 2;
            if (idx1 != -1) degree[idx1]++;
            if (idx2 != -1) degree[idx2]++;

            double pi_sum = 0.0;
            for (double p : pi) pi_sum += p;
            
            double curr_lb = one_tree_cost - 2 * pi_sum;

            if (curr_lb > best_lb) {
                best_lb = curr_lb;
            }

            // Check if valid tour (all degrees == 2)
            double grad_sq_norm = 0.0;
            bool is_valid_cycle = true;
            for (int i = 0; i < n; ++i) {
                int grad = degree[i] - 2;
                if (grad != 0) is_valid_cycle = false;
                grad_sq_norm += grad * grad;
            }

            if (is_valid_cycle) return curr_lb; // Optimal solution found
            
            if (grad_sq_norm == 0) grad_sq_norm = 1.0;

            // Polyak step size
            double step = alpha * (ub - curr_lb) / grad_sq_norm;

            // Update penalties
            for (int i = 0; i < n; ++i) {
                pi[i] += step * (degree[i] - 2);
            }

            // Decay alpha
            alpha *= 0.995;
            
            if (alpha < 0.0001) break;
        }

        return best_lb;
    }

} // namespace heuristic_utils