#ifndef HEURISTIC_UTILS_HPP
#define HEURISTIC_UTILS_HPP

#include <vector>
#include <unordered_map>

#include "types.hpp"

namespace heuristic_utils {

  enum class TwoOptMode { FirstImprovement, BestImprovement };

  struct MetaheuristicConfig {
    size_t n_starts = 0;
    size_t tenure = 0;
    size_t max_iter = 1000;
    size_t patience = 100;
    f64 pruning_threshold = 1.25;
    bool use_stagnation_recovery = true;
  };

  class TabuList {

  public:
    TabuList(size_t tenure, size_t tenure_oscillation_range)
      : m_tabu_list(3 * (tenure + tenure_oscillation_range) + 1)
      , m_tenure(tenure)
      , m_tenure_oscillation_range(tenure_oscillation_range) {}

    inline bool is_tabu(u16 i, u16 j, size_t curr_iter) const {
      auto it = m_tabu_list.find(hash(i, j));
      if (it == m_tabu_list.end()) {
        return false;
      }
      return curr_iter <= it->second;
    }

    inline void add_tabu(u16 i, u16 j, size_t curr_iter) {
      size_t tenure_offset = 0;
      if (m_tenure_oscillation_range != 0) {
        tenure_offset = curr_iter % m_tenure_oscillation_range;
      }

      m_tabu_list[hash(i, j)] = curr_iter + m_tenure + tenure_offset;

      if (m_tabu_list.size() > 3 * (m_tenure + m_tenure_oscillation_range)) {
        cleanup(curr_iter);
      }
    }

  private:
    std::unordered_map<u32, size_t> m_tabu_list;

    size_t m_tenure;
    size_t m_tenure_oscillation_range;
    
    inline u32 hash(u16 i, u16 j) const {
      // computes the same hash for (i, j) and (j, i) (i.e., it is symmetric)
      // and for (i, j) with i < j the result is |i (16 bits)|j (16 bits)|
      return i < j ? ((u32)i << 16) | (u32)j : ((u32)j << 16) | (u32)i;
    }

    void cleanup(size_t curr_iter);

  };

  f64 evaluate(
    const Tour &tour, const Matrix<f64> &costs, bool is_closed = true
  );

  f64 two_opt_delta(
    const Tour &tour, size_t i, size_t j, const Matrix<f64> &costs
  );

  size_t find_nearest_unvisited(
    u16 curr_node,
    const std::vector<u16> &candidate_nodes, 
    const std::vector<u8> &visited,
    const Matrix<f64> &costs
  );

  Tour nearest_neighbour(
    const std::vector<u16> &nodes, size_t start_idx, const Matrix<f64> &costs
  );

  Tour two_opt(
    const Tour &init_tour, 
    const Matrix<f64> &costs, 
    TwoOptMode mode = TwoOptMode::BestImprovement
  );

  Tour two_opt_tabu_search(
    const Tour &initial_tour, 
    const Matrix<f64> &costs, 
    size_t tenure,
    size_t tenure_oscillation_range,
    size_t max_iter,
    size_t patience,
    bool use_stagnation_recovery
  );

  Tour parallel_funneling_multi_start_metaheuristic(
    const std::vector<u16> &nodes,
    const Matrix<f64> &costs,
    const MetaheuristicConfig config = MetaheuristicConfig()
  );

  Tour sequential_pruning_multi_start_metaheuristic(
    const std::vector<u16> &nodes,
    const Matrix<f64> &costs,
    const MetaheuristicConfig config = MetaheuristicConfig(),
    size_t seed = 0
  );

};

#endif // HEURISTIC_UTILS
