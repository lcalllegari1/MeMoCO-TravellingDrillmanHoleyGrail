#ifndef HIERARCHICAL_TABU_SEARCH_HPP
#define HIERARCHICAL_TABU_SEARCH_HPP

#include "lib/types.hpp"
#include "lib/instance.hpp"
#include "heuristic_utils.hpp"

using namespace heuristic_utils;

class HierarchicalTabuSearch {

public:
  HierarchicalTabuSearch(const Instance &instance) : m_instance(instance) {}

  inline const std::vector<Tour>& get_cluster_orderings() const {
    return m_cluster_orderings;
  }

  Tour optimize(
    const MetaheuristicConfig &intercluster_base_config,
    const MetaheuristicConfig &intracluster_base_config,
    size_t n_cluster_orderings = 3
  );

  Tour optimize_parallel(
    const MetaheuristicConfig &intercluster_base_config,
    const MetaheuristicConfig &intracluster_base_config,
    size_t n_cluster_orderings = 3
  );

  void compute_top_k_cluster_orderings(
    const MetaheuristicConfig &inter_config, size_t k
  );

  void optimize_cluster_tours(const MetaheuristicConfig &intra_config);
  Tour stitch_cluster_tours(const Tour &cluster_ordering);
private: 
  const Instance m_instance;
  std::vector<Tour> m_cluster_orderings;
  std::vector<Tour> m_cluster_tours;

  size_t find_nearest_idx(u16 curr_node, const std::vector<u16> &candidate_nodes);
};

#endif // HIERARCHICAL_TABU_SEARCH_HPP