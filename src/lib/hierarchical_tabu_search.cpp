#include <cmath>
#include <numeric>
#include <algorithm>
#include <thread>
#include <future>
#include <iterator>

#include "lib/hierarchical_tabu_search.hpp"

Tour HierarchicalTabuSearch::optimize(
  const MetaheuristicConfig &intercluster_base_config,
  const MetaheuristicConfig &intracluster_base_config,
  size_t n_cluster_orderings
) {
  optimize_cluster_tours(intracluster_base_config);
  compute_top_k_cluster_orderings(
    intercluster_base_config, n_cluster_orderings
  );

  f64 best_cost = std::numeric_limits<f64>::max();
  Tour best_tour;
  const auto &costs = m_instance.hole_costs();
  for (const auto& ordering : m_cluster_orderings) {
    Tour curr_tour = stitch_cluster_tours(ordering);
    curr_tour = two_opt(curr_tour, costs, TwoOptMode::BestImprovement);

    f64 curr_cost = evaluate(curr_tour, costs);
    if (curr_cost < best_cost) {
      best_cost = curr_cost;
      best_tour = std::move(curr_tour);
    }
  }

  return best_tour;
}

Tour HierarchicalTabuSearch::optimize_parallel(
  const MetaheuristicConfig &intercluster_base_config,
  const MetaheuristicConfig &intracluster_base_config,
  size_t n_cluster_orderings
) {
  optimize_cluster_tours(intracluster_base_config);
  compute_top_k_cluster_orderings(
    intercluster_base_config, n_cluster_orderings
  );

  f64 best_cost = std::numeric_limits<f64>::max();
  Tour best_tour;
  const auto &costs = m_instance.hole_costs();

  auto process_candidate = [&](const Tour &ordering) -> std::pair<f64, Tour> {
    Tour curr_tour = stitch_cluster_tours(ordering);
    curr_tour = two_opt(curr_tour, costs, TwoOptMode::BestImprovement);

    f64 curr_cost = evaluate(curr_tour, costs);
    return std::make_pair(curr_cost, curr_tour);
  };

  std::vector<std::future<std::pair<f64, Tour>>> futures;
  futures.reserve(m_cluster_orderings.size());
  for (const auto& ordering : m_cluster_orderings) {
    futures.push_back(
      std::async(std::launch::async, process_candidate, ordering)
    );
  }

  for (auto &f : futures) {
    std::pair<f64, Tour> result = f.get();
    
    if (result.first < best_cost) {
      best_cost = result.first;
      best_tour = std::move(result.second);
    }
  }

  return best_tour;
}

size_t HierarchicalTabuSearch::find_nearest_idx(
  u16 curr_node, const std::vector<u16> &candidate_nodes
) {
  const size_t n_nodes = candidate_nodes.size();
  const auto &costs = m_instance.hole_costs();

  size_t next_idx = 0; 

  f64 best_cost = std::numeric_limits<f64>::max();
  for (size_t i = 0; i < n_nodes; ++i) {
    const f64 cost = costs(curr_node, candidate_nodes[i]);
    if (cost < best_cost) {
      best_cost = cost;
      next_idx = i;
    }
  }

  return next_idx;
}

void HierarchicalTabuSearch::compute_top_k_cluster_orderings(
  const MetaheuristicConfig &intercluster_base_config, size_t k
) {
  const size_t n_clusters = m_instance.n_clusters();
  std::vector<u16> clusters(n_clusters);
  std::iota(clusters.begin(), clusters.end(), 0); // [0,..., n_clusters - 1]

  const Matrix<f64> &cluster_costs = m_instance.cluster_costs();

  MetaheuristicConfig config = intercluster_base_config;
  for (size_t i = 0; i < k; i++) {
    Tour tour = sequential_pruning_multi_start_metaheuristic(
      clusters, cluster_costs, config, (i + k) * 0x4973
    );
    f64 cost = evaluate(tour, cluster_costs);

    bool unique = true;
    for (const auto &existing : m_cluster_orderings) {
      if (std::abs(evaluate(existing, cluster_costs) - cost) < 1e-3) {
        unique = false;
        break;
      }
    }

    if (unique) {
      m_cluster_orderings.push_back(std::move(tour));
    }
  }
}

void HierarchicalTabuSearch::optimize_cluster_tours(
  const MetaheuristicConfig &intracluster_base_config
) {
  const size_t n_clusters = m_instance.n_clusters();
  m_cluster_tours.resize(n_clusters);

  const auto &costs = m_instance.hole_costs();

  auto cluster_opt_task = [&](size_t cluster_idx) {
    const auto &holes = m_instance.get_holes(cluster_idx);

    MetaheuristicConfig config = intracluster_base_config;

    return sequential_pruning_multi_start_metaheuristic(
      holes, costs, config
    );
  };

  std::vector<std::future<void>> futures;
  futures.reserve(n_clusters);
  for (size_t i = 0; i < n_clusters; ++i) {
    futures.push_back(std::async(
      std::launch::async, [&, i, cluster_opt_task]() {
        m_cluster_tours[i] = cluster_opt_task(i);
      })
    );
  }

  for (auto &f : futures) {
    f.get();
  }
}

Tour HierarchicalTabuSearch::stitch_cluster_tours(
  const Tour &cluster_ordering
) {
  const size_t n_clusters = m_instance.n_clusters();
  const size_t n_holes = m_instance.n_holes();
  const auto &costs = m_instance.hole_costs();

  Tour global_tour;
  global_tour.reserve(n_holes);

  size_t curr_cluster_id = cluster_ordering[0];
  const Tour &curr_cluster_tour = m_cluster_tours[curr_cluster_id];
  global_tour.insert(
    global_tour.end(), curr_cluster_tour.begin(), curr_cluster_tour.end()
  );
  u16 curr_node = global_tour.back();

  for (size_t i = 1; i < n_clusters; ++i) {
    curr_cluster_id = cluster_ordering[i];
    const Tour &curr_cluster_tour = m_cluster_tours[curr_cluster_id];


    const Tour *next_cluster_nodes = nullptr;
    if (i + 1 < n_clusters) {
      next_cluster_nodes = &m_cluster_tours[cluster_ordering[i + 1]];
    }

    // now we have to rotate the current cluster tour in the best possible
    // way so that it attaches to the previous cluster (curr_node) and to 
    // the next cluster (next_node) (or the first one, if we are at the end)
    
    // the idea is to try and break all edges of the current cluster
    // tour, one at the time, and see whether stitching the previous 
    // and next cluster to the ends of this edge is good or not

    // in the end, we break the edge that is most convenient, i.e., the one
    // that yields the best net cost when broken and when its end nodes are 
    // used as attachments points for the previous and next cluster

    // we are essentially trying to find the best edge to break in order to
    // make the current closed cluster tour an open path, such that the start
    // and end nodes of this path are the best possible w.r.t. the previous
    // and next clusters current exit and entry respectively.

    // we have to keep in mind that the cluster tour has two directions, so 
    // we have to check them both to find the best way of bridging the 
    // previous cluster to the next one through the current one

    const size_t n_nodes = curr_cluster_tour.size();
    f64 best_delta = std::numeric_limits<f64>::max();
    bool best_is_rev = false;
    size_t best_j = 0;
    size_t prev_j = n_nodes - 1;
    for (size_t j = 0; j < n_nodes; ++j) { // we try all rotations
      // if we enter at position i, then the exit is the previous node in
      // the sequence, as we have to go from i all the way to the end and 
      // come back to where the edge has been broken a position i - 1
      // 
      // Example: 1 -> 2 -> 3 -> 4 (assume after 4 we go back to 1)
      // If we enter at 2, then we have to exit at 1, after visiting all 
      // the other nodes in the sequence. But because of symmetry, this 
      // tour is the same if followed backwards, starting from 4. We 
      // also need to check if breaking an edge in this direction is even 
      // better than doing so in the forward direction. The reason being 
      // that breaking the same edge (involving the same two end nodes)
      // in different directions, gives different paths. For instance, if
      // we consider the reversed tour 4 -> 3 -> 2 -> 1 (and then back to 4),
      // and we break the edge between 2 and 1, as before, we have a different
      // path than before, namely we now enter at 1, visit every node and 
      // finish at 2. So the same edge broken, but different directions, means
      // that we are effectively exchanging the entry and exit nodes.
      const u16 u = curr_cluster_tour[j];
      const u16 v = curr_cluster_tour[prev_j];
      prev_j = j;

      // When we break the edge between node 'u' and node 'v', we remove 
      // its cost from the current tour. This is what we gain by breaking 
      // that edge to make the tour open. This of course depends on what 
      // pair of nodes we consider, and we have to do this for every pair 
      // (i.e., every edge) so that we know what is the best edge to actually
      // cut. However, the best edge to cut does not depend on its cost only
      // but also on the bridging costs we introduce to connect the previous
      // cluster and the next cluster to 'u' or 'v' (what connections to make
      // depends on the direction, and we will consider both fwd and rev dir)
      const f64 broken_edge_cost = costs(v, u);

      // We first check the forward direction (the cluster tour as it is)
      u16 next_cluster_closest_node_fwd = global_tour[0];
      if (next_cluster_nodes) {
        size_t closest_idx = find_nearest_idx(v, *next_cluster_nodes);
        next_cluster_closest_node_fwd = (*next_cluster_nodes)[closest_idx];
      }

      const f64 fwd_bridges_cost = 
        costs(curr_node, u) + costs(v, next_cluster_closest_node_fwd);

      // we want to get the best open path of the current cluster (fwd dir)
      const f64 fwd_delta = fwd_bridges_cost - broken_edge_cost;
      if (fwd_delta < best_delta) {
        best_delta = fwd_delta;
        best_is_rev = false;
        best_j = j;
      }

      // Now we check the reverse direction
      u16 next_cluster_closest_node_rev = global_tour[0];
      if (next_cluster_nodes) {
        size_t closest_idx = find_nearest_idx(u, *next_cluster_nodes);
        next_cluster_closest_node_rev = (*next_cluster_nodes)[closest_idx];
      }

      const f64 rev_bridges_cost = 
        costs(curr_node, v) + costs(u, next_cluster_closest_node_rev);

      // we want to get the best open path of the current cluster (rev dir)
      const f64 rev_delta = rev_bridges_cost - broken_edge_cost;
      if (rev_delta < best_delta) {
        best_delta = rev_delta;
        best_is_rev = true;
        best_j = j;
      }
    }

    if (best_is_rev) { // rev: enter at v, exit at u
      // tour is [tour[best_i - 1]..tour[0]][tour[n_nodes - 1]..tour[best_i]]

      std::reverse_iterator<std::vector<u16>::const_iterator> rev_start(
        curr_cluster_tour.begin() + best_j
      ); // this points to tour[best_i - 1] (property of iterators)

      // insert from tour[best_i - 1] to tour[0] (visiting in reverse order!)
      if (best_j > 0) {
        global_tour.insert(
          global_tour.end(),
          rev_start,
          curr_cluster_tour.rend()
        );
      }

      // insert from tour[n_nodes - 1] to tour[best_i]
      global_tour.insert(
        global_tour.end(),
        curr_cluster_tour.rbegin(),
        rev_start
      );
    } else { // fwd: enter at u, exit at v
      // tour is [tour[best_i]..tour[n_nodes - 1][tour[0]..tour[best_i - 1]]

      // insert from tour[best_i] to tour[n_nodes]
      global_tour.insert(
        global_tour.end(),
        curr_cluster_tour.begin() + best_j,
        curr_cluster_tour.end()
      );

      // insert from tour[0] to tour[best_i - 1]
      if (best_j > 0) {
        global_tour.insert(
          global_tour.end(),
          curr_cluster_tour.begin(),
          curr_cluster_tour.begin() + best_j // note end is exclusive
        );
      }
    }

    curr_node = global_tour.back();
  }

  return global_tour;
}