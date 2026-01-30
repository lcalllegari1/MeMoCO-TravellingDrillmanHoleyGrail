#include <algorithm>
#include <limits>
#include <thread>
#include <mutex>
#include <cmath>
#include <future>

#include <chrono>
#include <iostream>
#include <iomanip>

#include "lib/heuristic_utils.hpp"
#include "lib/utils.hpp"

namespace heuristic_utils {

  void TabuList::cleanup(size_t curr_iter) {
    auto it = m_tabu_list.begin();
    while (it != m_tabu_list.end()) {
      if (curr_iter >= it->second) {
        it = m_tabu_list.erase(it);
      } else {
        ++it;
      }
    }
  }

  f64 evaluate(const Tour &tour, const Matrix<f64> &costs, bool is_closed) {
    f64 cost = 0.0;

    size_t n_nodes = tour.size();
    for (size_t i = 0; i < n_nodes - 1; ++i) {
      cost += costs(tour[i], tour[i + 1]);
    }
    if (is_closed) {
      cost += costs(tour.back(), tour.front());
    }

    return cost;
  }

  f64 two_opt_delta(
    const Tour &tour, size_t i, size_t j, const Matrix<f64> &costs
  ) {
    size_t n_nodes = tour.size(); 

    size_t prev = (i == 0) ? n_nodes - 1 : i - 1;
    size_t next = (j == n_nodes - 1) ? 0 : j + 1;

    return costs(tour[i], tour[next]) + costs(tour[prev], tour[j])
         - costs(tour[j], tour[next]) - costs(tour[prev], tour[i]);
  }

  size_t find_nearest_unvisited(
    u16 curr_node,
    const std::vector<u16> &candidate_nodes, 
    const std::vector<u8> &visited,
    const Matrix<f64> &costs
  ) {
    size_t next_idx = 0; 
    
    size_t n_nodes = candidate_nodes.size();
    f64 best_cost = std::numeric_limits<f64>::max();
    for (size_t i = 0; i < n_nodes; ++i) {
      if (visited[i]) {
        continue;
      }

      f64 cost = costs(curr_node, candidate_nodes[i]);
      if (cost < best_cost) {
        best_cost = cost;
        next_idx = i;
      }
    }

    return next_idx;
  }

  Tour nearest_neighbour(
    const std::vector<u16> &nodes, size_t start_idx, const Matrix<f64> &costs
  ) {
    size_t n_nodes = nodes.size();

    Tour tour; 
    tour.reserve(n_nodes);

    std::vector<u8> visited(n_nodes, 0);

    size_t curr_idx = start_idx; 
    tour.push_back(nodes[curr_idx]);
    visited[curr_idx] = 1;

    for (size_t step = 1; step < n_nodes; ++step) {
      curr_idx = find_nearest_unvisited(
        nodes[curr_idx], nodes, visited, costs
      );
      tour.push_back(nodes[curr_idx]);
      visited[curr_idx] = 1;
    }
    
    return tour;
  }

  Tour two_opt(
    const Tour &initial_tour, const Matrix<f64> &costs, TwoOptMode mode
  ) {
    size_t n_nodes = initial_tour.size();

    if (n_nodes <= 3) {
      return initial_tour;
    }

    Tour curr_tour = initial_tour;

    bool improvement_found = true;
    while (improvement_found) {
      improvement_found = false;

      f64 best_delta = -1e-9;
      size_t best_i = 0, best_j = 0;
      for (size_t i = 0; i < n_nodes - 1; ++i) {
        for (size_t j = i + 1; j < n_nodes; ++j) {
          // assuming 'costs' is symmetric, reversing the 
          // entire sequence would have no effect on the 
          // objective value (i.e., it would stay the same)
          if (i == 0 && j == n_nodes - 1) {
            continue; 
          }

          f64 delta = two_opt_delta(curr_tour, i, j, costs);
          if (delta < best_delta) {
            improvement_found = true;

            if (mode == TwoOptMode::FirstImprovement) {
              utils::rev_subseq(curr_tour, i, j); // apply 2-opt move
              goto next_while_iteration;
            }

            best_delta = delta;
            best_i = i; best_j = j;
          }
        }
      }

      if (improvement_found) {
        utils::rev_subseq(curr_tour, best_i, best_j); // apply 2-opt move
      }

      // jump to next iteration when mode is set to first improvement
      next_while_iteration:;
    }
    
    return curr_tour;
  }

  Tour two_opt_tabu_search(
    const Tour &initial_tour, 
    const Matrix<f64> &costs, 
    size_t tenure,
    size_t tenure_oscillation_range,
    size_t max_iter,
    size_t patience,
    bool use_stagnation_recovery
  ) {
    size_t n_nodes = initial_tour.size();

    if (n_nodes <= 3) {
      return initial_tour;
    }

    TabuList tabu_list(tenure, tenure_oscillation_range);

    Tour curr_tour = initial_tour; Tour best_tour = curr_tour;
    f64 curr_obj = evaluate(curr_tour, costs), best_obj = curr_obj;

    size_t iter_no_improv = 0;
    size_t kick_start = patience / 2;
    size_t kick_duration = std::min(
      (size_t)100, std::max((size_t)10, patience / 10)
    );

    for (size_t iter = 1; iter <= max_iter; ++iter) {
      if (iter % 100 == 0) {
        curr_obj = evaluate(curr_tour, costs);
      }

      bool use_strong_policy = false;
      if (
        use_stagnation_recovery && 
        iter_no_improv >= kick_start &&
        iter_no_improv < kick_start + kick_duration
      ) {
        use_strong_policy = true;
      }

      f64 best_delta = std::numeric_limits<f64>::max();
      size_t best_i = 0, best_j = 0;
      bool move_found = false;

      for (size_t i = 0; i < n_nodes - 1; ++i) {
        for (size_t j = i + 1; j < n_nodes; ++j) {
          // assuming 'costs' is symmetric, reversing the 
          // entire sequence would have no effect on the 
          // objective value (i.e., it would stay the same)
          if (i == 0 && j == n_nodes - 1) {
            continue;
          }

          f64 delta = two_opt_delta(curr_tour, i, j, costs);
          if (delta >= best_delta) { 
            // current 2-opt move is not strictly 
            // better than current best 2-opt move
            continue; 
          }

          // here current 2-opt move is strictly 
          // better than current best 2-opt move
          
          bool aspiration_satisfied = curr_obj + delta < best_obj - 1e-9;
          if (!aspiration_satisfied) { 
            // tabu list check on current 2-opt move

            // we consider the edges being connected as the result of the
            // current 2-opt move: (i - 1, j) and (i, j + 1), and we handle
            // the wrap around, if needed, by defining u, v as follows
            size_t u = (i == 0) ? n_nodes - 1 : i - 1; // refers to i - 1
            size_t v = (j == n_nodes - 1) ? 0 : j + 1; // refers to j + 1

            bool edge1_is_tabu = tabu_list.is_tabu(
              curr_tour[u], curr_tour[j], iter
            );
            bool edge2_is_tabu = tabu_list.is_tabu(
              curr_tour[i], curr_tour[v], iter
            );

            // check if the current 2-opt move is tabu, according to policy
            if (use_strong_policy) {
              if (edge1_is_tabu || edge2_is_tabu) continue;
            } else {
              if (edge1_is_tabu && edge2_is_tabu) continue;
            }
          }

          best_delta = delta; 
          best_i = i; best_j = j;
          move_found = true;
        }
      }

      if (!move_found) {
        // all moves where tabu, and no move satisfied the aspiration
        // criterium 
        if (++iter_no_improv >= patience) {
          break;
        }

        continue;
      }

      // here a 2-opt valid move has been found: it must be 
      // added to the tabu list and applied

      // for the tabu list insertion we consider the edges being broken 
      // as the result of the 2-opt move being applied: (best_i - 1, best_i) 
      // and (best_j, best_j + 1), and we handle the wrap around, if needed, 
      // by defining u, v as follows
      size_t u = (best_i == 0) ? n_nodes - 1 : best_i - 1;
      size_t v = (best_j == n_nodes - 1) ? 0 : best_j + 1; 

      tabu_list.add_tabu(curr_tour[u], curr_tour[best_i], iter);
      tabu_list.add_tabu(curr_tour[best_j], curr_tour[v], iter);

      // apply the 2-opt move
      utils::rev_subseq(curr_tour, best_i, best_j);

      // update curr values for next iteration
      curr_obj += best_delta;
      if (curr_obj < best_obj) {
        best_obj = curr_obj;
        best_tour = curr_tour;
        iter_no_improv = 0;
      } else {
        iter_no_improv++;
      }

      if (iter_no_improv >= patience) {
        break;
      }
    }

    return best_tour;
  }

  Tour parallel_funneling_multi_start_metaheuristic(
    const std::vector<u16> &nodes,
    const Matrix<f64> &costs,
    const MetaheuristicConfig config
  ) {
    const size_t n_nodes = nodes.size();

    if (n_nodes < 3) {
      return nodes;
    }

    size_t hwc = std::thread::hardware_concurrency();
    if (hwc == 0) {
      hwc = 4;
    }

    const size_t n_trials = std::min(n_nodes, (size_t)200);
    size_t n_survivors = config.n_starts;
    if (n_survivors == 0) {
      n_survivors = std::min(hwc, n_trials);
    } else if (n_survivors > n_trials) {
      n_survivors = n_trials;
    }

    const bool use_two_opt_filter = n_nodes < 300;

    // if we check every node, step is just 1, otherwise compute it
    const size_t step = n_trials == n_nodes ? 1 : utils::get_step(n_nodes);

    struct Candidate {
      f64 cost;
      Tour tour;
      bool operator<(const Candidate& other) const {
        return cost < other.cost;
      }
    };

    std::vector<Candidate> candidates;
    candidates.reserve(n_trials);
    for (size_t i = 0; i < n_trials; ++i) {
      size_t start_idx = (i * step) % n_nodes;

      Tour tour = nearest_neighbour(nodes, start_idx, costs);
      if (use_two_opt_filter) {
        tour = two_opt(tour, costs, TwoOptMode::BestImprovement);
      }
      candidates.push_back({evaluate(tour, costs), std::move(tour)});
    }

    std::sort(candidates.begin(), candidates.end());
    if (n_survivors < candidates.size()) {
      candidates.resize(n_survivors);
    }

    size_t tenure = config.tenure;
    if (tenure == 0) {
      tenure = std::max((size_t)5, n_nodes / 5);
    }
    size_t oscillation = tenure / 2;

    f64 best_cost = std::numeric_limits<f64>::max();
    Tour best_tour;

    std::mutex mutex;
    auto opt_task = [&](Candidate cand) -> void {
      Tour curr_tour = std::move(cand.tour);
      if (!use_two_opt_filter) {
        curr_tour = two_opt(curr_tour, costs, TwoOptMode::BestImprovement);
      }

      curr_tour = two_opt_tabu_search(
        curr_tour,
        costs,
        tenure,
        oscillation,
        config.max_iter,
        config.patience,
        config.use_stagnation_recovery
      ); 
      f64 curr_cost = evaluate(curr_tour, costs);

      std::lock_guard<std::mutex> lock(mutex);
      if (curr_cost < best_cost) {
        best_cost = curr_cost;
        best_tour = std::move(curr_tour);
      }
    };

    std::vector<std::future<void>> opt_futures;
    opt_futures.reserve(candidates.size());
    for (auto &cand : candidates) {
      opt_futures.push_back(
        std::async(std::launch::async, opt_task, std::move(cand))
      );
    }

    for (auto &f : opt_futures) {
      f.get();
    }

    return best_tour;
  }

  Tour sequential_pruning_multi_start_metaheuristic(
    const std::vector<u16> &nodes,
    const Matrix<f64> &costs,
    const MetaheuristicConfig config,
    size_t seed
  ) {
    size_t n_nodes = nodes.size();

    if (n_nodes < 3) {
      return nodes;
    }

    size_t n_starts = config.n_starts;
    if (n_starts == 0) {
      n_starts = std::min(n_nodes, (size_t)200);
    }
    size_t step = utils::get_step(n_nodes);

    size_t tenure = config.tenure;
    if (tenure == 0) {
      tenure = std::max((size_t)5, n_nodes / 5);
    }
    size_t oscillation = tenure / 2;

    f64 best_cost = std::numeric_limits<f64>::max();
    Tour best_tour;

    f64 curr_cost;
    Tour curr_tour;
    for (size_t i = 0; i < n_starts; ++i) {
      size_t start_idx = ((i * step) + seed) % n_nodes;
      curr_tour = nearest_neighbour(nodes, start_idx, costs);

      curr_cost = evaluate(curr_tour, costs);
      if (
        best_cost != std::numeric_limits<f64>::max() && 
        curr_cost > config.pruning_threshold * best_cost
      ) {
        continue;
      }

      curr_tour = two_opt(curr_tour, costs, TwoOptMode::BestImprovement);
      curr_tour = two_opt_tabu_search(
        curr_tour,
        costs,
        tenure,
        oscillation,
        config.max_iter,
        config.patience,
        config.use_stagnation_recovery
      );

      curr_cost = evaluate(curr_tour, costs);
      if (curr_cost < best_cost) {
        best_cost = curr_cost;
        best_tour = std::move(curr_tour);
      }
    }

    return best_tour;
  }

}
