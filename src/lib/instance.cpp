#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include "lib/instance.hpp"

Instance::Instance(const std::string &costs_path) {
  load_instance_name(costs_path);
  load_hole_costs(costs_path);
}

Instance::Instance(
  const std::string &costs_path, const std::string &clusters_path
) : Instance(costs_path) {
  load_clusters(clusters_path);
  build_cluster_costs();
}

void Instance::load_instance_name(const std::string &costs_path) {
  m_instance_name = costs_path.substr(0, costs_path.find_last_of("."));
}

void Instance::load_hole_costs(const std::string &costs_path) {
  std::ifstream file(costs_path);
  file >> m_num_holes; 
  
  std::vector<f64> hole_costs(m_num_holes * m_num_holes);
  for (size_t i = 0; i < m_num_holes; ++i) {
    for (size_t j = 0; j < m_num_holes; ++j) {
      file >> hole_costs[i * m_num_holes + j];
    }
  }
  m_hole_costs = Matrix<f64>(hole_costs, m_num_holes);
}

void Instance::load_clusters(const std::string &clusters_path) {
  std::ifstream file(clusters_path);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open cluster file " 
              << clusters_path << std::endl;
    return;
  }

  m_num_clusters = 0;
  m_hole_to_cluster.assign(m_num_holes, 0); 

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    std::istringstream iss(line);
    size_t n_members; 
    iss >> n_members;

    std::vector<u16> current_members; 
    current_members.reserve(n_members);

    u16 hole_id;
    while (iss >> hole_id) {
      current_members.push_back(hole_id);
      m_hole_to_cluster[hole_id] = static_cast<u16>(m_num_clusters);
    }

    m_cluster_holes.push_back(current_members);
    m_num_clusters++;
  }
}

void Instance::build_cluster_costs() {
  std::vector<f64> cluster_costs(m_num_clusters * m_num_clusters);

  for (size_t i = 0; i < m_num_clusters; ++i) {
    for (size_t j = i + 1; j < m_num_clusters; ++j) {
      f64 cost = min_intercluster_cost(i, j);
      cluster_costs[i * m_num_clusters + j] = cost;
      cluster_costs[j * m_num_clusters + i] = cost;
    }
  }
  
  m_cluster_costs = Matrix<f64>(cluster_costs, m_num_clusters);
}

f64 Instance::min_intercluster_cost(u16 i, u16 j) const {
  f64 min_cost = std::numeric_limits<f64>::max();

  const auto &i_nodes = get_holes(i);
  const auto &j_nodes = get_holes(j);

  for (auto u : i_nodes) {
    for (auto v : j_nodes) {
      f64 cost = hole_cost(u, v);
      if (cost < min_cost) {
        min_cost = cost;
      }
    }
  }

  return min_cost;
}
