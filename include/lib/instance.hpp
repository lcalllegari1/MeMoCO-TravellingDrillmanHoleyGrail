#ifndef INSTANCE_HPP
#define INSTANCE_HPP

#include <vector>
#include <string>

#include "types.hpp"

class Instance {

public: 
  Instance(const std::string &costs_path);
  Instance(const std::string &costs_path, const std::string &clusters_path);

  inline size_t n_holes() const { return m_num_holes; }
  inline size_t n_clusters() const { return m_num_clusters; }

  inline const Matrix<f64>& hole_costs() const {
    return m_hole_costs;
  }

  inline const Matrix<f64>& cluster_costs() const {
    return m_cluster_costs;
  }

  inline u16 get_cluster(u16 hole) const {
    return m_hole_to_cluster[hole];
  }

  inline const std::vector<u16>& get_holes(u16 cluster) const {
    return m_cluster_holes[cluster];
  }

  inline f64 hole_cost(u16 i, u16 j) const {
    return i == j ? 0 : m_hole_costs(i, j);
  }

  inline f64 cluster_cost(u16 i, u16 j) const {
    return i == j ? 0 : m_cluster_costs(i, j);
  }

  inline const std::string& instance_name() const {
    return m_instance_name;
  }

private:
  std::string m_instance_name;

  size_t m_num_holes; Matrix<f64> m_hole_costs;

  size_t m_num_clusters; Matrix<f64> m_cluster_costs;  

  std::vector<u16> m_hole_to_cluster;
  std::vector<std::vector<u16>> m_cluster_holes;

  f64 min_intercluster_cost(u16 i, u16 j) const;

  void load_instance_name(const std::string &costs_path);
  void load_hole_costs(const std::string &costs_path);
  void load_clusters(const std::string &clusters_path);
  void build_cluster_costs();

};

#endif // INSTANCE_HPP