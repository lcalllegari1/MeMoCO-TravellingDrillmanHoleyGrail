#ifndef SCF_SOLVER_HPP
#define SCF_SOLVER_HPP

#include <ilcplex/cplexx.h>

#include "instance.hpp"

using idx = int;

class SCFSolver {
public:
  /**
   * Public constructor that creates a CPLEX environment and a CPLEX
   * problem to solve the specified instance.
   */
  SCFSolver(const Instance &instance, bool verbose);

  /**
   * Public destructor that frees the memory of the CPLEX problem and
   * closes the CPLEX environment.
   */
  ~SCFSolver();

  /**
   * Solves the model associated with the instance m_instance.
   */
  double solve();

  void print_solution() const;
  void print_sequence_solution() const;

  std::vector<int> sequence_solution() const;

private:
  void print_solution_with_names() const;

  /**
   * Builds the model by creating all variables and adding all constraints.
   */
  void build_model();

  /**
   * Creates and adds the necessary y variables.
   */
  void add_y_vars();

  /**
   * Creates and adds the necessary x variables.
   */
  void add_x_vars();

  /**
   * Adds the flow constraints: for each node but the source, ensure that
   * the consumption of flow is exactly 1 unit.
   */
  void add_flow_cons();

  /**
   * Adds the degree constraints: for each node, make sure there is
   * exactly 1 incoming arc and exactly 1 outgoing arc.
   */
  void add_degree_cons();

  /**
   * Adds the linking constraints: whenever y_ij is set to zero, it must
   * be the case that x_ij is set to 0.
   */
  void add_linking_cons();

  /**
   * If set to 1, the solver prints out the model creation steps (e.g.
   * adding variables, adding constraints, ...)
   */
  bool m_verbose;

  /**
   * The instance to solve.
   */
  const Instance &m_instance;

  /**
   * CPLEX pointers for environment and problem, respectively.
   */
  CPXENVptr m_cpx_env = nullptr;
  CPXLPptr m_cpx_lp = nullptr;

  /**
   * counter for indexing variables in maps (see below), according to the
   * order in which they are created in the single CPLEX array of vars.
   */
  idx m_cpx_curr_var_pos = 0;

  /**
   * The maps to use as lookup table for recovering variable indices in
   * the CPLEX array.
   */
  std::vector<idx> m_ymap, m_xmap;
};

#endif // SCF_SOLVER_HPP