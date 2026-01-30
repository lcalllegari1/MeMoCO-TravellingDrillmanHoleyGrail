#include <iostream>

#include "lib/SCF_solver.hpp"

#include <ilcplex/cplexx.h>

#include <cmath>

SCFSolver::SCFSolver(const Instance &instance, bool verbose)
    : m_verbose(verbose), m_instance(instance), m_cpx_env(CPXXopenCPLEX(NULL)),
      m_cpx_lp(CPXXcreateprob(m_cpx_env, NULL, instance.instance_name().c_str())) {
  if (m_cpx_env == NULL or m_cpx_lp == NULL) {
    std::cerr << "Error: Could not create CPLEX environment or problem."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  m_ymap.resize(std::pow(instance.n_holes(), 2), -1);
  m_xmap.resize(std::pow(instance.n_holes(), 2), -1);

  if (m_verbose) {
    std::cout << "Starting to build the model for instance "
              << m_instance.instance_name() << " ..." << std::endl;
  }

  build_model();

  if (m_verbose) {
    std::cout << "... done. Model for instance " << m_instance.instance_name()
              << " was successfully created." << std::endl;
    std::cout << "INFO: To remove model creation steps logs (vars, cons, ...) "
              << "run with <verbose> set to 0" << std::endl;
  }
}

SCFSolver::~SCFSolver() {
  if (m_cpx_lp)
    CPXXfreeprob(m_cpx_env, &m_cpx_lp);
  if (m_cpx_env)
    CPXXcloseCPLEX(&m_cpx_env);
}

double SCFSolver::solve() {
  if (m_verbose) {
    std::cout << "Starting solve for instance " << m_instance.instance_name()
              << " ..." << std::endl;
  }

  int cpx_st = CPXXmipopt(m_cpx_env, m_cpx_lp);

  if (cpx_st != 0) {
    std::cout << "... failed. Solve for instance " << m_instance.instance_name()
              << " was unsuccessful." << std::endl;
  } else {
    if (m_verbose) {
      std::cout << "... done. Solve for instance " << m_instance.instance_name()
                << " was successfully completed." << std::endl;
      int sol_stat = CPXXgetstat(m_cpx_env, m_cpx_lp);
      std::cout << "SOLUTION STATUS: " << sol_stat << std::endl;
    }
  }

  double obj;
  CPXXgetobjval(m_cpx_env, m_cpx_lp, &obj);
  return obj;
}

void SCFSolver::build_model() {
  add_y_vars();
  add_x_vars();

  add_flow_cons();
  add_degree_cons();
  add_linking_cons();
}

void SCFSolver::add_y_vars() {
  size_t n = m_instance.n_holes();
  const char *y_name;
  std::string y_name_buf;
  double lb = 0.0, ub = 1.0, y_cost = 0;
  char y_type = 'B';
  int cpx_st, var_count = 0;
  if (m_verbose) {
    std::cout << "Creating y vars with: lb = " << lb << ", ub = " << ub
              << " and type = " << y_type << "." << std::endl;
  }
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i == j)
        continue; // we don't want loops, i.e. arcs from one node to
                  // itself.

      y_cost = m_instance.hole_cost(i, j);
      y_name_buf = "y_" + std::to_string(i) + "_" + std::to_string(j);
      y_name = y_name_buf.c_str();
      if (m_verbose)
        std::cout << "Creating " << y_name << " ... ";
      cpx_st = CPXXnewcols(
        m_cpx_env, /* CPLEX environment (env)     */
        m_cpx_lp,  /* CPLEX problem     (lp)      */
        1,         /* num vars created  (ccnt)    */
        &y_cost,   /* obj coefficients  (obj)     */
        &lb, &ub,  /* var bounds        (lb, ub)  */
        &y_type,   /* var type          (xctype)  */
        &y_name    /* var name          (colname) */
      );
      if (m_verbose) {
        if (cpx_st == 0) {
          std::cout << "done." << std::endl;
          var_count++;
        } else {
          std::cout << "failed." << std::endl;
          exit(EXIT_FAILURE);
        }
      }
      m_ymap[i * n + j] = m_cpx_curr_var_pos++;
    }
  }
  if (m_verbose) {
    std::cout << "All y vars (#" << var_count << ") were successfully created."
              << std::endl;
  }
}

void SCFSolver::add_x_vars() {
  size_t n = m_instance.n_holes();
  const char *x_name;
  std::string x_name_buf;
  double lb = 0.0, ub = CPX_INFBOUND;
  char x_type = 'C';
  int cpx_st, var_count = 0;
  if (m_verbose) {
    std::cout << "Creating x vars with: lb = " << lb << ", ub = " << ub
              << " and type = " << x_type << "." << std::endl;
  }
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i == j || j == 0)
        continue; // we don't want loops (1st condition), and we don't need
                  // to ship to the source (2nd condition).

      x_name_buf = "x_" + std::to_string(i) + "_" + std::to_string(j);
      x_name = x_name_buf.c_str();
      if (m_verbose)
        std::cout << "Creating " << x_name << " ... ";
      cpx_st = CPXXnewcols(
        m_cpx_env, /* CPLEX environment (env)     */
        m_cpx_lp,  /* CPLEX problem     (lp)      */
        1,         /* num vars created  (ccnt)    */
        NULL,      /* obj coefficients  (obj)     */
        &lb, &ub,  /* var bounds        (lb, ub)  */
        &x_type,   /* var type          (xctype)  */
        &x_name    /* var name          (colname) */
      );
      if (m_verbose) {
        if (cpx_st == 0) {
          std::cout << "done." << std::endl;
          var_count++;
        } else {
          std::cout << "failed." << std::endl;
          exit(EXIT_FAILURE);
        }
      }
      m_xmap[i * n + j] = m_cpx_curr_var_pos++;
    }
  }
  if (m_verbose) {
    std::cout << "All x vars (#" << var_count << ") were successfully created."
              << std::endl;
  }
}

void SCFSolver::add_flow_cons() {
  size_t n = m_instance.n_holes();
  const char *row_name;
  std::string row_name_buf;
  double rhs = 1.0;
  char sense = 'E';
  CPXNNZ rmatbeg = 0;
  std::vector<int> rmatind;
  std::vector<double> rmatval;
  idx idx_x_ik, idx_x_kj;
  int cpx_st, cons_count = 0;
  for (size_t k = 1; k < n;
       ++k) { // note we start from 1, since 0 is the source (starting node).
    rmatind.clear();
    rmatval.clear();
    // first sum
    for (size_t i = 0; i < n; ++i) {
      idx_x_ik = m_xmap[i * n + k];
      if (idx_x_ik == -1)
        continue; // only consider existing vars

      rmatind.push_back(idx_x_ik);
      rmatval.push_back(1.0);
    }
    // second sum (recall we have negative coefficients here)
    for (size_t j = 1; j < n;
         ++j) { // note we start from 1, since 0 is the source and does not
      // require any incoming flow.
      idx_x_kj = m_xmap[k * n + j];
      if (idx_x_kj == -1)
        continue; // only consider existing vars

      rmatind.push_back(idx_x_kj);
      rmatval.push_back(-1.0);
    }

    row_name_buf = "k = " + std::to_string(k) + ": sum_i x_ik - sum_j x_kj = 1";
    row_name = row_name_buf.c_str();
    if (m_verbose)
      std::cout << "Creating flow constraint " << row_name << " ... ";
    cpx_st = CPXXaddrows(
      m_cpx_env,      /* CPLEX environment      (env)     */
      m_cpx_lp,       /* CPLEX problem          (lp)      */
      0,              /* num vars created       (ccnt)    */
      1,              /* num cons created       (rcnt)    */
      rmatind.size(), /* cons num vars involved (nzcnt)   */
      &rhs,           /* cons right hand side   (rhs)     */
      &sense,         /* cons sense             (sense)   */
      &rmatbeg,       /* cons var matbeg        (rmatbeg) */
      rmatind.data(), /* cons var indices       (rmatind) */
      rmatval.data(), /* cons var coefficients  (rmatval) */
      NULL,           /* var name               (colname) */
      &row_name       /* cons name              (rowname) */
    );
    if (m_verbose) {
      if (cpx_st == 0) {
        std::cout << "done." << std::endl;
        cons_count++;
      } else {
        std::cout << "failed." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }
  if (m_verbose) {
    std::cout << "Flow constraints for all k (excluding source node 0) (#"
              << cons_count << ") were successfully created." << std::endl;
  }
}

void SCFSolver::add_degree_cons() {
  size_t n = m_instance.n_holes();
  const char *row_name;
  std::string row_name_buf;
  double rhs = 1.0;
  char sense = 'E';
  CPXNNZ rmatbeg = 0;
  std::vector<int> rmatind;
  std::vector<double> rmatval;
  idx idx_y_ij;
  int cpx_st, cons_count = 0;
  if (m_verbose) {
    std::cout << "Creating degree constraints with: rhs = " << rhs
              << " and sense = " << sense << " for all j." << std::endl;
  }
  for (size_t j = 0; j < n; ++j) {
    rmatind.clear();
    rmatval.clear();
    for (size_t i = 0; i < n; ++i) {
      idx_y_ij = m_ymap[i * n + j];

      if (idx_y_ij == -1)
        continue; // only create cons for existing y vars.

      rmatind.push_back(idx_y_ij);
      rmatval.push_back(1.0);
    }

    row_name_buf = "j = " + std::to_string(j) + ": sum_i x_ij = 1";
    row_name = row_name_buf.c_str();
    if (m_verbose)
      std::cout << "Creating degree constraint " << row_name << " ... ";
    cpx_st = CPXXaddrows(
      m_cpx_env,      /* CPLEX environment      (env)     */
      m_cpx_lp,       /* CPLEX problem          (lp)      */
      0,              /* num vars created       (ccnt)    */
      1,              /* num cons created       (rcnt)    */
      rmatind.size(), /* cons num vars involved (nzcnt)   */
      &rhs,           /* cons right hand side   (rhs)     */
      &sense,         /* cons sense             (sense)   */
      &rmatbeg,       /* cons var matbeg        (rmatbeg) */
      rmatind.data(), /* cons var indices       (rmatind) */
      rmatval.data(), /* cons var coefficients  (rmatval) */
      NULL,           /* var name               (colname) */
      &row_name       /* cons name              (rowname) */
    );
    if (m_verbose) {
      if (cpx_st == 0) {
        std::cout << "done." << std::endl;
        cons_count++;
      } else {
        std::cout << "failed." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }
  if (m_verbose) {
    std::cout << "Degree constraints for all j (#" << cons_count
              << ") were successfully created." << std::endl;
  }

  cons_count = 0;

  if (m_verbose) {
    std::cout << "Creating degree constraints with: rhs = " << rhs
              << " and sense = " << sense << " for all i." << std::endl;
  }
  for (size_t i = 0; i < n; ++i) {
    rmatind.clear();
    rmatval.clear();
    for (size_t j = 0; j < n; ++j) {
      idx_y_ij = m_ymap[i * n + j];

      if (idx_y_ij == -1)
        continue; // only create cons for existing y vars.

      rmatind.push_back(idx_y_ij);
      rmatval.push_back(1.0);
    }

    row_name_buf = "i = " + std::to_string(i) + ": sum_j x_ij = 1";
    row_name = row_name_buf.c_str();
    if (m_verbose)
      std::cout << "Creating degree constraint " << row_name << " ... ";
    cpx_st = CPXXaddrows(
      m_cpx_env,      /* CPLEX environment      (env)     */
      m_cpx_lp,       /* CPLEX problem          (lp)      */
      0,              /* num vars created       (ccnt)    */
      1,              /* num cons created       (rcnt)    */
      rmatind.size(), /* cons num vars involved (nzcnt)   */
      &rhs,           /* cons right hand side   (rhs)     */
      &sense,         /* cons sense             (sense)   */
      &rmatbeg,       /* cons var matbeg        (rmatbeg) */
      rmatind.data(), /* cons var indices       (rmatind) */
      rmatval.data(), /* cons var coefficients  (rmatval) */
      NULL,           /* var name               (colname) */
      &row_name       /* cons name              (rowname) */
    );
    if (m_verbose) {
      if (cpx_st == 0) {
        std::cout << "done." << std::endl;
        cons_count++;
      } else {
        std::cout << "failed." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }
  if (m_verbose) {
    std::cout << "Degree constraints for all i (#" << cons_count
              << ") were successfully created." << std::endl;
  }
}

void SCFSolver::add_linking_cons() {
  size_t n = m_instance.n_holes();
  const char *row_name;
  std::string row_name_buf;
  double rhs = 0.0;
  char sense = 'L';
  CPXNNZ rmatbeg = 0;
  std::vector<int> rmatind(2);
  std::vector<double> rmatval(2);
  idx idx_x_ij, idx_y_ij;
  int cpx_st, cons_count = 0;
  if (m_verbose) {
    std::cout << "Creating linking constraints with: rhs = " << rhs
              << " and sense = " << sense << "." << std::endl;
  }
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      idx_x_ij = m_xmap[i * n + j];
      idx_y_ij = m_ymap[i * n + j];

      if (idx_x_ij == -1)
        continue; // only create cons for existing x vars.

      rmatind[0] = idx_x_ij;
      rmatind[1] = idx_y_ij;
      rmatval[0] = 1.00;
      rmatval[1] = 1.0 - n;

      row_name_buf = "x_" + std::to_string(i) + "_" + std::to_string(j) +
                     " + (" + std::to_string((int)(1 - n)) + ")" + "y_" +
                     std::to_string(i) + "_" + std::to_string(j) + " <= 0";
      row_name = row_name_buf.c_str();
      if (m_verbose)
        std::cout << "Creating linking constraint " << row_name << " ... ";
      cpx_st = CPXXaddrows(
        m_cpx_env,      /* CPLEX environment      (env)     */
        m_cpx_lp,       /* CPLEX problem          (lp)      */
        0,              /* num vars created       (ccnt)    */
        1,              /* num cons created       (rcnt)    */
        rmatind.size(), /* cons num vars involved (nzcnt)   */
        &rhs,           /* cons right hand side   (rhs)     */
        &sense,         /* cons sense             (sense)   */
        &rmatbeg,       /* cons var matbeg        (rmatbeg) */
        rmatind.data(), /* cons var indices       (rmatind) */
        rmatval.data(), /* cons var coefficients  (rmatval) */
        NULL,           /* var name               (colname) */
        &row_name       /* cons name              (rowname) */
      );
      if (m_verbose) {
        if (cpx_st == 0) {
          std::cout << "done." << std::endl;
          cons_count++;
        } else {
          std::cout << "failed." << std::endl;
          exit(EXIT_FAILURE);
        }
      }
    }
  }
  if (m_verbose) {
    std::cout << "All linking constraints (#" << cons_count
              << ") were successfully created." << std::endl;
  }
}

std::vector<int> SCFSolver::sequence_solution() const {
  const int num_vars = CPXXgetnumcols(m_cpx_env, m_cpx_lp);
  std::vector<double> all_vars(num_vars);

  const int status =
    CPXXgetx(m_cpx_env, m_cpx_lp, all_vars.data(), 0, num_vars - 1);
  if (status) {
    throw std::runtime_error("CPLEX error fetching solution variables");
  }

  const size_t n = m_instance.n_holes();
  std::vector<int> sequence;
  sequence.reserve(n + 1);

  size_t current = 0;
  sequence.push_back(0);

  while (true) {
    bool found = false;

    for (size_t j = 0; j < n && !found; ++j) {
      const idx y_idx = m_ymap[current * n + j];
      if (y_idx == -1)
        continue;

      if (all_vars[y_idx] > 1e-6) {
        current = j;
        sequence.push_back(j);
        found = true;
      }
    }

    if (!found) {
      throw std::logic_error("No outgoing arc found in solution");
    }

    if (current == 0)
      break;
  }

  return sequence;
}

void SCFSolver::print_sequence_solution() const {
  std::vector<int> seq = sequence_solution();

  for (size_t i = 0; i < seq.size(); ++i) {
    std::cout << seq[i] << " ";
  }
  std::cout << "\n";
}

void SCFSolver::print_solution() const {
  double obj;
  CPXXgetobjval(m_cpx_env, m_cpx_lp, &obj);
  int num_vars = CPXXgetnumcols(m_cpx_env, m_cpx_lp);
  int num_cons = CPXXgetnumrows(m_cpx_env, m_cpx_lp);

  std::cout << "Objective Value = " << obj << "\n";
  std::cout << "Hole Sequence = ";
  print_sequence_solution();
  std::cout << "#Variables = " << num_vars << "\n";
  std::cout << "#Constraints = " << num_cons << "\n";

  print_solution_with_names();
}

void SCFSolver::print_solution_with_names() const {
  int status;

  CPXDIM numcols = CPXXgetnumcols(m_cpx_env, m_cpx_lp);
  if (numcols == 0) {
    printf("No columns to print.\n");
    return;
  }

  CPXSIZE storespace = 0;
  CPXSIZE surplus = 0;

  status = CPXXgetcolname(
    m_cpx_env, m_cpx_lp, nullptr, nullptr, storespace, &surplus, 0,
    numcols - 1);

  CPXDIM required_space = -surplus;
  if (status != 0 && status != CPXERR_NEGATIVE_SURPLUS) {
    fprintf(stderr, "Error querying name space: %d\n", status);
    return;
  }

  char **name = (char **)malloc(sizeof(char *) * numcols);
  char *namestore = (char *)malloc(sizeof(char) * required_space);

  storespace = required_space;
  status = CPXXgetcolname(
    m_cpx_env, m_cpx_lp, name, namestore, storespace, &surplus, 0, numcols - 1);
  if (status) {
    fprintf(stderr, "Error fetching names: %d\n", status);
    free(name);
    free(namestore);
    return;
  }

  double *x = (double *)malloc(sizeof(double) * numcols);
  if (x == nullptr) {
    fprintf(stderr, "Memory allocation for solution failed.\n");
    free(name);
    free(namestore);
    return;
  }

  status = CPXXgetx(m_cpx_env, m_cpx_lp, x, 0, numcols - 1);

  if (status) {
    fprintf(
      stderr,
      "Error fetching solution (Code %d). Has the problem been solved?\n",
      status);
  } else {
    printf("%-20s | %s\n", "Column Name", "Value");
    printf("---------------------|-------------\n");

    for (CPXDIM j = 0; j < numcols; j++) {
      if (x[j] > 1e-6 || x[j] < -1e-6) {
        printf("%-20s | %f\n", name[j], x[j]);
      }
    }
  }

  // --- Cleanup ---
  free(name);
  free(namestore);
  free(x);
}