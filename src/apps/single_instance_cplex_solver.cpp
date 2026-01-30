#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

#include "lib/SCF_solver.hpp"
#include "lib/instance.hpp"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <input_file> [--verbose <0|1|2>] [--average <num_runs>]\n";
    return 1;
  }

  std::string input_file = argv[1];
  int verbose_level = 1;
  int average_runs = 1;
  bool average_mode = false;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--verbose") {
      if (i + 1 < argc) {
        try {
          verbose_level = std::stoi(argv[i + 1]);
          if (verbose_level < 0 || verbose_level > 2)
            verbose_level = 1;
          ++i;
        } catch (...) {
          verbose_level = 1;
        }
      } else {
        verbose_level = 1;
      }
    } else if (arg == "--average") {
      average_mode = true;
      if (i + 1 < argc) {
        try {
          average_runs = std::stoi(argv[i + 1]);
          if (average_runs <= 0)
            average_runs = 5;
          ++i;
        } catch (...) {
          average_runs = 5;
        }
      } else {
        average_runs = 5;
      }
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  try {
    Instance inst(input_file);
    if (verbose_level >= 1) {
      std::cout << "Loaded instance: " << inst.instance_name() << "\n";
      std::cout << "Dimension: " << inst.n_holes() << "\n";
    }

    int runs = average_mode ? average_runs : 1;
    double total_build_time = 0.0;
    double total_solve_time = 0.0;
    double last_obj = 0.0;

    for (int r = 0; r < runs; ++r) {
      auto t1 = std::chrono::high_resolution_clock::now();
      SCFSolver solver(inst, false);
      auto t2 = std::chrono::high_resolution_clock::now();
      last_obj = solver.solve();
      auto t3 = std::chrono::high_resolution_clock::now();

      total_build_time += std::chrono::duration<double>(t2 - t1).count();
      total_solve_time += std::chrono::duration<double>(t3 - t2).count();
    }

    if (verbose_level >= 1) {
      std::cout << "\n=== Solution Details ===\n";
      SCFSolver solver(inst, verbose_level >= 2);
      last_obj = solver.solve();

      if (verbose_level >= 2) {
        std::cout << "\nFull variable values (non-zero only):\n";
        solver.print_solution();
      }

      std::cout << "\nSolution sequence: ";
      solver.print_sequence_solution();
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "Objective: " << last_obj << "\n";
    if (average_mode) {
      std::cout << "Average model creation time (" << runs
                << " runs) (seconds): " << (total_build_time / runs) << " \n";
      std::cout << "Average solve time (" << runs
                << " runs) (seconds): " << (total_solve_time / runs) << "\n";
    } else {
      std::cout << "Model creation time (seconds): " << total_build_time
                << "\n";
      std::cout << "Solve time (seconds): " << total_solve_time << "\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
