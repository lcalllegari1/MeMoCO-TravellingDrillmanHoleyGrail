#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lib/SCF_solver.hpp"
#include "lib/instance.hpp"

std::vector<std::string> list_dat_files(const std::string &folder) {
  std::vector<std::string> files;
  DIR *dir = opendir(folder.c_str());
  if (!dir) {
    std::cerr << "Cannot open folder: " << folder << "\n";
    return files;
  }
  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);
    if (name.size() > 4 && name.substr(name.size() - 4) == ".dat") {
      files.push_back(folder + "/" + name);
    }
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

std::string format_double(double val) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(9) << val;

  std::string s = oss.str();
  size_t dot = s.find('.');
  std::string int_part = s.substr(0, dot);
  std::string frac_part = s.substr(dot);

  if (int_part.size() < 4) {
    int_part = std::string(4 - int_part.size(), ' ') + int_part;
  }

  return int_part + frac_part;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr
      << "Usage: " << argv[0]
      << " <instances_folder> [--average <num_runs>] [--output <filename>]\n";
    return 1;
  }

  std::string folder = argv[1];
  int average_runs = 1;
  std::string output_file = "results.dat";

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--average" && i + 1 < argc) {
      average_runs = std::max(1, std::atoi(argv[++i]));
    } else if (arg == "--output" && i + 1 < argc) {
      output_file = argv[++i];
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  std::vector<std::string> instance_files = list_dat_files(folder);
  if (instance_files.empty()) {
    std::cerr << "No .dat files found in folder " << folder << "\n";
    return 1;
  }

  std::ofstream out(output_file);
  if (!out) {
    std::cerr << "Cannot open output file " << output_file << "\n";
    return 1;
  }

  struct Result {
    std::string name;
    double avg_creation;
    double avg_solve;
    double obj;
  };
  std::vector<Result> results(instance_files.size());

  std::cout << std::left << std::setw(30) << "Instance" << " | " << std::right
            << std::setw(15) << "Creation(s)" << " | " << std::setw(15)
            << "Solve(s)" << " | " << std::setw(15) << "Objective" << "\n";
  std::cout << std::string(30, '-') << "-|-" << std::string(15, '-') << "-|-"
            << std::string(15, '-') << "-|-" << std::string(15, '-') << "\n";

  for (size_t idx = 0; idx < instance_files.size(); ++idx) {
    const std::string &inst_file = instance_files[idx];
    double total_creation_time = 0.0;
    double total_solve_time = 0.0;
    double objective_value = 0.0;

    for (int run = 0; run < average_runs; ++run) {
      auto start_creation = std::chrono::high_resolution_clock::now();
      Instance inst(inst_file);
      SCFSolver solver(inst, false);
      auto end_creation = std::chrono::high_resolution_clock::now();
      total_creation_time +=
        std::chrono::duration<double>(end_creation - start_creation).count();

      auto start_solve = std::chrono::high_resolution_clock::now();
      objective_value = solver.solve();
      auto end_solve = std::chrono::high_resolution_clock::now();
      total_solve_time +=
        std::chrono::duration<double>(end_solve - start_solve).count();
    }

    std::string base_name = inst_file.substr(inst_file.find_last_of("/\\") + 1);
    size_t last_dot = base_name.find_last_of(".");
    if (last_dot != std::string::npos) {
        base_name = base_name.substr(0, last_dot);
    }
    
    results[idx].name = base_name;
    results[idx].avg_creation = total_creation_time / average_runs;
    results[idx].avg_solve = total_solve_time / average_runs;
    results[idx].obj = objective_value;

    std::cout << std::left << std::setw(30) << results[idx].name << " | "
              << std::right << std::setw(15)
              << format_double(results[idx].avg_creation) << " | "
              << std::setw(15) << format_double(results[idx].avg_solve) << " | "
              << std::setw(15) << format_double(results[idx].obj) << "\n";
  }

  for (const auto &res : results) {
    out << res.name << " " << res.avg_creation << " " << res.avg_solve << " "
        << res.obj << "\n";
  }

  std::cout << "All instances processed (average of " << average_runs 
            << " runs for each instance). Results saved to " << output_file 
            << "\n";
  return 0;
}