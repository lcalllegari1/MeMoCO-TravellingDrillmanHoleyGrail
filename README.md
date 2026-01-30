# Project Structure

```
TravellingDrillmanHoleyGrail-MeMoCO-project
├── data/ (see below for details)
├── include
│   └── lib
│       ├── held_karp_lb.hpp
│       ├── heuristic_utils.hpp
│       ├── hierarchical_tabu_search.hpp
│       ├── instance.hpp
│       ├── SCF_solver.hpp
│       ├── types.hpp
│       └── utils.hpp
├── Makefile
├── notebooks
│   ├── data
│   │   ├── demo_instance_n100_d1e87418.clusters
│   │   ├── demo_instance_n100_d1e87418.dat
│   │   ├── demo_instance_n100_d1e87418.json
│   │   └── tour.txt
│   └── demo.ipynb
├── scripts
│   ├── build_cost_matrix.py
│   ├── compute_clusters.py
│   ├── config
│   │   ├── board_layouts.yaml
│   │   └── drill_specs.yaml
│   ├── generate_board.py
│   ├── plot_board.py
│   ├── plot_clusters.py
│   └── plot_tour.py
└── src
    ├── apps
    │   ├── batch_cplex_solver.cpp
    │   ├── batch_heuristic_solver.cpp
    │   ├── demo_solver.cpp
    │   ├── lower_bound.cpp
    │   ├── params_grid_search.cpp
    │   ├── single_instance_cplex_solver.cpp
    │   └── single_instance_heuristic_solver.cpp
    └── lib
        ├── held_karp_lb.cpp
        ├── heuristic_utils.cpp
        ├── hierarchical_tabu_search.cpp
        ├── instance.cpp
        ├── SCF_solver.cpp
        └── utils.cpp
```

The `include` folder contains the headers, while the `src` folder contains the implementations (`lib`) and the executables (`apps`).

The `notebooks` folder contains a Python Notebook that allows for inspection of the overall pipeline applied to some instance.

The `scripts` folder contains Python code that allows instance-related tasks, from generation and computation of the cost matrices and clusters, to the visualization of the boards and the tours.

The `data` folder contains some examples of instances, and the organization of this folder is provided below for clarity.

```
data/
├── boards
│   ├── random
│   │   ├── n100_0fd7f9f9.json
│   │   ├── n10_3a1adae4.json
│   │   └── n50_0f490dd4.json
│   └── structured
│       ├── n100_0bb09919.json
│       └── n50_0d0caba8.json
├── clusters
│   ├── random
│   │   ├── n100_0fd7f9f9.clusters
│   │   ├── n10_3a1adae4.clusters
│   │   └── n50_0f490dd4.clusters
│   └── structured
│       ├── n100_0bb09919.clusters
│       └── n50_0d0caba8.clusters
├── costs
│   ├── random
│   │   ├── n100_0fd7f9f9.dat
│   │   ├── n10_3a1adae4.dat
│   │   └── n50_0f490dd4.dat
│   └── structured
│       ├── n100_0bb09919.dat
│       └── n50_0d0caba8.dat
└── plots
    ├── random
    │   ├── clusters
    │   │   ├── n100_0fd7f9f9.png
    │   │   ├── n10_3a1adae4.png
    │   │   └── n50_0f490dd4.png
    │   ├── n100_0fd7f9f9.png
    │   ├── n10_3a1adae4.png
    │   └── n50_0f490dd4.png
    └── structured
        ├── clusters
        │   ├── n100_0bb09919.png
        │   └── n50_0d0caba8.png
        ├── n100_0bb09919.png
        └── n50_0d0caba8.png
```

Further instructions to run the solvers are provided by trying to run the executables directly, or by reading the proper sections in the final report.