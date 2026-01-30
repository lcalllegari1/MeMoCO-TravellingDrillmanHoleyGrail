# ==========================================
# Settings
# ==========================================

CXX      := g++
# Updated to c++17 to support std::filesystem
CXXFLAGS := -std=c++11 -Ofast -Wall -Wextra -g
CXXFLAGS += -MMD -MP

# CPLEX Configuration
CPX_BASE    := /opt/ibm/ILOG/CPLEX_Studio2211
CPX_INCDIR  := $(CPX_BASE)/cplex/include
CPX_LIBDIR  := $(CPX_BASE)/cplex/lib/x86-64_linux/static_pic
# Added -L$(CPX_LIBDIR) so the linker finds the static lib
CPX_LDFLAGS := -L$(CPX_LIBDIR) -lcplex -lm -pthread -ldl

# Include both your local headers and CPLEX headers
INCLUDES := -Iinclude -I$(CPX_INCDIR)
SRC_DIR   := src
BUILD_DIR := build
BIN_DIR   := bin

# ==========================================
# Source Discovery
# ==========================================

# 1. Find all "Library" sources recursively (Keep this as is)
#    This finds everything in src/lib and subfolders
LIB_SRCS := $(shell find src/lib -name '*.cpp')

# 2. Find "App" sources NON-RECURSIVELY
#    $(wildcard ...) only looks in the specified folder, ignoring subfolders.
#    This picks up 'lower_bound.cpp' and 'test.cpp', but IGNORES 'tests/instance.cpp'
APP_SRCS := $(wildcard src/apps/*.cpp)

# ==========================================
# Object & Target Generation
# ==========================================

# Create list of library object files
LIB_OBJS := $(LIB_SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Calculate the final executable paths
# src/apps/test.cpp -> bin/test
APP_TARGETS := $(patsubst $(SRC_DIR)/apps/%.cpp, $(BIN_DIR)/%, $(APP_SRCS))

# Dependency files
ALL_SRCS := $(LIB_SRCS) $(APP_SRCS)
DEPS     := $(ALL_SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.d)

# ==========================================
# Object & Target Generation
# ==========================================

# Create list of library object files
# src/lib/utils.cpp -> build/lib/utils.o
LIB_OBJS := $(LIB_SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Calculate the final executable paths based on app filenames
# src/apps/test.cpp -> bin/test
# src/apps/solver.cpp -> bin/solver
APP_TARGETS := $(patsubst $(SRC_DIR)/apps/%.cpp, $(BIN_DIR)/%, $(APP_SRCS))

# Calculate all dependency files (libs + apps) for inclusion
ALL_SRCS := $(LIB_SRCS) $(APP_SRCS)
DEPS     := $(ALL_SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.d)

# ==========================================
# Build Rules
# ==========================================

# Default target: Build ALL executables
all: $(APP_TARGETS)
	@echo "Build complete. Created: $(notdir $(APP_TARGETS))"

# Generic Linking Rule for ANY app in src/apps
$(BIN_DIR)/%: $(BUILD_DIR)/apps/%.o $(LIB_OBJS)
	@echo "Linking executable: $@"
	@mkdir -p $(BIN_DIR)
	$(CXX) $< $(LIB_OBJS) $(CPX_LDFLAGS) -o $@

# Compile ANY source file (lib or app) into an object file
# This preserves the directory structure inside build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Convenience targets (allows you to type "make test" instead of "make bin/test")
# This defines a phony target for every app name
$(foreach app,$(APP_SRCS),$(eval $(notdir $(basename $(app))): $(BIN_DIR)/$(notdir $(basename $(app)))))

# Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Include dependency files
-include $(DEPS)

.PHONY: all clean