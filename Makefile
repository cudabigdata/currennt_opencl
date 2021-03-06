# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tamnguyen/Workspace/CUDA/currennt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tamnguyen/Workspace/CUDA/currennt

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running interactive CMake command-line interface..."
	/usr/bin/cmake -i .
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/tamnguyen/Workspace/CUDA/currennt/CMakeFiles /home/tamnguyen/Workspace/CUDA/currennt/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/tamnguyen/Workspace/CUDA/currennt/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named currennt

# Build rule for target.
currennt: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 currennt
.PHONY : currennt

# fast build rule for target.
currennt/fast:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/build
.PHONY : currennt/fast

currennt/src/main.o: currennt/src/main.cpp.o
.PHONY : currennt/src/main.o

# target to build an object file
currennt/src/main.cpp.o:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt/src/main.cpp.o
.PHONY : currennt/src/main.cpp.o

currennt/src/main.i: currennt/src/main.cpp.i
.PHONY : currennt/src/main.i

# target to preprocess a source file
currennt/src/main.cpp.i:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt/src/main.cpp.i
.PHONY : currennt/src/main.cpp.i

currennt/src/main.s: currennt/src/main.cpp.s
.PHONY : currennt/src/main.s

# target to generate assembly for a file
currennt/src/main.cpp.s:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt/src/main.cpp.s
.PHONY : currennt/src/main.cpp.s

currennt_lib/src/Configuration.o: currennt_lib/src/Configuration.cpp.o
.PHONY : currennt_lib/src/Configuration.o

# target to build an object file
currennt_lib/src/Configuration.cpp.o:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/Configuration.cpp.o
.PHONY : currennt_lib/src/Configuration.cpp.o

currennt_lib/src/Configuration.i: currennt_lib/src/Configuration.cpp.i
.PHONY : currennt_lib/src/Configuration.i

# target to preprocess a source file
currennt_lib/src/Configuration.cpp.i:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/Configuration.cpp.i
.PHONY : currennt_lib/src/Configuration.cpp.i

currennt_lib/src/Configuration.s: currennt_lib/src/Configuration.cpp.s
.PHONY : currennt_lib/src/Configuration.s

# target to generate assembly for a file
currennt_lib/src/Configuration.cpp.s:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/Configuration.cpp.s
.PHONY : currennt_lib/src/Configuration.cpp.s

currennt_lib/src/NeuralNetwork.o: currennt_lib/src/NeuralNetwork.cpp.o
.PHONY : currennt_lib/src/NeuralNetwork.o

# target to build an object file
currennt_lib/src/NeuralNetwork.cpp.o:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/NeuralNetwork.cpp.o
.PHONY : currennt_lib/src/NeuralNetwork.cpp.o

currennt_lib/src/NeuralNetwork.i: currennt_lib/src/NeuralNetwork.cpp.i
.PHONY : currennt_lib/src/NeuralNetwork.i

# target to preprocess a source file
currennt_lib/src/NeuralNetwork.cpp.i:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/NeuralNetwork.cpp.i
.PHONY : currennt_lib/src/NeuralNetwork.cpp.i

currennt_lib/src/NeuralNetwork.s: currennt_lib/src/NeuralNetwork.cpp.s
.PHONY : currennt_lib/src/NeuralNetwork.s

# target to generate assembly for a file
currennt_lib/src/NeuralNetwork.cpp.s:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/NeuralNetwork.cpp.s
.PHONY : currennt_lib/src/NeuralNetwork.cpp.s

currennt_lib/src/data_sets/DataSet.o: currennt_lib/src/data_sets/DataSet.cpp.o
.PHONY : currennt_lib/src/data_sets/DataSet.o

# target to build an object file
currennt_lib/src/data_sets/DataSet.cpp.o:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/data_sets/DataSet.cpp.o
.PHONY : currennt_lib/src/data_sets/DataSet.cpp.o

currennt_lib/src/data_sets/DataSet.i: currennt_lib/src/data_sets/DataSet.cpp.i
.PHONY : currennt_lib/src/data_sets/DataSet.i

# target to preprocess a source file
currennt_lib/src/data_sets/DataSet.cpp.i:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/data_sets/DataSet.cpp.i
.PHONY : currennt_lib/src/data_sets/DataSet.cpp.i

currennt_lib/src/data_sets/DataSet.s: currennt_lib/src/data_sets/DataSet.cpp.s
.PHONY : currennt_lib/src/data_sets/DataSet.s

# target to generate assembly for a file
currennt_lib/src/data_sets/DataSet.cpp.s:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/data_sets/DataSet.cpp.s
.PHONY : currennt_lib/src/data_sets/DataSet.cpp.s

currennt_lib/src/data_sets/DataSetFraction.o: currennt_lib/src/data_sets/DataSetFraction.cpp.o
.PHONY : currennt_lib/src/data_sets/DataSetFraction.o

# target to build an object file
currennt_lib/src/data_sets/DataSetFraction.cpp.o:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/data_sets/DataSetFraction.cpp.o
.PHONY : currennt_lib/src/data_sets/DataSetFraction.cpp.o

currennt_lib/src/data_sets/DataSetFraction.i: currennt_lib/src/data_sets/DataSetFraction.cpp.i
.PHONY : currennt_lib/src/data_sets/DataSetFraction.i

# target to preprocess a source file
currennt_lib/src/data_sets/DataSetFraction.cpp.i:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/data_sets/DataSetFraction.cpp.i
.PHONY : currennt_lib/src/data_sets/DataSetFraction.cpp.i

currennt_lib/src/data_sets/DataSetFraction.s: currennt_lib/src/data_sets/DataSetFraction.cpp.s
.PHONY : currennt_lib/src/data_sets/DataSetFraction.s

# target to generate assembly for a file
currennt_lib/src/data_sets/DataSetFraction.cpp.s:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/data_sets/DataSetFraction.cpp.s
.PHONY : currennt_lib/src/data_sets/DataSetFraction.cpp.s

currennt_lib/src/helpers/JsonClasses.o: currennt_lib/src/helpers/JsonClasses.cpp.o
.PHONY : currennt_lib/src/helpers/JsonClasses.o

# target to build an object file
currennt_lib/src/helpers/JsonClasses.cpp.o:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/helpers/JsonClasses.cpp.o
.PHONY : currennt_lib/src/helpers/JsonClasses.cpp.o

currennt_lib/src/helpers/JsonClasses.i: currennt_lib/src/helpers/JsonClasses.cpp.i
.PHONY : currennt_lib/src/helpers/JsonClasses.i

# target to preprocess a source file
currennt_lib/src/helpers/JsonClasses.cpp.i:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/helpers/JsonClasses.cpp.i
.PHONY : currennt_lib/src/helpers/JsonClasses.cpp.i

currennt_lib/src/helpers/JsonClasses.s: currennt_lib/src/helpers/JsonClasses.cpp.s
.PHONY : currennt_lib/src/helpers/JsonClasses.s

# target to generate assembly for a file
currennt_lib/src/helpers/JsonClasses.cpp.s:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/helpers/JsonClasses.cpp.s
.PHONY : currennt_lib/src/helpers/JsonClasses.cpp.s

currennt_lib/src/layers/InputLayer.o: currennt_lib/src/layers/InputLayer.cpp.o
.PHONY : currennt_lib/src/layers/InputLayer.o

# target to build an object file
currennt_lib/src/layers/InputLayer.cpp.o:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/layers/InputLayer.cpp.o
.PHONY : currennt_lib/src/layers/InputLayer.cpp.o

currennt_lib/src/layers/InputLayer.i: currennt_lib/src/layers/InputLayer.cpp.i
.PHONY : currennt_lib/src/layers/InputLayer.i

# target to preprocess a source file
currennt_lib/src/layers/InputLayer.cpp.i:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/layers/InputLayer.cpp.i
.PHONY : currennt_lib/src/layers/InputLayer.cpp.i

currennt_lib/src/layers/InputLayer.s: currennt_lib/src/layers/InputLayer.cpp.s
.PHONY : currennt_lib/src/layers/InputLayer.s

# target to generate assembly for a file
currennt_lib/src/layers/InputLayer.cpp.s:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/layers/InputLayer.cpp.s
.PHONY : currennt_lib/src/layers/InputLayer.cpp.s

currennt_lib/src/layers/Layer.o: currennt_lib/src/layers/Layer.cpp.o
.PHONY : currennt_lib/src/layers/Layer.o

# target to build an object file
currennt_lib/src/layers/Layer.cpp.o:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/layers/Layer.cpp.o
.PHONY : currennt_lib/src/layers/Layer.cpp.o

currennt_lib/src/layers/Layer.i: currennt_lib/src/layers/Layer.cpp.i
.PHONY : currennt_lib/src/layers/Layer.i

# target to preprocess a source file
currennt_lib/src/layers/Layer.cpp.i:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/layers/Layer.cpp.i
.PHONY : currennt_lib/src/layers/Layer.cpp.i

currennt_lib/src/layers/Layer.s: currennt_lib/src/layers/Layer.cpp.s
.PHONY : currennt_lib/src/layers/Layer.s

# target to generate assembly for a file
currennt_lib/src/layers/Layer.cpp.s:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/layers/Layer.cpp.s
.PHONY : currennt_lib/src/layers/Layer.cpp.s

currennt_lib/src/layers/PostOutputLayer.o: currennt_lib/src/layers/PostOutputLayer.cpp.o
.PHONY : currennt_lib/src/layers/PostOutputLayer.o

# target to build an object file
currennt_lib/src/layers/PostOutputLayer.cpp.o:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/layers/PostOutputLayer.cpp.o
.PHONY : currennt_lib/src/layers/PostOutputLayer.cpp.o

currennt_lib/src/layers/PostOutputLayer.i: currennt_lib/src/layers/PostOutputLayer.cpp.i
.PHONY : currennt_lib/src/layers/PostOutputLayer.i

# target to preprocess a source file
currennt_lib/src/layers/PostOutputLayer.cpp.i:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/layers/PostOutputLayer.cpp.i
.PHONY : currennt_lib/src/layers/PostOutputLayer.cpp.i

currennt_lib/src/layers/PostOutputLayer.s: currennt_lib/src/layers/PostOutputLayer.cpp.s
.PHONY : currennt_lib/src/layers/PostOutputLayer.s

# target to generate assembly for a file
currennt_lib/src/layers/PostOutputLayer.cpp.s:
	$(MAKE) -f CMakeFiles/currennt.dir/build.make CMakeFiles/currennt.dir/currennt_lib/src/layers/PostOutputLayer.cpp.s
.PHONY : currennt_lib/src/layers/PostOutputLayer.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... currennt"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... currennt/src/main.o"
	@echo "... currennt/src/main.i"
	@echo "... currennt/src/main.s"
	@echo "... currennt_lib/src/Configuration.o"
	@echo "... currennt_lib/src/Configuration.i"
	@echo "... currennt_lib/src/Configuration.s"
	@echo "... currennt_lib/src/NeuralNetwork.o"
	@echo "... currennt_lib/src/NeuralNetwork.i"
	@echo "... currennt_lib/src/NeuralNetwork.s"
	@echo "... currennt_lib/src/data_sets/DataSet.o"
	@echo "... currennt_lib/src/data_sets/DataSet.i"
	@echo "... currennt_lib/src/data_sets/DataSet.s"
	@echo "... currennt_lib/src/data_sets/DataSetFraction.o"
	@echo "... currennt_lib/src/data_sets/DataSetFraction.i"
	@echo "... currennt_lib/src/data_sets/DataSetFraction.s"
	@echo "... currennt_lib/src/helpers/JsonClasses.o"
	@echo "... currennt_lib/src/helpers/JsonClasses.i"
	@echo "... currennt_lib/src/helpers/JsonClasses.s"
	@echo "... currennt_lib/src/layers/InputLayer.o"
	@echo "... currennt_lib/src/layers/InputLayer.i"
	@echo "... currennt_lib/src/layers/InputLayer.s"
	@echo "... currennt_lib/src/layers/Layer.o"
	@echo "... currennt_lib/src/layers/Layer.i"
	@echo "... currennt_lib/src/layers/Layer.s"
	@echo "... currennt_lib/src/layers/PostOutputLayer.o"
	@echo "... currennt_lib/src/layers/PostOutputLayer.i"
	@echo "... currennt_lib/src/layers/PostOutputLayer.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

