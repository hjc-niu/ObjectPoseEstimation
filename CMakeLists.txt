# Specifies the minimum version of CMake
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

################################################################################
# --- EDIT THE VERSION HERE AFTER CHANGING THE CODE ---
# Specifies the version of project
set(OPE_VERSION_MAJOR 1)
set(OPE_VERSION_MINOR 0)
set(OPE_VERSION_PATCH 0)
set(OPE_VERSION       "${OPE_VERSION_MAJOR}.${OPE_VERSION_MINOR}.${OPE_VERSION_PATCH}")
message("### INFO: The version of project is ${OPE_VERSION} ###")
################################################################################

# Set project name
set(OPE_PROJECT_NAME "object_pose_estimation_v${OPE_VERSION}")
message("### INFO: The name of project is ${OPE_PROJECT_NAME} ###")

PROJECT(${OPE_PROJECT_NAME})

# Find PCL library
# sudo apt install libpcl-dev
# Ubuntu 20.04 -> v1.10.0
# Ubuntu 18.04 -> v1.8.1
# Ubuntu 16.04 -> v1.7.2
find_package(PCL 1.2 REQUIRED)

# Specifies the path of the head and library files of PCL
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add the directories of the head files
# Note:
# the directories of the head files must be specified before calling add_executable
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Find out all source files from the current directory
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SOURCE_FILES)
message("### INFO: The version of project is ${SOURCE_FILES} ###")

# Specifies the output path of the executable program
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Add the source files
add_executable(${OPE_PROJECT_NAME} ${SOURCE_FILES})

# Link the executable file with the PCL library files
target_link_libraries(${OPE_PROJECT_NAME} ${PCL_LIBRARIES})
