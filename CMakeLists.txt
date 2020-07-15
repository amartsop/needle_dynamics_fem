cmake_minimum_required( VERSION 3.10 )

project( needle_dynamics )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#Optimization parameters
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Os")

# For gnuplot
find_package(Boost COMPONENTS filesystem system iostreams REQUIRED)

# Armadillo linking
find_package(Armadillo REQUIRED)

# Include directories
include_directories(
    ./include/
)

set(ALL_LIBS
  ${ARMADILLO_LIBRARIES} ${Boost_LIBRARIES})

set(SOURCES
  ./src/dynamics_math.cpp
  ./src/rigid_body.cpp
  ./src/flexible_beam_fem.cpp
  ./src/beam_element.cpp
  ./src/euler_rotations.cpp
  ./src/system_fem.cpp
  ./src/input_trajectory.cpp
)

  
# target
add_executable(main main.cpp ${SOURCES})

target_include_directories(main PRIVATE ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(main ${ALL_LIBS})
