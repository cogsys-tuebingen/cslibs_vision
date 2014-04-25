cmake_minimum_required(VERSION 2.8.3)
project(utils_vision)

set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED COMPONENTS cv_bridge utils_plugin utils_param cmake_modules)

find_package(OpenCV REQUIRED)

find_package(Eigen QUIET)
if(NOT ${Eigen_FOUND})
  find_package(Eigen3 REQUIRED)
endif()
include_directories(${EIGEN_INCLUDE_DIRS})

find_package(Boost COMPONENTS program_options serialization REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES utils_vision logger
    CATKIN_DEPENDS utils_plugin utils_param
    DEPENDS Eigen
)

include_directories(include
    ${catkin_INCLUDE_DIRS}
)

##### COMMON
set(GLOBALS
    include/utils_vision/common/global.hpp)

##### LOGGER
add_library(logger
    include/utils_vision/common/global.hpp
    src/common/logger.cpp)

##### CORE
add_library(utils_vision
    src/utils/extractors_default.cpp

    src/config/config.cpp
    src/config/configurated_tools.cpp
    src/config/reconfigurable.cpp
    src/config/types.cpp
    src/data/angle.cpp
    src/data/directory_io.cpp
    src/data/frame.cpp
    src/data/frame_buffer.cpp
    src/data/frame_io.cpp
    src/data/matchable.cpp
    src/data/matchable_pose.cpp
    src/data/painter.cpp
    src/data/pose.cpp
    src/utils/extractor.cpp
    src/utils/extractor_factory.cpp
    src/utils/extractor_manager.cpp
    src/utils/hough_peak.cpp
    src/utils/matcher.cpp
    src/utils/match_scorer.cpp
    src/utils/match_scorer_homography.cpp
    src/utils/match_scorer_clustering.cpp
    src/utils/match_scorer_reprojection.cpp
    src/utils/match_scorer_factory.cpp
    include/utils_vision/utils/opencv_utils.hpp
    src/utils/grusig_descriptor.cpp
    src/utils/rectangle_cluster.cpp
    ${QT_UTIL}
    ${GLOBALS})
target_link_libraries(utils_vision yaml-cpp ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})
target_link_libraries(utils_vision logger)



install(TARGETS utils_vision
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(TARGETS logger
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(DIRECTORY include/utils_vision
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

