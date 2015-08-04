cmake_minimum_required(VERSION 2.8.3)
project(utils_vision)

## Enforce that we use C++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(catkin REQUIRED COMPONENTS cv_bridge utils_param cmake_modules)

# note to ubuntu 14 users: if opencv is installed at /usr/local and is not found:    sudo ldconfig /usr/local/lib
set(USED_OpenCV_COMPONENTS core highgui imgproc flann ml video)
find_package(OpenCV COMPONENTS nonfree ${USED_OpenCV_COMPONENTS} QUIET CONFIG)

if(${OpenCV_FOUND})
    add_definitions("-DCV_NON_FREE=1")
else()
    add_definitions("-DCV_NON_FREE=0")
    message("OpenCV nonfree was not found, SIFT and SURF will not be available. If you need them, compile OpenCV from source and remove CMakeCache.txt!")
    find_package(OpenCV REQUIRED COMPONENTS ${USED_OpenCV_COMPONENTS} CONFIG)
endif()

find_package(Eigen3 REQUIRED)

find_package(Boost COMPONENTS program_options serialization REQUIRED)

catkin_package(
    INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIR}
    LIBRARIES utils_vision logger
    CATKIN_DEPENDS utils_param
    DEPENDS Eigen3
)

include_directories(include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
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
    src/utils/extractors_default.cpp
    src/utils/extractor_factory.cpp
    src/utils/extractor_manager.cpp
    src/utils/flood.cpp
    src/utils/hough_peak.cpp
    src/utils/matcher.cpp
    src/utils/match_scorer.cpp
    src/utils/match_scorer_homography.cpp
    src/utils/match_scorer_clustering.cpp
    src/utils/match_scorer_reprojection.cpp
    src/utils/match_scorer_factory.cpp
    src/utils/perspective_transform.cpp
    src/utils/preemptiveSLIC.cpp
    src/utils/randomforest.cpp
    src/utils/undistortion.cpp

    #include/utils_vision/utils/cluster_boundaries.hpp
    #include/utils_vision/utils/color_functions.hpp
    #include/utils_vision/utils/heatmap.hpp
    #include/utils_vision/utils/histogram.hpp
    #include/utils_vision/utils/kernel.hpp
    #include/utils_vision/utils/log_odds.hpp
    #include/utils_vision/utils/noise_filter.hpp
    #include/utils_vision/utils/opencv_utils.hpp
    #include/utils_vision/textures/texture_descriptor.hpp
    #include/utils_vision/textures/lbp.hpp
    #include/utils_vision/textures/ltp.hpp
    #include/utils_vision/textures/wld.hpp
    #include/utils_vision/textures/homogenity.hpp

    src/utils/grusig_descriptor.cpp
    src/utils/rectangle_cluster.cpp
    src/utils/camera_calibration.cpp
    ${QT_UTIL}
    ${GLOBALS})
target_link_libraries(utils_vision yaml-cpp ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})
target_link_libraries(utils_vision logger)

add_executable(utils_vision_test_stereo
    src/tests/test_stereo.cpp
)

target_link_libraries(utils_vision_test_stereo
    ${OpenCV_LIBRARIES}
)

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

