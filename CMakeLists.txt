cmake_minimum_required(VERSION 2.8.3)
project(cslibs_vision)

## Enforce that we use C++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    add_definitions(-std=c++11)
elseif(COMPILER_SUPPORTS_CXX0X)
    add_definitions(-std=c++0x)
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(catkin REQUIRED COMPONENTS cv_bridge)

# note to ubuntu 14 users: if opencv is installed at /usr/local and is not found:    sudo ldconfig /usr/local/lib
set(USED_OpenCV_COMPONENTS core highgui imgproc flann ml video calib3d)
find_package(OpenCV COMPONENTS nonfree ${USED_OpenCV_COMPONENTS} QUIET CONFIG)

list(FIND OpenCV_LIB_COMPONENTS opencv_nonfree NONFREE_INDEX)
set(NONFREE_FOUND NOT ${NONFREE_INDEX} EQUAL -1)

if(${OpenCV_FOUND} AND ${NONFREE_FOUND})
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
    LIBRARIES ${PROJECT_NAME} logger
    CATKIN_DEPENDS
    DEPENDS Eigen3
)

include_directories(include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

##### COMMON
set(GLOBALS
    include/cslibs_vision/common/global.hpp)

##### LOGGER
add_library(logger
    include/cslibs_vision/common/global.hpp
    src/common/logger.cpp)

##### CORE
add_library(cslibs_vision
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
    src/utils/flood.cpp
    src/utils/hough_peak.cpp
    src/utils/matcher.cpp
    src/utils/match_scorer.cpp
    src/utils/match_scorer_homography.cpp
    src/utils/match_scorer_clustering.cpp
    src/utils/match_scorer_reprojection.cpp
    src/utils/perspective_transform.cpp
    src/utils/preemptiveSLIC.cpp
    src/utils/randomforest.cpp
    src/utils/undistortion.cpp

    #include/cslibs_vision/utils/cluster_boundaries.hpp
    #include/cslibs_vision/utils/color_functions.hpp
    #include/cslibs_vision/utils/heatmap.hpp
    #include/cslibs_vision/utils/histogram.hpp
    #include/cslibs_vision/utils/kernel.hpp
    #include/cslibs_vision/utils/log_odds.hpp
    #include/cslibs_vision/utils/noise_filter.hpp
    #include/cslibs_vision/utils/opencv_utils.hpp
    #include/cslibs_vision/textures/texture_descriptor.hpp
    #include/cslibs_vision/textures/lbp.hpp
    #include/cslibs_vision/textures/ltp.hpp
    #include/cslibs_vision/textures/wld.hpp
    #include/cslibs_vision/textures/homogenity.hpp

    src/utils/grusig_descriptor.cpp
    src/utils/rectangle_cluster.cpp
    src/utils/camera_calibration.cpp
    ${QT_UTIL}
    ${GLOBALS})
target_link_libraries(cslibs_vision yaml-cpp ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})
target_link_libraries(cslibs_vision logger)

add_custom_target(show_cslibs_vision_header_in_qt SOURCES
    include/cslibs_vision/data/angle.h
    include/cslibs_vision/data/directory_io.h
    include/cslibs_vision/data/frame_buffer.h
    include/cslibs_vision/data/frame.h
    include/cslibs_vision/data/frame_io.h
    include/cslibs_vision/data/matchable.h
    include/cslibs_vision/data/matchable_pose.h
    include/cslibs_vision/data/painter.h
    include/cslibs_vision/data/pose.h
    include/cslibs_vision/data/robot.h

    include/cslibs_vision/features/magnitude.hpp
    include/cslibs_vision/features/homogenity.hpp
    include/cslibs_vision/features/lbp.hpp
    include/cslibs_vision/features/ltp.hpp
    include/cslibs_vision/features/wld.hpp
    include/cslibs_vision/features/hog.hpp
    include/cslibs_vision/features/acf.hpp

    include/cslibs_vision/utils/camera_calibration.h
    include/cslibs_vision/utils/cluster_boundaries.hpp
    include/cslibs_vision/utils/color_functions.hpp
    include/cslibs_vision/utils/extractor.h
    include/cslibs_vision/utils/flood.h
    include/cslibs_vision/utils/grusig_descriptor.h
    include/cslibs_vision/utils/heatmap.hpp
    include/cslibs_vision/utils/histogram.hpp
    include/cslibs_vision/utils/hough_peak.h
    include/cslibs_vision/utils/kernel.hpp
    include/cslibs_vision/utils/lbp.h
    include/cslibs_vision/utils/log_odds.hpp
    include/cslibs_vision/utils/matcher.h
    include/cslibs_vision/utils/match_scorer_clustering.h
    include/cslibs_vision/utils/match_scorer.h
    include/cslibs_vision/utils/match_scorer_homography.h
    include/cslibs_vision/utils/match_scorer_reprojection.h
    include/cslibs_vision/utils/noise_filter.hpp
    include/cslibs_vision/utils/opencv_utils.hpp
    include/cslibs_vision/utils/perspective_transform.h
    include/cslibs_vision/utils/preemptiveSLIC.h
    include/cslibs_vision/utils/randomforest.h
    include/cslibs_vision/utils/rectangle_cluster.h
    include/cslibs_vision/utils/stereo_parameters.hpp
    include/cslibs_vision/utils/undistortion.h
)


install(TARGETS ${PROJECT_NAME}  logger
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY include/cslibs_vision
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

