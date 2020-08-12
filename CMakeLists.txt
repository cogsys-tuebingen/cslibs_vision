cmake_minimum_required(VERSION 3.5.0)
project(cslibs_vision)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

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

find_package(catkin REQUIRED)

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
)

##### LOGGER
add_library(logger
    include/cslibs_vision/common/global.hpp
    src/common/logger.cpp)
targeT_include_directories(logger
    PUBLIC
        include
)

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

    src/utils/grusig_descriptor.cpp
    src/utils/rectangle_cluster.cpp
    src/utils/camera_calibration.cpp

    ${QT_UTIL}
)
target_link_libraries(cslibs_vision yaml-cpp ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})
target_link_libraries(cslibs_vision logger)
target_include_directories(cslibs_vision
    PUBLIC
        include
    SYSTEM PUBLIC
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)
target_compile_options(cslibs_vision
    PRIVATE
        -Wno-ignored-attributes
)


install(TARGETS ${PROJECT_NAME}  logger
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY include/cslibs_vision/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
