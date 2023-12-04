cmake_minimum_required(VERSION 3.5)
project(gv_recon LANGUAGES CXX CUDA)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(OpenGL_GL_PREFERENCE LEGACY)
# set(CUDAToolkit_LIBRARIES "/usr/local/cuda-11.3/lib64/")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
# find_package(octomap REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(gpu_voxels REQUIRED)
find_package(icl_core REQUIRED)
find_package(statgrab REQUIRED)
find_package(PCL 1.10 REQUIRED COMPONENTS common io)
find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Open3D REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(TBB REQUIRED)
#find_package(TooN   REQUIRED)
add_definitions(-D_USE_MATH_DEFINES)
add_definitions(${icl_core_DEFINITIONS} ${gpu_voxels_DEFINITIONS})

###########
## Build ##
###########
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${STATGRAB_INCLUDE_DIR}
  ${OCTOMAP_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${gpu_voxels_INCLUDE_DIRS}
  ${message_filters_INCLUDE_DIRS}
	include
	#thirdparty
  #getopt
  "/usr/local/cuda-11.3/samples/common/inc"
)
#message(${gpu_voxels_LIBRARIES})

if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}") 
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

###################################  CUDA  ###################################
if(CUDA_FOUND)
  #message("cuda found")
  set(CUDA_PROPAGATE_HOST_FLAGS ON)

  # for call gpu_voxels kernel
  set(kfusion_cuda_srcs
	    src/cuda/DeviceKernel.cu
      src/cuda/marchingCubes_kernel.cu
      )
  cuda_add_library(kfusion-cuda STATIC
              ${kfusion_cuda_srcs} OPTIONS -O3
              )
  target_link_libraries(kfusion-cuda)

endif(CUDA_FOUND)

##########################################################################
LIST(APPEND SOURCEFILES
    src/SyncedSubNode.cpp
    src/marchingCubes.cpp
    # include/glDefines.h
    src/main.cpp
    )

add_executable(az_yoso ${SOURCEFILES})

target_link_directories(az_yoso PUBLIC
  ${Open3D_LIBRARY_DIRS}
  )

target_link_libraries(az_yoso
                      kfusion-cuda
                      ${OPENGL_LIBRARIES}
                      # ${GLEW_LIBRARIES}
                      ${OpenCV_LIBS}
                      ${STATGRAB_LIBRARIES}
                      # ${OCTOMAP_LIBRARIES}
                      ${PCL_LIBRARIES}
                      ${icl_core_LIBRARIES}
                      ${gpu_voxels_LIBRARIES}
                      ${GLUT_LIBRARIES}
                      Open3D::Open3D
                      )
ament_target_dependencies(az_yoso
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  message_filters
  )
##########################################################################



install(TARGETS
  az_yoso
  DESTINATION lib/${PROJECT_NAME}
  )

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
  )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()



# Macro for ament package
ament_package()
