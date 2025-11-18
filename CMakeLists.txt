cmake_minimum_required(VERSION 3.18)
project(MyPolySolveApp LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 对 Release 模式下打开优化
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_CXX_STANDARD 17)

# # for amgcl
# # install MKL
# set(MKLROOT "/u/1/chenyang/intel/oneapi/mkl/2025.0")  
# set(MKL_INCLUDE_DIRS     "${MKLROOT}/include")
# set(MKL_LIBRARY_DIRS     "${MKLROOT}/lib/intel64")
# # set(MKL_INCLUDE_DIR "/u/1/chenyang/intel/oneapi/mkl/latest/include") 

# set(MKL_CORE_LIBRARY "/u/1/chenyang/intel/oneapi/mkl/2025.0/lib/intel64/libmkl_core.so")
# set(MKL_INTEL_LP64_LIBRARY "/u/1/chenyang/intel/oneapi/mkl/2025.0/lib/intel64/libmkl_intel_lp64.so")
# set(MKL_TBB_THREADING_LIBRARY "/u/1/chenyang/intel/oneapi/mkl/2025.0/lib/intel64/libmkl_tbb_thread.so")

# set(MKL_LINKING_STRATEGY "DYNAMIC")
# set(MKL_INTERFACE_LAYER  "LP64")
# set(MKL_THREADING_LAYER  "TBB")

# # install boost
# set(Boost_ROOT "/u/1/chenyang/boost/boost_1_83_0")
# set(Boost_INCLUDE_DIRS "/u/1/chenyang/boost/boost_1_83_0/stage/include")
# set(Boost_LIBRARY_DIRS "/u/1/chenyang/boost/boost_1_83_0/stage/lib")
# set(Boost_NO_SYSTEM_PATHS ON)

# #*********************OLD
# add_library(mkl::mkl UNKNOWN IMPORTED GLOBAL)
# # set_target_properties(mkl::mkl PROPERTIES
# #   INTERFACE_INCLUDE_DIRECTORIES "/u/1/chenyang/intel/oneapi/mkl/2025.0/include"
# #   IMPORTED_LOCATION            "/u/1/chenyang/intel/oneapi/mkl/2025.0/lib/intel64/libmkl_core.so"
# #   INTERFACE_LINK_LIBRARIES     "/u/1/chenyang/intel/oneapi/mkl/2025.0/lib/intel64/libmkl_intel_lp64.so;/u/1/chenyang/intel/oneapi/mkl/2025.0/lib/intel64/libmkl_tbb_thread.so"
# # )
# set_target_properties(mkl::mkl PROPERTIES
#   INTERFACE_INCLUDE_DIRECTORIES "/opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/include"
#   IMPORTED_LOCATION            "/opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64/libmkl_core.so"
#   INTERFACE_LINK_LIBRARIES     "/opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64/libmkl_intel_lp64.so;/opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64/libmkl_tbb_thread.so"
# )
# #*********************OLD

set(MKLROOT "/opt/intel/compilers_and_libraries_2019.5.281/linux/mkl")
set(MKL_INCLUDE_DIR "${MKLROOT}/include")
set(MKL_LIB_DIR     "${MKLROOT}/lib/intel64")


add_library(mkl::mkl UNKNOWN IMPORTED GLOBAL)
set_target_properties(mkl::mkl PROPERTIES
  IMPORTED_LOCATION            "${MKL_LIB_DIR}/libmkl_core.so"
  INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES     "${MKL_LIB_DIR}/libmkl_intel_lp64.so;${MKL_LIB_DIR}/libmkl_tbb_thread.so;${MKL_LIB_DIR}/libmkl_core.so"
  )

set(TBBROOT "/opt/intel/tbb")
set(TBB_LIB_DIR     "${TBBROOT}/lib/intel64_lin/gcc4.7")  # 注意根据你的系统用的子目录可能不同


# set(Boost_ROOT "/u/1/chenyang/boost/boost_1_83_0" CACHE PATH "")
set(Boost_NO_BOOST_CMAKE ON CACHE BOOL "")
set(Boost_INCLUDE_DIRS "/u/1/chenyang/boost/boost_1_83_0/stage/include")
set(Boost_LIBRARY_DIRS "/u/1/chenyang/boost/boost_1_83_0/stage/lib")
# set(Boost_ROOT "/usr/local/lib/cmake/Boost-1.71.0")
# set(Boost_INCLUDE_DIRS "/usr/local/lib/cmake/Boost-1.71.0/stage/include")
# set(Boost_LIBRARY_DIRS "/usr/local/lib/cmake/Boost-1.71.0/stage/lib")
# set(Boost_NO_SYSTEM_PATHS ON CACHE BOOL "")
find_package(Boost REQUIRED COMPONENTS serialization)
# add_library(Boost::boost INTERFACE IMPORTED GLOBAL)
# target_include_directories(Boost::boost INTERFACE ${Boost_INCLUDE_DIRS})
# target_link_libraries(Boost::boost INTERFACE ${Boost_LIBRARIES})


include_directories(
    include             
    /u/1/chenyang/benchmark/libs/polysolve
)

add_subdirectory(/u/1/chenyang/benchmark/libs/polysolve)

# include_directories(
#     include             
#     /u/1/chenyang/benchmark/libs/polyfem_debug_set/polysolve)

# add_subdirectory(/u/1/chenyang/benchmark/libs/polyfem_debug_set/polysolve)



set(TestMainSource
    src/main.cpp     
    src/test.cpp 
)
add_executable(TestMain ${TestMainSource})
# target_link_libraries(TestMain polysolve)
target_include_directories(TestMain PRIVATE "${MKL_INCLUDE_DIR}")
target_link_libraries(TestMain PRIVATE
    polysolve
    mkl::mkl
    pthread
    m
    dl
    "${TBB_LIB_DIR}/libtbb.so"   
)

set_target_properties(TestMain PROPERTIES
  LINK_FLAGS "-Wl,--no-as-needed"
)










set(TestMatSource
    src/main_test_mat.cpp     
    src/save_problem.cpp     
)
add_executable(TestMat ${TestMatSource})
# target_link_libraries(TestMat polysolve)
target_include_directories(TestMat PRIVATE "${MKL_INCLUDE_DIR}")
target_link_libraries(TestMat PRIVATE
    polysolve
    mkl::mkl
    pthread
    m
    dl
    "${TBB_LIB_DIR}/libtbb.so"   
)
set_target_properties(TestMat PROPERTIES
  LINK_FLAGS "-Wl,--no-as-needed"
)



set(TestMatTimeSource
    src/main_test_mat_time.cpp     
    src/save_problem.cpp     
)
add_executable(TestMatTime ${TestMatTimeSource})
target_include_directories(TestMatTime PRIVATE "${MKL_INCLUDE_DIR}")
target_link_libraries(TestMatTime PRIVATE
    polysolve
    mkl::mkl
    pthread
    m
    dl
    "${TBB_LIB_DIR}/libtbb.so"   
)
set_target_properties(TestMatTime PROPERTIES
  LINK_FLAGS "-Wl,--no-as-needed"
)

set(TestMatCNSource
    src/main_test_mat_CN.cpp     
    src/save_problem.cpp     
)
add_executable(TestMatCN ${TestMatCNSource})
target_include_directories(TestMatCN PRIVATE "${MKL_INCLUDE_DIR}")
target_link_libraries(TestMatCN PRIVATE
    polysolve
    mkl::mkl
    pthread
    m
    dl
    "${TBB_LIB_DIR}/libtbb.so"   
)
set_target_properties(TestMatCN PROPERTIES
  LINK_FLAGS "-Wl,--no-as-needed"
)

set(TestMatLoggerSource
    src/main_test_mat_logger.cpp     
    src/save_problem.cpp     
)
add_executable(TestMatLogger ${TestMatLoggerSource})
target_include_directories(TestMatLogger PRIVATE "${MKL_INCLUDE_DIR}")
target_link_libraries(TestMatLogger PRIVATE
    polysolve
    mkl::mkl
    pthread
    m
    dl
    "${TBB_LIB_DIR}/libtbb.so"   
)
set_target_properties(TestMatLogger PROPERTIES
  LINK_FLAGS "-Wl,--no-as-needed"
)