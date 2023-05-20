cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TARGET_SOC "rk356x")
set(CMAKE_C_COMPILER "aarch64")
set(LIB_ARCH "aarch64")
# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")


# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/runtime/RK356X/${CMAKE_SYSTEM_NAME}/librknn_api)
#set(RKNN_RT_LIB /usr/lib/librknnrt.so)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)



message(${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.soaa)
include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)

message("")
# opencv
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)

#rga
#set(RGA_PATH ${CMAKE_SOURCE_DIR}/RK356X/include/librga)
set(RGA_PATH ${CMAKE_SOURCE_DIR}/librga/include)
set(RGA_LIB ${CMAKE_SOURCE_DIR}/librga/libs/Linux/gcc-aarch64/librga.a)
include_directories( ${RGA_PATH})
include_directories( ${CMAKE_SOURCE_DIR}/include)
message(${RGA_PATH}aaa)


# rknn_yolov5_demo
add_executable(rknn_yolov5_demo
        src/main.cpp
        src/postprocess.cpp
)
message(  ${OpenCV_LIBS})
target_link_libraries(rknn_yolov5_demo
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
)


