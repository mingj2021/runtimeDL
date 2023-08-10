set(TARGET_NAME algorithms)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ALGORITHMS)
#file(GLOB ALGORITHMS "*.cpp")

add_library(${TARGET_NAME} SHARED ${ALGORITHMS})

target_link_libraries(${TARGET_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${TARGET_NAME} TorchVision::TorchVision)
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)