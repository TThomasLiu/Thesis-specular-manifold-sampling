add_library(fm_torch_bridge SHARED
    torch_bridge.cpp
)

target_compile_options(fm_torch_bridge PRIVATE -stdlib=libstdc++)
target_link_options(fm_torch_bridge PRIVATE -stdlib=libstdc++)
target_link_libraries(fm_torch_bridge PRIVATE ${TORCH_LIBRARIES})
target_include_directories(fm_torch_bridge PRIVATE ${TORCH_INCLUDE_DIRS})

set_target_properties(fm_torch_bridge PROPERTIES
POSITION_INDEPENDENT_CODE ON
INSTALL_RPATH "/usr/local/libtorch/lib"
BUILD_WITH_INSTALL_RPATH TRUE
)
set_target_properties(fm_torch_bridge PROPERTIES CXX_VISIBILITY_PRESET default)

add_dist(fm_torch_bridge)