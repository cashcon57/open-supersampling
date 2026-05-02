# OSS-Gaussian Vulkan port build — Sprint 7 / T7.V.1.
#
# Builds two artifacts:
#   1. rasterizer.spv         — SPIR-V compiled from rasterizer.comp via glslangValidator.
#   2. liboss_gaussian_vk     — C++ host harness (rasterizer.cpp) linking the Vulkan loader.
#
# Requires the Vulkan SDK on PATH (provides glslangValidator + vulkan headers).
# On Arch / CachyOS / SteamOS dev setups: `sudo pacman -S vulkan-devel shaderc`.
# On Ubuntu / Debian: `apt install libvulkan-dev glslang-tools`.

cmake_minimum_required(VERSION 3.20)
project(oss_gaussian_vk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)

# --- SPIR-V compilation -------------------------------------------------------
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

set(COMP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/rasterizer.comp)
set(SPV_OUT  ${CMAKE_CURRENT_BINARY_DIR}/rasterizer.spv)

add_custom_command(
    OUTPUT  ${SPV_OUT}
    COMMAND ${GLSLANG_VALIDATOR} -V ${COMP_SRC} -o ${SPV_OUT}
    DEPENDS ${COMP_SRC}
    COMMENT "Compiling rasterizer.comp -> rasterizer.spv"
    VERBATIM
)
add_custom_target(rasterizer_spv ALL DEPENDS ${SPV_OUT})

# --- Host library -------------------------------------------------------------
add_library(oss_gaussian_vk STATIC rasterizer.cpp)
target_link_libraries(oss_gaussian_vk PUBLIC Vulkan::Vulkan)
target_compile_options(oss_gaussian_vk PRIVATE -Wall -Wextra -Werror)
add_dependencies(oss_gaussian_vk rasterizer_spv)
