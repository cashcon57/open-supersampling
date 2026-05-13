# =============================================================================
#  OSS-Gaussian Interception DLL
#  Copyright 2026 OSS-Gaussian contributors
#  Licensed under the Apache License, Version 2.0 (see ./LICENSE)
#
#  Modeled on OptiScaler project layout (https://github.com/optiscaler/OptiScaler).
#  Builds a Windows DLL named `dxgi.dll` for game-local proxy injection.
# =============================================================================

cmake_minimum_required(VERSION 3.20)

project(oss_gaussian_interception
    VERSION     0.1.0
    LANGUAGES   CXX
    DESCRIPTION "OSS-Gaussian D3D12 interception layer (Cyberpunk 2077, DLSS substitution)")

# -----------------------------------------------------------------------------
#  Toolchain guards. Windows + x64 only.
# -----------------------------------------------------------------------------
if(NOT WIN32)
    message(FATAL_ERROR "oss_gaussian_interception is Windows-only. Configure on the 3080 Ti box.")
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "x64 build required (CMAKE_SIZEOF_VOID_P=${CMAKE_SIZEOF_VOID_P}).")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Static CRT to keep the DLL self-contained when dropped into the game folder.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options(/W4 /permissive- /Zc:__cplusplus /utf-8 /EHsc)
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
endif()

# -----------------------------------------------------------------------------
#  Detours (vendored, see third_party/Detours/.gitkeep).
#  When the source is dropped in by T2.2, this block links it as a static lib.
# -----------------------------------------------------------------------------
set(DETOURS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Detours")
if(EXISTS "${DETOURS_ROOT}/src/detours.cpp")
    file(GLOB DETOURS_SRC
        "${DETOURS_ROOT}/src/detours.cpp"
        "${DETOURS_ROOT}/src/modules.cpp"
        "${DETOURS_ROOT}/src/disasm.cpp"
        "${DETOURS_ROOT}/src/image.cpp"
        "${DETOURS_ROOT}/src/creatwth.cpp"
        "${DETOURS_ROOT}/src/disolx86.cpp"
        "${DETOURS_ROOT}/src/disolx64.cpp"
        "${DETOURS_ROOT}/src/disolarm.cpp"
        "${DETOURS_ROOT}/src/disolarm64.cpp"
        "${DETOURS_ROOT}/src/disolia64.cpp")
    add_library(detours STATIC ${DETOURS_SRC})
    target_include_directories(detours PUBLIC "${DETOURS_ROOT}/src")
    if(MSVC)
        target_compile_options(detours PRIVATE /W0)
    endif()
else()
    message(WARNING "Detours source not present at ${DETOURS_ROOT}. "
                    "Run T2.2 (vendor microsoft/Detours) before building.")
    add_library(detours INTERFACE)
endif()

# -----------------------------------------------------------------------------
#  The DLL.
#
#  OUTPUT_NAME is `dxgi` so the produced binary is `dxgi.dll`, matching the
#  OptiScaler-recommended proxy name for Cyberpunk 2077
#  (see https://github.com/optiscaler/OptiScaler/wiki/Cyberpunk-2077).
# -----------------------------------------------------------------------------
set(INTERCEPTION_SOURCES
    oss_capture.cpp
    exr_writer.cpp
    perceptual_hash.cpp
    src/dllmain.cpp
    src/d3d12_hooks.cpp
    src/staging_copy.cpp
    src/dll_config.cpp
    src/dxgi_proxy.cpp
    src/version_proxy.cpp
    src/ngx_passthrough.cpp
    src/ngx_exports.cpp
    src/ngx_frame_capture.cpp
    src/sl_interposer_proxy.cpp
    src/ffx_fsr3_proxy.cpp
    src/g_buffer_extractor.cpp
    src/log.cpp)

set(INTERCEPTION_HEADERS
    oss_capture.h
    include/oss_gaussian_interception.h
    src/d3d12_hooks.h
    src/staging_copy.h
    src/dll_config.h
    src/dxgi_proxy.h
    src/version_proxy.h
    src/ngx_passthrough.h
    src/ngx_types.h
    src/ngx_frame_capture.h
    src/sl_interposer_proxy.h
    src/ffx_fsr3_proxy.h
    src/ffx_backend_dx12_proxy.h
    src/g_buffer_extractor.h
    src/log.h)

add_library(oss_gaussian_interception SHARED
    ${INTERCEPTION_SOURCES}
    ${INTERCEPTION_HEADERS})

set_target_properties(oss_gaussian_interception PROPERTIES
    OUTPUT_NAME              "dxgi"
    ARCHIVE_OUTPUT_NAME      "oss_gaussian_interception"
    PREFIX                   ""
    WINDOWS_EXPORT_ALL_SYMBOLS OFF)

# Packaging consumes the canonical artifact name `oss_capture.dll` and the
# installer drops it as the per-game proxy name (`dxgi.dll` for Cyberpunk).
add_custom_command(TARGET oss_gaussian_interception POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:oss_gaussian_interception>"
            "$<TARGET_FILE_DIR:oss_gaussian_interception>/oss_capture.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:oss_gaussian_interception>"
            "$<TARGET_FILE_DIR:oss_gaussian_interception>/oss_capture_ngx.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:oss_gaussian_interception>"
            "$<TARGET_FILE_DIR:oss_gaussian_interception>/nvngx_dlss.dll"
    COMMENT "Staging capture DLL artifacts dxgi.dll, oss_capture.dll, oss_capture_ngx.dll, and nvngx_dlss.dll")

target_include_directories(oss_gaussian_interception
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(oss_gaussian_interception
    PRIVATE
        detours
        version
        shlwapi)

find_package(OpenEXR CONFIG QUIET)
if(TARGET OpenEXR::OpenEXR)
    target_compile_definitions(oss_gaussian_interception PRIVATE OSS_CAPTURE_HAS_OPENEXR=1)
    target_link_libraries(oss_gaussian_interception PRIVATE OpenEXR::OpenEXR)
else()
    message(WARNING "OpenEXR not found; oss_capture_write_exr builds the deterministic schema stub. "
                    "Install OpenEXR on the Windows build host for real multi-channel EXR output.")
endif()

# .def file added by T2.3 will live at src/dxgi_proxy.def. Wire it conditionally:
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/dxgi_proxy.def")
    target_sources(oss_gaussian_interception PRIVATE src/dxgi_proxy.def)
endif()

add_library(oss_gaussian_version_interception SHARED
    ${INTERCEPTION_SOURCES}
    ${INTERCEPTION_HEADERS}
    src/version_proxy.def)

set_target_properties(oss_gaussian_version_interception PROPERTIES
    OUTPUT_NAME              "version"
    ARCHIVE_OUTPUT_NAME      "oss_gaussian_version_interception"
    PREFIX                   ""
    WINDOWS_EXPORT_ALL_SYMBOLS OFF)

target_include_directories(oss_gaussian_version_interception
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(oss_gaussian_version_interception
    PRIVATE
        detours
        version
        shlwapi)

if(TARGET OpenEXR::OpenEXR)
    target_compile_definitions(oss_gaussian_version_interception PRIVATE OSS_CAPTURE_HAS_OPENEXR=1)
    target_link_libraries(oss_gaussian_version_interception PRIVATE OpenEXR::OpenEXR)
endif()

add_library(oss_gaussian_streamline_interposer SHARED
    ${INTERCEPTION_SOURCES}
    ${INTERCEPTION_HEADERS}
    src/sl_interposer_proxy.def)

set_target_properties(oss_gaussian_streamline_interposer PROPERTIES
    OUTPUT_NAME              "sl.interposer"
    ARCHIVE_OUTPUT_NAME      "oss_gaussian_streamline_interposer"
    PREFIX                   ""
    WINDOWS_EXPORT_ALL_SYMBOLS OFF)

target_include_directories(oss_gaussian_streamline_interposer
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(oss_gaussian_streamline_interposer
    PRIVATE
        detours
        version
        shlwapi)

target_compile_definitions(oss_gaussian_streamline_interposer
    PRIVATE
        OSS_GAUSSIAN_NO_SYSTEM_PROXY_ATTACH=1
        OSS_GAUSSIAN_DEFER_NATIVE_ATTACH=1
        OSS_GAUSSIAN_PROXY_EXPORT_LIGHTWEIGHT_INIT=1)

if(TARGET OpenEXR::OpenEXR)
    target_compile_definitions(oss_gaussian_streamline_interposer PRIVATE OSS_CAPTURE_HAS_OPENEXR=1)
    target_link_libraries(oss_gaussian_streamline_interposer PRIVATE OpenEXR::OpenEXR)
endif()

add_custom_command(TARGET oss_gaussian_streamline_interposer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:oss_gaussian_streamline_interposer>"
            "$<TARGET_FILE_DIR:oss_gaussian_streamline_interposer>/oss_capture_sl.dll"
    COMMENT "Staging Streamline capture artifact sl.interposer.dll / oss_capture_sl.dll")

set(FSR3_INTERCEPTION_SOURCES
    oss_capture.cpp
    exr_writer.cpp
    perceptual_hash.cpp
    src/dllmain.cpp
    src/d3d12_hooks.cpp
    src/staging_copy.cpp
    src/dll_config.cpp
    src/ngx_passthrough.cpp
    src/ngx_frame_capture.cpp
    src/ffx_fsr3_proxy.cpp
    src/g_buffer_extractor.cpp
    src/log.cpp)

add_library(oss_gaussian_fsr3_interception SHARED
    ${FSR3_INTERCEPTION_SOURCES}
    ${INTERCEPTION_HEADERS}
    src/ffx_fsr3_proxy.def)

set_target_properties(oss_gaussian_fsr3_interception PROPERTIES
    OUTPUT_NAME              "ffx_fsr3_x64"
    ARCHIVE_OUTPUT_NAME      "oss_gaussian_fsr3_interception"
    PREFIX                   ""
    WINDOWS_EXPORT_ALL_SYMBOLS OFF)

target_include_directories(oss_gaussian_fsr3_interception
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(oss_gaussian_fsr3_interception
    PRIVATE
        detours
        version
        shlwapi)

target_compile_definitions(oss_gaussian_fsr3_interception
    PRIVATE
        OSS_GAUSSIAN_NO_SYSTEM_PROXY_ATTACH=1
        OSS_GAUSSIAN_DEFER_NATIVE_ATTACH=1
        OSS_GAUSSIAN_PROXY_EXPORT_LIGHTWEIGHT_INIT=1)

if(TARGET OpenEXR::OpenEXR)
    target_compile_definitions(oss_gaussian_fsr3_interception PRIVATE OSS_CAPTURE_HAS_OPENEXR=1)
    target_link_libraries(oss_gaussian_fsr3_interception PRIVATE OpenEXR::OpenEXR)
endif()

add_custom_command(TARGET oss_gaussian_fsr3_interception POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:oss_gaussian_fsr3_interception>"
            "$<TARGET_FILE_DIR:oss_gaussian_fsr3_interception>/oss_capture_fsr3.dll"
    COMMENT "Staging FSR3 capture artifact ffx_fsr3_x64.dll / oss_capture_fsr3.dll")

add_library(oss_gaussian_fsr3upscaler_interception SHARED
    ${FSR3_INTERCEPTION_SOURCES}
    ${INTERCEPTION_HEADERS}
    src/ffx_fsr3upscaler_proxy.def)

set_target_properties(oss_gaussian_fsr3upscaler_interception PROPERTIES
    OUTPUT_NAME              "ffx_fsr3upscaler_x64"
    ARCHIVE_OUTPUT_NAME      "oss_gaussian_fsr3upscaler_interception"
    PREFIX                   ""
    WINDOWS_EXPORT_ALL_SYMBOLS OFF)

target_include_directories(oss_gaussian_fsr3upscaler_interception
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(oss_gaussian_fsr3upscaler_interception
    PRIVATE
        detours
        version
        shlwapi)

target_compile_definitions(oss_gaussian_fsr3upscaler_interception
    PRIVATE
        OSS_GAUSSIAN_NO_SYSTEM_PROXY_ATTACH=1
        OSS_GAUSSIAN_DEFER_NATIVE_ATTACH=1
        OSS_GAUSSIAN_PROXY_EXPORT_LIGHTWEIGHT_INIT=1)

if(TARGET OpenEXR::OpenEXR)
    target_compile_definitions(oss_gaussian_fsr3upscaler_interception PRIVATE OSS_CAPTURE_HAS_OPENEXR=1)
    target_link_libraries(oss_gaussian_fsr3upscaler_interception PRIVATE OpenEXR::OpenEXR)
endif()

add_custom_command(TARGET oss_gaussian_fsr3upscaler_interception POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:oss_gaussian_fsr3upscaler_interception>"
            "$<TARGET_FILE_DIR:oss_gaussian_fsr3upscaler_interception>/oss_capture_fsr3upscaler.dll"
    COMMENT "Staging FSR3 upscaler capture artifact ffx_fsr3upscaler_x64.dll / oss_capture_fsr3upscaler.dll")

add_library(oss_gaussian_ffx_backend_dx12_interception SHARED
    src/ffx_backend_dx12_proxy.cpp
    src/ffx_backend_dx12_proxy.h
    src/log.cpp
    src/log.h
    src/ffx_backend_dx12_proxy.def)

set_target_properties(oss_gaussian_ffx_backend_dx12_interception PROPERTIES
    OUTPUT_NAME              "ffx_backend_dx12_x64"
    ARCHIVE_OUTPUT_NAME      "oss_gaussian_ffx_backend_dx12_interception"
    PREFIX                   ""
    WINDOWS_EXPORT_ALL_SYMBOLS OFF)

target_include_directories(oss_gaussian_ffx_backend_dx12_interception
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(oss_gaussian_ffx_backend_dx12_interception
    PRIVATE shlwapi)

add_custom_command(TARGET oss_gaussian_ffx_backend_dx12_interception POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:oss_gaussian_ffx_backend_dx12_interception>"
            "$<TARGET_FILE_DIR:oss_gaussian_ffx_backend_dx12_interception>/oss_capture_ffx_backend_dx12.dll"
    COMMENT "Staging FFX DX12 backend artifact ffx_backend_dx12_x64.dll / oss_capture_ffx_backend_dx12.dll")

add_library(oss_gaussian_ffx_api_interception SHARED
    src/ffx_api_proxy.cpp
    src/ffx_api_proxy.h
    src/log.cpp
    src/log.h
    src/ffx_api_proxy.def)

set_target_properties(oss_gaussian_ffx_api_interception PROPERTIES
    OUTPUT_NAME              "amd_fidelityfx_dx12"
    ARCHIVE_OUTPUT_NAME      "oss_gaussian_ffx_api_interception"
    PREFIX                   ""
    WINDOWS_EXPORT_ALL_SYMBOLS OFF)

target_include_directories(oss_gaussian_ffx_api_interception
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(oss_gaussian_ffx_api_interception
    PRIVATE shlwapi)

add_custom_command(TARGET oss_gaussian_ffx_api_interception POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:oss_gaussian_ffx_api_interception>"
            "$<TARGET_FILE_DIR:oss_gaussian_ffx_api_interception>/oss_capture_ffx_api.dll"
    COMMENT "Staging generic FidelityFX API artifact amd_fidelityfx_dx12.dll / oss_capture_ffx_api.dll")

# -----------------------------------------------------------------------------
#  Install rules — copies the built DLL to a stage dir for packaging.
# -----------------------------------------------------------------------------
install(TARGETS oss_gaussian_interception
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin)

install(FILES "$<TARGET_FILE_DIR:oss_gaussian_interception>/oss_capture.dll"
    DESTINATION .
    COMPONENT CaptureInstaller)

install(FILES "$<TARGET_FILE:oss_gaussian_interception>"
    DESTINATION .
    COMPONENT CaptureInstaller)

install(FILES "$<TARGET_FILE_DIR:oss_gaussian_interception>/nvngx_dlss.dll"
    DESTINATION .
    COMPONENT CaptureInstaller)

install(FILES "$<TARGET_FILE:oss_gaussian_version_interception>"
    DESTINATION .
    COMPONENT CaptureInstaller)

install(FILES "$<TARGET_FILE:oss_gaussian_streamline_interposer>"
    DESTINATION .
    COMPONENT CaptureInstaller)

install(FILES "$<TARGET_FILE:oss_gaussian_fsr3_interception>"
    DESTINATION .
    COMPONENT CaptureInstaller)

install(FILES "$<TARGET_FILE:oss_gaussian_ffx_backend_dx12_interception>"
    DESTINATION .
    COMPONENT CaptureInstaller)

install(FILES "$<TARGET_FILE:oss_gaussian_ffx_api_interception>"
    DESTINATION .
    COMPONENT CaptureInstaller)

install(FILES include/oss_gaussian_interception.h
    DESTINATION include)

install(FILES LICENSE README.md
    DESTINATION .)

# -----------------------------------------------------------------------------
#  Tests (T2.3). Builds a small executable that LoadLibrary's the produced
#  dxgi.dll and calls each forwarded export with safe args. Run with:
#      ctest --test-dir build -C Release --output-on-failure
# -----------------------------------------------------------------------------
enable_testing()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dxgi_proxy.cpp")
    add_executable(test_dxgi_proxy tests/test_dxgi_proxy.cpp)
    target_link_libraries(test_dxgi_proxy PRIVATE shlwapi)
    add_dependencies(test_dxgi_proxy oss_gaussian_interception)
    add_test(
        NAME    test_dxgi_proxy
        COMMAND test_dxgi_proxy "$<TARGET_FILE:oss_gaussian_interception>")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ngx_exports.cpp")
    add_executable(test_ngx_exports tests/test_ngx_exports.cpp)
    add_dependencies(test_ngx_exports oss_gaussian_interception)
    add_test(
        NAME    test_ngx_exports
        COMMAND test_ngx_exports "$<TARGET_FILE:oss_gaussian_interception>")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_version_proxy.cpp")
    add_executable(test_version_proxy tests/test_version_proxy.cpp)
    add_dependencies(test_version_proxy oss_gaussian_version_interception)
    add_test(
        NAME    test_version_proxy
        COMMAND test_version_proxy "$<TARGET_FILE:oss_gaussian_version_interception>")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sl_interposer_proxy.cpp")
    add_executable(test_sl_interposer_proxy tests/test_sl_interposer_proxy.cpp)
    add_dependencies(test_sl_interposer_proxy oss_gaussian_streamline_interposer)
    add_test(
        NAME    test_sl_interposer_proxy
        COMMAND test_sl_interposer_proxy "$<TARGET_FILE:oss_gaussian_streamline_interposer>")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ffx_fsr3_proxy.cpp")
    add_executable(test_ffx_fsr3_proxy tests/test_ffx_fsr3_proxy.cpp)
    add_dependencies(test_ffx_fsr3_proxy oss_gaussian_fsr3_interception)
    add_test(
        NAME    test_ffx_fsr3_proxy
        COMMAND test_ffx_fsr3_proxy "$<TARGET_FILE:oss_gaussian_fsr3_interception>")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ffx_fsr3upscaler_proxy.cpp")
    add_executable(test_ffx_fsr3upscaler_proxy tests/test_ffx_fsr3upscaler_proxy.cpp)
    add_dependencies(test_ffx_fsr3upscaler_proxy oss_gaussian_fsr3upscaler_interception)
    add_test(
        NAME    test_ffx_fsr3upscaler_proxy
        COMMAND test_ffx_fsr3upscaler_proxy "$<TARGET_FILE:oss_gaussian_fsr3upscaler_interception>")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ffx_backend_dx12_proxy.cpp")
    add_executable(test_ffx_backend_dx12_proxy tests/test_ffx_backend_dx12_proxy.cpp)
    add_dependencies(test_ffx_backend_dx12_proxy oss_gaussian_ffx_backend_dx12_interception)
    add_test(
        NAME    test_ffx_backend_dx12_proxy
        COMMAND test_ffx_backend_dx12_proxy "$<TARGET_FILE:oss_gaussian_ffx_backend_dx12_interception>")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ffx_api_proxy.cpp")
    add_executable(test_ffx_api_proxy tests/test_ffx_api_proxy.cpp)
    add_dependencies(test_ffx_api_proxy oss_gaussian_ffx_api_interception)
    add_test(
        NAME    test_ffx_api_proxy
        COMMAND test_ffx_api_proxy "$<TARGET_FILE:oss_gaussian_ffx_api_interception>")
endif()

set(ROOT_CAPTURE_TEST "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/capture/test_capture_unit.cpp")
if(EXISTS "${ROOT_CAPTURE_TEST}")
    add_executable(test_capture_unit
        "${ROOT_CAPTURE_TEST}"
        oss_capture.cpp
        exr_writer.cpp
        perceptual_hash.cpp
        src/g_buffer_extractor.cpp
        src/log.cpp)
    target_compile_definitions(test_capture_unit PRIVATE OSS_CAPTURE_UNIT_TEST=1)
    target_include_directories(test_capture_unit
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(TARGET OpenEXR::OpenEXR)
        target_compile_definitions(test_capture_unit PRIVATE OSS_CAPTURE_HAS_OPENEXR=1)
        target_link_libraries(test_capture_unit PRIVATE OpenEXR::OpenEXR)
    endif()
    add_test(NAME test_capture_unit COMMAND test_capture_unit)
endif()
