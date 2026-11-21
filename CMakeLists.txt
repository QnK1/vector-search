cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(vector_search LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
    cli11
    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.zip
)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Header-only Boost modules needed for Boost.Interprocess file mapping
set(BOOST_TAG boost-1.86.0)
set(BOOST_LIBS config assert core static_assert throw_exception predef integer type_traits winapi move intrusive container interprocess)

foreach(lib ${BOOST_LIBS})
    FetchContent_Declare(
        boost_${lib}
        URL https://github.com/boostorg/${lib}/archive/refs/tags/${BOOST_TAG}.zip
        SOURCE_SUBDIR non_existent_subdir_headers_only
    )
endforeach()

FetchContent_MakeAvailable(cli11 googletest)

add_library(boost_interprocess_headers INTERFACE)
foreach(lib ${BOOST_LIBS})
    FetchContent_MakeAvailable(boost_${lib})
    target_include_directories(boost_interprocess_headers SYSTEM INTERFACE ${boost_${lib}_SOURCE_DIR}/include)
endforeach()

add_library(vecsearch
    src/dataset.cpp
    src/ivf_index.cpp
)
target_include_directories(vecsearch PUBLIC include)
target_link_libraries(vecsearch PUBLIC boost_interprocess_headers)

if(MSVC)
    target_compile_options(vecsearch PRIVATE /W4 /utf-8)
else()
    target_compile_options(vecsearch PRIVATE -Wall -Wextra)
endif()

add_executable(vsearch src/main.cpp)
target_link_libraries(vsearch PRIVATE vecsearch CLI11::CLI11)
if(MSVC)
    target_compile_options(vsearch PRIVATE /W4 /utf-8)
endif()

enable_testing()
add_executable(vsearch_tests tests/test_vecsearch.cpp)
target_link_libraries(vsearch_tests PRIVATE vecsearch GTest::gtest_main)
if(MSVC)
    target_compile_options(vsearch_tests PRIVATE /W4 /utf-8)
endif()

include(GoogleTest)
gtest_discover_tests(vsearch_tests)

file(GLOB_RECURSE ALL_HEADERS "${CMAKE_SOURCE_DIR}/include/*.hpp")
file(GLOB_RECURSE ALL_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/tests/*.cpp")

find_program(CLANG_FORMAT_EXE clang-format)
if(CLANG_FORMAT_EXE)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_HEADERS} ${ALL_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-format"
    )
endif()

find_program(CLANG_TIDY_EXE clang-tidy)
if(CLANG_TIDY_EXE)
    set(TIDY_INCLUDES -I${CMAKE_SOURCE_DIR}/include -I${cli11_SOURCE_DIR}/include -I${googletest_SOURCE_DIR}/googletest/include)
    foreach(lib ${BOOST_LIBS})
        list(APPEND TIDY_INCLUDES -isystem ${boost_${lib}_SOURCE_DIR}/include)
    endforeach()

    add_custom_target(lint
        COMMAND ${CLANG_TIDY_EXE} --quiet "-header-filter=${CMAKE_SOURCE_DIR}/include/.*" ${ALL_SOURCES} -- -std=c++23 ${TIDY_INCLUDES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-tidy"
    )
endif()

