if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_CMD cargo build --verbose)
    set(TARGET_DIR "debug")
else ()
    set(CARGO_CMD cargo build --release --verbose)
    set(TARGET_DIR "release")
    set(RUST_FLAGS "-C" "target-cpu=native")
endif ()

if(USE_AVX512)
	set(RUST_FLAG  "${CARGO_CMD} --features=nightly")
endif()

if(ENABLE_LTO)
    set(RUST_FLAGS "${RUST_FLAGS} -Clinker-plugin-lto" "-Clinker=clang-17" "-Clink-arg=-fuse-ld=lld-17")
endif()

set(RUST_PART_LIB "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/libconcrete_fft.a")

set(RUST_PART_CXX "${CMAKE_CURRENT_BINARY_DIR}/concrete-fft.cpp")
add_custom_command(
    OUTPUT ${RUST_PART_CXX} ${RUST_PART_LIB} ${CMAKE_CURRENT_BINARY_DIR}/cxx.hpp
    COMMAND CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} RUSTFLAGS="${RUST_FLAGS}" ${CARGO_CMD}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/concrete-fft/src/lib.rs.cc ${RUST_PART_CXX}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/concrete-fft/src/lib.rs.h ${CMAKE_CURRENT_BINARY_DIR}/concrete-fft.hpp
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/rust/cxx.h ${CMAKE_CURRENT_BINARY_DIR}/cxx.hpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_library(concrete-fft STATIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/fft_processor_concrete.cpp  ${RUST_PART_CXX})
target_include_directories(concrete-fft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp ${CMAKE_CURRENT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(concrete-fft pthread dl ${RUST_PART_LIB})
