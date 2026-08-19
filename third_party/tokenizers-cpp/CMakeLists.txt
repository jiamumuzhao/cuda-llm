cmake_minimum_required(VERSION 3.16)
project(vendored_tokenizers_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_program(TOKENIZERS_CARGO cargo REQUIRED)
set(TOKENIZERS_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(TOKENIZERS_BUILD ${CMAKE_CURRENT_BINARY_DIR}/rust)
set(TOKENIZERS_RUST_LIB ${TOKENIZERS_BUILD}/release/libtokenizers_c.a)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cargo-config.toml
  "[source.crates-io]\nreplace-with = \"vendored-sources\"\n\n"
  "[source.vendored-sources]\ndirectory = \"${TOKENIZERS_ROOT}/vendor\"\n")

add_custom_command(
  OUTPUT ${TOKENIZERS_RUST_LIB}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TOKENIZERS_BUILD}
  COMMAND ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${TOKENIZERS_BUILD}
          ${TOKENIZERS_CARGO} build --release --offline
          --config ${CMAKE_CURRENT_BINARY_DIR}/cargo-config.toml
          --manifest-path ${TOKENIZERS_ROOT}/rust/Cargo.toml
  WORKING_DIRECTORY ${TOKENIZERS_ROOT}/rust
  DEPENDS ${TOKENIZERS_ROOT}/rust/Cargo.toml ${TOKENIZERS_ROOT}/rust/src/lib.rs
          ${CMAKE_CURRENT_BINARY_DIR}/cargo-config.toml
)
add_custom_target(tokenizers_rust DEPENDS ${TOKENIZERS_RUST_LIB})

add_library(tokenizers_c STATIC IMPORTED GLOBAL)
set_target_properties(tokenizers_c PROPERTIES IMPORTED_LOCATION ${TOKENIZERS_RUST_LIB})
add_dependencies(tokenizers_c tokenizers_rust)

add_library(tokenizers_cpp STATIC src/huggingface_tokenizer.cc)
target_include_directories(tokenizers_cpp PUBLIC ${TOKENIZERS_ROOT}/include)
target_link_libraries(tokenizers_cpp PUBLIC tokenizers_c ${CMAKE_DL_LIBS})
add_dependencies(tokenizers_cpp tokenizers_rust)
