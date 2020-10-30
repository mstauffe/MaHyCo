cmake_minimum_required( VERSION 3.1 )

project( EucclhydRemapProject VERSION 1.0 LANGUAGES CXX )

# Disable in-source builds to prevent source tree corruption.
if( " ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}" )
  message( FATAL_ERROR "FATAL: In-source builds are not allowed.
                       You should create a separate directory for build files." )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL Clang OR
      CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-c++11-narrowing")
endif()

enable_testing()

set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_POSITION_INDEPENDENT_CODE ON )

add_subdirectory( "./cppnabla" )
add_subdirectory( "./eucclhyd_remap" )
add_subdirectory( "./vnr_remap" )
add_subdirectory( "./remap" )
add_subdirectory( "./particle_scheme" )
add_subdirectory( "./lecture_donnees" )
add_subdirectory( "./tests" )

add_executable( mahyco )
target_sources( mahyco 
    PRIVATE
        main.cc
)
target_link_libraries( mahyco PRIVATE eucclhydremap vnrremap remap lecture_donnees particles)
