# SPDX-FileCopyrightText: 2022 The dune-localfefunction developers
# mueller@ibb.uni-stuttgart.de SPDX-License-Identifier: LGPL-2.1-or-later
option(DUNE_LOCALFEFUNCTIONS_ENABLE_TESTING OFF)
if(DUNE_LOCALFEFUNCTIONS_ENABLE_TESTING)
  find_package(autodiff REQUIRED)
  find_package(Eigen3 3.3.9 REQUIRED)

  option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)

  file(GLOB programSourceFiles CONFIGURE_DEPENDS *.cc)
  list(REMOVE_ITEM programSourceFiles
       ${CMAKE_CURRENT_SOURCE_DIR}/testFacilities.cc)

  add_library(TestFacilities OBJECT
              ${CMAKE_CURRENT_SOURCE_DIR}/testFacilities.cc)

  target_compile_features(TestFacilities PRIVATE cxx_std_20)

  foreach(programSourceFile ${programSourceFiles})
    get_filename_component(programName ${programSourceFile} NAME_WLE)

    dune_add_test(
      NAME
      ${programName}
      SOURCES
      ${programSourceFile}
      $<TARGET_OBJECTS:TestFacilities>
      LINK_LIBRARIES
      Eigen3::Eigen
      dune-localfefunctions)
    if(ENABLE_TEST_COVERAGE)
      message("Enable Test Coverage")
      target_compile_options(${programName} PUBLIC -O0 -g -fprofile-arcs
                                                   -ftest-coverage)
      target_link_options(${programName} PUBLIC -fprofile-arcs -ftest-coverage)

    endif()
    target_compile_options(
      ${programName}
      PRIVATE -ftemplate-backtrace-limit=0
              -fdiagnostics-show-template-tree
              -fno-elide-type
              -fconcepts-diagnostics-depth=2
              -Wall
              -Wextra
              -Wshadow
              -Wnon-virtual-dtor
              -pedantic)

    target_compile_features(${programName} PRIVATE cxx_std_20)
  endforeach()
endif()
