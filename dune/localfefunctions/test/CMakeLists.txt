# SPDX-FileCopyrightText: 2022 The dune-localfefunction developers
# mueller@ibb.uni-stuttgart.de SPDX-License-Identifier: LGPL-2.1-or-later

option(DUNE_LOCALFEFUNCTIONS_ENABLE_TESTING OFF)
if(DUNE_LOCALFEFUNCTIONS_ENABLE_TESTING)
  find_package(autodiff REQUIRED)
  find_package(Eigen3 3.3.9 REQUIRED)
  find_package(PythonLibs REQUIRED)
  include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})

  option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)

  file(GLOB programSourceFiles CONFIGURE_DEPENDS *.cc)
  list(REMOVE_ITEM programSourceFiles
       ${CMAKE_CURRENT_SOURCE_DIR}/testFacilities.cc)
  list(REMOVE_ITEM programSourceFiles
       ${CMAKE_CURRENT_SOURCE_DIR}/testmanifoldsblockvector.cc)
  list(REMOVE_ITEM programSourceFiles
       ${CMAKE_CURRENT_SOURCE_DIR}/testbadinvokations.cc)

  add_library(TestFacilities OBJECT
              ${CMAKE_CURRENT_SOURCE_DIR}/testFacilities.cc)
  target_link_libraries(TestFacilities PRIVATE ${PYTHON_LIBRARIES})
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
      ${PYTHON_LIBRARIES})
    if(ENABLE_TEST_COVERAGE)
      message("Enable Test Coverage")
      target_compile_options(${programName} PUBLIC -O0 -g -fprofile-arcs
                                                   -ftest-coverage)
      target_link_options(${programName} PUBLIC -fprofile-arcs -ftest-coverage)

    endif()
    # target_compile_options( ${programName} PRIVATE
    # -ftemplate-backtrace-limit=0 -fdiagnostics-show-template-tree
    # -fno-elide-type -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic)

    target_compile_features(${programName} PRIVATE cxx_std_20)
  endforeach()

  dune_add_test(
    NAME
    testbadinvokations
    SOURCES
    testbadinvokations.cc
    $<TARGET_OBJECTS:TestFacilities>
    LINK_LIBRARIES
    Eigen3::Eigen
    EXPECT_COMPILE_FAIL)
  target_compile_features(testbadinvokations PRIVATE cxx_std_20)

  dune_add_test(
    NAME
    testmanifoldsblockvector
    SOURCES
    testmanifoldsblockvector.cc
    $<TARGET_OBJECTS:TestFacilities>
    LINK_LIBRARIES
    Eigen3::Eigen
    ${PYTHON_LIBRARIES}
    COMPILE_ONLY)
  target_compile_features(testmanifoldsblockvector PRIVATE cxx_std_20)
endif()
