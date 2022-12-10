# SPDX-FileCopyrightText: 2022 The dune-localfefunction developers
# mueller@ibb.uni-stuttgart.de SPDX-License-Identifier: LGPL-2.1-or-later
option(DUNE_LOCALFEFUNCTIONS_ENABLE_TESTING OFF)
if(DUNE_LOCALFEFUNCTIONS_ENABLE_TESTING)
  find_package(autodiff REQUIRED)
  find_package(Eigen3 3.3.9 REQUIRED)

  option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
  dune_add_test(
    NAME
    expressiontest
    SOURCES
    expressionTest.cc
    testFacilities.cc
    LINK_LIBRARIES
    Eigen3::Eigen
    dune-localfefunctions)
  if(ENABLE_TEST_COVERAGE)
    message("Enable Test Coverage")
    target_compile_options(expressiontest PUBLIC -O0 -g -fprofile-arcs
                                                 -ftest-coverage)
    target_link_options(expressiontest PUBLIC -fprofile-arcs -ftest-coverage)

  endif()
  target_compile_options(
    expressiontest
    PRIVATE -ftemplate-backtrace-limit=0 -fdiagnostics-show-template-tree
            -fno-elide-type -fconcepts-diagnostics-depth=2)

  target_compile_features(expressiontest PRIVATE cxx_std_20)

  dune_add_test(
    NAME
    cachedlocalbasis
    SOURCES
    cachedlocalbasis.cc
    testFacilities.cc
    LINK_LIBRARIES
    Eigen3::Eigen
    dune-localfefunctions)
  if(ENABLE_TEST_COVERAGE)
    message("Enable Test Coverage")
    target_compile_options(cachedlocalbasis PUBLIC -O0 -g -fprofile-arcs
                                                   -ftest-coverage)
    target_link_options(cachedlocalbasis PUBLIC -fprofile-arcs -ftest-coverage)

  endif()
  target_compile_options(
    cachedlocalbasis
    PRIVATE -ftemplate-backtrace-limit=0 -fdiagnostics-show-template-tree
            -fno-elide-type -fconcepts-diagnostics-depth=2)

  target_compile_features(cachedlocalbasis PRIVATE cxx_std_20)

endif()
