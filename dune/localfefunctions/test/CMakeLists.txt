if(DUNE_LOCALFEFUNCTIONS_ENABLE_TESTING)
  find_package(autodiff REQUIRED)
  find_package(Eigen3 3.3.9 REQUIRED)

  option(DUNE_LOCALFEFUNCTIONS_ENABLE_TESTING 0)
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

endif()
