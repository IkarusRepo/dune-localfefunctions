if(DUNE_LOCALFEFUNCTIONS_ENABLE_TESTING)
  find_package(autodiff REQUIRED)
  find_package(Eigen3 3.3.9 REQUIRED)
  find_package(Ikarus REQUIRED)

  option(DUNE_LOCALFEFUNCTIONS_ENABLE_TESTING 0)

  dune_add_test(
    NAME
    expressiontest
    SOURCES
    expressionTest.cc
    LINK_LIBRARIES
    Eigen3::Eigen
    Ikarus::Ikarus
    dune-localfefunctions)

  target_compile_options(
    expressiontest
    PRIVATE -ftemplate-backtrace-limit=0 -fdiagnostics-show-template-tree
            -fno-elide-type -fconcepts-diagnostics-depth=2)

  target_compile_features(expressiontest PRIVATE cxx_std_20)

endif()
