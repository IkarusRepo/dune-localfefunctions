
find_package(autodiff REQUIRED)
find_package(MPI QUIET)
dune_add_test(SOURCES expressionTest.cc)

target_compile_features(expressionTest PRIVATE cxx_std_20)

