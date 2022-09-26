
find_package(autodiff REQUIRED)
find_package(MPI QUIET)
dune_add_test(SOURCES expressionTest.cc)
