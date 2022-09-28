
find_package(autodiff REQUIRED)
find_package(Eigen3 3.3.9 REQUIRED)
#find_package(MPI QUIET)
find_package(Ikarus REQUIRED)

dune_add_test(NAME expressiontest SOURCES expressionTest.cc LINK_LIBRARIES Eigen3::Eigen Ikarus::Ikarus dune-localfefunctions)
#target_link_libraries(expressiontest PRIVATE Ikarus::Ikarus)
#target_link_libraries(expressiontest PUBLIC dune-localfefunctions)
#target_link_libraries(expressiontest PUBLIC Eigen3::Eigen)
#target_link_libraries(expressiontest PUBLIC autodiff::autodiff)
#target_include_directories(expressiontest     PRIVATE   ${Eigen3_INCLUDE_DIRS}        )

target_compile_features(expressiontest PRIVATE cxx_std_20)

