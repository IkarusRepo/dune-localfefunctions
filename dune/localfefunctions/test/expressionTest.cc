#include "config.h"
#include <dune/common/parallel/mpihelper.hh>

#include <dune/common/test/testsuite.hh>
int main (int argc, char** argv)try {
  using namespace Dune;
  MPIHelper::instance(argc, argv);

  TestSuite testSuite;

  testSuite.check(false, "IsFalse");
  testSuite.require(false, "IsFalse");

  return 0;
}catch (Dune::Exception& e) {
  std::cerr << "Dune reported error: " << e << std::endl;
}