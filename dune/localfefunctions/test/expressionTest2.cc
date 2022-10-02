
#include <config.h>

#include "factories.hh"

#include <array>
#include <complex>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "dune/localfefunctions/eigenDuneTransformations.hh"
#include "dune/localfefunctions/linearAlgebraHelper.hh"
#include "dune/localfefunctions/localBasis/localBasis.hh"
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/geometry/multilineargeometry.hh>
#include <dune/localfunctions/lagrange/pqkfactory.hh>

// #include <ikarus/utils/linearAlgebraHelper.hh>
//#include <ikarus/utils/multiIndex.hh>
using Dune::TestSuite;
int main(int argc, char **argv) {
  using namespace Dune;
  Dune::MPIHelper::instance(argc, argv);
  TestSuite t;
  constexpr int domainDim = 2;
  constexpr int worldDim = 2;
  std::vector<Dune::FieldVector<double,worldDim>> corners;

  Dune::FieldVector<double,domainDim> gp({0.5,0.5});
  const auto& refElement = Dune::ReferenceElements<double,domainDim>::general(  Dune::GeometryTypes::quadrilateral );

  CornerFactory<worldDim>::construct(corners, refElement.size(domainDim) );
  std::cout<<corners.size()<<std::endl;
  auto geometry = std::make_shared<const Dune::MultiLinearGeometry<double,domainDim,worldDim>>(refElement,corners);
  auto J = toEigenMatrix(geometry->jacobianTransposed(gp));
  auto Jinv = toEigenMatrix(geometry->jacobianInverseTransposed(gp));
  std::cout<<J<<std::endl;
  std::cout<<Jinv<<std::endl;
  std::cout<<J*Jinv<<std::endl;
  std::cout<<Jinv.transpose()*Jinv<<std::endl;
  std::cout<<"J*orthonormalizeMatrixColumns(J).transpose()"<<std::endl;
  std::cout<<J*orthonormalizeMatrixColumns(J.transpose())<<std::endl;
  std::cout<<"orthonormalizeMatrixColumns(J.transpose())"<<std::endl;
  std::cout<<orthonormalizeMatrixColumns(J.transpose())<<std::endl;

  auto tranformMat = (J*orthonormalizeMatrixColumns(J.transpose()).eval()).eval().transpose();
  using FECache = Dune::PQkLocalFiniteElementCache<double, double, domainDim, 1>;
  FECache feCache;
  const auto &fe      = feCache.get(Dune::GeometryTypes::quadrilateral);
  auto localBasis     = Dune::LocalBasis(fe.localBasis());

 Dune::FieldMatrix<double,Eigen::Dynamic,domainDim> dN;
  localBasis.evaluateJacobian(gp,dN);
  dN = dN*tranformMat.inverse();
  Dune::FieldMatrix<double,worldDim,domainDim> JByHand;
  std::cout<<"dN"<<std::endl;
  std::cout<<dN<<std::endl;
  JByHand.setZero();
  for (int i = 0; i < domainDim; ++i) {
    for (int j = 0; j < refElement.size(domainDim); ++j) {
      for (int k = 0; k < worldDim; ++k) {
        JByHand.col(i)[k] += dN(j,i)*corners[j][k];
      }
    }

  }
std::cout<<"JByHand"<<std::endl;
std::cout<<JByHand<<std::endl;
  return t.exit();
}