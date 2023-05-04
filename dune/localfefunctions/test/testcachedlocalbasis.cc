// SPDX-FileCopyrightText: 2022 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <config.h>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
using Dune::TestSuite;

#include "testFacilities.hh"

#include <complex>

#include <dune/common/classname.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/localfefunctions/cachedlocalBasis/cachedlocalBasis.hh>
#include <dune/localfefunctions/eigenDuneTransformations.hh>
#include <dune/localfefunctions/linearAlgebraHelper.hh>

#include <Eigen/Core>

using namespace Dune::Functions::BasisFactory;

template <typename LB, bool isCopy = false>
auto testLocalBasis(LB& localBasis, const Dune::GeometryType& type) {
  TestSuite t("testLocalBasis");
  using namespace autodiff;
  using namespace Testing;
  using namespace Dune;

  constexpr int gridDim = LB::gridDim;
  const auto& rule      = Dune::QuadratureRules<double, gridDim>::rule(type, 3);

  for (const auto& gp : rule) {
    /// Check spatial derivatives
    {
      /// Check if spatial derivatives are really derivatives
      /// Perturb in a random direction in the elements parameter space and check spatial derivative
      auto func = [&](auto& gpOffset_) {
        DefaultLinearAlgebra::template VariableSizedVector<double> N;
        localBasis.evaluateFunction(gp.position() + toDune(gpOffset_), N);

        return toEigen(N);
      };
      auto jacobianLambda = [&](auto& gpOffset_) {
        DefaultLinearAlgebra::template VarFixSizedMatrix<double, gridDim> dN;
        localBasis.evaluateJacobian(gp.position() + toDune(gpOffset_), dN);
        return toEigen(dN);
      };

      Eigen::Vector<double, gridDim> ipOffset = (Eigen::Vector<double, gridDim>::Random()).normalized() / 8;

      auto nonLinOpSpatialAll = NonLinearOperator(func, jacobianLambda, ipOffset);
      t.check(checkJacobian(nonLinOpSpatialAll, 1e-2));
      if constexpr (gridDim > 1) {
        std::cout << "Test Second Derivatives" << std::endl;

        for (int i = 0; const auto [firstDirection, secondDirection] : Dune::voigtNotationContainer<gridDim>) {
          std::cout << "Test Mixed Directions: " << firstDirection << " " << secondDirection << std::endl;
          auto jacobianLambda1D = [&](const auto& gpOffset_) {
            DefaultLinearAlgebra::template VarFixSizedMatrix<double, gridDim> dN;
            Dune::FieldVector<double, gridDim> gpOffset2D;
            std::ranges::fill(gpOffset2D, 0);
            gpOffset2D[firstDirection] = gpOffset_[0];
            localBasis.evaluateJacobian(gp.position() + gpOffset2D, dN);
            return toEigen(Dune::eval(col(dN, secondDirection)));
          };
          constexpr int secondDerivatives = gridDim * (gridDim + 1) / 2;
          auto hessianLambda              = [&](const auto& gpOffset_) {
            DefaultLinearAlgebra::template VarFixSizedMatrix<double, secondDerivatives> ddN;
            Dune::FieldVector<double, gridDim> gpOffset2D;
            std::ranges::fill(gpOffset2D, 0);
            gpOffset2D[firstDirection] = gpOffset_[0];
            localBasis.evaluateSecondDerivatives(gp.position() + gpOffset2D, ddN);
            return toEigen(Dune::eval(col(ddN, i)));
          };

          Eigen::Vector<double, 1> ipOffset1D(1);

          auto nonLinOpHg = NonLinearOperator(jacobianLambda1D, hessianLambda, ipOffset1D);

          t.check(checkJacobian(nonLinOpHg, 1e-2));
          ++i;
        }
      }
    }
  }
  // Unbound basis checks
  t.check(not localBasis.isBound(0));
  t.check(not localBasis.isBound(1));
  t.check(not localBasis.isBound(2));
  try {
    localBasis.evaluateFunction(0);
    t.check(false, "The prior function call should have thrown! You should not end up here.");
  } catch (const std::logic_error&) {
  }
  try {
    localBasis.evaluateJacobian(0);
    t.check(false, "The prior function call should have thrown! You should not end up here.");
  } catch (const std::logic_error&) {
  }
  if constexpr (gridDim > 1) {
    try {
      localBasis.evaluateSecondDerivatives(0);
      t.check(false, "The prior function call should have thrown! You should not end up here.");
    } catch (const std::logic_error&) {
    }
    localBasis.bind(rule, Dune::bindDerivatives(0, 1, 2));
    t.check(localBasis.isBound(0));
    t.check(localBasis.isBound(1));
    t.check(localBasis.isBound(2));
  } else {
    localBasis.bind(rule, Dune::bindDerivatives(0, 1));
    t.check(localBasis.isBound(0));
    t.check(localBasis.isBound(1));
  }

  return t;
}

template <int domainDim, int order>
auto localBasisTestConstructor(const Dune::GeometryType& geometryType, [[maybe_unused]] size_t nNodalTestPointsI = 6) {
  using namespace Dune::Indices;

  using FECache = Dune::PQkLocalFiniteElementCache<double, double, domainDim, order>;
  FECache feCache;
  const auto& fe  = feCache.get(geometryType);
  auto localBasis = Dune::CachedLocalBasis(fe.localBasis());

  return testLocalBasis(localBasis, geometryType);
}

int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);
  TestSuite t;
  using namespace Dune::GeometryTypes;
  std::cout << "Test line with linear ansatz functions" << std::endl;
  t.subTest(localBasisTestConstructor<1, 1>(line));
  std::cout << "Test line with quadratic ansatz functions" << std::endl;
  t.subTest(localBasisTestConstructor<1, 2>(line));
  std::cout << "Test triangle with linear ansatz functions" << std::endl;
  t.subTest(localBasisTestConstructor<2, 1>(triangle));
  std::cout << "Test triangle with quadratic ansatz functions" << std::endl;
  t.subTest(localBasisTestConstructor<2, 2>(triangle));
  std::cout << "Test quadrilateral with linear ansatz functions" << std::endl;
  t.subTest(localBasisTestConstructor<2, 1>(quadrilateral));
  std::cout << "Test quadrilateral with quadratic ansatz functions" << std::endl;
  t.subTest(localBasisTestConstructor<2, 2>(quadrilateral));
  std::cout << "Test hexahedron with linear ansatz functions" << std::endl;
  t.subTest(localBasisTestConstructor<3, 1>(hexahedron));
  std::cout << "Test hexahedron with quadratic ansatz functions" << std::endl;
  t.subTest(localBasisTestConstructor<3, 2>(hexahedron));

  return t.exit();
}
