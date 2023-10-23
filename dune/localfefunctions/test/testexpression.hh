// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <config.h>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
using Dune::TestSuite;

#include "fecache.hh"
#include "testFacilities.hh"
#include "testHelpers.hh"
#include "testfactories.hh"

#include <array>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <complex>
#include <vector>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/transpose.hh>
#include <dune/geometry/multilineargeometry.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrix.hh>
#include <dune/localfefunctions/impl/projectionBasedLocalFunction.hh>
#include <dune/localfefunctions/impl/standardLocalFunction.hh>
#include <dune/localfefunctions/localFunctionName.hh>
#include <dune/matrix-vector/transpose.hh>

#include <Eigen/Core>

template <typename M, typename id>
struct ManiFoldIDPair {
  using Manifold = M;
  using ID       = id;
};

template <template <int> typename M, auto id>
struct ManiFoldTemplateIDPair {
  template <int dim>
  using ManiFoldIDPair = ManiFoldIDPair<M<dim>, decltype(id)>;
};

auto singleStandardLocalFunction = [](auto& localBasis, auto& vBlockedLocal0, auto& geometry, auto&& ID) {
  return Dune::StandardLocalFunction(localBasis, vBlockedLocal0, geometry, ID);
};

auto singleProjectionBasedLocalFunction = [](auto& localBasis, auto& vBlockedLocal0, auto& geometry, auto&& ID) {
  return Dune::ProjectionBasedLocalFunction(localBasis, vBlockedLocal0, geometry, ID);
};

auto doubleStandardLocalFunctionDouble
    = [](auto& localBasis0, auto& vBlockedLocal0, auto&& ID0, auto&, auto&, auto&& ID1, auto& geometry) {
        auto f = Dune::StandardLocalFunction(localBasis0, vBlockedLocal0, geometry, ID0);
        auto g = Dune::StandardLocalFunction(localBasis0, vBlockedLocal0, geometry, ID1);
        return std::make_tuple(f, g);
      };

auto doubleStandardLocalFunctionDistinct = [](auto& localBasis0, auto& vBlockedLocal0, auto&& ID0, auto& localBasis1,
                                              auto& vBlockedLocal1, auto&& ID1, auto& geometry) {
  auto f = Dune::StandardLocalFunction(localBasis0, vBlockedLocal0, geometry, ID0);
  auto g = Dune::StandardLocalFunction(localBasis1, vBlockedLocal1, geometry, ID1);
  return std::make_tuple(f, g);
};

template <typename LF>
TestSuite testLocalFunction(const LF& lf, bool isCopy = false) {
  auto localFunctionName = Dune::localFunctionName(lf);
  TestSuite t(std::string(isCopy ? "Copy " : "") + localFunctionName);
  std::cout << "Testing: " + std::string(isCopy ? "Copy " : "") + localFunctionName << std::endl;
  const double tol = 1e-12;
  using namespace Dune::DerivativeDirections;
  using namespace autodiff;
  using namespace Dune;
  using namespace Testing;
  const auto& coeffs     = lf.node().coefficientsRef();
  const size_t coeffSize = coeffs.size();
  auto geometry          = lf.node().geometry();

  constexpr int gridDim = LF::gridDim;
  //  constexpr int worldDimension         = LF::worldDimension;
  using Manifold                       = typename std::remove_cvref_t<decltype(coeffs)>::value_type;
  constexpr int localFunctionValueSize = LF::Traits::valueSize;
  constexpr int coeffValueSize         = Manifold::valueSize;
  //  using ctype                          = typename Manifold::ctype;
  constexpr int coeffCorrectionSize = Manifold::correctionSize;

  // dynamic sized vectors before the loop
  Eigen::VectorXdual2nd xvr(coeffs.size() * coeffValueSize);
  xvr.setZero();
  Eigen::VectorXd gradienWRTCoeffs;
  Eigen::MatrixXd hessianWRTCoeffs;
  Eigen::VectorXd gradienWRTCoeffsSpatialAll;
  Eigen::MatrixXd hessianWRTCoeffsSpatialAll;
  std::array<Eigen::MatrixXd, gridDim> hessianWRTCoeffsTwoTimesSingleSpatial;
  std::array<Eigen::VectorXd, gridDim> gradientWRTCoeffsTwoTimesSingleSpatial;
  for (const auto& [ipIndex, ip] : lf.viewOverIntegrationPoints()) {
    /// Check spatial derivatives
    /// Check spatial derivatives return sizes
    {
      if constexpr (requires {
                      lf.evaluateDerivative(ipIndex, Dune::wrt(spatialAll),
                                            Dune::on(DerivativeDirections::referenceElement));
                    }) {
        using SpatialAllDerivative = std::remove_cvref_t<decltype(Dune::eval(
            lf.evaluateDerivative(ipIndex, Dune::wrt(spatialAll), Dune::on(DerivativeDirections::referenceElement))))>;
        static_assert(Cols<SpatialAllDerivative>::value == gridDim);
        static_assert(Rows<SpatialAllDerivative>::value == localFunctionValueSize);
      }
      if constexpr (requires {
                      lf.evaluateDerivative(ipIndex, Dune::wrt(spatial(0)),
                                            Dune::on(DerivativeDirections::referenceElement));
                    }) {
        using SpatialSingleDerivative = std::remove_cvref_t<decltype(Dune::eval(
            lf.evaluateDerivative(ipIndex, Dune::wrt(spatial(0)), Dune::on(DerivativeDirections::referenceElement))))>;
        static_assert(Cols<SpatialSingleDerivative>::value == 1);
        static_assert(Rows<SpatialSingleDerivative>::value == localFunctionValueSize);
      }

      /// Check if spatial derivatives are really derivatives
      /// Perturb in a random direction in the elements parameter space and check spatial derivative
      auto func = [&](auto& gpOffset_) {
        return toEigen(lf.evaluate(toFieldVector(gpOffset_), Dune::on(DerivativeDirections::referenceElement)));
      };
      auto spatialDerivAll = [&](auto& gpOffset_) {
        return toEigen(Dune::eval(lf.evaluateDerivative(toFieldVector(gpOffset_), Dune::wrt(spatialAll),
                                                        Dune::on(DerivativeDirections::referenceElement))));
      };

      Eigen::Vector<double, gridDim> ipOffset = (Eigen::Vector<double, gridDim>::Random()).normalized() / 16;
      try {
        auto nonLinOpSpatialAll = NonLinearOperator(func, spatialDerivAll, ipOffset);

        t.check(checkJacobian(nonLinOpSpatialAll, 1e-2), "Check spatial derivative in all directions");
      } catch (const Dune::NotImplemented& exception) {
        std::cout
            << "SpatialDerivative in all directions not tested, since it is not implemented by the local function "
                   + ("(" + localFunctionName + ")")
            << std::endl;
      }

      /// Perturb each spatial direction and check with derivative value
      for (int i = 0; i < gridDim; ++i) {
        Eigen::Vector<double, 1> ipOffsetSingle(ipOffset[i]);
        auto derivDerivSingleI = [&](auto gpOffset_) {
          auto offSetSingle = ipOffset;
          offSetSingle[i] += gpOffset_[0];
          return toEigen(Dune::eval(lf.evaluateDerivative(toFieldVector(offSetSingle), Dune::wrt(spatial(i)),
                                                          Dune::on(DerivativeDirections::referenceElement))));
        };

        auto funcSingle = [&](const auto& gpOffset_) {
          auto offSetSingle = ipOffset;
          offSetSingle[i] += gpOffset_[0];
          return toEigen(lf.evaluate(toFieldVector(offSetSingle), Dune::on(DerivativeDirections::referenceElement)));
        };

        try {
          auto nonLinOpSpatialSingle = NonLinearOperator(funcSingle, derivDerivSingleI, ipOffsetSingle);
          t.check(checkJacobian(nonLinOpSpatialSingle, 1e-2), "Check single spatial derivative");
        } catch (const Dune::NotImplemented& exception) {
          std::cout << "Single SpatialDerivative not tested, since it is not implemented by the local function "
                           + ("(" + localFunctionName + ")")
                    << std::endl;
        }
      }
    }

    /// Check coeff and spatial derivatives

    const auto alongVec = localFunctionValueSize == 1 ? createOnesVector<double, localFunctionValueSize>()
                                                      : createRandomVector<double, localFunctionValueSize>();
    const auto alongMat = localFunctionValueSize == 1 ? createOnesMatrix<double, localFunctionValueSize, gridDim>()
                                                      : createRandomMatrix<double, localFunctionValueSize, gridDim>();
    /// Rebind local function to second order dual number
    auto lfDual2nd                   = lf.rebindClone(dual2nd());
    auto lfDual2ndLeafNodeCollection = collectLeafNodeLocalFunctions(lfDual2nd);

    auto localFdual2nd = [&](const auto& x) {
      lfDual2ndLeafNodeCollection.addToCoeffsInEmbedding(x);
      auto value = inner(lfDual2nd.evaluate(ipIndex, Dune::on(DerivativeDirections::referenceElement)), alongVec);
      lfDual2ndLeafNodeCollection.addToCoeffsInEmbedding(-x);
      return value;
    };

    auto localFdual2ndSpatialSingle = [&](const auto& x, int i) {
      lfDual2ndLeafNodeCollection.addToCoeffsInEmbedding(x);
      auto value = inner(lfDual2nd.evaluateDerivative(ipIndex, Dune::wrt(spatial(i)),
                                                      Dune::on(DerivativeDirections::referenceElement)),
                         alongVec);

      lfDual2ndLeafNodeCollection.addToCoeffsInEmbedding(-x);
      return value;
    };

    auto localFdual2ndSpatialAll = [&](const auto& x) {
      lfDual2ndLeafNodeCollection.addToCoeffsInEmbedding(x);
      auto value = inner(lfDual2nd.evaluateDerivative(ipIndex, Dune::wrt(spatialAll),
                                                      Dune::on(DerivativeDirections::referenceElement)),
                         alongMat);

      lfDual2ndLeafNodeCollection.addToCoeffsInEmbedding(-x);
      return value;
    };

    dual2nd u;

    autodiff::hessian(localFdual2nd, autodiff::wrt(xvr), autodiff::at(xvr), u, gradienWRTCoeffs, hessianWRTCoeffs);
    bool spatialSingleImplemented{true}, spatialAllImplemented{true};
    try {
      autodiff::hessian(localFdual2ndSpatialAll, autodiff::wrt(xvr), autodiff::at(xvr), u, gradienWRTCoeffsSpatialAll,
                        hessianWRTCoeffsSpatialAll);
    } catch (const Dune::NotImplemented& exception) {
      std::cout << "SpatialDerivative in all directions not tested, since it is not implemented by the local function "
                       + ("(" + localFunctionName + ")")
                << std::endl;
      spatialSingleImplemented = false;
    }
    try {
      for (int d = 0; d < gridDim; ++d)
        autodiff::hessian(localFdual2ndSpatialSingle, autodiff::wrt(xvr), autodiff::at(xvr, d), u,
                          gradientWRTCoeffsTwoTimesSingleSpatial[d], hessianWRTCoeffsTwoTimesSingleSpatial[d]);

    } catch (const Dune::NotImplemented& exception) {
      std::cout << "Single SpatialDerivative not tested, since it is not implemented by the local function "
                       + ("(" + localFunctionName + ")")
                << std::endl;
      spatialAllImplemented = false;
    }

    for (size_t i = 0; i < coeffSize; ++i) {
      const auto BLAi = coeffs[i].orthonormalFrame();
      const auto jacobianWRTCoeffslf
          = lf.evaluateDerivative(ipIndex, Dune::wrt(coeff(i)), Dune::on(DerivativeDirections::referenceElement));
      static_assert(Rows<decltype(jacobianWRTCoeffslf)>::value == localFunctionValueSize);
      static_assert(Cols<decltype(jacobianWRTCoeffslf)>::value == coeffCorrectionSize);
      const auto jacobianWRTCoeffs = transposeEvaluated(leftMultiplyTranspose(alongVec, jacobianWRTCoeffslf));
      static_assert(Rows<decltype(jacobianWRTCoeffs)>::value == coeffCorrectionSize);  // note the  transpose above
      static_assert(Cols<decltype(jacobianWRTCoeffs)>::value == 1);
      auto jacobianWRTCoeffsAD
          = Dune::eval(transposeEvaluated(BLAi) * segment<coeffValueSize>(gradienWRTCoeffs, i * coeffValueSize));
      t.require(isApproxSame(jacobianWRTCoeffs, jacobianWRTCoeffsAD, tol), "Test first derivative wrt coeff")
          << "jacobianWRTCoeffs:\n"
          << jacobianWRTCoeffs << "\n jacobianWRTCoeffsAD: \n"
          << jacobianWRTCoeffsAD << "\nwithout alongVec:\n"
          << jacobianWRTCoeffslf;

      if (spatialSingleImplemented)
        for (int d = 0; d < gridDim; ++d) {
          const auto jacoWrtCoeffAndSpatiallf = lf.evaluateDerivative(ipIndex, Dune::wrt(coeff(i), spatial(d)),
                                                                      Dune::on(DerivativeDirections::referenceElement));
          static_assert(Cols<decltype(jacoWrtCoeffAndSpatiallf)>::value == coeffCorrectionSize);
          static_assert(Rows<decltype(jacoWrtCoeffAndSpatiallf)>::value == localFunctionValueSize);
          const auto jacoWrtCoeffAndSpatial
              = transposeEvaluated(leftMultiplyTranspose(alongVec, jacoWrtCoeffAndSpatiallf));
          const auto jacoWrtSpatialAndCoeff = transposeEvaluated(
              leftMultiplyTranspose(alongVec, lf.evaluateDerivative(ipIndex, Dune::wrt(spatial(d), coeff(i)),
                                                                    Dune::on(DerivativeDirections::referenceElement))));

          static_assert(Cols<decltype(jacoWrtSpatialAndCoeff)>::value == 1);
          static_assert(Rows<decltype(jacoWrtSpatialAndCoeff)>::value == coeffCorrectionSize);
          static_assert(Rows<decltype(jacoWrtCoeffAndSpatial)>::value == Rows<decltype(jacoWrtSpatialAndCoeff)>::value
                        and Cols<decltype(jacoWrtCoeffAndSpatial)>::value
                                == Cols<decltype(jacoWrtSpatialAndCoeff)>::value);

          t.check(
              isApproxSame(jacoWrtCoeffAndSpatial, jacoWrtSpatialAndCoeff, tol),
              "Check that passing in different order does not change anything for the derivatives (spatial single)");

          auto jacoWrtCoeffAndSpatialExpected
              = Dune::eval(transposeEvaluated(BLAi)
                           * segment<coeffValueSize>(gradientWRTCoeffsTwoTimesSingleSpatial[d], i * coeffValueSize));
          t.check(isApproxSame(jacoWrtCoeffAndSpatial, jacoWrtCoeffAndSpatialExpected, tol),
                  "Test mixed first derivative wrt coeff and first derivative wrt single spatial");
        }

      if (spatialAllImplemented and spatialSingleImplemented) {
        const auto jacoWrtSpatialAllAndCoeff = lf.evaluateDerivative(ipIndex, Dune::wrt(spatialAll, coeff(i)),
                                                                     Dune::on(DerivativeDirections::referenceElement));

        static_assert(Cols<decltype(jacoWrtSpatialAllAndCoeff[0])>::value == coeffCorrectionSize);
        static_assert(Rows<decltype(jacoWrtSpatialAllAndCoeff[0])>::value == localFunctionValueSize);

        typename DefaultLinearAlgebra::template FixedSizedMatrix<double, 1, coeffCorrectionSize>
            jacoWrtSpatialAllAndCoeffProd;
        setZero(jacoWrtSpatialAllAndCoeffProd);

        for (int d = 0; d < gridDim; ++d)
          jacoWrtSpatialAllAndCoeffProd += leftMultiplyTranspose(col(alongMat, d), jacoWrtSpatialAllAndCoeff[d]);

        t.check(
            isApproxSame((jacoWrtSpatialAllAndCoeffProd),
                         transposeEvaluated(transposeEvaluated(BLAi)
                                            * segment<coeffValueSize>(gradienWRTCoeffsSpatialAll, i * coeffValueSize)),
                         tol),
            "Test spatiall all and coeff derivative")
            << "jacoWrtSpatialAllAndCoeffProd\n"
            << jacoWrtSpatialAllAndCoeffProd << "\ntransposeEvaluated(transposeEvaluated(BLAi) \n"
            << transposeEvaluated(BLAi) << "\nsegment<coeffValueSize>(gradienWRTCoeffsSpatialAll, i * coeffValueSize)\n"
            << segment<coeffValueSize>(gradienWRTCoeffsSpatialAll, i * coeffValueSize)
            << "\n gradienWRTCoeffsSpatialAll\n"
            << gradienWRTCoeffsSpatialAll;

        // Check if spatialAll returns the same as the single spatial derivatives
        const auto Warray  = lf.evaluateDerivative(ipIndex, Dune::wrt(coeff(i), spatialAll),
                                                   Dune::on(DerivativeDirections::referenceElement));
        const auto Warray2 = lf.evaluateDerivative(ipIndex, Dune::wrt(spatialAll, coeff(i)),
                                                   Dune::on(DerivativeDirections::referenceElement));
        for (int j = 0; j < gridDim; ++j)
          t.check(isApproxSame(Warray[j], Warray2[j], tol),
                  "Check that passing in different order does not change anything for the derivatives (spatial all)")
              << "Warray[j]\n"
              << Warray[j] << "\nWarray2[j]\n"
              << Warray2[j];

        std::array<std::remove_cvref_t<decltype(Warray[0])>, gridDim> WarraySingle;
        for (int s = 0; s < gridDim; ++s)
          WarraySingle[s] = lf.evaluateDerivative(ipIndex, Dune::wrt(coeff(i), spatial(s)),
                                                  Dune::on(DerivativeDirections::referenceElement));

        for (int j = 0; j < gridDim; ++j)
          t.check(isApproxSame(Warray[j], WarraySingle[j], tol),
                  "Check that spatial all and spatial single coincide with coeff derivative");
      }
      for (size_t j = 0; j < coeffSize; ++j) {
        const auto BLAj                      = coeffs[j].orthonormalFrame();
        const auto jacobianWRTCoeffsTwoTimes = lf.evaluateDerivative(
            ipIndex, Dune::wrt(coeff(i, j)), Dune::along(alongVec), Dune::on(DerivativeDirections::referenceElement));
        static_assert(Cols<decltype(jacobianWRTCoeffsTwoTimes)>::value == coeffCorrectionSize);
        static_assert(Rows<decltype(jacobianWRTCoeffsTwoTimes)>::value == coeffCorrectionSize);
        const auto jacobianWRTCoeffsTwoTimesExpected = Dune::eval(
            transposeEvaluated(BLAi)
                * block<coeffValueSize, coeffValueSize>(hessianWRTCoeffs, i * coeffValueSize, j * coeffValueSize) * BLAj
            + (i == j) * coeffs[i].weingarten(segment<coeffValueSize>(gradienWRTCoeffs, i * coeffValueSize)));

        const bool passed = isApproxSame(jacobianWRTCoeffsTwoTimes, jacobianWRTCoeffsTwoTimesExpected, tol);
        t.check(passed, "Test second derivatives wrt coeffs" + std::string((i == j) ? " for i==j " : " for i!=j"));
        if (not passed) {
          std::cout << "Actual: \n"
                    << jacobianWRTCoeffsTwoTimes << "\n Expected: \n"
                    << jacobianWRTCoeffsTwoTimesExpected << "\n Diff: \n"
                    << jacobianWRTCoeffsTwoTimes - jacobianWRTCoeffsTwoTimesExpected << "\n Weingarten part: \n"
                    << (i == j) * coeffs[i].weingarten(segment<coeffValueSize>(gradienWRTCoeffs, i * coeffValueSize))
                    << std::endl;
          std::cout << "\n Total Hessian:\n" << hessianWRTCoeffs << std::endl;
        }

        if (spatialAllImplemented) {
          const auto jacobianWRTCoeffsTwoTimesSpatialAll
              = lf.evaluateDerivative(ipIndex, Dune::wrt(coeff(i, j), spatialAll), Dune::along(alongMat),
                                      Dune::on(DerivativeDirections::referenceElement));
          static_assert(Cols<decltype(jacobianWRTCoeffsTwoTimesSpatialAll)>::value == coeffCorrectionSize);
          static_assert(Rows<decltype(jacobianWRTCoeffsTwoTimesSpatialAll)>::value == coeffCorrectionSize);
          const auto jacobianWRTCoeffsTwoTimesSpatialAllExpected = Dune::eval(
              transposeEvaluated(BLAi)
                  * block<coeffValueSize, coeffValueSize>(hessianWRTCoeffsSpatialAll, i * coeffValueSize,
                                                          j * coeffValueSize)
                  * BLAj
              + (i == j)
                    * coeffs[j].weingarten(segment<coeffValueSize>(gradienWRTCoeffsSpatialAll, i * coeffValueSize)));

          /// if the order of the function value is less then quadratic then this should yield a vanishing derivative
          if (lf.order() < quadratic) {
            t.check(two_norm(jacobianWRTCoeffsTwoTimesSpatialAll) < tol,
                    "For first order linear local functions the second derivative should vanish");
            t.check(two_norm(jacobianWRTCoeffsTwoTimesSpatialAllExpected) < tol,
                    "For first order linear local functions the second derivative should vanish");
          } else {
            const bool passed2
                = isApproxSame(jacobianWRTCoeffsTwoTimesSpatialAll, jacobianWRTCoeffsTwoTimesSpatialAllExpected, tol);
            t.check(passed2, "Test third derivatives wrt coeffs, coeffs and spatialall");
            if (not passed2)
              std::cout << "Actual: \n"
                        << jacobianWRTCoeffsTwoTimesSpatialAll << "\n Expected: \n"
                        << jacobianWRTCoeffsTwoTimesSpatialAllExpected << "\n Diff: \n"
                        << jacobianWRTCoeffsTwoTimesSpatialAll - jacobianWRTCoeffsTwoTimesSpatialAllExpected
                        << std::endl;
          }
        }
        if (spatialSingleImplemented)
          for (int d = 0; d < gridDim; ++d) {
            const auto jacobianWRTCoeffsTwoTimesSingleSpatial
                = lf.evaluateDerivative(ipIndex, Dune::wrt(coeff(i, j), spatial(d)), Dune::along(alongVec),
                                        Dune::on(DerivativeDirections::referenceElement));
            static_assert(Cols<decltype(jacobianWRTCoeffsTwoTimesSingleSpatial)>::value == coeffCorrectionSize);
            static_assert(Rows<decltype(jacobianWRTCoeffsTwoTimesSingleSpatial)>::value == coeffCorrectionSize);
            const auto jacobianWRTCoeffsTwoTimesSingleSpatialExpected
                = Dune::eval(transposeEvaluated(BLAi)
                                 * block<coeffValueSize, coeffValueSize>(hessianWRTCoeffsTwoTimesSingleSpatial[d],
                                                                         i * coeffValueSize, j * coeffValueSize)
                                 * BLAj
                             + (i == j)
                                   * coeffs[j].weingarten(segment<coeffValueSize>(
                                       gradientWRTCoeffsTwoTimesSingleSpatial[d], i * coeffValueSize)));
            const bool passed3 = isApproxSame(jacobianWRTCoeffsTwoTimesSingleSpatial,
                                              jacobianWRTCoeffsTwoTimesSingleSpatialExpected, tol);
            t.check(passed3, "Test third derivatives wrt coeffs, coeffs and spatial single");
            if (not passed3)
              std::cout << "Actual: \n"
                        << jacobianWRTCoeffsTwoTimesSingleSpatial << "\n Expected: \n"
                        << jacobianWRTCoeffsTwoTimesSingleSpatialExpected << "\n Diff: \n"
                        << jacobianWRTCoeffsTwoTimesSingleSpatial - jacobianWRTCoeffsTwoTimesSingleSpatialExpected
                        << std::endl;
          }
      }
    }
  }
  std::puts("done.\n");
  if (not isCopy) {  //  test the cloned local function
    const auto lfCopy     = lf.clone();
    const auto& coeffCopy = lfCopy.node().coefficientsRef();
    for (size_t i = 0; i < coeffSize; ++i)
      t.check(coeffCopy[i] == coeffs[i], "Copied coeffs coincide");

    t.check(&coeffCopy != &coeffs, "Copied coeffs differ by address");

    t.subTest(testLocalFunction(lfCopy, true));
  }
  return t;
}

template <typename Manifold, int domainDim, int order>
auto createVectorOfNodalValues(const Dune::GeometryType& geometryType, size_t nNodalTestPointsI = 1) {
  static constexpr int worldDim = Manifold::valueSize;
  using namespace Dune;
  using namespace Dune::Indices;
  const auto& refElement = Dune::ReferenceElements<double, domainDim>::general(geometryType);
  std::vector<Dune::FieldVector<double, worldDim>> corners;
  CornerFactory<worldDim>::construct(corners, refElement.size(domainDim));
  auto geometry = std::make_shared<const Dune::MultiLinearGeometry<double, domainDim, worldDim>>(refElement, corners);

  Dune::BlockVector<Manifold> testNodalPoints;

  FECache<domainDim, order> feCache;
  const auto& fe             = feCache.get(geometryType);
  const size_t nNodes        = fe.size();
  const int nNodalTestPoints = std::max(nNodalTestPointsI, nNodes);
  Dune::ValueFactory<Manifold>::construct(testNodalPoints, nNodalTestPoints);
  Dune::BlockVector<Manifold> vBlockedLocal(nNodes);

  for (size_t j = 0; j < fe.size(); ++j)
    vBlockedLocal[j] = testNodalPoints[j];

  return vBlockedLocal;
}

template <int domainDim, int order, typename Expr, typename ExprTest, typename BaseFunctionConstructor,
          typename ManifoldIDPair0, typename ManifoldIDPair1, bool doDefaultTests = true>
auto localFunctionTestConstructorNew(const Dune::GeometryType& geometryType, Expr& expr, ExprTest& exprSpecialTests,
                                     const BaseFunctionConstructor& baseFunctionConstructor,
                                     size_t nNodalTestPointsI = 1) {
  TestSuite t;
  using namespace Dune;
  using namespace Dune::Indices;
  using Manifold0 = typename ManifoldIDPair0::Manifold;
  using ID0       = typename ManifoldIDPair0::ID;

  using Manifold1 = typename ManifoldIDPair1::Manifold;
  using ID1       = typename ManifoldIDPair1::ID;

  static constexpr int worldDim = Manifold0::valueSize;
  const auto& refElement        = Dune::ReferenceElements<double, domainDim>::general(geometryType);

  std::vector<Dune::FieldVector<double, worldDim>> corners;
  CornerFactory<worldDim>::construct(corners, refElement.size(domainDim));
  auto geometry = std::make_shared<const Dune::MultiLinearGeometry<double, domainDim, worldDim>>(refElement, corners);

  FECache<domainDim, order> feCache;
  const auto& fe  = feCache.get(geometryType);
  auto localBasis = Dune::CachedLocalBasis(fe.localBasis());

  auto vBlockedLocal0 = createVectorOfNodalValues<Manifold0, domainDim, order>(geometryType, nNodalTestPointsI);
  auto vBlockedLocal1 = createVectorOfNodalValues<Manifold1, domainDim, order>(geometryType, nNodalTestPointsI);

  const auto& rule = Dune::QuadratureRules<double, domainDim>::rule(fe.type(), 2);
  localBasis.bind(rule, bindDerivatives(0, 1));

  auto [f, g] = baseFunctionConstructor(localBasis, vBlockedLocal0, ID0{}, localBasis, vBlockedLocal1, ID1{}, geometry);
  auto h      = expr(f, g);
  t.subTest(exprSpecialTests(h, vBlockedLocal0, vBlockedLocal1, fe));
  if constexpr (doDefaultTests) t.subTest(testLocalFunction(h));

  return t;
}

template <int domainDim, int order, typename Expr, typename ExprTest, typename BaseFunctionConstructor,
          typename ManifoldIDPair, bool doDefaultTests = true>
auto localFunctionTestConstructorNew(const Dune::GeometryType& geometryType, Expr& expr, ExprTest& exprSpecialTests,
                                     const BaseFunctionConstructor& baseFunctionConstructor,
                                     size_t nNodalTestPointsI = 1) {
  TestSuite t;
  using namespace Dune;
  using namespace Dune::Indices;
  using Manifold                = typename ManifoldIDPair::Manifold;
  using ID                      = typename ManifoldIDPair::ID;
  static constexpr int worldDim = Manifold::valueSize;
  const auto& refElement        = Dune::ReferenceElements<double, domainDim>::general(geometryType);

  std::vector<Dune::FieldVector<double, worldDim>> corners;
  CornerFactory<worldDim>::construct(corners, refElement.size(domainDim));
  auto geometry = std::make_shared<const Dune::MultiLinearGeometry<double, domainDim, worldDim>>(refElement, corners);

  FECache<domainDim, order> feCache;
  const auto& fe  = feCache.get(geometryType);
  auto localBasis = Dune::CachedLocalBasis(fe.localBasis());

  auto vBlockedLocal0 = createVectorOfNodalValues<Manifold, domainDim, order>(geometryType, nNodalTestPointsI);

  const auto& rule = Dune::QuadratureRules<double, domainDim>::rule(fe.type(), 2);
  localBasis.bind(rule, bindDerivatives(0, 1));

  auto f = baseFunctionConstructor(localBasis, vBlockedLocal0, geometry, ID{});

  auto h = expr(f);
  t.subTest(exprSpecialTests(h, vBlockedLocal0, fe));
  if constexpr (doDefaultTests) t.subTest(testLocalFunction(h));

  return t;
}

using namespace Dune::GeometryTypes;

template <typename Expr, typename ExprTests, typename BaseFunctionConstructor, bool doDefaultTests = true,
          typename... ManiFoldTemplateIDPairS>
auto testExpressionsOnTriangle(Expr& expr, ExprTests& exprTests,
                               const BaseFunctionConstructor& baseFunctionConstructor) {
  TestSuite t("testExpressionsOnTriangle");

  std::cout << "triangle with linear ansatz functions and 2d local function" << std::endl;
  t.subTest(
      localFunctionTestConstructorNew<2, 1, Expr, ExprTests, BaseFunctionConstructor,
                                      typename ManiFoldTemplateIDPairS::template ManiFoldIDPair<2>..., doDefaultTests>(
          triangle, expr, exprTests, baseFunctionConstructor));
  std::cout << "triangle with quadratic ansatz functions and 3d local function" << std::endl;
  t.subTest(
      localFunctionTestConstructorNew<2, 1, Expr, ExprTests, BaseFunctionConstructor,
                                      typename ManiFoldTemplateIDPairS::template ManiFoldIDPair<3>..., doDefaultTests>(
          triangle, expr, exprTests, baseFunctionConstructor));
  return t;
}

template <typename Expr, typename ExprTests, typename BaseFunctionConstructor, bool doDefaultTests = true,
          typename... ManiFoldTemplateIDPairS>
auto testExpressionsOnQuadrilateral(Expr& expr, ExprTests& exprTests,
                                    const BaseFunctionConstructor& baseFunctionConstructor) {
  TestSuite t("testExpressionsOnQuadrilateral");
  std::cout << "quadrilateral with linear ansatz functions and 2d local function" << std::endl;
  t.subTest(
      localFunctionTestConstructorNew<2, 1, Expr, ExprTests, BaseFunctionConstructor,
                                      typename ManiFoldTemplateIDPairS::template ManiFoldIDPair<2>..., doDefaultTests>(
          quadrilateral, expr, exprTests, baseFunctionConstructor));
  std::cout << "quadrilateral with quadratic ansatz functions and 3d local function" << std::endl;
  t.subTest(
      localFunctionTestConstructorNew<2, 2, Expr, ExprTests, BaseFunctionConstructor,
                                      typename ManiFoldTemplateIDPairS::template ManiFoldIDPair<3>..., doDefaultTests>(
          quadrilateral, expr, exprTests, baseFunctionConstructor));
  return t;
}

template <typename Expr, typename ExprTests, typename BaseFunctionConstructor, bool doDefaultTests = true,
          typename... ManiFoldTemplateIDPairS>
auto testExpressionsOnHexahedron(Expr& expr, ExprTests& exprTests,
                                 const BaseFunctionConstructor& baseFunctionConstructor) {
  TestSuite t("testExpressionsOnHexahedron");

  std::cout << "hexahedron with linear ansatz functions and 3d local function" << std::endl;
  t.subTest(
      localFunctionTestConstructorNew<3, 1, Expr, ExprTests, BaseFunctionConstructor,
                                      typename ManiFoldTemplateIDPairS::template ManiFoldIDPair<3>..., doDefaultTests>(
          hexahedron, expr, exprTests, baseFunctionConstructor));
  std::cout << "hexahedron with quadratic ansatz functions and 3d local function" << std::endl;
  t.subTest(
      localFunctionTestConstructorNew<3, 2, Expr, ExprTests, BaseFunctionConstructor,
                                      typename ManiFoldTemplateIDPairS::template ManiFoldIDPair<3>..., doDefaultTests>(
          hexahedron, expr, exprTests, baseFunctionConstructor));

  return t;
}

template <typename Expr, typename ExprTests, typename BaseFunctionConstructor, bool doDefaultTests = true,
          typename... ManiFoldTemplateIDPairS>
auto testExpressionsOnLine(Expr& expr, ExprTests& exprTests, const BaseFunctionConstructor& baseFunctionConstructor) {
  TestSuite t("testExpressionsOnLine");
  std::cout << "line with linear ansatz functions and 1d local function" << std::endl;
  t.subTest(
      localFunctionTestConstructorNew<1, 1, Expr, ExprTests, BaseFunctionConstructor,
                                      typename ManiFoldTemplateIDPairS::template ManiFoldIDPair<1>..., doDefaultTests>(
          line, expr, exprTests, baseFunctionConstructor));
  std::cout << "line with quadratic ansatz functions and 3d local function" << std::endl;
  t.subTest(
      localFunctionTestConstructorNew<1, 2, Expr, ExprTests, BaseFunctionConstructor,
                                      typename ManiFoldTemplateIDPairS::template ManiFoldIDPair<3>..., doDefaultTests>(
          line, expr, exprTests, baseFunctionConstructor));
  return t;
}

template <int domainDim, int order, int worldDim, typename Expr, typename ExprTests, typename BaseFunctionConstructor,
          bool doDefaultTests = true, typename... ManiFoldTemplateIDPairS>
auto testExpressionsOnCustomGeometry(const Dune::GeometryType& geometryType, Expr& expr, ExprTests& exprTests,
                                     const BaseFunctionConstructor& baseFunctionConstructor) {
  using namespace std::string_literals;
  TestSuite t("testExpressionsOnCustomGeometry" + " domainDim: "s + std::to_string(domainDim) + " order: "s
              + std::to_string(order) + " worldDim: "s + std::to_string(worldDim));
  std::cout << "line with linear ansatz functions and 1d local function" << std::endl;
  t.subTest(localFunctionTestConstructorNew<domainDim, order, Expr, ExprTests, BaseFunctionConstructor,
                                            typename ManiFoldTemplateIDPairS::template ManiFoldIDPair<worldDim>...,
                                            doDefaultTests>(geometryType, expr, exprTests, baseFunctionConstructor));
  return t;
}
