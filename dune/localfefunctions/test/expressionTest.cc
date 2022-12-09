// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <config.h>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
using Dune::TestSuite;

#include "factories.hh"
#include "testFacilities.hh"
#include "testHelpers.hh"

#include <array>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <complex>
#include <vector>

#include <dune/common/classname.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/transpose.hh>
#include <dune/functions/functionspacebases/basistags.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/geometry/multilineargeometry.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrix.hh>
#include <dune/localfefunctions/expressions.hh>
#include <dune/localfefunctions/impl/projectionBasedLocalFunction.hh>
#include <dune/localfefunctions/impl/standardLocalFunction.hh>
#include <dune/localfefunctions/localFunctionName.hh>
#include <dune/localfefunctions/manifolds/realTuple.hh>
#include <dune/localfefunctions/manifolds/unitVector.hh>
#include <dune/matrix-vector/transpose.hh>

#include <Eigen/Core>

template <typename LF>
TestSuite testLocalFunction(const LF &lf, bool isCopy = false) {
  auto localFunctionName = Dune::localFunctionName(lf);
  TestSuite t(std::string(isCopy ? "Copy " : "") + localFunctionName);
  std::cout << "Testing: " + std::string(isCopy ? "Copy " : "") + localFunctionName << std::endl;
  const double tol = 1e-12;
  using namespace Dune::DerivativeDirections;
  using namespace autodiff;
  using namespace Dune;
  using namespace Testing;
  const auto &coeffs     = lf.node().coefficientsRef();
  const size_t coeffSize = coeffs.size();
  auto geometry          = lf.node().geometry();

  constexpr int gridDim                = LF::gridDim;
  constexpr int worldDimension         = LF::worldDimension;
  using Manifold                       = typename std::remove_cvref_t<decltype(coeffs)>::value_type;
  constexpr int localFunctionValueSize = LF::Traits::valueSize;
  constexpr int coeffValueSize         = Manifold::valueSize;
  using ctype                          = typename Manifold::ctype;
  constexpr int coeffCorrectionSize    = Manifold::correctionSize;

  // dynamic sized vectors before the loop
  Eigen::VectorXdual2nd xvr(valueSize(coeffs));
  xvr.setZero();
  Eigen::VectorXd gradienWRTCoeffs;
  Eigen::MatrixXd hessianWRTCoeffs;
  Eigen::VectorXd gradienWRTCoeffsSpatialAll;
  Eigen::MatrixXd hessianWRTCoeffsSpatialAll;
  std::array<Eigen::MatrixXd, gridDim> hessianWRTCoeffsTwoTimesSingleSpatial;
  std::array<Eigen::VectorXd, gridDim> gradientWRTCoeffsTwoTimesSingleSpatial;
  for (const auto &[ipIndex, ip] : lf.viewOverIntegrationPoints()) {
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
      auto func = [&](auto &gpOffset_) {
        return toEigen(lf.evaluate(toFieldVector(gpOffset_), Dune::on(DerivativeDirections::referenceElement)));
      };
      auto spatialDerivAll = [&](auto &gpOffset_) {
        return toEigen(Dune::eval(lf.evaluateDerivative(toFieldVector(gpOffset_), Dune::wrt(spatialAll),
                                                        Dune::on(DerivativeDirections::referenceElement))));
      };

      Eigen::Vector<double, gridDim> ipOffset = (Eigen::Vector<double, gridDim>::Random()).normalized() / 16;
      try {
        auto nonLinOpSpatialAll = NonLinearOperator(func, spatialDerivAll, ipOffset);

        t.check(checkJacobian(nonLinOpSpatialAll, 1e-2), "Check spatial derivative in all directions");
      } catch (const Dune::NotImplemented &exception) {
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

        auto funcSingle = [&](const auto &gpOffset_) {
          auto offSetSingle = ipOffset;
          offSetSingle[i] += gpOffset_[0];
          return toEigen(lf.evaluate(toFieldVector(offSetSingle), Dune::on(DerivativeDirections::referenceElement)));
        };

        try {
          auto nonLinOpSpatialSingle = NonLinearOperator(funcSingle, derivDerivSingleI, ipOffsetSingle);
          t.check(checkJacobian(nonLinOpSpatialSingle, 1e-2), "Check single spatial derivative");
        } catch (const Dune::NotImplemented &exception) {
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

    auto localFdual2nd = [&](const auto &x) {
      lfDual2ndLeafNodeCollection.addToCoeffsInEmbedding(x);
      auto value = inner(lfDual2nd.evaluate(ipIndex, Dune::on(DerivativeDirections::referenceElement)), alongVec);
      lfDual2ndLeafNodeCollection.addToCoeffsInEmbedding(-x);
      return value;
    };

    auto localFdual2ndSpatialSingle = [&](const auto &x, int i) {
      lfDual2ndLeafNodeCollection.addToCoeffsInEmbedding(x);
      auto value = inner(lfDual2nd.evaluateDerivative(ipIndex, Dune::wrt(spatial(i)),
                                                      Dune::on(DerivativeDirections::referenceElement)),
                         alongVec);

      lfDual2ndLeafNodeCollection.addToCoeffsInEmbedding(-x);
      return value;
    };

    auto localFdual2ndSpatialAll = [&](const auto &x) {
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
    } catch (const Dune::NotImplemented &exception) {
      std::cout << "SpatialDerivative in all directions not tested, since it is not implemented by the local function "
                       + ("(" + localFunctionName + ")")
                << std::endl;
      spatialSingleImplemented = false;
    }
    try {
      for (int d = 0; d < gridDim; ++d)
        autodiff::hessian(localFdual2ndSpatialSingle, autodiff::wrt(xvr), autodiff::at(xvr, d), u,
                          gradientWRTCoeffsTwoTimesSingleSpatial[d], hessianWRTCoeffsTwoTimesSingleSpatial[d]);

    } catch (const Dune::NotImplemented &exception) {
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
          if constexpr (lf.order() < quadratic) {
            t.check(two_norm(jacobianWRTCoeffsTwoTimesSpatialAll) < tol,
                    "For first order linear local functions the second derivative should vanish");
            t.check(two_norm(jacobianWRTCoeffsTwoTimesSpatialAllExpected) < tol,
                    "For first order linear local functions the second derivative should vanish");
          } else {
            const bool passed
                = isApproxSame(jacobianWRTCoeffsTwoTimesSpatialAll, jacobianWRTCoeffsTwoTimesSpatialAllExpected, tol);
            t.check(passed, "Test third derivatives wrt coeffs, coeffs and spatialall");
            if (not passed)
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
            const bool passed = isApproxSame(jacobianWRTCoeffsTwoTimesSingleSpatial,
                                             jacobianWRTCoeffsTwoTimesSingleSpatialExpected, tol);
            t.check(passed, "Test third derivatives wrt coeffs, coeffs and spatial single");
            if (not passed)
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
    const auto &coeffCopy = lfCopy.node().coefficientsRef();
    for (size_t i = 0; i < coeffSize; ++i)
      t.check(coeffCopy[i] == coeffs[i], "Copied coeffs coincide");

    t.check(&coeffCopy != &coeffs, "Copied coeffs differ by address");

    t.subTest(testLocalFunction(lfCopy, true));
  }
  return t;
}

template <int domainDim, int worldDim, int order>
auto localFunctionTestConstructor(const Dune::GeometryType &geometryType, size_t nNodalTestPointsI = 1) {
  TestSuite t;
  using namespace Dune;
  using namespace Dune::Indices;
  const auto &refElement = Dune::ReferenceElements<double, domainDim>::general(geometryType);
  std::vector<Dune::FieldVector<double, worldDim>> corners;
  CornerFactory<worldDim>::construct(corners, refElement.size(domainDim));
  auto geometry = std::make_shared<const Dune::MultiLinearGeometry<double, domainDim, worldDim>>(refElement, corners);

  using Manifold     = Dune::RealTuple<double, worldDim>;
  using Manifold2    = Dune::UnitVector<double, worldDim>;
  using VectorType   = DefaultLinearAlgebra::FixedSizedVector<double, worldDim>;
  using MatrixType   = DefaultLinearAlgebra::FixedSizedMatrix<double, worldDim, worldDim>;
  constexpr int size = Manifold::valueSize;
  Dune::BlockVector<Manifold> testNodalPoints;
  using FECache = Dune::PQkLocalFiniteElementCache<double, double, domainDim, order>;
  FECache feCache;
  const auto &fe      = feCache.get(geometryType);
  auto localBasis     = Dune::CachedLocalBasis(fe.localBasis());
  const size_t nNodes = fe.size();
  Dune::BlockVector<Manifold> testNodalPoints1, testNodalPoints2;
  const int nNodalTestPoints = std::max(nNodalTestPointsI, nNodes);
  Dune::ValueFactory<Manifold>::construct(testNodalPoints1, nNodalTestPoints);
  Dune::ValueFactory<Manifold>::construct(testNodalPoints2, nNodalTestPoints);

  Dune::BlockVector<Manifold2> testNodalPoints3;
  Dune::ValueFactory<Manifold2>::construct(testNodalPoints3, nNodalTestPoints);

  Dune::BlockVector<Manifold> vBlockedLocal(nNodes);
  Dune::BlockVector<Manifold> vBlockedLocal2(nNodes);
  Dune::BlockVector<Manifold2> vBlockedLocal3(nNodes);

  const auto &rule = Dune::QuadratureRules<double, domainDim>::rule(fe.type(), 2);
  localBasis.bind(rule, bindDerivatives(0, 1));

  // More thorough testing by swapping indices and testing again
  //  for (size_t i = 0; i < multIndex.cycles(); ++i, ++multIndex) {
  //    auto sortedMultiIndex = multIndex;
  //    std::ranges::sort(sortedMultiIndex);
  //    if (std::ranges::adjacent_find(sortedMultiIndex)
  //        != sortedMultiIndex.end())  // skip multiIndices with duplicates. Since otherwise duplicate points are
  //                                    // interpolated the jacobian is ill-defined
  //      continue;

  for (size_t j = 0; j < fe.size(); ++j) {
    vBlockedLocal[j]  = testNodalPoints1[j];
    vBlockedLocal2[j] = testNodalPoints2[j];
    vBlockedLocal3[j] = testNodalPoints3[j];
  }

  auto f = Dune::StandardLocalFunction(localBasis, vBlockedLocal, geometry);
  t.subTest(testLocalFunction(f));
  static_assert(f.order() == linear);
  static_assert(countNonArithmeticLeafNodes(f) == 1);

  auto g = Dune::StandardLocalFunction(localBasis, vBlockedLocal, geometry);
  static_assert(countNonArithmeticLeafNodes(g) == 1);
  static_assert(g.order() == linear);

  auto h = f + g;
  t.subTest(testLocalFunction(h));
  static_assert(h.order() == linear);
  for (size_t k = 0; k < fe.size(); ++k) {
    t.check(h.node(_0).coefficientsRef()[k] == vBlockedLocal[k],
            "Check if coeffref returns the correct coeffs in slot 0");
    t.check(h.node(_1).coefficientsRef()[k] == vBlockedLocal[k],
            "Check if coeffref returns the correct coeffs in slot 1");
  }
  static_assert(std::tuple_size_v<decltype(collectNonArithmeticLeafNodes(h))> == 2);
  static_assert(countNonArithmeticLeafNodes(h) == 2);
  static_assert(decltype(h)::id[0] == 0 and decltype(h)::id[1] == 0);

  auto ft2 = 2 * f;
  t.subTest(testLocalFunction(ft2));
  static_assert(ft2.order() == linear);

  auto f23 = 2 * f * 3;
  t.subTest(testLocalFunction(f23));
  static_assert(f23.order() == linear);

  auto mf = -f;
  t.subTest(testLocalFunction(mf));
  static_assert(f.order() == mf.order());

  if constexpr (domainDim == worldDim) {
    auto eps = linearStrains(f);
    t.subTest(testLocalFunction(eps));
    static_assert(eps.order() == linear);

    auto gls = greenLagrangeStrains(f);
    t.subTest(testLocalFunction(gls));

    static_assert(gls.order() == quadratic);
  }

  auto k = -dot(f + f, 3.0 * (g / 5.0) * 5.0);
  t.subTest(testLocalFunction(k));
  static_assert(k.order() == quadratic);
  static_assert(std::tuple_size_v<decltype(collectNonArithmeticLeafNodes(k))> == 3);
  static_assert(countNonArithmeticLeafNodes(k) == 3);

  auto dotfg = dot(f, g);
  t.subTest(testLocalFunction(dotfg));
  static_assert(countNonArithmeticLeafNodes(dotfg) == 2);
  static_assert(dotfg.order() == quadratic);
  static_assert(decltype(dotfg)::id[0] == 0 and decltype(dotfg)::id[1] == 0);

  auto normSq = normSquared(f);
  t.subTest(testLocalFunction(normSq));
  static_assert(normSq.order() == quadratic);

  auto logg = log(dotfg);
  t.subTest(testLocalFunction(logg));

  auto powf = pow<3>(dotfg);
  t.subTest(testLocalFunction(powf));

  auto powfgsqrtdotfg = sqrt(powf);
  t.subTest(testLocalFunction(powfgsqrtdotfg));

  if constexpr (size > 1)  // Projection-Based only makes sense in 2d+
  {
    auto gP = Dune::ProjectionBasedLocalFunction(localBasis, vBlockedLocal3, geometry);
    static_assert(gP.order() == nonLinear);
    t.subTest(testLocalFunction(gP));
  }

  using namespace Dune::DerivativeDirections;

  const double tol = 1e-13;

  auto f2 = Dune::StandardLocalFunction(localBasis, vBlockedLocal, geometry, _0);
  auto g2 = Dune::StandardLocalFunction(localBasis, vBlockedLocal2, geometry, _1);
  static_assert(countNonArithmeticLeafNodes(f2) == 1);
  static_assert(countNonArithmeticLeafNodes(g2) == 1);

  auto k2 = dot(f2 + g2, g2);
  static_assert(countNonArithmeticLeafNodes(k2) == 3);
  static_assert(decltype(k2)::id[0] == 0 and decltype(k2)::id[1] == 1 and decltype(k2)::id[2] == 1);

  auto b2 = collectNonArithmeticLeafNodes(k2);
  static_assert(std::tuple_size_v<decltype(b2)> == 3);

  for (int gpIndex = 0; [[maybe_unused]] auto &gp : rule) {
    const auto &N  = localBasis.evaluateFunction(gpIndex);
    const auto &dN = localBasis.evaluateJacobian(gpIndex);
    t.check(Dune::FloatCmp::eq(inner(f2.evaluate(gpIndex, Dune::on(DerivativeDirections::referenceElement))
                                         + g2.evaluate(gpIndex, Dune::on(DerivativeDirections::referenceElement)),
                                     g2.evaluate(gpIndex, Dune::on(DerivativeDirections::referenceElement))),
                               coeff(k2.evaluate(gpIndex, Dune::on(DerivativeDirections::referenceElement)), 0, 0)),
            "Check function value");
    auto df2 = f2.evaluateDerivative(gpIndex, wrt(spatial(0)), Dune::on(DerivativeDirections::referenceElement));
    auto dg2 = g2.evaluateDerivative(gpIndex, wrt(spatial(0)), Dune::on(DerivativeDirections::referenceElement));
    auto g2E = g2.evaluate(gpIndex, Dune::on(DerivativeDirections::referenceElement));
    auto f2E = f2.evaluate(gpIndex, Dune::on(DerivativeDirections::referenceElement));
    auto resSingleSpatial = Dune::eval(inner(df2 + dg2, g2E) + inner(f2E + g2E, dg2));
    t.check(Dune::FloatCmp::eq(
        resSingleSpatial,
        coeff(k2.evaluateDerivative(gpIndex, wrt(spatial(0)), Dune::on(DerivativeDirections::referenceElement)), 0, 0),
        tol));

    auto df2A = f2.evaluateDerivative(gpIndex, wrt(spatialAll), Dune::on(DerivativeDirections::referenceElement));
    auto dg2A = g2.evaluateDerivative(gpIndex, wrt(spatialAll), Dune::on(DerivativeDirections::referenceElement));

    auto resSpatialAll
        = eval(transposeEvaluated(leftMultiplyTranspose(df2A + dg2A, g2E)) + leftMultiplyTranspose(f2E + g2E, dg2A));
    static_assert(Cols<decltype(resSpatialAll)>::value == domainDim);
    static_assert(Rows<decltype(resSpatialAll)>::value == 1);

    t.check(isApproxSame(
        resSpatialAll,
        k2.evaluateDerivative(gpIndex, wrt(spatialAll), Dune::on(DerivativeDirections::referenceElement)), tol));

    for (size_t iC = 0; iC < fe.size(); ++iC) {
      const VectorType dfdi = g2.evaluate(gpIndex, Dune::on(DerivativeDirections::referenceElement)) * N[iC];

      const auto dkdi = Dune::eval(transposeEvaluated(
          k2.evaluateDerivative(gpIndex, wrt(coeff(_0, iC)), Dune::on(DerivativeDirections::referenceElement))));

      t.check(isApproxSame(dfdi, dkdi, tol));

      for (size_t jC = 0; jC < fe.size(); ++jC) {
        const MatrixType dkdij         = k2.evaluateDerivative(gpIndex, wrt(coeff(_0, iC, _1, jC)),
                                                               Dune::on(DerivativeDirections::referenceElement));
        const MatrixType dkdijExpected = createScaledIdentityMatrix<double, worldDim, worldDim>(N[jC] * N[iC]);
        t.check(isApproxSame(dkdijExpected, dkdij, tol));

        const MatrixType dkdij2         = k2.evaluateDerivative(gpIndex, wrt(coeff(_0, iC, _0, jC)),
                                                                Dune::on(DerivativeDirections::referenceElement));
        const MatrixType dkdijExpected2 = createZeroMatrix<double, worldDim, worldDim>();
        t.check(isApproxSame(dkdijExpected2, dkdij2, tol));
        const MatrixType dkdij3         = k2.evaluateDerivative(gpIndex, wrt(coeff(_1, iC, _1, jC)),
                                                                Dune::on(DerivativeDirections::referenceElement));
        const MatrixType dkdijExpected3 = createScaledIdentityMatrix<double, worldDim, worldDim>(2 * N[iC] * N[jC]);
        t.check(isApproxSame(dkdijExpected3, dkdij3, tol));

        const MatrixType dkdSij         = k2.evaluateDerivative(gpIndex, wrt(spatial(0), coeff(_0, iC, _1, jC)),
                                                                Dune::on(DerivativeDirections::referenceElement));
        const MatrixType dkdSijR        = k2.evaluateDerivative(gpIndex, wrt(coeff(_0, iC, _1, jC), spatial(0)),
                                                                Dune::on(DerivativeDirections::referenceElement));
        const MatrixType dkdSijExpected = createScaledIdentityMatrix<double, worldDim, worldDim>(
            coeff(dN, jC, 0) * N[iC] + N[jC] * coeff(dN, iC, 0));
        t.check(isApproxSame(dkdSijR, dkdSij, tol));
        t.check(isApproxSame(dkdSijExpected, dkdSij, tol));
      }
    }
    ++gpIndex;
  }
  //    }
  return t;
}

// Most of the following tests are commented out due to very long compile times and long runtimes in debug mode we hope
//  to still capture most the bugs
using namespace Dune::GeometryTypes;
auto testExpressionsOnLine() {
  TestSuite t("testExpressionsOnLine");
  std::cout << "line with linear ansatz functions and 1d local function" << std::endl;
  localFunctionTestConstructor<1, 1, 1>(line);
  //  localFunctionTestConstructor<1, 2, 1>(line);  // line with linear ansatz functions and 2d lf
  //  std::cout << "line with linear ansatz functions and 3d local function" << std::endl;
  //  localFunctionTestConstructor<1, 3, 1>(line);
  //    std::cout << "line with quadratic ansatz functions and 1d local function" << std::endl;
  //    localFunctionTestConstructor<1, 1, 2>(line);
  //  localFunctionTestConstructor<1, 2, 2>(line);  // line with quadratic ansatz functions and 2d lf
  std::cout << "line with quadratic ansatz functions and 3d local function" << std::endl;
  t.subTest(localFunctionTestConstructor<1, 3, 2>(line));
  return t;
}

auto testExpressionsOnTriangle() {
  TestSuite t("testExpressionsOnTriangle");

  //  std::cout << "triangle with linear ansatz functions and 1d local function" << std::endl;
  //  localFunctionTestConstructor<2, 1, 1>(triangle);
  std::cout << "triangle with linear ansatz functions and 2d local function" << std::endl;
  t.subTest(localFunctionTestConstructor<2, 2, 1>(triangle));
  //  std::cout << "triangle with linear ansatz functions and 3d local function" << std::endl;
  //  localFunctionTestConstructor<2, 3, 1>(triangle);
  //  std::cout << "triangle with quadratic ansatz functions and 1d local function" << std::endl;
  //  localFunctionTestConstructor<2, 1, 2>(triangle);
  //  localFunctionTestConstructor<2, 2, 2>(triangle);  // triangle with quadratic ansatz functions and 2d lf
  std::cout << "triangle with quadratic ansatz functions and 3d local function" << std::endl;
  t.subTest(localFunctionTestConstructor<2, 3, 2>(triangle));
  return t;
}

auto testExpressionsOnQuadrilateral() {
  TestSuite t("testExpressionsOnQuadrilateral");
  //  std::cout << "quadrilateral with linear ansatz functions and 1d local function" << std::endl;
  //  localFunctionTestConstructor<2, 1, 1>(quadrilateral);
  std::cout << "quadrilateral with linear ansatz functions and 2d local function" << std::endl;
  t.subTest(localFunctionTestConstructor<2, 2, 1>(quadrilateral));
  //  std::cout << "quadrilateral with linear ansatz functions and 3d local function" << std::endl;
  //  localFunctionTestConstructor<2, 3, 1>(quadrilateral);
  //  std::cout << "quadrilateral with quadratic ansatz functions and 1d local function" << std::endl;
  //  localFunctionTestConstructor<2, 1, 2>(quadrilateral);
  //  localFunctionTestConstructor<2, 2, 2>(quadrilateral);  // quadrilateral with quadratic ansatz functions and 2d lf
  std::cout << "quadrilateral with quadratic ansatz functions and 3d local function" << std::endl;
  t.subTest(localFunctionTestConstructor<2, 3, 2>(quadrilateral));
  return t;
}

auto testExpressionsOnHexahedron() {
  TestSuite t("testExpressionsOnHexahedron");
  //  std::cout << "hexahedron with linear ansatz functions and 1d local function" << std::endl;
  //  localFunctionTestConstructor<3, 1, 1>(hexahedron);  // hexahedron with linear ansatz functions and 1d lf
  //  localFunctionTestConstructor<3, 2, 1>(hexahedron);  // hexahedron with linear ansatz functions and 2d lf
  std::cout << "hexahedron with linear ansatz functions and 3d local function" << std::endl;
  t.subTest(localFunctionTestConstructor<3, 3, 1>(hexahedron));
  //  std::cout << "hexahedron with quadratic ansatz functions and 1d local function" << std::endl;
  //  localFunctionTestConstructor<3, 1, 2>(hexahedron);
  //  localFunctionTestConstructor<3, 2, 2>(hexahedron);  // hexahedron with quadratic ansatz functions and 2d lf
  //  std::cout << "hexahedron with quadratic ansatz functions and 3d local function" << std::endl;
  //  localFunctionTestConstructor<3, 3, 2>(hexahedron);  // hexahedron with quadratic ansatz functions and 3d lf
  return t;
}

int main(int argc, char **argv) {
  Dune::MPIHelper::instance(argc, argv);
  TestSuite t;

  using namespace std::chrono;
  using namespace std;
  auto start = high_resolution_clock::now();
  t.subTest(testExpressionsOnLine());
  t.subTest(testExpressionsOnTriangle());
  t.subTest(testExpressionsOnQuadrilateral());
  t.subTest(testExpressionsOnHexahedron());
  auto stop     = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "The test execution took: " << duration.count() << endl;
  return t.exit();
}
