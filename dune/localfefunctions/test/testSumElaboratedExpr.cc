// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <config.h>

#include "testexpression.hh"

#include <dune/localfefunctions/expressions.hh>
#include <dune/localfefunctions/manifolds/realTuple.hh>

using namespace Dune::GeometryTypes;

template <int dim>
using RealT = Dune::RealTuple<double, dim>;

auto testElaboratedSum() {
  TestSuite t;
  using namespace Dune;
  using namespace Dune::Indices;

  auto expr = [](auto& f, auto& g) { return dot(f + g, g); };

  auto exprTest = [](auto& h, auto& vBlockedLocal0, auto& vBlockedLocal1, [[maybe_unused]] auto& fe) {
    TestSuite tL("ElaboratedSumExprTests");
    using HRawType                = std::remove_cvref_t<decltype(h)>;
    static constexpr int worldDim = HRawType::worldDimension;

    for (size_t k = 0; k < vBlockedLocal0.size(); ++k) {
      tL.check(h.node(_0).coefficientsRef()[k] == vBlockedLocal0[k],
               "Check if coeffref returns the correct coeffs in slot 0")
          << "L" << h.node(_0).coefficientsRef()[k] << "\n R: " << vBlockedLocal0[k];
      tL.check(h.node(_1).coefficientsRef()[k] == vBlockedLocal1[k],
               "Check if coeffref returns the correct coeffs in slot 1")
          << "L" << h.node(_1).coefficientsRef()[k] << "\n R: " << vBlockedLocal1[k];
    }

    static_assert(std::tuple_size_v<decltype(collectNonArithmeticLeafNodes(h))> == 3);

    using FRawType = std::remove_cvref_t<decltype(h.node(_0))>;
    using GRawType = std::remove_cvref_t<decltype(h.node(_1))>;
    static_assert(countNonArithmeticLeafNodes<HRawType>() == 3);
    tL.check(HRawType::id[0] == 0 and HRawType::id[1] == 1 and HRawType::id[2] == 1);

    static_assert(HRawType::order(_0) == linear);
    static_assert(HRawType::order(_1) == linear);
    static_assert(HRawType::order(_2) == constant);
    static_assert(FRawType::order(_0) == linear);
    static_assert(FRawType::order(_1) == constant);
    static_assert(countNonArithmeticLeafNodes<FRawType>() == 1);
    static_assert(GRawType::order(_0) == constant);
    static_assert(GRawType::order() == constant);
    static_assert(countNonArithmeticLeafNodes<GRawType>() == 1);

    using VectorType = DefaultLinearAlgebra::FixedSizedVector<double, worldDim>;
    using MatrixType = DefaultLinearAlgebra::FixedSizedMatrix<double, worldDim, worldDim>;

    using namespace Dune::DerivativeDirections;
    const double tol = 1e-13;
    auto& localBasis = h.node(_0).basis();
    auto rule        = h.viewOverIntegrationPoints();
    const auto& f    = h.node(_0);
    const auto& g    = h.node(_1);

    for (auto [gpIndex, gp] : rule) {
      const auto& N  = localBasis.evaluateFunction(gpIndex);
      const auto& dN = localBasis.evaluateJacobian(gpIndex);
      auto f2E       = f.evaluate(gpIndex, Dune::on(DerivativeDirections::referenceElement));
      auto g2E       = g.evaluate(gpIndex, Dune::on(DerivativeDirections::referenceElement));
      auto hE        = h.evaluate(gpIndex, Dune::on(DerivativeDirections::referenceElement));
      tL.check(Dune::FloatCmp::eq(inner(f2E + g2E, g2E), coeff(hE, 0, 0)), "Check function value") << "f2E:\n"
                                                                                                   << f2E << "\ng2E:\n"
                                                                                                   << g2E << "\nhE:\n"
                                                                                                   << hE;
      auto df2 = f.evaluateDerivative(gpIndex, wrt(spatial(0)), Dune::on(DerivativeDirections::referenceElement));
      auto dg2 = g.evaluateDerivative(gpIndex, wrt(spatial(0)), Dune::on(DerivativeDirections::referenceElement));
      auto resSingleSpatial = Dune::eval(inner(df2 + dg2, g2E) + inner(f2E + g2E, dg2));
      tL.check(Dune::FloatCmp::eq(
          resSingleSpatial,
          coeff(h.evaluateDerivative(gpIndex, wrt(spatial(0)), Dune::on(DerivativeDirections::referenceElement)), 0, 0),
          tol));

      auto df2A = f.evaluateDerivative(gpIndex, wrt(spatialAll), Dune::on(DerivativeDirections::referenceElement));
      auto dg2A = g.evaluateDerivative(gpIndex, wrt(spatialAll), Dune::on(DerivativeDirections::referenceElement));

      auto resSpatialAll
          = eval(transposeEvaluated(leftMultiplyTranspose(df2A + dg2A, g2E)) + leftMultiplyTranspose(f2E + g2E, dg2A));
      //      static_assert(Cols<decltype(resSpatialAll)>::value == domainDim);
      static_assert(Rows<decltype(resSpatialAll)>::value == 1);

      tL.check(isApproxSame(
          resSpatialAll,
          h.evaluateDerivative(gpIndex, wrt(spatialAll), Dune::on(DerivativeDirections::referenceElement)), tol));

      for (size_t iC = 0; iC < fe.size(); ++iC) {
        const VectorType dfdi = g.evaluate(gpIndex, Dune::on(DerivativeDirections::referenceElement)) * N[iC];

        const auto dkdi = Dune::eval(transposeEvaluated(
            h.evaluateDerivative(gpIndex, wrt(coeff(_0, iC)), Dune::on(DerivativeDirections::referenceElement))));

        tL.check(isApproxSame(dfdi, dkdi, tol));

        for (size_t jC = 0; jC < fe.size(); ++jC) {
          const MatrixType dkdij         = h.evaluateDerivative(gpIndex, wrt(coeff(_0, iC, _1, jC)),
                                                                Dune::on(DerivativeDirections::referenceElement));
          const MatrixType dkdijExpected = createScaledIdentityMatrix<double, worldDim, worldDim>(N[jC] * N[iC]);
          tL.check(isApproxSame(dkdijExpected, dkdij, tol));

          const MatrixType dkdij2         = h.evaluateDerivative(gpIndex, wrt(coeff(_0, iC, _0, jC)),
                                                                 Dune::on(DerivativeDirections::referenceElement));
          const MatrixType dkdijExpected2 = createZeroMatrix<double, worldDim, worldDim>();
          tL.check(isApproxSame(dkdijExpected2, dkdij2, tol));
          const MatrixType dkdij3         = h.evaluateDerivative(gpIndex, wrt(coeff(_1, iC, _1, jC)),
                                                                 Dune::on(DerivativeDirections::referenceElement));
          const MatrixType dkdijExpected3 = createScaledIdentityMatrix<double, worldDim, worldDim>(2 * N[iC] * N[jC]);
          tL.check(isApproxSame(dkdijExpected3, dkdij3, tol));

          const MatrixType dkdSij         = h.evaluateDerivative(gpIndex, wrt(spatial(0), coeff(_0, iC, _1, jC)),
                                                                 Dune::on(DerivativeDirections::referenceElement));
          const MatrixType dkdSijR        = h.evaluateDerivative(gpIndex, wrt(coeff(_0, iC, _1, jC), spatial(0)),
                                                                 Dune::on(DerivativeDirections::referenceElement));
          const MatrixType dkdSijExpected = createScaledIdentityMatrix<double, worldDim, worldDim>(
              coeff(dN, jC, 0) * N[iC] + N[jC] * coeff(dN, iC, 0));
          tL.check(isApproxSame(dkdSijR, dkdSij, tol));
          tL.check(isApproxSame(dkdSijExpected, dkdSij, tol));
        }
      }
    }

    return tL;
  };

  using ManiFoldIDP  = ManiFoldTemplateIDPair<RealT, _0>;
  using ManiFoldIDP2 = ManiFoldTemplateIDPair<RealT, _1>;

  using Expr     = decltype(expr);
  using ExprTest = decltype(exprTest);

  using FunctionConstructor = decltype(doubleStandardLocalFunctionDistinct);
  t.subTest(testExpressionsOnLine<Expr, ExprTest, FunctionConstructor, false, ManiFoldIDP, ManiFoldIDP2>(
      expr, exprTest, doubleStandardLocalFunctionDistinct));
  t.subTest(testExpressionsOnTriangle<Expr, ExprTest, FunctionConstructor, false, ManiFoldIDP, ManiFoldIDP2>(
      expr, exprTest, doubleStandardLocalFunctionDistinct));
  t.subTest(testExpressionsOnQuadrilateral<Expr, ExprTest, FunctionConstructor, false, ManiFoldIDP, ManiFoldIDP2>(
      expr, exprTest, doubleStandardLocalFunctionDistinct));
  t.subTest(testExpressionsOnHexahedron<Expr, ExprTest, FunctionConstructor, false, ManiFoldIDP, ManiFoldIDP2>(
      expr, exprTest, doubleStandardLocalFunctionDistinct));
  return t;
}

int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);
  TestSuite t;

  using namespace std::chrono;
  using namespace std;
  auto start = high_resolution_clock::now();
  t.subTest(testElaboratedSum());
  auto stop     = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "The test execution took: " << duration.count() << endl;
  return t.exit();
}
