// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <config.h>

#include "testexpression.hh"

#include <dune/localfefunctions/expressions.hh>
#include <dune/localfefunctions/manifolds/realTuple.hh>

template <int dim>
using RealT = Dune::RealTuple<double, dim>;

auto testDot() {
  TestSuite t;
  using namespace Dune;
  using namespace Dune::Indices;

  auto expr = [](auto& f, auto& g) { return -dot(f + f, 3.0 * (g / 5.0) * 5.0); };

  auto exprTest = [](auto& h, auto& vBlockedLocal0, auto& vBlockedLocal1, [[maybe_unused]] auto& fe) {
    TestSuite tL("dotExprTests");

    for (size_t k = 0; k < vBlockedLocal0.size(); ++k) {
      tL.check(h.node(_0).coefficientsRef()[k] == vBlockedLocal0[k],
               "Check if coeffref returns the correct coeffs in slot 0")
          << "L" << h.node(_0).coefficientsRef()[k] << "\n R: " << vBlockedLocal0[k];
      tL.check(h.node(_1).coefficientsRef()[k] == vBlockedLocal0[k],
               "Check if coeffref returns the correct coeffs in slot 1")
          << "L" << h.node(_1).coefficientsRef()[k] << "\n R: " << vBlockedLocal0[k];
      tL.check(h.node(_2).coefficientsRef()[k] == vBlockedLocal1[k],
               "Check if coeffref returns the correct coeffs in slot 2")
          << "L" << h.node(_2).coefficientsRef()[k] << "\n R: " << vBlockedLocal1[k];
    }

    static_assert(std::tuple_size_v<decltype(collectNonArithmeticLeafNodes(h))> == 3);

    using HRawType  = std::remove_cvref_t<decltype(h)>;
    using F1RawType = std::remove_cvref_t<decltype(h.node(_0))>;
    using F2RawType = std::remove_cvref_t<decltype(h.node(_1))>;
    using GRawType  = std::remove_cvref_t<decltype(h.node(_2))>;
    static_assert(countNonArithmeticLeafNodes<HRawType>() == 3);
    static_assert(HRawType::id[0] == 0 and HRawType::id[1] == 0 and HRawType::id[2] == arithmetic
                  and HRawType::id[3] == 0);

    static_assert(HRawType::order() == quadratic);
    static_assert(countNonArithmeticLeafNodes<F1RawType>() == 1);
    static_assert(countNonArithmeticLeafNodes<F2RawType>() == 1);
    static_assert(F1RawType::order() == linear);
    static_assert(F2RawType::order() == linear);
    static_assert(GRawType::order() == linear);
    static_assert(countNonArithmeticLeafNodes<GRawType>() == 1);

    return tL;
  };

  using ManiFoldIDP = ManiFoldTemplateIDPair<RealT, _0>;

  using Expr     = decltype(expr);
  using ExprTest = decltype(exprTest);
  using FC       = decltype(doubleStandardLocalFunctionDistinct);
  t.subTest(testExpressionsOnLine<Expr, ExprTest, FC, true, ManiFoldIDP, ManiFoldIDP>(
      expr, exprTest, doubleStandardLocalFunctionDistinct));
  t.subTest(testExpressionsOnTriangle<Expr, ExprTest, FC, true, ManiFoldIDP, ManiFoldIDP>(
      expr, exprTest, doubleStandardLocalFunctionDistinct));
  t.subTest(testExpressionsOnQuadrilateral<Expr, ExprTest, FC, true, ManiFoldIDP, ManiFoldIDP>(
      expr, exprTest, doubleStandardLocalFunctionDistinct));
  t.subTest(testExpressionsOnHexahedron<Expr, ExprTest, FC, true, ManiFoldIDP, ManiFoldIDP>(
      expr, exprTest, doubleStandardLocalFunctionDistinct));
  return t;
}

int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);
  TestSuite t;

  using namespace std::chrono;
  using namespace std;
  auto start = high_resolution_clock::now();
  t.subTest(testDot());
  auto stop     = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "The test execution took: " << duration.count() << endl;
  return t.exit();
}
