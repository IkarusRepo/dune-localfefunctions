// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <config.h>

#include "testexpression.hh"

#include <dune/localfefunctions/expressions.hh>
#include <dune/localfefunctions/manifolds/realTuple.hh>

template <int dim>
using RealT = Dune::RealTuple<double, dim>;

auto testSum() {
  TestSuite t;
  using namespace Dune;
  using namespace Dune::Indices;

  auto expr = [](auto& f, auto& g) { return f + g; };

  auto exprTest = [](auto& h, auto& vBlockedLocal0, auto& vBlockedLocal1, [[maybe_unused]] auto& fe) {
    TestSuite tL("SumExprTests");

    for (size_t k = 0; k < vBlockedLocal0.size(); ++k) {
      tL.check(h.node(_0).coefficientsRef()[k] == vBlockedLocal0[k],
               "Check if coeffref returns the correct coeffs in slot 0")
          << "L" << h.node(_0).coefficientsRef()[k] << "\n R: " << vBlockedLocal0[k];
      tL.check(h.node(_1).coefficientsRef()[k] == vBlockedLocal1[k],
               "Check if coeffref returns the correct coeffs in slot 1")
          << "L" << h.node(_1).coefficientsRef()[k] << "\n R: " << vBlockedLocal1[k];
    }

    static_assert(std::tuple_size_v<decltype(collectNonArithmeticLeafNodes(h))> == 2);
    using HRawType = std::remove_cvref_t<decltype(h)>;
    using FRawType = std::remove_cvref_t<decltype(h.node(_0))>;
    using GRawType = std::remove_cvref_t<decltype(h.node(_1))>;
    static_assert(countNonArithmeticLeafNodes<HRawType>() == 2);
    tL.check(HRawType::id[0] == 0 and HRawType::id[1] == 0);

    static_assert(HRawType::order() == linear);
    static_assert(FRawType::order() == linear);
    static_assert(countNonArithmeticLeafNodes<FRawType>() == 1);
    static_assert(GRawType::order() == linear);
    static_assert(countNonArithmeticLeafNodes<GRawType>() == 1);

    return tL;
  };

  using ManiFoldIDP = ManiFoldTemplateIDPair<RealT, _0>;

  using Expr     = decltype(expr);
  using ExprTest = decltype(exprTest);

  using FunctionConstructor = decltype(doubleStandardLocalFunctionDistinct);
  t.subTest(testExpressionsOnLine<Expr, ExprTest, FunctionConstructor, true, ManiFoldIDP, ManiFoldIDP>(
      expr, exprTest, doubleStandardLocalFunctionDistinct));
  t.subTest(testExpressionsOnTriangle<Expr, ExprTest, FunctionConstructor, true, ManiFoldIDP, ManiFoldIDP>(
      expr, exprTest, doubleStandardLocalFunctionDistinct));
  t.subTest(testExpressionsOnQuadrilateral<Expr, ExprTest, FunctionConstructor, true, ManiFoldIDP, ManiFoldIDP>(
      expr, exprTest, doubleStandardLocalFunctionDistinct));
  t.subTest(testExpressionsOnHexahedron<Expr, ExprTest, FunctionConstructor, true, ManiFoldIDP, ManiFoldIDP>(
      expr, exprTest, doubleStandardLocalFunctionDistinct));
  return t;
}

int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);
  TestSuite t;

  using namespace std::chrono;
  using namespace std;
  auto start = high_resolution_clock::now();
  t.subTest(testSum());
  auto stop     = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "The test execution took: " << duration.count() << endl;
  return t.exit();
}
