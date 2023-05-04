// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <config.h>

#include "testexpression.hh"

#include <dune/localfefunctions/expressions.hh>
#include <dune/localfefunctions/manifolds/realTuple.hh>

template <int dim>
using RealT = Dune::RealTuple<double, dim>;

auto testScaleExpr() {
  TestSuite t("testScaleExpr");
  using namespace Dune;
  using namespace Dune::Indices;

  auto expr = [](auto& f) { return 2 * f; };

  auto expr2 = [](auto& f) { return 2 * f * 3; };

  auto exprTest = [](auto& h, auto&, [[maybe_unused]] auto& fe) {
    TestSuite tL("ScaleExprSpecialTests");

    static_assert(std::tuple_size_v<decltype(collectNonArithmeticLeafNodes(h))> == 1);
    using HRawType = std::remove_cvref_t<decltype(h)>;
    static_assert(countNonArithmeticLeafNodes<HRawType>() == 1);
    static_assert(HRawType::id[0] == arithmetic);
    static_assert(HRawType::id[1] == 0);

    static_assert(HRawType::order() == linear);

    return tL;
  };

  using ManiFoldIDP = ManiFoldTemplateIDPair<RealT, _0>;

  using Expr     = decltype(expr);
  using ExprTest = decltype(exprTest);
  using FC       = decltype(singleStandardLocalFunction);

  t.subTest(testExpressionsOnLine<Expr, ExprTest, FC, true, ManiFoldIDP>(expr, exprTest, singleStandardLocalFunction));
  t.subTest(
      testExpressionsOnTriangle<Expr, ExprTest, FC, true, ManiFoldIDP>(expr, exprTest, singleStandardLocalFunction));
  t.subTest(testExpressionsOnQuadrilateral<Expr, ExprTest, FC, true, ManiFoldIDP>(expr, exprTest,
                                                                                  singleStandardLocalFunction));
  t.subTest(
      testExpressionsOnHexahedron<Expr, ExprTest, FC, true, ManiFoldIDP>(expr, exprTest, singleStandardLocalFunction));

  using Expr2 = decltype(expr2);
  t.subTest(
      testExpressionsOnLine<Expr2, ExprTest, FC, true, ManiFoldIDP>(expr2, exprTest, singleStandardLocalFunction));
  t.subTest(
      testExpressionsOnTriangle<Expr2, ExprTest, FC, true, ManiFoldIDP>(expr2, exprTest, singleStandardLocalFunction));
  t.subTest(testExpressionsOnQuadrilateral<Expr2, ExprTest, FC, true, ManiFoldIDP>(expr2, exprTest,
                                                                                   singleStandardLocalFunction));
  t.subTest(testExpressionsOnHexahedron<Expr2, ExprTest, FC, true, ManiFoldIDP>(expr2, exprTest,
                                                                                singleStandardLocalFunction));
  return t;
}

int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);
  TestSuite t;

  using namespace std::chrono;
  using namespace std;
  auto start = high_resolution_clock::now();
  t.subTest(testScaleExpr());

  auto stop     = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "The test execution took: " << duration.count() << endl;
  return t.exit();
}
