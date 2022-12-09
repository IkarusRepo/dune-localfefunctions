// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include <functional>
#include <string>

#include "dune/localfefunctions/linearAlgebraHelper.hh"
#include <dune/common/float_cmp.hh>
#include <dune/functions/analyticfunctions/polynomial.hh>

#include <Eigen/Dense>

namespace Testing {

  template <typename Fun, typename Arg>
  using ReturnType = std::invoke_result_t<Fun, Arg>;
  /*
   * Simplified version of
   * https://github.com/ikarus-project/ikarus/blob/main/src/include/ikarus/linearAlgebra/nonLinearOperator.hh
   *
   * NonLinearOperator is a class taking linear algebra function and their arguments.
   * The fcuntion are assumed to be derivatvies of each other. */
  template <typename F, typename DF, typename Arg>
  class NonLinearOperator {
  public:
    using ValueResult      = ReturnType<F, Arg&>;
    using DerivativeResult = ReturnType<DF, Arg&>;
    using ParameterValue   = std::remove_cvref_t<Arg>;

    explicit NonLinearOperator(const F& f_, const DF& df_, const Arg& arg_) : f{f_}, df{df_}, arg{arg_} {}

    /* Evaluates all functions. Usually called if the parameters changes */
    void updateAll() {
      derivativesEvaluated_.first  = f(arg);
      derivativesEvaluated_.second = df(arg);
    }

    /* Evaluates the n-th function */
    template <int n>
    requires(n < 2) void update() {
      if constexpr (n == 0)
        derivativesEvaluated_.first = f(arg);
      else
        derivativesEvaluated_.second = df(arg);
    }

    /* Returns the value of the zeros function, e.g. the energy value as reference */
    auto& value() { return derivativesEvaluated_.first; }
    /* Returns the derivative value, e.g. the gradient of an energy */
    auto& derivative() { return derivativesEvaluated_.second; }
    /* Returns the first parameter value */
    auto& firstParameter() { return arg; }

  private:
    F f;
    DF df;
    Arg arg;

    std::pair<ValueResult, DerivativeResult> derivativesEvaluated_;
  };

  std::tuple<Dune::Functions::Polynomial<double>, double> polyfit(const Eigen::Ref<const Eigen::VectorXd>& x,
                                                                  const Eigen::Ref<const Eigen::VectorXd>& y, int deg);

  /*
   * This function is inspired from
   * https://github.com/NicolasBoumal/manopt/blob/master/manopt/tools/identify_linear_piece.m
   */
  std::tuple<Dune::Functions::Polynomial<double>, decltype(Eigen::seq(0, 0))> findLineSegment(const Eigen::VectorXd& x,
                                                                                              const Eigen::VectorXd& y,
                                                                                              int segmentSize);

  double getSlope(const std::function<double(double)>& ftfunc, int slopeOfReference);

  /*
   * Simplified version of
   * https://github.com/ikarus-project/ikarus/blob/main/src/include/ikarus/utils/functionSanityChecks.hh The
   * checkjacobian function is inspired by http://sma.epfl.ch/~nboumal/book/  Chapter 4.8 and
   * https://github.com/NicolasBoumal/manopt/blob/master/manopt/tools/checkdiff.m
   */
  template <typename NonlinearOperator>
  bool checkJacobian(NonlinearOperator& nonLinOp, double tol) {
    nonLinOp.template updateAll();
    auto& x          = nonLinOp.firstParameter();
    const auto xOld  = x;
    using UpdateType = typename NonlinearOperator::ParameterValue;
    typename NonlinearOperator::ParameterValue b;
    if constexpr (Dune::Rows<UpdateType>::value != 1) {
      b.resizeLike(nonLinOp.derivative().row(0).transpose());
      b.setRandom();
      b /= b.norm();
    } else
      b(0, 0) = 1;

    nonLinOp.updateAll();
    const auto e = nonLinOp.value();

    const auto jacofv = (nonLinOp.derivative() * b).eval();

    auto ftfunc = [&](auto t) {
      x += t * b;
      nonLinOp.template update<0>();
      auto value = (nonLinOp.value() - e - t * jacofv).norm();
      x          = xOld;
      return value;
    };

    const double slope = getSlope(ftfunc, 2);

    const bool checkPassed = Dune::FloatCmp::le(2.0, slope, tol);

    nonLinOp.template updateAll();
    return checkPassed;
  }

}  // namespace Testing
