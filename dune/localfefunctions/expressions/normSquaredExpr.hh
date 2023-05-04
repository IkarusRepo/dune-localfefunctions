// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include "rebind.hh"

#include <dune/localfefunctions/expressions/unaryExpr.hh>
//#include <ikarus/manifolds/realTuple.hh>
//#include <ikarus/utils/linearAlgebraHelper.hh>
namespace Dune {

  template <typename E1>
  class NormSquaredExpr : public UnaryExpr<NormSquaredExpr, E1> {
  public:
    using Base = UnaryExpr<NormSquaredExpr, E1>;
    using Base::Base;
    using Traits        = LocalFunctionTraits<NormSquaredExpr>;
    using LinearAlgebra = typename Base::E1Raw::LinearAlgebra;
    /** \brief Type used for coordinates */
    using ctype                    = typename Traits::ctype;
    static constexpr int valueSize = 1;
    static constexpr int gridDim   = Traits::gridDim;

    template <size_t ID_ = 0>
    static constexpr int orderID = std::min(2 * Base::E1Raw::template order<ID_>(), nonlinear);

    template <typename LFArgs>
    auto evaluateValueOfExpression(const LFArgs &lfArgs) const {
      const auto m = evaluateFunctionImpl(this->m(), lfArgs);
      return typename DefaultLinearAlgebra::template FixedSizedMatrix<ctype, 1, 1>(two_norm2(m));
    }

    template <int DerivativeOrder, typename LFArgs>
    auto evaluateDerivativeOfExpression(const LFArgs &lfArgs) const {
      const auto u = evaluateFunctionImpl(this->m(), lfArgs);
      if constexpr (DerivativeOrder == 1)  // d(squaredNorm(u))/dx = 2 * u_x * u
      {
        const auto u_x = evaluateDerivativeImpl(this->m(), lfArgs);
        auto res       = Dune::eval(leftMultiplyTranspose(u, u_x));
        res *= 2;
        return res;
      } else if constexpr (DerivativeOrder == 2) {  // dd(squaredNorm(u))/(dxdy) =  2 *u_{x,y} * u + 2*u_x*u_y
        const auto &[u_x, u_y] = evaluateFirstOrderDerivativesImpl(this->m(), lfArgs);
        if constexpr (LFArgs::hasNoSpatial and LFArgs::hasTwoCoeff) {
          const auto alonguArgs = replaceAlong(lfArgs, along(u));
          const auto u_xyAlongu = eval(evaluateDerivativeImpl(this->m(), alonguArgs));

          auto res = u_xyAlongu;
          res += leftMultiplyTranspose(u_x, u_y);
          res *= 2;
          return res;
        } else if constexpr (LFArgs::hasOneSpatial and LFArgs::hasSingleCoeff) {
          const auto u_xy = evaluateDerivativeImpl(this->m(), lfArgs);
          if constexpr (LFArgs::hasOneSpatialSingle and LFArgs::hasSingleCoeff) {
            return Dune::eval(2 * (leftMultiplyTranspose(u, u_xy) + leftMultiplyTranspose(u_x, u_y)));
          } else if constexpr (LFArgs::hasOneSpatialAll and LFArgs::hasSingleCoeff) {
            std::array<std::remove_cvref_t<decltype(Dune::eval(leftMultiplyTranspose(u, u_xy[0])))>, gridDim> res;
            for (int i = 0; i < gridDim; ++i)
              res[i] = 2 * (leftMultiplyTranspose(u, u_xy[i]) + leftMultiplyTranspose(col(u_x, i), u_y));
            return res;
          }
        }
      } else if constexpr (DerivativeOrder == 3) {  // dd(squaredNorm(u))/(dxdydz) = 2*( u_{x,y,z} * v + u_{x,y} * v_z +
                                                    // u_{x,z}*v_y + u_x*v_{y,z})
        if constexpr (LFArgs::hasOneSpatialSingle) {
          const auto argsForDyz = lfArgs.extractSecondWrtArgOrFirstNonSpatial();

          const auto &[u_x, u_y, u_z] = evaluateFirstOrderDerivativesImpl(this->m(), lfArgs);
          const auto &[u_xy, u_xz]    = evaluateSecondOrderDerivativesImpl(this->m(), lfArgs);

          const auto alonguArgs             = replaceAlong(lfArgs, along(u));
          const auto argsForDyzalongu_xArgs = replaceAlong(argsForDyz, along(u_x));

          const auto u_xyzAlongu = evaluateDerivativeImpl(this->m(), alonguArgs);
          const auto u_yzAlongux = evaluateDerivativeImpl(this->m(), argsForDyzalongu_xArgs);

          return Dune::eval(
              2 * (u_xyzAlongu + leftMultiplyTranspose(u_xy, u_z) + leftMultiplyTranspose(u_xz, u_y) + u_yzAlongux));
        } else if constexpr (LFArgs::hasOneSpatialAll) {
          // check that the along argument has the correct size
          const auto &alongMatrix = std::get<0>(lfArgs.alongArgs.args);

          //          static_assert(cols(alongMatrix) == gridDim);
          //          static_assert(rows(alongMatrix) == 1);

          const typename DefaultLinearAlgebra::template FixedSizedMatrix<ctype, Base::E1Raw::valueSize, gridDim> uTimesA
              = u * alongMatrix;
          static_assert(Rows<decltype(uTimesA)>::value == Base::E1Raw::valueSize);
          static_assert(Cols<decltype(uTimesA)>::value == gridDim);

          const auto &[gradu, u_c0, u_c1]  = evaluateFirstOrderDerivativesImpl(this->m(), lfArgs);
          const auto &[gradu_c0, gradu_c1] = evaluateSecondOrderDerivativesImpl(this->m(), lfArgs);

          const auto graduTimesA = Dune::eval(gradu * transpose(alongMatrix));
          //          using graduTimesAType  = std::remove_cvref_t<decltype(graduTimesA)>;

          static_assert(Rows<decltype(graduTimesA)>::value == Base::E1Raw::valueSize);
          static_assert(Cols<decltype(graduTimesA)>::value == 1);

          const auto argsForDyz = lfArgs.extractSecondWrtArgOrFirstNonSpatial();

          const auto alonguAArgs          = replaceAlong(lfArgs, along(uTimesA));
          const auto alonggraduTimesAArgs = replaceAlong(argsForDyz, along(graduTimesA));

          const auto u_xyzAlongu            = evaluateDerivativeImpl(this->m(), alonguAArgs);
          const auto u_c0c1AlongGraduTimesA = evaluateDerivativeImpl(this->m(), alonggraduTimesAArgs);
          std::remove_cvref_t<decltype(eval(u_xyzAlongu))> res;

          res = u_xyzAlongu + u_c0c1AlongGraduTimesA;
          for (int i = 0; i < gridDim; ++i)
            res += (leftMultiplyTranspose(u_c1, gradu_c0[i]) + leftMultiplyTranspose(u_c0, gradu_c1[i]))
                   * coeff(alongMatrix, 0, i);

          res *= 2;
          return res;
        } else
          static_assert(
              LFArgs::hasOneSpatialSingle or LFArgs::hasOneSpatialAll,
              "Only a spatial single direction or all spatial directions are supported. You should not end up here.");
      } else
        static_assert(DerivativeOrder > 3 or DerivativeOrder < 1,
                      "Only first, second and third order derivatives are supported.");
    }
  };

  template <typename E1>
  struct LocalFunctionTraits<NormSquaredExpr<E1>> {
    using E1Raw = std::remove_cvref_t<E1>;
    /** \brief Size of the function value */
    static constexpr int valueSize = 1;
    /** \brief Type for the points for evaluation, usually the integration points */
    using DomainType = std::common_type_t<typename E1Raw::DomainType>;
    /** \brief Type used for coordinates */
    using ctype = std::common_type_t<typename E1Raw::ctype>;
    /** \brief Dimension of the grid */
    static constexpr int gridDim = E1Raw::gridDim;
    /** \brief Dimension of the world where this function is mapped to from the reference element */
    static constexpr int worldDimension = E1Raw::worldDimension;
  };

  template <typename E1>
  requires IsLocalFunction<E1>
  constexpr auto normSquared(E1 &&u) { return NormSquaredExpr<E1>(std::forward<E1>(u)); }

}  // namespace Dune
