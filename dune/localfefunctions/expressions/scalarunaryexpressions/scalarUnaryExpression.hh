/*
 * This file is part of the Ikarus distribution (https://github.com/ikarus-project/ikarus).
 * Copyright (c) 2022. The Ikarus developers.
 *
 * This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 */

#pragma once
#include <dune/localfefunctions/expressions/unaryExpr.hh>

namespace Dune {

  template <typename E1, typename Func>
  class ScalarUnaryExpr : public UnaryExpr<ScalarUnaryExpr, E1, Func> {
  public:
    using Base = UnaryExpr<ScalarUnaryExpr, E1, Func>;
    using Base::Base;
    using Traits                   = LocalFunctionTraits<ScalarUnaryExpr>;
    static constexpr int valueSize = 1;
    static constexpr int gridDim   = Traits::gridDim;
    using ctype                    = typename Traits::ctype;
    using LinearAlgebra = typename Base::E1Raw::LinearAlgebra;

    using E1Raw = std::remove_cvref_t<E1>;

    template <size_t ID_ = 0>
    static constexpr int orderID = nonLinear;

    template <typename LFArgs>
    auto evaluateValueOfExpression(const LFArgs& lfArgs) const {
      return typename LinearAlgebra::template FixedSizedMatrix<ctype, 1, 1>(Func::value(coeff(evaluateFunctionImpl(this->m(), lfArgs),0,0)));
    }

    template <int DerivativeOrder, typename LFArgs>
    auto evaluateDerivativeOfExpression(const LFArgs& lfArgs) const {
      const auto u = coeff(evaluateFunctionImpl(this->m(), lfArgs),0,0);
      if constexpr (DerivativeOrder == 1)  // d(f(u(x)))/(dx) =  u_x *D (f(u(x))
      {
        const auto u_x = evaluateDerivativeImpl(this->m(), lfArgs);
        return Dune::eval(u_x * Func::derivative(u));
      } else if constexpr (DerivativeOrder == 2) {  // d^2(f(u(x,y)))/(dxdy) = - u_x*u_y* D^2 (f(u(x,y))+
                                                    // u_xy* D(f(u(x,y))
        const auto& [u_x, u_y]    = evaluateFirstOrderDerivativesImpl(this->m(), lfArgs);
        const auto u_xy           = evaluateDerivativeImpl(this->m(), lfArgs);
        const auto u_yTimesfactor = Dune::eval(u_y * Func::secondDerivative(u));
        if constexpr (LFArgs::hasOneSpatialAll and LFArgs::hasSingleCoeff) {
          std::array<std::remove_cvref_t<decltype(Dune::eval(coeff(u_x, 0,0) * u_y))>, gridDim> res;
          for (int i = 0; i < gridDim; ++i)
            res[i] = Dune::eval(col(u_x, i)[0] * u_yTimesfactor + u_xy[i] * Func::derivative(u));
          return res;
        } else {  // one spatial and one coeff derivative
          return Dune::eval(leftMultiplyTranspose(u_x, u_yTimesfactor) + u_xy * Func::derivative(u));
        }
      } else if constexpr (DerivativeOrder == 3) {  // d^3(f(u(x,y,z)))/(dxdydz) =(u_x*u_y*u_z)* D^3 (f(u(x,y,z)) +
                                                    // u_xz*u_y* D^2 (f(u(x,y,z)) + u_x*u_yz *D^2 (f(u(x,y,z)) +
                                                    // u_xy*u_z*D^2 (f(u(x,y,z))+ u_xyz*D (f(u(x,y,z))
        const auto& [u_x, u_y, u_z] = evaluateFirstOrderDerivativesImpl(this->m(), lfArgs);
        const auto& [u_xy, u_xz]    = evaluateSecondOrderDerivativesImpl(this->m(), lfArgs);

        const auto u_xyz = evaluateDerivativeImpl(this->m(), lfArgs);

        const auto argsForDyz = lfArgs.extractSecondWrtArgOrFirstNonSpatial();
        const auto u_yz       = evaluateDerivativeImpl(this->m(), argsForDyz);

        if constexpr (LFArgs::hasOneSpatialSingle) {
          static_assert(Cols<decltype(u_x)>::value == 1);
          static_assert(Rows<decltype(u_x)>::value == 1);

          return eval(u_xyz * Func::derivative(u)
                      + (coeff(u_x,0,0) * leftMultiplyTranspose(u_y, u_z)) * Func::thirdDerivative(u)
                      + ((leftMultiplyTranspose(u_y, u_xz) + coeff(u_x,0,0) * transposeEvaluated(u_yz)
                          + leftMultiplyTranspose(u_xy, u_z)))
                            * Func::secondDerivative(u));
        } else if constexpr (LFArgs::hasOneSpatialAll) {
          const auto& alongMatrix = std::get<0>(lfArgs.alongArgs.args);
          std::remove_cvref_t<decltype(u_xyz)> res;
          res = u_xyz * Func::derivative(u);

          for (int i = 0; i < gridDim; ++i)
            res += coeff(alongMatrix, 0, i)
                   * (coeff(u_x,0,i) * leftMultiplyTranspose(u_y, u_z) * Func::thirdDerivative(u)
                      + (leftMultiplyTranspose(u_y, u_xz[i]) + coeff(u_x,0,i) * transposeEvaluated(u_yz)
                         + leftMultiplyTranspose(u_xy[i], u_z))
                            * Func::secondDerivative(u));
          return res;
        }
      }
    }
  };

  template <typename E1, typename Func>
  struct LocalFunctionTraits<ScalarUnaryExpr<E1, Func>> : public LocalFunctionTraits<std::remove_cvref_t<E1>> {};

}  // namespace Dune
