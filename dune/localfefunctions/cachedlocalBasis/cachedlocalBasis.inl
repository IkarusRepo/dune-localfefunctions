

/*
 * This file is part of the Ikarus distribution (https://github.com/IkarusRepo/Ikarus).
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

namespace Dune {

  template <Concepts::LocalBasis DuneLocalBasis>
  void CachedLocalBasis<DuneLocalBasis>::evaluateFunction(const DomainType& local, Dune::BlockVector<RangeFieldType>& N) const {
    duneLocalBasis->evaluateFunction(local, Ndune);
    N.resize(Ndune.size(), 1);
    N.setZero();
    for (size_t i = 0; i < Ndune.size(); ++i)
      N[i] = Ndune[i][0];
  }

  template <Concepts::LocalBasis DuneLocalBasis>
  void CachedLocalBasis<DuneLocalBasis>::evaluateJacobian(const DomainType& local, Dune::BlockVector<JacobianType>& dN) const {
    duneLocalBasis->evaluateJacobian(local, dNdune);
    dN.resize(dNdune.size());

    for (auto i = 0U; i < dNdune.size(); ++i)
      for (int j = 0; j < gridDim; ++j)
        dN[i][j] = dNdune[i][0][j];
  }

  template <Concepts::LocalBasis DuneLocalBasis>
  const Dune::QuadraturePoint<typename CachedLocalBasis<DuneLocalBasis>::DomainFieldType, CachedLocalBasis<DuneLocalBasis>::gridDim>& CachedLocalBasis<DuneLocalBasis>::indexToIntegrationPoint(int i) const
  {
    if(isBound())
      return rule.value()[i];
    else
      assert(false && "You need to call bind first");
  }

  /*
   * This function returns the second derivatives of the ansatz functions.
   * The assumed order is in Voigt notation, e.g. for 3d ansatzfunctions N_xx,N_yy,N_zz,N_yz,N_xz, N_xy
   */
  template <Concepts::LocalBasis DuneLocalBasis>
  void CachedLocalBasis<DuneLocalBasis>::evaluateSecondDerivatives(const DomainType& local, Dune::BlockVector<SecondDerivativeType>& ddN) const {
    std::array<unsigned int, gridDim> order;
    std::ranges::fill(order, 0);

    for (int i = 0; i < gridDim; ++i) { //Diagonal terms
      order[i] = 2;
      duneLocalBasis->partial(order,local, ddNdune);
      for (size_t j = 0; j < ddNdune.size(); ++j)
        ddN[j][i]=ddNdune[j][0];

      order[i] = 0;
    }

    std::ranges::fill(order, 1);
    for (int i = 0; i < gridDim*(gridDim-1)/2; ++i) { //off-diagonal terms
      if constexpr (gridDim>2)
        order[i] = 0;
      duneLocalBasis->partial(order,local, ddNdune);
      for (size_t j = 0; j < ddNdune.size(); ++j)
        ddN[j][i+gridDim]=ddNdune[j][0];
      order[i] = 1;
    }

  }

  template <Concepts::LocalBasis DuneLocalBasis>
  template <typename Derived1, typename Derived2>
  void CachedLocalBasis<DuneLocalBasis>::evaluateFunctionAndJacobian(const DomainType& local, Eigen::PlainObjectBase<Derived1>& N,
                                   Eigen::PlainObjectBase<Derived2>& dN) const {
    evaluateFunction(local, N);
    evaluateJacobian(local, dN);
  }

  template <Concepts::LocalBasis DuneLocalBasis>
  void CachedLocalBasis<DuneLocalBasis>::bind(const Dune::QuadratureRule<DomainFieldType, gridDim>& p_rule, std::set<int>&& ints) {
    rule             = p_rule;
    boundDerivatives = ints;
    Nbound           = std::make_optional<typename decltype(Nbound)::value_type> ();
    dNbound           = std::make_optional<typename decltype(dNbound)::value_type> ();
    ddNbound           = std::make_optional<typename decltype(ddNbound)::value_type> ();
    dNbound.value().resize(rule.value().size());
    ddNbound.value().resize(rule.value().size());
    Nbound.value().resize(rule.value().size());

    for (int i = 0; auto& gp : rule.value()) {
      if (boundDerivatives.value().contains(0)) evaluateFunction(gp.position(), Nbound.value()[i]);
      if (boundDerivatives.value().contains(1)) evaluateJacobian(gp.position(), dNbound.value()[i]);
      if (boundDerivatives.value().contains(2)) evaluateSecondDerivatives(gp.position(), ddNbound.value()[i]);
      ++i;
    }
  }

}