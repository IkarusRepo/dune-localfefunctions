// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later
#include <dune/localfefunctions/eigenDuneTransformations.hh>

namespace Dune {

  template <Concepts::LocalBasis DuneLocalBasis>
  void CachedLocalBasis<DuneLocalBasis>::evaluateFunction(const DomainType& local, AnsatzFunctionType& N) const {
    duneLocalBasis->evaluateFunction(local, Ndune);
    N.resize(Ndune.size());
    for (size_t i = 0; i < Ndune.size(); ++i)
      N[i] = Ndune[i][0];
  }

  template <Concepts::LocalBasis DuneLocalBasis>
  void CachedLocalBasis<DuneLocalBasis>::evaluateJacobian(const DomainType& local, JacobianType& dN) const {
    duneLocalBasis->evaluateJacobian(local, dNdune);
    resize(dN,dNdune.size());

    for (auto i = 0U; i < dNdune.size(); ++i)
      for (int j = 0; j < gridDim; ++j)
        coeff(dN,i,j) = dNdune[i][0][j];
  }

  template <Concepts::LocalBasis DuneLocalBasis>
  const Dune::QuadraturePoint<typename CachedLocalBasis<DuneLocalBasis>::DomainFieldType, CachedLocalBasis<DuneLocalBasis>::gridDim>& CachedLocalBasis<DuneLocalBasis>::indexToIntegrationPoint(int i) const
  {
    if(isBound())
      return rule.value()[i];
    else
      assert(false && "You need to call bind first");
    __builtin_unreachable();
  }

  /*
   * This function returns the second derivatives of the ansatz functions.
   * The assumed order is in Voigt notation, e.g. for 3d ansatzfunctions N_xx,N_yy,N_zz,N_yz,N_xz, N_xy
   */
  template <Concepts::LocalBasis DuneLocalBasis>
  void CachedLocalBasis<DuneLocalBasis>::evaluateSecondDerivatives(const DomainType& local, SecondDerivativeType& ddN) const {
    std::array<unsigned int, gridDim> order;
    std::ranges::fill(order, 0);
      resize(ddN,dNdune.size());

    for (int i = 0; i < gridDim; ++i) { //Diagonal terms
      order[i] = 2;
      duneLocalBasis->partial(order,local, ddNdune);
      for (size_t j = 0; j < ddNdune.size(); ++j)
          coeff(ddN,j,i)=ddNdune[j][0];

      order[i] = 0;
    }

    std::ranges::fill(order, 1);
    for (int i = 0; i < gridDim*(gridDim-1)/2; ++i) { //off-diagonal terms
      if constexpr (gridDim>2)
        order[i] = 0;
      duneLocalBasis->partial(order,local, ddNdune);
      for (size_t j = 0; j < ddNdune.size(); ++j)
          coeff(ddN,j,i+gridDim)=ddNdune[j][0];
      order[i] = 1;
    }

  }

  template <Concepts::LocalBasis DuneLocalBasis>
  void CachedLocalBasis<DuneLocalBasis>::evaluateFunctionAndJacobian(const DomainType& local, AnsatzFunctionType& N,
                                                                     JacobianType& dN) const {
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
