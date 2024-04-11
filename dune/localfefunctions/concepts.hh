// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

//
// Created by lex on 9/28/22.
//

#pragma once
namespace Dune::Concepts {

  template <typename DomainTypeOrIntegrationPointIndex, typename DomainType>
  concept IsIntegrationPointIndexOrIntegrationPointPosition
      = std::is_same_v<DomainTypeOrIntegrationPointIndex, DomainType>
        or std::numeric_limits<DomainTypeOrIntegrationPointIndex>::is_integer;

  template <typename LocalBasisImpl>
  concept LocalBasis = requires(LocalBasisImpl& duneLocalBasis) {
    typename LocalBasisImpl::Traits::RangeType;
    typename LocalBasisImpl::Traits::JacobianType;
    LocalBasisImpl::Traits::dimDomain;
    typename LocalBasisImpl::Traits::DomainType;

    typename LocalBasisImpl::Traits::DomainFieldType;
    typename LocalBasisImpl::Traits::RangeFieldType;

    duneLocalBasis.evaluateFunction(std::declval<typename LocalBasisImpl::Traits::DomainType>(),
                                    std::declval<std::vector<typename LocalBasisImpl::Traits::RangeType>&>());
    duneLocalBasis.evaluateJacobian(std::declval<typename LocalBasisImpl::Traits::DomainType>(),
                                    std::declval<std::vector<typename LocalBasisImpl::Traits::JacobianType>&>());

    //                           duneLocalBasis.partial(std::declval<typename
    //                           LocalBasisImpl::Traits::DomainType>(),
    //                                                           std::declval<std::vector<typename
    //                                                           LocalBasisImpl::Traits::JacobianType>&>());
  };

  template <typename L, typename R>
  concept MultiplyAble = requires(L x, R y) { x* y; };

  template <typename L, typename R>
  concept AddAble = requires(L x, R y) { x + y; };

  template <typename L, typename R>
  concept SubstractAble = requires(L x, R y) { x - y; };

  template <typename L, typename R>
  concept MultiplyAssignAble = requires(L x, R y) { x *= y; };

  template <typename L, typename R>
  concept DivideAssignAble = requires(L x, R y) { x /= y; };

  template <typename L, typename R>
  concept AddAssignAble = requires(L x, R y) { x += y; };

  template <typename L, typename R>
  concept SubstractAssignAble = requires(L x, R y) { x -= y; };

  template <typename L, typename R>
  concept DivideAble = requires(L x, R y) { x / y; };

  template <typename L>
  concept NegateAble = requires(L x) { -x; };

  template <typename L>
  concept TransposeAble = requires(L x) { transpose(x); };

}  // namespace Dune::Concepts
