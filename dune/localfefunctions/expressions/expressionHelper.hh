// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

namespace Dune {

  /** Helper to evaluate the local basis ansatz function and gradient with an integration point index or coordinate
   * vector*/
  template <typename DomainTypeOrIntegrationPointIndex, typename Basis>
  auto returnIntegrationPointPosition(const DomainTypeOrIntegrationPointIndex& localOrIpId, const Basis& basis) {
    if constexpr (std::is_same_v<DomainTypeOrIntegrationPointIndex, typename Basis::DomainType>) {
      return localOrIpId;
    } else if constexpr (std::numeric_limits<DomainTypeOrIntegrationPointIndex>::is_integer) {
      return basis.indexToIntegrationPoint(localOrIpId).position();
    } else
      static_assert(
          std::is_same_v<DomainTypeOrIntegrationPointIndex,
                         typename Basis::DomainType> or std::is_same_v<DomainTypeOrIntegrationPointIndex, int>,
          "The argument you passed should be an id for the integration point or the point coordinates");
  }
}  // namespace Dune
