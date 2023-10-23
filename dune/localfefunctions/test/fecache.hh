// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include <dune/common/version.hh>
#if DUNE_VERSION_GTE(DUNE_FUNCTIONS, 2, 10)
#  include <dune/localfunctions/lagrange/lagrangelfecache.hh>
template <int domainDim, int order>
using FECache = Dune::LagrangeLocalFiniteElementCache<double, double, domainDim, order>;
#else
#  include <dune/localfunctions/lagrange/pqkfactory.hh>
template <int domainDim, int order>
using FECache = Dune::PQkLocalFiniteElementCache<double, double, domainDim, order>;
#endif
