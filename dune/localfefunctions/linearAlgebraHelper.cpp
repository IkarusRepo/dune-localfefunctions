// SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "linearAlgebraHelper.hh"

namespace Dune {
  Dune::DerivativeDirections::ZeroMatrix transpose(const Dune::DerivativeDirections::ZeroMatrix&) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  Dune::DerivativeDirections::ZeroMatrix operator+(Dune::DerivativeDirections::ZeroMatrix,
                                                   Dune::DerivativeDirections::ZeroMatrix) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  Dune::DerivativeDirections::ZeroMatrix operator-(Dune::DerivativeDirections::ZeroMatrix,
                                                   Dune::DerivativeDirections::ZeroMatrix) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  Dune::DerivativeDirections::ZeroMatrix eval(const Dune::DerivativeDirections::ZeroMatrix&) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }

  Dune::DerivativeDirections::ZeroMatrix operator-(Dune::DerivativeDirections::ZeroMatrix) {
    return Dune::DerivativeDirections::ZeroMatrix();
  }
}  // namespace Dune
