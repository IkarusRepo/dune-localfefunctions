# SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
# SPDX-License-Identifier: LGPL-2.1-or-later

# install headers
install(
  FILES scalarLogExpr.hh scalarPowExpr.hh scalarSqrtExpr.hh
        scalarUnaryExpression.hh
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}/dune/localfefunctions/expressions/scalarunaryexpressions
)
