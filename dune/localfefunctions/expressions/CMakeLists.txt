# SPDX-FileCopyrightText: 2022 The dune-localfefunction developers mueller@ibb.uni-stuttgart.de
# SPDX-License-Identifier: LGPL-2.1-or-later

# install headers
install(
  FILES binaryExpr.hh
        constant.hh
        dotProductExpr.hh
        exprChecks.hh
        expressionHelper.hh
        greenLagrangeStrains.hh
        linearStrainsExpr.hh
        negateExpr.hh
        normSquaredExpr.hh
        rebind.hh
        scaleExpr.hh
        sumExpr.hh
        unaryExpr.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/localfefunctions/expressions)

add_subdirectory(scalarunaryexpressions)
