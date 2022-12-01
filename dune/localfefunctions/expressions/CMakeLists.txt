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
