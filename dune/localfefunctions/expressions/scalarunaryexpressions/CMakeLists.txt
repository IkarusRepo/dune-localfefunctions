# install headers
install(
  FILES scalarLogExpr.hh scalarPowExpr.hh scalarSqrtExpr.hh
        scalarUnaryExpression.hh
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}/dune/localfefunctions/expressions/scalarunaryexpressions
)
