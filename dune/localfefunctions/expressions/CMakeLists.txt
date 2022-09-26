#install headers
install(FILES
        binaryExpr.hh
        constant.hh
        dotProductExpr.hh
        exprChecks.hh
        negateExpr.hh
        normSquaredExpr.hh
        rebind.hh
        scaleExpr.hh
        sqrtExpr.hh
        sumExpr.hh
        unaryExpr.hh
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/localfefunctions/expressions)
