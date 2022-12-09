# SPDX-FileCopyrightText: 2022 The dune-localfefunction developers
# mueller@ibb.uni-stuttgart.de SPDX-License-Identifier: LGPL-2.1-or-later

# install headers

file(
  GLOB_RECURSE headers
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS *.hh)

install(FILES ${headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/localfefunctions)

file(
  GLOB_RECURSE sources
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS *.cpp)

target_sources(dune-localfefunctions PRIVATE ${sources})

target_link_dune_default_libraries(${PROJECT_NAME})

add_subdirectory(expressions)
add_subdirectory(impl)
add_subdirectory(test)
add_subdirectory(cachedlocalBasis)
add_subdirectory(manifolds)
