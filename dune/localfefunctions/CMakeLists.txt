#install headers

file(
        GLOB_RECURSE headers
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        CONFIGURE_DEPENDS *.hh
)

install(FILES ${headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/localfefunctions)

file(
        GLOB_RECURSE sources
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        CONFIGURE_DEPENDS *.cpp
)

target_sources(dune-localfefunctions PRIVATE
        ${sources})

dune_register_package_flags(
        INCLUDE_DIRS
        ${Eigen3_INCLUDE_DIRS}
        LIBRARIES
        Eigen3::Eigen
)

target_link_dune_default_libraries(${PROJECT_NAME})


add_subdirectory(expressions)
add_subdirectory(impl)
add_subdirectory(test)