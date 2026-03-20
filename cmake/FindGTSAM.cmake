include_guard()

# Forward version and EXACT arguments to GTSAM's own config-mode package,
# which handles version checking via GTSAMConfigVersion.cmake.
set(_gtsam_find_args)
if(GTSAM_FIND_VERSION)
  list(APPEND _gtsam_find_args ${GTSAM_FIND_VERSION})
  if(GTSAM_FIND_VERSION_EXACT)
    list(APPEND _gtsam_find_args EXACT)
  endif()
endif()

find_package(GTSAM CONFIG ${_gtsam_find_args})
unset(_gtsam_find_args)

if(GTSAM_FOUND AND TARGET gtsam)
  # Promote to global visibility so the target is usable outside this
  # directory scope.
  set_target_properties(gtsam PROPERTIES IMPORTED_GLOBAL TRUE)

  # GTSAM's config exports a bare 'gtsam' target. Create the namespaced
  # GTSAM::GTSAM interface that gtsam_points and downstream consumers expect.
  if(NOT TARGET GTSAM::GTSAM)
    add_library(GTSAM::GTSAM INTERFACE IMPORTED GLOBAL)
    set_target_properties(GTSAM::GTSAM PROPERTIES
      INTERFACE_LINK_LIBRARIES gtsam
    )
  endif()

  set(GTSAM_INCLUDE_DIRS ${GTSAM_INCLUDE_DIR})
endif()
