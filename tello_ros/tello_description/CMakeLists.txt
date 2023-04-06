cmake_minimum_required(VERSION 3.5)
project(tello_description)

find_package(ament_cmake REQUIRED)

#=============
# Generate N urdf files, each with unique robot and frame names
# Source file is in ${CMAKE_CURRENT_SOURCE_DIR}/urdf/tello.xml
# Generated files are in ${CMAKE_CURRENT_BINARY_DIR}/urdf/telloX.urdf
#=============

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/urdf")

foreach (INDEX RANGE 0 8)
  if (${INDEX} EQUAL 0)
    set(SUFFIX "")
    set(TOPIC_NS "solo")
  else ()
    set(SUFFIX "_${INDEX}")
    set(TOPIC_NS "drone${INDEX}")
  endif ()
  set(URDF_FILE "${CMAKE_CURRENT_BINARY_DIR}/urdf/tello${SUFFIX}.urdf")
  message(STATUS "creating rules for ${URDF_FILE}")
  add_custom_command(
    OUTPUT ${URDF_FILE}
    COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/replace.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/urdf/tello.xml" "suffix=${SUFFIX}" "topic_ns=${TOPIC_NS}" ">" "${URDF_FILE}"
    DEPENDS urdf/tello.xml
    COMMENT "Generate ${URDF_FILE}"
    VERBATIM
  )
  # TODO investigate why configure_file isn't working
  # configure_file("urdf/tello.xml" ${URDF_FILE})
  add_custom_target(generate_urdf${SUFFIX} ALL DEPENDS ${URDF_FILE})
endforeach ()

#=============
# Generate 5 urdf files, each with unique red gate and frame names
# Source file is in ${CMAKE_CURRENT_SOURCE_DIR}/urdf/rgate.xml
# Generated files are in ${CMAKE_CURRENT_BINARY_DIR}/urdf/rgate_X.urdf
#=============

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/urdf")

foreach (INDEX RANGE 0 5)
  if (${INDEX} EQUAL 0)
    set(SUFFIX "")
    set(TOPIC_NS "solo")
  else ()
    set(SUFFIX "_${INDEX}")
    set(TOPIC_NS "drone${INDEX}")
  endif ()
  set(URDF_FILE "${CMAKE_CURRENT_BINARY_DIR}/urdf/rgate${SUFFIX}.urdf")
  message(STATUS "creating rules for ${URDF_FILE}")
  add_custom_command(
    OUTPUT ${URDF_FILE}
    COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/replace.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/urdf/rgate.xml" "suffix=${SUFFIX}" "topic_ns=${TOPIC_NS}" ">" "${URDF_FILE}"
    DEPENDS urdf/rgate.xml
    COMMENT "Generate ${URDF_FILE}"
    VERBATIM
  )
  # TODO investigate why configure_file isn't working
  # configure_file("urdf/tello.xml" ${URDF_FILE})
  add_custom_target(generate_rgate_urdf${SUFFIX} ALL DEPENDS ${URDF_FILE})
endforeach ()

#=============
# Generate 3 urdf files, each with unique green gate and frame names
# Source file is in ${CMAKE_CURRENT_SOURCE_DIR}/urdf/rgate.xml
# Generated files are in ${CMAKE_CURRENT_BINARY_DIR}/urdf/rgate_X.urdf
#=============

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/urdf")

foreach (INDEX RANGE 0 5)
  if (${INDEX} EQUAL 0)
    set(SUFFIX "")
    set(TOPIC_NS "solo")
  else ()
    set(SUFFIX "_${INDEX}")
    set(TOPIC_NS "drone${INDEX}")
  endif ()
  set(URDF_FILE "${CMAKE_CURRENT_BINARY_DIR}/urdf/ggate${SUFFIX}.urdf")
  message(STATUS "creating rules for ${URDF_FILE}")
  add_custom_command(
    OUTPUT ${URDF_FILE}
    COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/replace.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/urdf/ggate.xml" "suffix=${SUFFIX}" "topic_ns=${TOPIC_NS}" ">" "${URDF_FILE}"
    DEPENDS urdf/ggate.xml
    COMMENT "Generate ${URDF_FILE}"
    VERBATIM
  )
  # TODO investigate why configure_file isn't working
  # configure_file("urdf/tello.xml" ${URDF_FILE})
  add_custom_target(generate_ggate_urdf${SUFFIX} ALL DEPENDS ${URDF_FILE})
endforeach ()

#=============
# Generate 3 urdf files, each with unique green gate and frame names
# Source file is in ${CMAKE_CURRENT_SOURCE_DIR}/urdf/rgate.xml
# Generated files are in ${CMAKE_CURRENT_BINARY_DIR}/urdf/rgate_X.urdf
#=============

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/urdf")

foreach (INDEX RANGE 0 5)
  if (${INDEX} EQUAL 0)
    set(SUFFIX "")
    set(TOPIC_NS "solo")
  else ()
    set(SUFFIX "_${INDEX}")
    set(TOPIC_NS "drone${INDEX}")
  endif ()
  set(URDF_FILE "${CMAKE_CURRENT_BINARY_DIR}/urdf/bgate${SUFFIX}.urdf")
  message(STATUS "creating rules for ${URDF_FILE}")
  add_custom_command(
    OUTPUT ${URDF_FILE}
    COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/replace.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/urdf/bgate.xml" "suffix=${SUFFIX}" "topic_ns=${TOPIC_NS}" ">" "${URDF_FILE}"
    DEPENDS urdf/bgate.xml
    COMMENT "Generate ${URDF_FILE}"
    VERBATIM
  )
  # TODO investigate why configure_file isn't working
  # configure_file("urdf/tello.xml" ${URDF_FILE})
  add_custom_target(generate_bgate_urdf${SUFFIX} ALL DEPENDS ${URDF_FILE})
endforeach ()

#=============
# Install
#=============

install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/urdf"
  DESTINATION share/${PROJECT_NAME}
)

#=============
# Run ament macros
#=============

ament_package()
