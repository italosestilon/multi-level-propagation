set(PYTHON ${PYTHON_ROOT}/bin/python)
message(STATUS ${PYTHON})
if (PYTHON)
   set (SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
   set (SETUP_PY ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
  
   set (PY_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build/timestamp)

   configure_file(${SETUP_PY_IN} ${SETUP_PY})

   add_custom_command(OUTPUT "${PY_OUTPUT}"
                   COMMAND ${PYTHON} ${SETUP_PY} build_ext
                   COMMAND ${CMAKE_COMMAND} -E touch ${PY_OUTPUT})

   add_custom_target(label_propagation_py ALL DEPENDS ${PY_OUTPUT})

   install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")

endif()

