# ============================================================================
#  bin2c_wrapper.cmake
# ============================================================================
#  Portable wrapper around the CUDA `bin2c` tool.  bin2c writes its C array
#  to stdout, so we capture stdout into ${OUTPUT} via execute_process — this
#  works identically on Windows and Unix without shell redirection.
#
#  Expected -D arguments:
#    BIN2C     full path to bin2c executable
#    PTX_FILE  path to the PTX file to embed
#    OUTPUT    output .c file containing the embedded byte array
#    VAR_NAME  C identifier for the generated array
# ============================================================================

execute_process(
    COMMAND "${BIN2C}" --padd 0 --type char --name "${VAR_NAME}" "${PTX_FILE}"
    OUTPUT_FILE "${OUTPUT}"
    RESULT_VARIABLE bin2c_result
)

if(NOT bin2c_result EQUAL 0)
    message(FATAL_ERROR "bin2c_wrapper: bin2c failed (exit ${bin2c_result})")
endif()
