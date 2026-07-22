# ============================================================================
#  bin2c_wrapper.cmake
# ============================================================================
#  Portable wrapper around the CUDA `bin2c` tool.  bin2c writes its C array
#  to stdout, so we capture stdout into ${OUTPUT} via execute_process — this
#  works identically on Windows and Unix without shell redirection.
#
#  PTX_FILE may be a single path OR a CMake semicolon-separated list of paths
#  (as produced by $<TARGET_OBJECTS:…> when the OBJECT library has multiple
#  source files).  All PTX files are concatenated into a single temporary file
#  before bin2c is invoked so that only one C array is emitted.
#
#  Expected -D arguments:
#    BIN2C     full path to bin2c executable
#    PTX_FILE  single PTX path, or ;-separated list of PTX paths
#    OUTPUT    output .c file containing the embedded byte array
#    VAR_NAME  C identifier for the generated array
# ============================================================================

# PTX is plain text — concatenate all files into one before embedding.
set(CONCAT_PTX "${OUTPUT}.tmp.ptx")
file(WRITE "${CONCAT_PTX}" "")    # create / reset

foreach(PTX_F ${PTX_FILE})
    if(NOT EXISTS "${PTX_F}")
        message(FATAL_ERROR "bin2c_wrapper: PTX file not found: ${PTX_F}")
    endif()
    file(READ "${PTX_F}" PTX_CONTENT)
    file(APPEND "${CONCAT_PTX}" "${PTX_CONTENT}")
endforeach()

execute_process(
    COMMAND "${BIN2C}" --padd 0 --type char --name "${VAR_NAME}" "${CONCAT_PTX}"
    OUTPUT_FILE "${OUTPUT}"
    RESULT_VARIABLE bin2c_result
)

if(NOT bin2c_result EQUAL 0)
    message(FATAL_ERROR "bin2c_wrapper: bin2c failed (exit ${bin2c_result})")
endif()
