include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)

#######################################
#######################################
file(GLOB HEADERS *.h *.cuh)
source_group("Headers" FILES ${HEADERS})

cuda_add_executable( cu_MatrixFactorization
${HEADERS}
mf.cu
OPTIONS --cudart=shared  -lineinfo -arch sm_52 -use_fast_math -maxrregcount 104
DEBUG -Xptxas -v -g -G 
)


target_link_libraries( 
	cu_MatrixFactorization

	${CUDA_CUDA_LIBRARY}
	${CUDA_CUDART_LIBRARY}
	${CUDA_CURAND_LIBRARY}
	${NVML_DEBUG_LIBRARIES}
	${NVTX_DEBUG_LIBRARIES}
	${FREEGLUT_DEBUG_LIBRARIES}
	${GLEW_DEBUG_LIBRARIES}
)


add_custom_command(TARGET cu_MatrixFactorization POST_BUILD     
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
"${CUDA_TOOLKIT_SAMPLES_DIR}/bin/win64/Release/freeglut.dll"  
$<TARGET_FILE_DIR:cu_MatrixFactorization>)          

add_custom_command(TARGET cu_MatrixFactorization POST_BUILD     
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
"${CUDA_TOOLKIT_SAMPLES_DIR}/bin/win64/Release/glew64.dll"  
$<TARGET_FILE_DIR:cu_MatrixFactorization>)   