CFLAG = -I${mkOpenblasInc} -L${mkOpenblasLib} -lopenblas

autoprofile_nomiss: test_profiler.cpp test_profiler.cpp ../../src/mmm.cpp ../../src/mmm_blas.cpp
	g++ test_profiler.cpp ../../src/profiler.cpp ../../src/mmm.cpp ../../src/mmm_blas.cpp -o autoprofile_nomiss ${CFLAG}