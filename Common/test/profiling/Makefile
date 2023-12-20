CFLAG = -I${mkOpenblasInc} -L${mkOpenblasLib} -lopenblas

autoprofile_nomiss: test_profiler.cpp test_profiler.cpp ../../src/mmm.cpp ../../src/mmm_blas.cpp
	g++ test_profiler.cpp ../../src/profiler.cpp ../../src/mmm.cpp ../../src/mmm_blas.cpp -o autoprofile_nomiss ${CFLAG}


gmultiT: gmultiT.cpp ../../src/mmm.cpp
	@echo "Compiling and linking gmultiT.cpp, mmm.cpp"
	@g++ gmultiT.cpp ../../src/mmm.cpp -o gmultiT
	@echo "Done! To execute, type ./gmultiT  dim datatype optimization  tile_dim  num_threads  valgrind"

clear:
	rm -f naive loopI tiling multiT o_blas oblas avx avxT gmultiT
