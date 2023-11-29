CFLAG = -I${mkOpenblasInc} -L${mkOpenblasLib} -lopenblas



# making of test_mmm_blas.cpp
mmmBLAS: test_mmm_blas.o mmm.o
	@echo "Linking..."
	@g++ test_mmm_blas.o mmm.o -o mmmBLAS ${CFLAG}
	@echo "Done!"

test_mmm_blas.o: test_mmm_blas.cpp
	@echo "Compiling test_mmm_blas..."
	@g++ test_mmm_blas.cpp -c

mmm.o: ../src/mmm.cpp
	@echo "Compiling mmm.cpp"
	@g++ ../src/mmm.cpp -c

# making clean
clean:
	@echo "Removing everything but the source files"
	@rm -f test_mmm_blas.o mmm.o mmmBLAS test_MatrixFlat.o MatrixFlat

# making of test_MatrixFlat.cpp
MatrixFlat: MatrixFlat.o
	@echo "Linking..."
	@g++ test_MatrixFlat.o -o MatrixFlat
	@echo "Done!"
MatrixFlat.o :
	@echo "Compiling test_MatrixFlat.cpp..."
	@g++ test_MatrixFlat.cpp -c