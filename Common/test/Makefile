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

# making clear
clear:
	@echo "Removing everything but the source files"
	@rm -f test_mmm_blas.o mmm.o mmmBLAS UnitTest_MatrixFlat.o UnitTest_MatrixFlat

# making of Unit_Test_MatrixFlat.cpp
UnitTest_MatrixFlat: UnitTest_MatrixFlat.o
	@echo "Linking..."
	@g++ UnitTest_MatrixFlat.o -o UnitTest_MatrixFlat
	@echo "Done!"
UnitTest_MatrixFlat.o :
	@echo "Compiling test_MatrixFlat.cpp..."
	@g++ UnitTest_MatrixFlat.cpp -c

# Making of UnitTest_mmm_naive.cpp

UnitTest_mmm_naive: UnitTest_mmm_naive.o mmm.o
	@echo "Linking..."
	@g++ UnitTest_mmm_naive.o mmm.o -o UnitTest_mmm_naive ${CFLAG}
	@echo "Done! To run the test call ./UnitTest_mmm_naive "

UnitTest_mmm_naive.o: UnitTest_mmm_naive.cpp
	@echo "Building UnitTest_mmm_naive.cpp..."
	@g++ UnitTest_mmm_naive.cpp -c