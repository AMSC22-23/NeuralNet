#CFLAG:  flags needed to link the openblas library:
CFLAG = -I${mkOpenblasInc} -L${mkOpenblasLib} -lopenblas
#FLAG1X1: optimization flags (1 = filippo, x = any algorithm, 1 = code optimization)
FLAG1X1 = -O3 -march=native -ffast-math



mmm.o: ../src/mmm.cpp
	@echo "Compiling mmm.cpp"
	@g++ ../src/mmm.cpp -c

mmm_1X1.o:  ../src/mmm.cpp
	@echo "Compiling mmm.cpp with optimizations..."
	@g++ ../src/mmm.cpp -c ${FLAG1X1}

# making of Unit_Test_MatrixFlat.cpp
UnitTest_MatrixFlat: UnitTest_MatrixFlat.o
	@echo "Linking..."
	@g++ UnitTest_MatrixFlat.o -o UnitTest_MatrixFlat
	@echo "Done! To run the test call ./UnitTest_MatrixFlat"

UnitTest_MatrixFlat.o :
	@echo "Compiling UnitTest_MatrixFlat.cpp..."
	@g++ UnitTest_MatrixFlat.cpp -c

# Making of UnitTest_mmm_naive.cpp

# making of Unit_Test_mmm_naive.cpp
UnitTest_mmm_naive: UnitTest_mmm_naive.o mmm.o
	@echo "Linking..."
	@g++ UnitTest_mmm_naive.o mmm.o -o UnitTest_mmm_naive ${CFLAG}
	@echo "Done! To run the test call ./UnitTest_mmm_naive MATRIXDIM "

UnitTest_mmm_naive_101: UnitTest_mmm_naive_101.o mmm_1X1.o
	@echo "Linking..."
	@g++ UnitTest_mmm_naive.o mmm.o -o UnitTest_mmm_naive_101 ${CFLAG} ${FLAG1X1}
	@echo "Done! To run the test call ./UnitTest_mmm_naive_101 MATRIXDIM "
	@echo "Optimizations level 1X1 : (-O3 -march=native -ffast-math) "

UnitTest_mmm_naive_101.o:
	@echo "Building UnitTest_mmm_naive.cpp with optimizations ..."
	@g++ UnitTest_mmm_naive.cpp -c ${FLAG1X1}


UnitTest_mmm_naive.o: UnitTest_mmm_naive.cpp
	@echo "Building UnitTest_mmm_naive.cpp..."
	@g++ UnitTest_mmm_naive.cpp -c

# making of ale_test.cpp
ale_test: ale_test.cpp
	@echo "Building and linking ale_test... "
	@g++ -std=c++20 ale_test.cpp ../src/matrixProd_AVX.cpp  -mavx2 -std=c++20 -o ale_test
	@echo "Done! To run the test call ./ale_test"



# making clear
clear:
	@echo "Removing everything but the source files"
	@rm -f mmm.o UnitTest_MatrixFlat.o UnitTest_MatrixFlat UnitTest_mmm_naive UnitTest_mmm_naive.o
