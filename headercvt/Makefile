build: headercvt

headercvt: headercvt.cpp
	clang++ -std=c++1y -D_GLIBCXX_USE_CXX11_ABI=$(use_cxx11_abi) -Wall -Wextra -O3 -pedantic-errors $< -lclangTooling -lclangFrontendTool -lclangFrontend -lclangDriver -lclangSerialization -lclangCodeGen -lclangParse -lclangSema -lclangStaticAnalyzerFrontend -lclangStaticAnalyzerCheckers -lclangStaticAnalyzerCore -lclangAnalysis -lclangEdit -lclangAST -lclangLex -lclangBasic -lclang `llvm-config --libs --system-libs` -fno-rtti -o $@



func_decl.pxi: headercvt
	./headercvt stub.c -- $(CLPY_HEADERCVT_INCLUDE_DIRS)
preprocessor_defines.pxi:
types.pxi:



DEPLOY_PATH=../clpy/backend/opencl

$(DEPLOY_PATH)/%.pxi: %.pxi
	cp -f $< $@

deploy: \
	$(DEPLOY_PATH)/func_decl.pxi \
	$(DEPLOY_PATH)/preprocessor_defines.pxi \
	$(DEPLOY_PATH)/types.pxi


clean:
	rm headercvt 2>/dev/null || true
	rm *.pxi  2>/dev/null || true
