EIGEN_PATH = /usr/local/include/Eigen
MKL_PATH = /opt/intel/mkl
BGEN_PATH = /usr/local

heter:heter.cpp
	g++ -std=c++17 -w -O3 -m64 -fopenmp -fpermissive -DEIGEN_NO_DEBUG heter.cpp -I $(BGEN_PATH)/include/bgen -I $(EIGEN_PATH) -I $(MKL_PATH)/include -o heter -lboost_filesystem -lboost_system -Wl,--start-group $(BGEN_PATH)/lib/libbgen.so $(MKL_PATH)/lib/intel64/libmkl_intel_lp64.so $(MKL_PATH)/lib/intel64/libmkl_gnu_thread.so $(MKL_PATH)/lib/intel64/libmkl_core.so -Wl,--end-group
