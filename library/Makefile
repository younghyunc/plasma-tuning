-include make.inc

CC = g++
CXXFLAGS =
INCLUDES = -I${PLASMA_INC_PATH} -I${PAPI_INC_PATH}
LDFLAGS = -L${PLASMA_LIB_PATH} -L${PAPI_LIB_PATH}

all:
	g++ ${CXXFLAGS} ${INCLUDES} -shared -fPIC -o plasma_tuning.so plasma_tuning.cpp plasma_tuning.h ${LDFLAGS} -lplasma -lpapi -fopenmp

clean:
	rm -rf plasma_tuning.so
