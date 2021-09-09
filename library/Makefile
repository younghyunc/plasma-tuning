-include make.inc

CC = g++
CXXFLAGS =
INCLUDES = -I${PLASMA_INC_PATH}
LDFLAGS = -L${PLASMA_LIB_PATH}

all:
	g++ ${CXXFLAGS} ${INCLUDES} -shared -fPIC -o plasma_tuning.so plasma_tuning.cpp plasma_tuning.h ${LDFLAGS} -lplasma -fopenmp

clean:
	rm -rf plasma_tuning.so
