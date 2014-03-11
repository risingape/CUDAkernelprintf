CUDA = /opt/cuda/lib64
SDK = /home/guido/cuda/C

#CUDA	:= $(CUDA_INSTALL_PATH)
#SDK	:= $(CUDA_INSTALL_PATH)/../sdk/C

INC	:= -I$(CUDA)/include -I$(SDK)/common/inc
LIB	:= -L$(CUDA)/lib64   -L$(SDK)/lib

NVCCFLAGS := -arch=sm_12 --ptxas-options=-v
LIBS	:=  -lcudart -lpthread

all:	dprintf

dprintf:	test.cu
	nvcc test.cu -o test $(INC) $(LIB) $(NVCCFLAGS) $(LIBS) 

clean:
	rm test

