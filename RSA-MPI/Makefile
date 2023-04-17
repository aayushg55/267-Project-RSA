NVCCFLAGS = -arch=sm_80
NVCFLAGS = -gpu=cc80

mpi_rsa: main.cu
	CC $(NVCFLAGS) main.cu -o $@
clean:
	rm mpi_rsa
