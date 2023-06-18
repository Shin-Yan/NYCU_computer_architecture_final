CC = gcc
NVCC = nvcc
LINKER_DIR = -L/usr/local/cuda/lib
OCELOT=`OcelotConfig -l`
CUDA_ARCH = -arch=sm_20
TARGET = gpu_version1 gpu_version2

all: $(TARGET)

vm: gpu_vm1 gpu_vm2

gpu_version1: main_gpu1.o matrix_gpu1.o
	$(NVCC) -o gpu_version1 main_gpu1.o matrix_gpu1.o

main_gpu1.o: main_gpu1.cu
	$(NVCC) -c main_gpu1.cu -I .

matrix_gpu1.o: matrix_gpu1.cu
	$(NVCC) -c matrix_gpu1.cu -I .

gpu_vm1: main_gpu_vm1.o matrix_gpu_vm1.o
	$(CC) -o gpu_vm1 main_gpu_vm1.o matrix_gpu_vm1.o $(LINKER_DIR) $(OCELOT)

main_gpu_vm1.o: main_gpu1.cu
	$(NVCC) -c main_gpu1.cu $(CUDA_ARCH) -I . -o main_gpu_vm1.o

matrix_gpu_vm1.o: matrix_gpu1.cu
	$(NVCC) -c matrix_gpu1.cu $(CUDA_ARCH) -I . -o matrix_gpu_vm1.o

gpu_version2: main_gpu2.o matrix_gpu2.o
	$(NVCC) -o gpu_version2 main_gpu2.o matrix_gpu2.o

main_gpu2.o: main_gpu2.cu
	$(NVCC) -c main_gpu2.cu -I .

matrix_gpu2.o: matrix_gpu2.cu
	$(NVCC) -c matrix_gpu2.cu -I .

gpu_vm2: main_gpu_vm2.o matrix_gpu_vm2.o
	$(CC) -o gpu_vm2 main_gpu_vm2.o matrix_gpu_vm2.o $(LINKER_DIR) $(OCELOT)

main_gpu_vm2.o: main_gpu2.cu
	$(NVCC) -c main_gpu2.cu $(CUDA_ARCH) -I . -o main_gpu_vm2.o

matrix_gpu_vm2.o: matrix_gpu2.cu
	$(NVCC) -c matrix_gpu2.cu $(CUDA_ARCH) -I . -o matrix_gpu_vm2.o

.PHONY: clean
clean:
	rm -rf *.o gpu_version1 gpu_version2 gpu_vm1 gpu_vm2
