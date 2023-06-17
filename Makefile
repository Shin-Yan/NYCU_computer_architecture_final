CC = gcc
NVCC = nvcc
LINKER_DIR = -L/usr/local/cuda/lib
OCELOT=`OcelotConfig -l`
CUDA_ARCH = -arch=sm_20
TARGET = cpu_version gpu_version
all: $(TARGET)
vm: cpu_vm gpu_vm

cpu_version: main.o matrix_cpu.o 
	$(CC) -o cpu_version main.o matrix_cpu.o -lm

gpu_version: main_gpu.o matrix_gpu.o
	$(NVCC) -o gpu_version main_gpu.o matrix_gpu.o

main.o: main.c
	$(CC) -c main.c -I .

main_gpu.o: main_gpu.cu
	$(NVCC) -c main_gpu.cu -I .

matrix_cpu.o: matrix_cpu.c
	$(CC) -c matrix_cpu.c -I .

matrix_gpu.o: matrix_gpu.cu
	$(NVCC) -c matrix_gpu.cu -I .

cpu_vm: main.o matrix_cpu.o
	$(CC) -o cpu_vm main.o matrix_cpu.o -lm

gpu_vm: main_gpu_vm.o matrix_gpu_vm.o
	$(NVCC) -o main_gpu.o matrix_gpu.o gpu_vm $(LINKER_DIR) $(OCELOT)

main_gpu_vm.o: main_gpu.cu
	$(NVCC) -c main_gpu.cu $(CUDA_ARCH) -I . -o main_gpu_vm.o

matrix_gpu_vm.o: matrix_gpu.cu
	$(NVCC) -c matrix_gpu.cu $(CUDA_ARCH) -I . -o matrix_gpu_vm.o

.PHONY: clean
clean:
	rm -rf *.o cpu_version gpu_version cpu_vm gpu_vm
