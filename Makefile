CC = gcc
NVCC = nvcc
TARGET = cpu_version gpu_version
all: $(TARGET)

cpu_version: main.o matrix_cpu.o 
	$(CC) -o cpu_version main.o matrix_cpu.o

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

.PHONY: clean
clean:
	rm -rf main.o matrix_cpu.o cpu_version matrix_gpu.o gpu_version