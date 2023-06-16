CC = gcc
TARGET = cpu_version
all: TARGET

TARGET: main.o matrix_cpu.o
	gcc -o $(TARGET) main.o matrix_cpu.o

main.o: main.c
	gcc -c main.c

matrix.o: matrix_cpu.c
	gcc -c matrix_cpu.c

.PHONY: clean
clean:
	rm -rf main.o matrix_cpu.o cpu_version