CC = gcc
TARGET = cpu_version
all: TARGET

TARGET: main.o matrix.o
	gcc -o $(TARGET) main.o matrix.o

main.o: main.c
	gcc -c main.c

matrix.o: matrix.c
	gcc -c matrix.c

.PHONY: clean
clean:
	rm -rf main.o matrix.o cpu_version
