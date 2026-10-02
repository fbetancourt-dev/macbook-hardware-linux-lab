CC = gcc
CFLAGS = -O3 -mavx2 -mf16c -mfma -fopenmp -Wall -Wextra
LDFLAGS = -lOpenCL -lm -fopenmp

TARGET = gemv_benchmark

all: $(TARGET)

$(TARGET): gemv_benchmark.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) test_vload_half

run: $(TARGET)
	RUSTICL_ENABLE=nouveau ./$(TARGET)

.PHONY: all clean run
