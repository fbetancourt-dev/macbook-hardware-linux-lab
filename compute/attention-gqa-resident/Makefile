CC = gcc
CFLAGS = -O3 -mavx2 -mf16c -mfma -fopenmp -Wall -Wextra
LDFLAGS = -lOpenCL -lm -fopenmp

TARGET = attention_gqa_benchmark

all: $(TARGET)

$(TARGET): attention_gqa_benchmark.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	RUSTICL_ENABLE=nouveau ./$(TARGET)

.PHONY: all clean run
