CC = gcc
CFLAGS = -O3 -mavx2 -mf16c -mfma -fopenmp -Wall -Wextra
LDFLAGS = -lOpenCL -lm -fopenmp

TARGET = test_real_layers

all: $(TARGET)

$(TARGET): test_real_layers.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	RUSTICL_ENABLE=nouveau ./$(TARGET)

.PHONY: all clean run
