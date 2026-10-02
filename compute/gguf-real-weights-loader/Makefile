CC = gcc
CFLAGS = -O3 -mavx2 -mf16c -mfma -fopenmp -Wall -Wextra
LDFLAGS = -lOpenCL -lm -fopenmp

TARGETS = test_real_layers test_28_layers_real

all: $(TARGETS)

test_real_layers: test_real_layers.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test_28_layers_real: test_28_layers_real.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)

run: test_28_layers_real
	RUSTICL_ENABLE=nouveau ./test_28_layers_real

.PHONY: all clean run
