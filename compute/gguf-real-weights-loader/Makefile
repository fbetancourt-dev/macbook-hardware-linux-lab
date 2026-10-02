CC = gcc
CFLAGS = -O3 -mavx2 -mf16c -mfma -fopenmp -Wall -Wextra
LDFLAGS = -lOpenCL -lm -fopenmp

TARGETS = test_real_layers test_28_layers_real test_gemv_q6_k test_full_model_logits

all: $(TARGETS)

test_real_layers: test_real_layers.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test_28_layers_real: test_28_layers_real.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test_gemv_q6_k: test_gemv_q6_k.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test_full_model_logits: test_full_model_logits.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)

run: test_full_model_logits
	RUSTICL_ENABLE=nouveau ./test_full_model_logits

.PHONY: all clean run

