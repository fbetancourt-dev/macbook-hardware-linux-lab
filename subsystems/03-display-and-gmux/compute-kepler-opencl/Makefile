CC = gcc
CFLAGS = -O3 -Wall -Wextra -fopenmp -mavx2
LDFLAGS = -lOpenCL -lm -fopenmp

TARGETS = heat_equation_2d nbody_simulation

all: $(TARGETS)

heat_equation_2d: heat_equation_2d.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nbody_simulation: nbody_simulation.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run-heat: heat_equation_2d
	RUSTICL_ENABLE=nouveau ./heat_equation_2d

run-nbody: nbody_simulation
	RUSTICL_ENABLE=nouveau ./nbody_simulation

clean:
	rm -f $(TARGETS)

.PHONY: all run-heat run-nbody clean
