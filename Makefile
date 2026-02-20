CC      = gcc
CFLAGS  = -O3 -march=native -Wall -Wextra -std=c11
LDFLAGS = -lm -lpthread

RUNTIME = runtime/soma_runtime.c
INC     = -Iruntime

.PHONY: all clean test bench

all: hello_agent swarm_benchmark

hello_agent: examples/hello_agent.c $(RUNTIME)
	$(CC) $(CFLAGS) $(INC) -o $@ $^ $(LDFLAGS)

swarm_benchmark: examples/swarm_benchmark.c $(RUNTIME)
	$(CC) $(CFLAGS) $(INC) -o $@ $^ $(LDFLAGS)

test: hello_agent swarm_benchmark
	@echo "--- hello_agent ---"
	./hello_agent
	@echo ""
	@echo "--- swarm 16 agents ---"
	./swarm_benchmark 16
	@echo ""
	@echo "--- swarm 64 agents ---"
	./swarm_benchmark 64
	@echo ""
	@echo "--- swarm 128 agents ---"
	./swarm_benchmark 128

bench: swarm_benchmark
	@echo "Scaling benchmark:"
	@for n in 1 8 16 32 64 128 200; do \
		printf "  %-4d agents: " $$n; \
		./swarm_benchmark $$n 2>/dev/null | grep "wall time"; \
	done

clean:
	rm -f hello_agent swarm_benchmark *.o
