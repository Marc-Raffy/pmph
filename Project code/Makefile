COMPILER?=nvcc
STD_FLAG?=-std=c++11
TEST_RADIX=test-radix

.PHONY: clean all run

default: compile run

compile: $(TEST_RADIX)

$(TEST_RADIX): kernel.cuh
	$(COMPILER) $(STD_FLAG) -o $(TEST_RADIX) main.cu

all: $(TEST_RADIX)

run: $(TEST_RADIX)
	./$(TEST_RADIX)

run-RADIX: $(TEST_RADIX)
	./$(TEST_RADIX)

clean:
	rm -f $(TEST_RADIX)
