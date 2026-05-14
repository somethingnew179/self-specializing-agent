GEN_SRC := materials/problem-c-generator/files/generator.cpp
GEN_BIN := build/problem-c-generator

.PHONY: generator clean

generator: $(GEN_BIN)

$(GEN_BIN): $(GEN_SRC) materials/problem-c-generator/files/testlib.h
	mkdir -p build
	c++ -O2 -std=c++17 -Wall -Wextra -I materials/problem-c-generator/files -o $(GEN_BIN) $(GEN_SRC)

clean:
	rm -rf build generated
