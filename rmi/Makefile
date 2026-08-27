CXX      ?= g++
# -ffp-contract=off is load-bearing, not cosmetic: it stops the compiler from
# contracting `slope * key + intercept` into an FMA at some call sites and not
# others. Training and lookup must compute bit-identical predictions, or a
# stored error bound can be off by one and a key goes missing.
CXXFLAGS ?= -std=c++17 -O3 -march=native -ffp-contract=off -Wall -Wextra \
            -Iinclude -Isrc
LDFLAGS  ?=

LIB_SRCS := src/rmi.cpp src/dataset.cpp
BINARIES := rmi_main rmi_bench

.PHONY: all clean debug
all: $(BINARIES)

rmi_main: $(LIB_SRCS) src/main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

rmi_bench: $(LIB_SRCS) bench/benchmark.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Assertions on (monotone-routing check in RMI::train).
debug: CXXFLAGS := -std=c++17 -O0 -g -ffp-contract=off -Wall -Wextra -Iinclude -Isrc
debug: clean $(BINARIES)

clean:
	rm -f $(BINARIES)
