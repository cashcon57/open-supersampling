# OSS-Gaussian Metal port build — Sprint 7 / T7.M.1.
#
# Compiles `rasterizer.metal` to `rasterizer.metallib`. Optionally builds the
# Swift host harness (`rasterizer_test`) if Swift is available.
#
# Requires Xcode Command Line Tools:
#     xcode-select --install
#
# Verify:
#     xcrun -sdk macosx metal --version
#
# Targets:
#     make           — build metallib
#     make harness   — build Swift host harness
#     make clean     — remove build artifacts

METAL    ?= xcrun -sdk macosx metal
METALLIB ?= xcrun -sdk macosx metallib
SWIFTC   ?= swiftc

METAL_FLAGS    = -Werror -ffast-math -gline-tables-only
METALLIB_FLAGS =
SWIFT_FLAGS    = -O -framework Metal -framework Foundation

.PHONY: all harness clean
all: rasterizer.metallib

rasterizer.air: rasterizer.metal
	$(METAL) $(METAL_FLAGS) -c $< -o $@

rasterizer.metallib: rasterizer.air
	$(METALLIB) $(METALLIB_FLAGS) $< -o $@

harness: rasterizer.metallib rasterizer.swift
	$(SWIFTC) $(SWIFT_FLAGS) rasterizer.swift -o rasterizer_test

clean:
	rm -f rasterizer.air rasterizer.metallib rasterizer_test
