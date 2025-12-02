# =============================================================================
# Makefile - SCIARA-FV2 CUDA TILED VERSION
# =============================================================================

NVCC       := nvcc
NVCC_FLAGS := -O3 -arch=sm_52 -Xcompiler -fopenmp

INCLUDES   := -I../common -I../../cpu -I../..
CPU_SRC    := ../../cpu/Sciara.cpp \
              ../../cpu/io.cpp \
              ../../cpu/cal2DBuffer.cpp \
              ../../cpu/cal2DBufferIO.cpp \
              ../../cpu/GISInfo.cpp \
              ../../cpu/vent.cpp \
              ../../cpu/configurationPathLib.cpp

CUDA_SRC   := sciara_fv2.cu

TARGET     := sciara_tiled_wHalo

# =============================================================================
# Build
# =============================================================================
all: $(TARGET)

$(TARGET): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $(TARGET) \
		$(CUDA_SRC) $(CPU_SRC)

# =============================================================================
# Run example (edit arguments as needed)
# =============================================================================
run: $(TARGET)
	./$(TARGET) ../../data/2006/2006_000000000000.cfg output.cfg 20000 200 0.001

# =============================================================================
# Clean
# =============================================================================
clean:
	rm -f $(TARGET) *.o

.PHONY: all clean run
