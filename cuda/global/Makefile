# ============================================================
#                  SCIARA-FV2 GPU Makefile
# ============================================================

# GPU architecture
ARCH        = sm_52

# Compiler
NVCC        = nvcc

# Include directories
INCLUDES    = -I../common -I../../cpu -I../..

# CPU source files
CPU_SRCS    = ../../cpu/Sciara.cpp \
              ../../cpu/io.cpp \
              ../../cpu/cal2DBuffer.cpp \
              ../../cpu/cal2DBufferIO.cpp \
              ../../cpu/GISInfo.cpp \
              ../../cpu/vent.cpp \
              ../../cpu/configurationPathLib.cpp

# CUDA source file
CUDA_SRC    = sciara_fv2.cu

# Output executable
TARGET      = sciara_global

# Compiler flags
NVCCFLAGS   = -O3 -arch=$(ARCH) $(INCLUDES)

# ------------------------------------------------------------
# Simulation parameters (modifiable by the user)
# ------------------------------------------------------------

INPUT       = ../../data/2006/2006_000000000000.cfg
OUTPUT      = output.cfg
STEPS       = 20000
REDUCE      = 200
THRESHOLD   = 0.001

# ------------------------------------------------------------
# Rules
# ------------------------------------------------------------

# Build everything
all: build

# Compile
build:
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(CUDA_SRC) $(CPU_SRCS)
	@echo "✔ Build completed."

# Run the simulation
run: build
	./$(TARGET) $(INPUT) $(OUTPUT) $(STEPS) $(REDUCE) $(THRESHOLD)

# Profile with nvprof
profile: build
	nvprof ./$(TARGET) $(INPUT) $(OUTPUT) 1000 $(REDUCE) $(THRESHOLD)

# CUDA memcheck
memcheck: build
	cuda-memcheck ./$(TARGET) $(INPUT) $(OUTPUT) 200 $(REDUCE) $(THRESHOLD)

# Clean
clean:
	rm -f $(TARGET)
	@echo "✔ Clean completed."

