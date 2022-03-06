NVCC := nvcc
CC := g++
CUDA_SRC_FILES := $(wildcard src/cuda/*.cu)
CUDA_OBJ_FILES := $(patsubst src/%.cu, build/%.o, $(CUDA_SRC_FILES))
CUDA_ARCH := sm_86

ifeq ($(DEBUG), 1)
	CC_FLAGS := -g
endif

# compile cuda
$(CUDA_OBJ_FILES): build/%.o: src/%.cu
	$(NVCC) $(CC_FLAGS) -arch=$(CUDA_ARCH) -c -o $@ $<

# compile cpp & link
build/nvastro: $(CUDA_OBJ_FILES) src/*.cpp
	$(CC) $(CC_FLAGS) -o $@ $^ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart $(shell Magick++-config --cppflags --cxxflags --ldflags --libs)

#test
build/test_cuda: $(CUDA_OBJ_FILES) src/test_cuda.cu
	$(NVCC) $(CC_FLAGS) -arch=$(CUDA_ARCH) -c -o build/test_cuda.o src/test_cuda.cu
	$(NVCC) $(CC_FLAGS) -arch=$(CUDA_ARCH) -o build/test_cuda $(CUDA_OBJ_FILES) build/test_cuda.o

.PHONY: nvastro
nvastro: build/nvastro

.PHONY: test_cuda
test_cuda: build/test_cuda
	./build/test_cuda

.PHONY: run
run: build/nvastro
	./build/nvastro

.PHONY: clean
clean:
	rm -f build/*