NVCC := nvcc
CC := g++
CUDA_SRC_FILES := $(wildcard src/cuda/*.cu)
CUDA_OBJ_FILES := $(patsubst src/%.cu, build/%.o, $(SRC_FILES))
CUDA_ARCH := sm_86

ifeq ($(DEBUG), 1)
	CC_FLAGS := -g
endif

# compile cuda
$(CUDA_OBJ_FILES): build/%.o: src/%.cu
	$(NVCC) $(CC_FLAGS) -arch=$(CUDA_ARCH) -c -o $@ $<

# compile cpp & link
build/nvastro: $(CUDA_OBJ_FILES) src/*.cpp
	$(CC) $(CC_FLAGS) -g -o $@ $^ $(shell Magick++-config --cppflags --cxxflags --ldflags --libs)

.PHONY: nvastro
nvastro: build/nvastro

.PHONY: run
run: build/nvastro
	./build/nvastro

.PHONY: clean
clean:
	rm -f build/*