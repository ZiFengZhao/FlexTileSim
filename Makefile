########################################################
# NPU Simulator Makefile
########################################################

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -g

SRC_DIR = src
BUILD_DIR = build
DRAMSIM_DIR = external/DRAMsim3
BOOKSIM_DIR = external/Booksim2/src

INCLUDES = -I./include \
           -I./$(DRAMSIM_DIR)/src \
           -I./$(BOOKSIM_DIR) \
           -I./$(BOOKSIM_DIR)/arbiters \
           -I./$(BOOKSIM_DIR)/allocators \
           -I./$(BOOKSIM_DIR)/routers \
           -I./$(BOOKSIM_DIR)/networks \
           -I./$(BOOKSIM_DIR)/power 

BOOKSIM_LIB = $(BOOKSIM_DIR)/libbooksim.a

LIBS = $(BOOKSIM_LIB) \
       -L./$(DRAMSIM_DIR) -ldramsim3 \
       -lpthread

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

TARGET = $(BUILD_DIR)/npu_sim

all: build run

build: $(TARGET)
	@echo "Build complete! Generated npu simulator: $(TARGET)"

$(TARGET): $(OBJS) $(BOOKSIM_LIB)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJS) $(LIBS)

$(BOOKSIM_LIB): 
	@echo "Building Booksim2 library from makefile.lib..."
	$(MAKE) -C $(BOOKSIM_DIR) -f Makefile.lib clean  # 先清理
	$(MAKE) -C $(BOOKSIM_DIR) -f Makefile.lib        # 编译生成 libbooksim.a
	@if [ ! -f $(BOOKSIM_LIB) ]; then \
		echo "Error: BookSim library build failed!"; \
		exit 1; \
	fi
	@echo "BookSim library built successfully: $(BOOKSIM_LIB)"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

run: $(TARGET)
	@echo "Running NPU Simulator..."
	./$(TARGET) ./config/alexnet_conv3_cfg.txt

run_alexnet: $(TARGET)
	@echo "Running NPU Simulator with alexnet ..."
	./$(TARGET) ./config/alexnet_cfg.txt

booksim:
	$(MAKE) -C $(BOOKSIM_DIR) -f Makefile.lib clean
	$(MAKE) -C $(BOOKSIM_DIR) -f Makefile.lib

clean_booksim:
	$(MAKE) -C $(BOOKSIM_DIR) -f Makefile.lib clean

clean:
	rm -rf $(BUILD_DIR)

distclean: clean clean_booksim
	rm -f $(BOOKSIM_LIB)

.PHONY: all build run run_alexnet booksim clean clean_booksim distclean

help:
	@echo "Available targets:"
	@echo "  all                    - Build and run (default)"
	@echo "  build                  - Build only"
	@echo "  run                    - Run with example config"
	@echo "  run_alexnet  - Run with alexnet dual core config"
	@echo "  booksim                - Build BookSim library only"
	@echo "  clean                  - Clean all"
	@echo "  clean_booksim          - Clean BookSim only"
	@echo "  distclean              - Deep clean (remove library)"