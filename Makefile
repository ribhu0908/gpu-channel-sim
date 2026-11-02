CC      ?= gcc
CFLAGS  ?= -std=c11 -O2 -g -Wall -Wextra -Werror -Wshadow -Wconversion -Wno-sign-conversion
INC     := -Iinclude
SRC     := src/gpu_sim.c
TEST    := tests/test_gpu.c
LDLIBS  := -lpthread
BUILD   := build

.PHONY: all test asan tsan portable demo clean

all: test

$(BUILD):
	mkdir -p $(BUILD)

test: | $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(SRC) $(TEST) -o $(BUILD)/tests $(LDLIBS)
	./$(BUILD)/tests

# AddressSanitizer + UndefinedBehaviorSanitizer
asan: | $(BUILD)
	$(CC) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
	      -fno-sanitize-recover=all $(INC) $(SRC) $(TEST) -o $(BUILD)/tests_asan $(LDLIBS)
	./$(BUILD)/tests_asan

# ThreadSanitizer: proves the engine / watchdog / host threads are race-free
tsan: | $(BUILD)
	$(CC) $(CFLAGS) -O1 -fsanitize=thread $(INC) $(SRC) $(TEST) -o $(BUILD)/tests_tsan $(LDLIBS)
	TSAN_OPTIONS=halt_on_error=1 ./$(BUILD)/tests_tsan

# Exercises the macOS code path (realtime-clock condvars) on Linux
portable: | $(BUILD)
	$(CC) $(CFLAGS) -DGPU_SIM_REALTIME_CONDS $(INC) $(SRC) $(TEST) -o $(BUILD)/tests_rt $(LDLIBS)
	./$(BUILD)/tests_rt

demo: | $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(SRC) examples/demo.c -o $(BUILD)/demo $(LDLIBS)
	./$(BUILD)/demo

clean:
	rm -rf $(BUILD)
