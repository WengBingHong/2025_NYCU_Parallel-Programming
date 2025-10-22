# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O3 -mavx2 -Wall -Wextra -Werror -std=gnu11 -lpthread

# Target executable
TARGET = pi.out

# Source file
SRC = pi.c

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

# Run example
run: $(TARGET)
	@echo "Running with 4 threads and 100000000 tosses..."
	@./$(TARGET) 4 100000000

# Clean up
clean:
	rm -f $(TARGET)

