CC=gcc
FLAGS=-std=c11 -Wall -Wextra -O3
LIBS=-march=native  # or -msse4

TARGET=sse

all: $(TARGET)

sse: sse.c
	$(CC) $(FLAGS) $(LIBS) -o $@ $^

clean:
	rm -f $(TARGET) *.o