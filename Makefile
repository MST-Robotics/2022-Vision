SOURCES=$(shell find ./ -type f -name '*.cpp')
HEADERS=$(shell find ./ -type f -name '*.h')
OBJECTS=$(SOURCES:%.cpp=%.o)
EXE=VISION
DESTDIR?=/home/pi/

DEPS_CFLAGS?=$(shell env PKG_CONFIG_PATH=/usr/local/frc/lib/pkgconfig pkg-config --cflags wpilibc)
FLAGS?=-std=c++17 -Wno-psabi
DEPS_LIBS?=$(shell env PKG_CONFIG_PATH=/usr/local/frc/lib/pkgconfig pkg-config --libs wpilibc)

all: ${EXE}

.PHONY: debug clean

debug: FLAGS=-std=c++17 -Wno-psabi -Wall -Wextra -g -fsanitize=address -fno-omit-frame-pointer
debug: exportenv
debug: clean
debug: ${EXE}

build: ${EXE}

install: build
	cp ${EXE} runCamera ${DESTDIR}

clean:
	-@rm -f ${EXE}
	-@rm -f ${OBJECTS}

depend: ${}

${EXE}: ${OBJECTS}
	${CXX} -pthread -o $@ $^ ${DEPS_LIBS} ${FLAGS} -Wl,--unresolved-symbols=ignore-in-shared-libs

.cpp.o:
	${CXX} -pthread -Og -c -o $@ ${FLAGS} ${DEPS_CFLAGS} $<

exportenv:
	-@export ASAN_SYMBOLIZER_PATH='which llvm-symbolizer'
	-@export ASAN_OPTIONS=symbolize=1
