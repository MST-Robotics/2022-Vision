SOURCES=$(shell find ./Code -type f -name '*.cpp')
HEADERS=$(shell find ./Code -type f -name '*.h')
OBJECTS=$(SOURCES:%.cpp=%.o)
EXE=VISION
DESTDIR?=/home/pi/

KERNEL_NAME=rpi_$(shell uname -m)
TENSORFLOW_PATH=./Resources/tensorflow
EDGETPU_PATH=./Resources/libedgetpu/tflite/public
LIBEDGETPU_PATH=/usr/lib/arm-linux-gnueabihf
TFDEPS_CFLAGS+=\
	-I$(EDGETPU_PATH) \
	-I$(TENSORFLOW_PATH) \
	-I$(TENSORFLOW_PATH)/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-L$(TENSORFLOW_PATH)/tensorflow/lite/tools/make/gen/${KERNEL_NAME}/lib \
	-L$(LIBEDGETPU_PATH) \
	-ltensorflow-lite -l:libedgetpu.so.1.0 -lpthread -lm -ldl

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
	${CXX} -pthread -o $@ $^ ${TFDEPS_CFLAGS} ${DEPS_LIBS} ${FLAGS} -Wl,--unresolved-symbols=ignore-in-shared-libs

.cpp.o:
	${CXX} -pthread -Og -c -o $@ ${FLAGS} ${DEPS_CFLAGS} ${TFDEPS_CFLAGS} $<

exportenv:
	-@export ASAN_SYMBOLIZER_PATH='which llvm-symbolizer'
	-@export ASAN_OPTIONS=symbolize=1
