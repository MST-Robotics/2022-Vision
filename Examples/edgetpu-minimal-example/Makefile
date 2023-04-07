SOURCES=$(shell find ./ -type f -name '*.cc')
HEADERS=$(shell find ./ -type f -name '*.h')
OBJECTS=$(SOURCES:%.cc=%.o)
EXE=EDGETPU_TEST
DESTDIR?=/home/pi/

KERNEL_NAME=rpi_$(shell uname -m)
TENSORFLOW_PATH=../../Resources/tensorflow
EDGETPU_PATH=../../Resources/libedgetpu/tflite/public
LIBEDGETPU_PATH=/usr/lib/arm-linux-gnueabihf
TFDEPS_CFLAGS+=\
	-I$(EDGETPU_PATH) \
	-I$(TENSORFLOW_PATH) \
	-I$(TENSORFLOW_PATH)/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-L$(TENSORFLOW_PATH)/tensorflow/lite/tools/make/gen/${KERNEL_NAME}/lib \
	-L$(LIBEDGETPU_PATH) \
	-ltensorflow-lite -l:libedgetpu.so.1.0 -lpthread -lm -ldl

FLAGS?=-std=c++17 -Wno-psabi

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
	${CXX} -pthread -o $@ $^ ${TFDEPS_CFLAGS} ${FLAGS} -Wl,--unresolved-symbols=ignore-in-shared-libs

.cc.o:
	${CXX} -pthread -Og -c -o $@ ${FLAGS} ${DEPS_CFLAGS} ${TFDEPS_CFLAGS} $<

exportenv:
	-@export ASAN_SYMBOLIZER_PATH='which llvm-symbolizer'
	-@export ASAN_OPTIONS=symbolize=1
