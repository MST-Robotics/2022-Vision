SOURCES=$(shell find ./Code -type f -name '*.cpp')
HEADERS=$(shell find ./Code -type f -name '*.h')
OBJECTS=$(SOURCES:%.cpp=%.o)
EXE=VISION
CXX=aarch64-linux-gnu-g++
DESTDIR?=/home/pi/

KERNEL_NAME=linux_$(shell uname -m)
TENSORFLOW_PATH=./Resources/tensorflow
EDGETPU_PATH=./Resources/libedgetpu/tflite/public
LIBEDGETPU_PATH=/usr/lib/aarch64-linux-gnu
TFDEPS_CFLAGS+=\
	-I$(EDGETPU_PATH) \
	-I$(TENSORFLOW_PATH) \
	-I$(TENSORFLOW_PATH)/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-L$(TENSORFLOW_PATH)/tensorflow/lite/tools/make/gen/${KERNEL_NAME}/lib \
	-L$(LIBEDGETPU_PATH)
TFDEPS_LIBS+=-ltensorflow-lite -l:libedgetpu.so.1.0 -lpthread -lm -ldl

DEPS_CFLAGS?=-I./Resources/wpilib/include/ -I./Resources/wpilib/include/opencv4/ -L./Resources/wpilib/lib
FLAGS?=-std=c++20
DEPS_LIBS?=-Llib -lwpilibc -lwpiHal -lapriltag -lcameraserver -lntcore -lcscore -lopencv_gapi -lopencv_highgui -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video -lopencv_dnn -lopencv_videoio -lopencv_imgcodecs -lopencv_aruco -lopencv_calib3d -lopencv_features2d -lopencv_imgproc -lopencv_flann -lopencv_core -lwpimath -lwpinet -lwpiutil -latomic

all: ${EXE}

.PHONY: debug clean

debug: FLAGS=-std=c++20 -Wno-psabi -Wall -Wextra -g -fsanitize=address -fno-omit-frame-pointer
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
	${CXX} -pthread -o $@ $^ ${TFDEPS_CFLAGS} ${TFDEPS_LIBS} ${DEPS_CFLAGS} ${DEPS_LIBS} ${FLAGS} -Wl,--unresolved-symbols=ignore-in-shared-libs

.cpp.o:
	${CXX} -pthread -Og -c -o $@ ${DEPS_CFLAGS} ${TFDEPS_CFLAGS} ${FLAGS} $<

exportenv:
	-@export ASAN_SYMBOLIZER_PATH='which llvm-symbolizer'
	-@export ASAN_OPTIONS=symbolize=1
