QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

SOURCES += \
    network.cpp \
    train.cpp

HEADERS += \
    layer.h \
    loss.h \
    network.h

CUDA_SOURCES += \
    layer.cu \
    loss.cu


#================================================================
#CUDA COMPILER SETUP                            @ only debug mode
#================================================================

# Define output directories
CUDA_OBJECTS_DIR = .

# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG   = "/MDd"
#MSVCRT_LINK_FLAG_RELEASE = "/MD"

# CUDA settings
CUDA_DIR = $$(CUDA_PATH)            # Path to cuda toolkit install
SYSTEM_NAME = x64                   # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64                    # '32' or '64', depending on your system
CUDA_ARCH = all                     # Type of CUDA architecture
NVCC_OPTIONS = --use_fast_math

# include paths
INCLUDEPATH += $$CUDA_DIR/include

QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

CUDA_LIBS  = cublas cublasLt cuda cudadevrt cudart cudart_static cudnn cudnn64_8 cudnn_adv_infer \
             cudnn_adv_infer64_8 cudnn_adv_train cudnn_adv_train64_8 cudnn_cnn_infer cudnn_cnn_infer64_8 \
             cudnn_cnn_train cudnn_cnn_train64_8 cudnn_ops_infer cudnn_ops_infer64_8 cudnn_ops_train \
             cudnn_ops_train64_8 cufft cufftw cufilt curand cusolver cusolverMg cusparse nppc nppial \
             nppicc nppidei nppif nppig nppim nppist nppisu nppitc npps nvblas nvjpeg nvml \
             nvptxcompiler_static nvrtc-builtins_static nvrtc nvrtc_static OpenCL

for(lib, CUDA_LIBS) {
    CUDA_LIB += -l$$lib
}
LIBS += $$CUDA_LIB

# Debug mode
cuda.input = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj

cuda.commands = $$CUDA_DIR\bin\nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                      --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                      --compile -cudart static -g -DWIN32 -D_MBCS \
                      -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
                      -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
                      -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}

cuda.dependency_type = TYPE_C

QMAKE_EXTRA_COMPILERS += cuda


