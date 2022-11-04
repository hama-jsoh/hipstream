# python version
ARG PYTHON_MAJOR_VERSION=3
ARG PYTHON_MINOR_VERSION=10
ARG PYTHON_VERSION=${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}

# pytorch version
ARG TORCH_VERSION="v1.12.0"

# torchvision version
ARG TORCHVISION_VERSION="v0.13.0"

# tensorrt version
ARG TENSORRT_VERSION="8.4.1"

# deepstream version
ARG DEEPSTREAM_VERSION="6.0.1"
ARG PYDS_VERSION="v1.1.1"

# cuda architecture
# see https://en.wikipedia.org/wiki/CUDA
ARG CUDA_ARCH_LIST="7.5;8.6"

FROM nvcr.io/nvidia/deepstream:${DEEPSTREAM_VERSION}-devel as base

ARG PYTHON_VERSION
ARG CUDA_ARCH_LIST
ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i "s/archive.ubuntu/mirror.kakao/g" /etc/apt/sources.list \
 # Update gpg keys
 # see https://github.com/NVIDIA/nvidia-docker/issues/1631
 && rm /etc/apt/sources.list.d/cuda.list \
 && apt-key del 7fa2af80 \
 && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
 && dpkg -i cuda-keyring_1.0-1_all.deb \
 && apt-get update \
 # Install packages from apt
 && apt-get install -y --no-install-recommends \
      git \
      build-essential \
      pkg-config \
      software-properties-common \
      tzdata \
      locales \
      ffmpeg \
      graphviz \
      libcairo2-dev \
      libgirepository1.0-dev \
      libgstreamer1.0-dev \
      libgstreamer-plugins-base1.0-dev \
      libgstreamer-plugins-bad1.0-dev \
      gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly \
      gstreamer1.0-libav \
      gstreamer1.0-tools \
      libjpeg-turbo8-dev \
      libpng-dev \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt install -y --no-install-recommends \
      python${PYTHON_VERSION}-venv \
      python${PYTHON_VERSION}-dev \
 && apt-get upgrade -y \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf var/lib/apt/lists/*

# setup env
RUN locale-gen "en_US.UTF-8"
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    PATH="/venv/bin:${PATH}" \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}

# setup timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
 && dpkg-reconfigure --frontend noninteractive tzdata

# setup python venv
WORKDIR /
RUN python${PYTHON_VERSION} -m venv venv \
 && pip3 install --upgrade pip

FROM base as builder

ARG PYTHON_MAJOR_VERSION
ARG PYTHON_MINOR_VERSION
ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TENSORRT_VERSION
ARG DEEPSTREAM_VERSION
ARG PYDS_VERSION
ARG CUDA_ARCH_LIST

# Install mkl from intel repo
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
 && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
 && rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
 && echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends intel-mkl-64bit-2020.4-912

# Install compile-requirements
COPY requirements/compile.txt requirements/compile.txt
RUN pip3 install -r requirements/compile.txt

# Build pytorch from source
RUN git clone --recursive --branch ${TORCH_VERSION} \
    https://github.com/pytorch/pytorch \
 && cd pytorch \
 && python3 -m pip install -r requirements.txt \
 && MAX_JOBS="$(nproc)" USE_STATIC_MKL="1" \
    python3 setup.py install --verbose

# Build torchvision from source
RUN git clone --branch ${TORCHVISION_VERSION} https://github.com/pytorch/vision \
 && cd vision \
 && python3 setup.py install --verbose

# Build TensorRT python-bindings from source
RUN git clone --recursive --branch ${TENSORRT_VERSION} \
    https://github.com/NVIDIA/TensorRT \
 && mkdir -p /external \
 && cd /external \
 && git clone --branch v2.8.1 https://github.com/pybind/pybind11.git \
 && cp -r /usr/include/python${PYTHON_VERSION}/ /external/python${PYTHON_VERSION} \
 && cd /TensorRT/python \
 && EXT_PATH=/external \
    TRT_OSSPATH=/TensorRT \
    PYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION} \
    PYTHON_MINOR_VERSION=${PYTHON_MINOR_VERSION} \
    TARGET_ARCHITECTURE=x86_64 \
    bash build.sh \
 && cd build && pip3 install . --verbose

# Build Deepstream python-bindings from source
RUN git clone --recursive --branch ${PYDS_VERSION} \
    https://github.com/NVIDIA-AI-IOT/deepstream_python_apps \
 && sed -i "s/PYTHON_MINVERS_ALLOWED\s6\s8/PYTHON_MINVERS_ALLOWED 6 10/g" deepstream_python_apps/bindings/CMakeLists.txt \
 && cd deepstream_python_apps/bindings \
 && mkdir build && cd build \
 && cmake .. -DDS_VERSION=6.0.1 \
             -DPYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION} \
             -DPYTHON_MINOR_VERSION=${PYTHON_MINOR_VERSION} \
             -DPIP_PLATFORM=linux_x86_64 \
             -DDS_PATH=/opt/nvidia/deepstream/deepstream \
 && make -j$(nproc) \
 && pip3 install ./pyds*.whl

# Install runtime-requirements
COPY requirements/runtime.txt requirements/runtime.txt
RUN MAKEFLAGS="-j$(nproc)" \
    pip3 install -r requirements/runtime.txt

FROM base as runtime

# see https://github.com/dusty-nv/jetson-inference/issues/6
RUN ln -sf /usr/lib/x86_64-linux-gnu/glib-2.0/include/glibconfig.h /usr/include/glib-2.0/glibconfig.h

COPY --from=builder /venv/ /venv/

# Remove unnecessary files
# This layer will be squashed using `docker-squash`
RUN rm -rf /opt/nvidia/deepstream/deepstream/samples && rm -rf /usr/src/tensorrt/
