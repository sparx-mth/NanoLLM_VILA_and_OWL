# Depth Anything 3 (TensorRT) + ROS2 Humble for Jetson Orin NX (JetPack 6 / L4T R36.x)
# l4t-jetpack ships CUDA, cuDNN, TensorRT and trtexec.
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive LANG=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        locales curl gnupg2 lsb-release git build-essential cmake \
    && locale-gen en_US.UTF-8 \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base ros-humble-cv-bridge ros-humble-image-transport \
        python3-colcon-common-extensions python3-rosdep libopencv-dev \
        ros-humble-tf2-ros ros-humble-tf2-sensor-msgs ros-humble-message-filters \
        ros-humble-geometry-msgs ros-humble-sensor-msgs ros-humble-pcl-conversions \
        libpcl-dev libyaml-cpp-dev nlohmann-json3-dev libcpp-httplib-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true

WORKDIR /depth_anything_ws
# Source is deployed with the repo (docker/da3/src, daphnaa fork @ 52131cc), no GitHub needed.
COPY src/ros2-depth-anything-v3-trt src/ros2-depth-anything-v3-trt

# TensorRT/CUDA are provided by the base image; skip them in rosdep.
RUN rosdep update && rosdep install --from-paths src -y --ignore-src \
        --skip-keys "tensorrt cuda libnvinfer" || true

# Upstream tests an undefined TENSORRT_VERSION_MAJOR, so it always takes the TRT<8 path
# (buildEngineWithConfig, removed in TRT 10). Use TensorRT's own NV_TENSORRT_MAJOR.
RUN sed -i 's/TENSORRT_VERSION_MAJOR/NV_TENSORRT_MAJOR/g' \
    src/ros2-depth-anything-v3-trt/depth_anything_v3/src/tensorrt_common/tensorrt_common.cpp

RUN . /opt/ros/humble/setup.sh && \
    # libnvinfer references DLA libs that the nvidia runtime only injects at `docker run`,
    # not at build time; allow those unresolved shlib symbols at link time.
    colcon build --packages-select depth_anything_v3 --cmake-args -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--allow-shlib-undefined"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
