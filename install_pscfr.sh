#!/bin/bash

# RCI cluster modules
ml Clang/12.0.1-GCCcore-10.3.0
ml CMake/3.20.1-GCCcore-10.3.0
ml Python/3.9.5-GCCcore-10.3.0
ml magma/2.6.1-foss-2021a-CUDA-11.3.1
ml SciPy-bundle/2021.05-foss-2021a
ml typing-extensions/3.10.0.0-GCCcore-10.3.0
ml protobuf-python/3.17.3-GCCcore-10.3.0
ml matplotlib/3.4.2-foss-2021a

OPEN_SPIEL_BUILD_WITH_ACPC=ON \
OPEN_SPIEL_BUILD_WITH_LIBNOP=ON \
OPEN_SPIEL_BUILD_WITH_PAPERS=ON \
OPEN_SPIEL_BUILD_WITH_PYTHON=ON \
./install.sh

