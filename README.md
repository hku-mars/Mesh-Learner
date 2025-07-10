# **[IROS 2025] Mesh-Learner: Texturing Mesh with Spherical Harmonics**

## 1. Introduction

## 2. Environment Setup
> 1 min setup: we offer the docker image ready for compilation and run
```bash
  docker pull alexwanwan/mesh_learner
  sudo docker run --ipc=host -p 10301:10400 -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:0 -e GDK_SCALE -e GDK_DPI_SCALE --name mesh_learner_docker --runtime=nvidia -v [your_path] --gpus all mesh_learner:1.0 /bin/bash
