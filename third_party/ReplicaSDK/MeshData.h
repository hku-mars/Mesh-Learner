// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once

// #include <pangolin/image/managed_image.h>
#include <Eigen/Core>

typedef struct FaceInfo {
  std::vector<uint32_t> faceIdx;
  float areaSum;
} FaceInfo;

typedef struct TriangleInfo {
    Eigen::Vector3f vert[3]; // counter clock wise
    Eigen::Vector3f norm;
    uint32_t vertIdx[3];
} TriangleInfo;


struct MeshData {
  MeshData(size_t polygonStride = 3) : polygonStride(polygonStride) {}

  // MeshData(const MeshData& other) {
  //   if (other.vbo.IsValid())
  //     vbo.CopyFrom(other.vbo);

  //   if (other.ibo.IsValid())
  //     ibo.CopyFrom(other.ibo);

  //   if (other.nbo.IsValid())
  //     nbo.CopyFrom(other.nbo);

  //   if (other.cbo.IsValid())
  //     cbo.CopyFrom(other.cbo);

  //   polygonStride = other.polygonStride;
  // }

  // MeshData(MeshData&& other) {
  //   *this = std::move(other);
  // }

  // void operator=(MeshData&& other) {
  //   vbo = (std::move(other.vbo));
  //   ibo = (std::move(other.ibo));
  //   nbo = (std::move(other.nbo));
  //   cbo = (std::move(other.cbo));
  //   polygonStride = other.polygonStride;
  // }

  // pangolin::ManagedImage<Eigen::Vector4f> vbo;
  std::vector<float> m_vboVec;
  // pangolin::ManagedImage<Eigen::Vector4d> vboDouble;
  // pangolin::ManagedImage<uint32_t> ibo;
  // pangolin::ManagedImage<int32_t> iboInt32;
  std::vector<int32_t> m_iboVec;
  // pangolin::ManagedImage<Eigen::Vector4f> nbo;
  // pangolin::ManagedImage<Eigen::Vector3f> nbo;
  // pangolin::ManagedImage<Eigen::Vector4d> nboDouble;
  // pangolin::ManagedImage<Eigen::Matrix<unsigned char, 4, 1>> cbo;
  std::vector<unsigned char> m_cboVec;
  // pangolin::ManagedImage<Eigen::Vector3f> m_faceNormal;
  std::vector<float> m_faceNormalVec;
  std::vector<FaceInfo>m_ptIdx2FaceInfoHashMap;
  // pangolin::ManagedImage<float> m_faceArea;
  size_t polygonStride;
};
