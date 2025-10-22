// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "Assert.h"
#include "MeshData.h"

#include "oneapi/tbb/concurrent_hash_map.h"
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include <oneapi/tbb/concurrent_vector.h>

#include <cuco/static_map.cuh>
#include <cuco/static_multimap.cuh>
#include <cuco/static_set.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <cuda_runtime.h>

#include "../../include/my_common.hpp"
#include <glm/glm.hpp>

#define XSTR(x) #x
#define STR(x) XSTR(x)

using namespace oneapi::tbb;
namespace cg = cooperative_groups;

using my_atomic_ctr_type     = cuda::atomic<std::size_t, cuda::thread_scope_device>;

using my_counter_allocator_type = typename std::allocator_traits<cuco::cuda_allocator<char>>::rebind_alloc<my_atomic_ctr_type>;

struct my_counter_deleter {
    my_counter_deleter(my_counter_allocator_type& a) : allocator{a} {}

    my_counter_deleter(my_counter_deleter const&) = default;

    void operator()(my_atomic_ctr_type* ptr) { allocator.deallocate(ptr, 1); }

    my_counter_allocator_type& allocator;
  };

enum MeshType
{
    MESH_TYPE_IM_MESH = 0,
    MESH_TYPE_REPLICA,
    MESH_TYPE_BLENDER_OUT,
    MESH_TYPE_M2_MAPPING,
    MESH_TYPE_MAX,
};

struct EdgeKey {
  uint32_t m_v1;
  uint32_t m_v2;

  EdgeKey(uint32_t v1, uint32_t v2) {
    m_v1 = v1;
    m_v2 = v2;
  }

  bool operator==(const EdgeKey& other) const {
    return (m_v1 == other.m_v1 && m_v2 == other.m_v2) || (m_v1 == other.m_v2 && m_v2 == other.m_v1);
  }
};

// typedef struct EdgeKeyCuda {
//   int32_t m_e0{-1};
//   int32_t m_e1{-1};

//   __host__ __device__ EdgeKeyCuda(int32_t e0_, int32_t e1_): m_e0{e0_}, m_e1{e1_} {};
//   __host__ __device__ EdgeKeyCuda() {};

//   __device__ bool operator==(EdgeKeyCuda const& other) const
//   {
//     return (((m_e0 == other.m_e0) && (m_e1 == other.m_e1)) || ((m_e0 == other.m_e1) && (m_e1 == other.m_e0)));
//   }
// } EdgeKeyCuda;

// typedef struct EdgeKeyEqual {
//   __device__ bool operator()(EdgeKeyCuda const& lhs, EdgeKeyCuda const& rhs) const noexcept
//   {
//     return lhs == rhs;
//   }
// } EdgeKeyEqual;



typedef struct EdgeKeyEqualTest {
  __device__ bool operator()(EdgeKeyCuda const& lhs, EdgeKeyCuda const& rhs) const noexcept
  {
    return lhs.m_e0 == rhs.m_e1;
  }
} EdgeKeyEqualTest;


// typedef struct EdgeKeyCudaHasher {
//   cuco::default_hash_function<uint32_t> hasher;
//   __device__ uint32_t operator()(EdgeKeyCuda const& input) const noexcept
//   {
//     return hasher(input.m_e0) ^ hasher(input.m_e1);
//   }
// } EdgeKeyCudaHasher;


struct HashEdgeKey {
  size_t operator()(const EdgeKey& edge) const {
    return std::hash<unsigned int>()(edge.m_v1) ^ std::hash<unsigned int>()(edge.m_v2);
  }
};



template <typename Int32It, typename floatIt>
  __device__ void update_triangle_info_triangle(TriangleInfoDevice* triangleInfoDeviceInput,
    Int32It iboDeviceFirst,
    floatIt vboDeviceFirst,
    floatIt faceNormalDeviceFirst,
    uint32_t tid)
  {
      TriangleInfoDevice tmp1;

      tmp1.vert[0] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 3 + 0)) * 3 + 0);
      tmp1.vert[1] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 3 + 0)) * 3 + 1);
      tmp1.vert[2] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 3 + 0)) * 3 + 2);

      tmp1.vert[3] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 3 + 1)) * 3 + 0);
      tmp1.vert[4] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 3 + 1)) * 3 + 1);
      tmp1.vert[5] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 3 + 1)) * 3 + 2);

      tmp1.vert[6] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 3 + 2)) * 3 + 0);
      tmp1.vert[7] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 3 + 2)) * 3 + 1);
      tmp1.vert[8] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 3 + 2)) * 3 + 2);


      tmp1.norm[0] = *(faceNormalDeviceFirst + tid * 3 + 0);
      tmp1.norm[1] = *(faceNormalDeviceFirst + tid * 3 + 1);
      tmp1.norm[2] = *(faceNormalDeviceFirst + tid * 3 + 2);

      tmp1.vertIdx[0] = (uint32_t)(*(iboDeviceFirst + tid * 3 + 0));
      tmp1.vertIdx[1] = (uint32_t)(*(iboDeviceFirst + tid * 3 + 1));
      tmp1.vertIdx[2] = (uint32_t)(*(iboDeviceFirst + tid * 3 + 2));

      triangleInfoDeviceInput[tid] = tmp1;
  }


  template <typename Int32It, typename floatIt>
  __device__ void update_triangle_info_quad(TriangleInfoDevice* triangleInfoDeviceInput,
    Int32It iboDeviceFirst,
    floatIt vboDeviceFirst,
    floatIt faceNormalDeviceFirst,
    uint32_t tid)
  {
      TriangleInfoDevice tmp1;
      TriangleInfoDevice tmp2;

      tmp1.vert[0] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 0)) * 3 + 0);
      tmp1.vert[1] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 0)) * 3 + 1);
      tmp1.vert[2] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 0)) * 3 + 2);

      tmp1.vert[3] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 1)) * 3 + 0);
      tmp1.vert[4] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 1)) * 3 + 1);
      tmp1.vert[5] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 1)) * 3 + 2);

      tmp1.vert[6] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 2)) * 3 + 0);
      tmp1.vert[7] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 2)) * 3 + 1);
      tmp1.vert[8] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 2)) * 3 + 2);



      // tmp1.vert[0] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 0];
      // tmp1.vert[1] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 1];
      // tmp1.vert[2] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 2];

      // tmp1.vert[3] = vboDevice[iboDevice[tid * 4 + 1] * 3 + 0];
      // tmp1.vert[4] = vboDevice[iboDevice[tid * 4 + 1] * 3 + 1];
      // tmp1.vert[5] = vboDevice[iboDevice[tid * 4 + 1] * 3 + 2];

      // tmp1.vert[6] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 0];
      // tmp1.vert[7] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 1];
      // tmp1.vert[8] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 2];

      tmp1.norm[0] = *(faceNormalDeviceFirst + tid * 2 * 3 + 0);
      tmp1.norm[1] = *(faceNormalDeviceFirst + tid * 2 * 3 + 1);
      tmp1.norm[2] = *(faceNormalDeviceFirst + tid * 2 * 3 + 2);

      // tmp1.norm[0] = faceNormalDevice[tid * 2 * 3 + 0];
      // tmp1.norm[1] = faceNormalDevice[tid * 2 * 3 + 1];
      // tmp1.norm[2] = faceNormalDevice[tid * 2 * 3 + 2];

      tmp1.vertIdx[0] = (uint32_t)(*(iboDeviceFirst + tid * 4 + 0));
      tmp1.vertIdx[1] = (uint32_t)(*(iboDeviceFirst + tid * 4 + 1));
      tmp1.vertIdx[2] = (uint32_t)(*(iboDeviceFirst + tid * 4 + 2));

      // tmp1.vertIdx[0] = iboDevice[tid * 4 + 0];
      // tmp1.vertIdx[1] = iboDevice[tid * 4 + 1];
      // tmp1.vertIdx[2] = iboDevice[tid * 4 + 2];


      tmp2.vert[0] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 0)) * 3 + 0);
      tmp2.vert[1] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 0)) * 3 + 1);
      tmp2.vert[2] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 0)) * 3 + 2);

      // tmp2.vert[0] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 0];
      // tmp2.vert[1] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 1];
      // tmp2.vert[2] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 2];

      tmp2.vert[3] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 2)) * 3 + 0);
      tmp2.vert[4] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 2)) * 3 + 1);
      tmp2.vert[5] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 2)) * 3 + 2);

      // tmp2.vert[3] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 0];
      // tmp2.vert[4] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 1];
      // tmp2.vert[5] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 2];

      tmp2.vert[6] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 3)) * 3 + 0);
      tmp2.vert[7] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 3)) * 3 + 1);
      tmp2.vert[8] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 3)) * 3 + 2);

      // tmp2.vert[6] = vboDevice[iboDevice[tid * 4 + 3] * 3 + 0];
      // tmp2.vert[7] = vboDevice[iboDevice[tid * 4 + 3] * 3 + 1];
      // tmp2.vert[8] = vboDevice[iboDevice[tid * 4 + 3] * 3 + 2];


      tmp2.norm[0] = *(faceNormalDeviceFirst + tid * 2 * 3 + 3);
      tmp2.norm[1] = *(faceNormalDeviceFirst + tid * 2 * 3 + 4);
      tmp2.norm[2] = *(faceNormalDeviceFirst + tid * 2 * 3 + 5);

      // tmp2.norm[0] = faceNormalDevice[tid * 2 * 3 + 3];
      // tmp2.norm[1] = faceNormalDevice[tid * 2 * 3 + 4];
      // tmp2.norm[2] = faceNormalDevice[tid * 2 * 3 + 5];

      tmp2.vertIdx[0] = (uint32_t)(*(iboDeviceFirst + tid * 4 + 0));
      tmp2.vertIdx[1] = (uint32_t)(*(iboDeviceFirst + tid * 4 + 2));
      tmp2.vertIdx[2] = (uint32_t)(*(iboDeviceFirst + tid * 4 + 3));

      // tmp2.vertIdx[0] = iboDevice[tid * 4 + 0];
      // tmp2.vertIdx[1] = iboDevice[tid * 4 + 2];
      // tmp2.vertIdx[2] = iboDevice[tid * 4 + 3];


      triangleInfoDeviceInput[tid * 2 + 0] = tmp1;
      triangleInfoDeviceInput[tid * 2 + 1] = tmp2;
  }


  template <typename vertMap, typename tileType, typename TriArrIt>
  __device__ void update_vert_map_quad(vertMap& vertMapInsertView, tileType& tile, TriArrIt& triArrFirst, uint32_t tid)
  {
    vertMapInsertView.insert(tile, cuco::pair{(int32_t)((thrust::raw_reference_cast(*(triArrFirst + tid))).vertIdx[0]), (int32_t)(tid)});
    vertMapInsertView.insert(tile, cuco::pair{(int32_t)((thrust::raw_reference_cast(*(triArrFirst + tid))).vertIdx[1]), (int32_t)(tid)});
    vertMapInsertView.insert(tile, cuco::pair{(int32_t)((thrust::raw_reference_cast(*(triArrFirst + tid))).vertIdx[2]), (int32_t)(tid)});

    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 4 + 0)), (int32_t)(tid * 2 + 0)});
    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 4 + 0)), (int32_t)(tid * 2 + 1)});

    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 4 + 1)), (int32_t)(tid * 2 + 0)});

    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 4 + 2)), (int32_t)(tid * 2 + 0)});
    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 4 + 2)), (int32_t)(tid * 2 + 1)});

    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 4 + 3)), (int32_t)(tid * 2 + 1)});

    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 0]), (int32_t)(tid * 2 + 0)});
    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 0]), (int32_t)(tid * 2 + 1)});

    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 1]), (int32_t)(tid * 2 + 0)});

    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 2]), (int32_t)(tid * 2 + 0)});
    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 2]), (int32_t)(tid * 2 + 1)});

    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 3]), (int32_t)(tid * 2 + 1)});
  }

  
  template <typename vertMap, typename tileType, typename InputIt>
  __device__ void update_vert_map_triangle(vertMap& vertMapInsertView, tileType& tile, InputIt& iboDeviceFirst, uint32_t tid)
  {
    vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 3 + 0)), (int32_t)(tid)});
    vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 3 + 1)), (int32_t)(tid)});
    vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 3 + 2)), (int32_t)(tid)});
  }



  template <typename TriArrIt>
  __device__ uint8_t FindLongestEdgeCudaEdgeMap(uint32_t tid, TriArrIt triArrBegin)
  {
      float epsilon = 1e-6;
    
      glm::vec3 pt0((thrust::raw_reference_cast(*(triArrBegin + tid))).vert[0],
                    (thrust::raw_reference_cast(*(triArrBegin + tid))).vert[1],
                    (thrust::raw_reference_cast(*(triArrBegin + tid))).vert[2]);

      glm::vec3 pt1((thrust::raw_reference_cast(*(triArrBegin + tid))).vert[3],
                    (thrust::raw_reference_cast(*(triArrBegin + tid))).vert[4],
                    (thrust::raw_reference_cast(*(triArrBegin + tid))).vert[5]);

      glm::vec3 pt2((thrust::raw_reference_cast(*(triArrBegin + tid))).vert[6],
                    (thrust::raw_reference_cast(*(triArrBegin + tid))).vert[7],
                    (thrust::raw_reference_cast(*(triArrBegin + tid))).vert[8]);

      glm::vec3 dist01 = pt0 - pt1;
      glm::vec3 dist12 = pt1 - pt2;
      glm::vec3 dist02 = pt0 - pt2;

      if (fabsf(glm::length(dist01) - glm::length(dist12)) <= epsilon &&
          fabsf(glm::length(dist12) - glm::length(dist02)) <= epsilon) {
          return 0;
      } else if ((glm::length(dist01) - glm::length(dist12)) > epsilon &&
                (glm::length(dist01) - glm::length(dist02)) > epsilon) {
          return 2; 
      } else if ((glm::length(dist12) - glm::length(dist01)) > epsilon &&
                (glm::length(dist12) - glm::length(dist02)) > epsilon) {
          return 0;
      } else {
          return 1;
      }
  }

  template <typename edgeMap, typename tileType, typename SetRef, typename TriArrIt>
  __device__ void update_edge_map_quad(tileType& tile, edgeMap& edgeMapInsertView, SetRef& edgeKeysSet, uint32_t tid, TriArrIt triArrBegin)
  {
    uint8_t topPtIdx = FindLongestEdgeCudaEdgeMap(tid, triArrBegin);

    auto edgeLeft = EdgeKeyCuda((thrust::raw_reference_cast(*(triArrBegin + tid))).vertIdx[topPtIdx],
                             (thrust::raw_reference_cast(*(triArrBegin + tid))).vertIdx[(topPtIdx + 1) % 3]);
    // auto edge01 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 0]), (uint32_t)(iboDevice[tid * 4 + 1]));
    edgeMapInsertView.insert(tile, cuco::pair{edgeLeft, ConstructValueMapKey(tid, EDGE_TYPE_CUDA_LEFT)});

    auto edgeBottom = EdgeKeyCuda((thrust::raw_reference_cast(*(triArrBegin + tid))).vertIdx[(topPtIdx + 1) % 3],
                                  (thrust::raw_reference_cast(*(triArrBegin + tid))).vertIdx[(topPtIdx + 2) % 3]);
    // auto edge12 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 1]), (uint32_t)(iboDevice[tid * 4 + 2]));
    edgeMapInsertView.insert(tile, cuco::pair{edgeBottom, ConstructValueMapKey(tid, EDGE_TYPE_CUDA_BOTTOM)});

    auto edgeRight = EdgeKeyCuda((thrust::raw_reference_cast(*(triArrBegin + tid))).vertIdx[(topPtIdx + 2) % 3],
                                 (thrust::raw_reference_cast(*(triArrBegin + tid))).vertIdx[topPtIdx]);
    // auto edge02 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 0]), (uint32_t)(iboDevice[tid * 4 + 2]));
    edgeMapInsertView.insert(tile, cuco::pair{edgeRight, ConstructValueMapKey(tid, EDGE_TYPE_CUDA_RIGHT)});

    // //triangle 2
    // edgeMapInsertView.insert(tile, cuco::pair{edge02, (int32_t)(tid * 2 + 1)});

    // auto edge23 = EdgeKeyCuda((*(iboVecFirst + tid * 4 + 2)), (*(iboVecFirst + tid * 4 + 3)));
    // // auto edge23 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 2]), (uint32_t)(iboDevice[tid * 4 + 3]));
    // edgeMapInsertView.insert(tile, cuco::pair{edge23, (int32_t)(tid * 2 + 1)});

    // auto edge03 = EdgeKeyCuda((*(iboVecFirst + tid * 4 + 0)), (*(iboVecFirst + tid * 4 + 3)));
    // // auto edge03 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 0]), (uint32_t)(iboDevice[tid * 4 + 3]));
    // edgeMapInsertView.insert(tile, cuco::pair{edge03, (int32_t)(tid * 2 + 1)});

    edgeKeysSet.insert(tile, edgeLeft);
    edgeKeysSet.insert(tile, edgeBottom);
    edgeKeysSet.insert(tile, edgeRight);
    // edgeKeysSet.insert(tile, edge23);
    // edgeKeysSet.insert(tile, edge03);
  }

// __device__ int32_t ConstructValueMapKey(uint32_t tid, EdgeTypeEdgeMap edgeType)
// {
//   int32_t high30bits = (tid << 2);

//   return (high30bits | edgeType);
// }

template <typename edgeMap, typename tileType, typename SetRef, typename InputIt, typename TriArrIt>
  __device__ void update_edge_map_triangle(InputIt& iboVecFirst, tileType& tile, edgeMap& edgeMapInsertView, SetRef& edgeKeysSet, uint32_t tid, TriArrIt triArrBegin)
  {
    uint8_t topPtIdx = FindLongestEdgeCudaEdgeMap(tid, triArrBegin);

    auto edgeLeft = EdgeKeyCuda((*(iboVecFirst + tid * 3 + (topPtIdx))), (*(iboVecFirst + tid * 3 + (topPtIdx + 1) % 3)));
    edgeMapInsertView.insert(tile, cuco::pair{edgeLeft, ConstructValueMapKey(tid, EDGE_TYPE_CUDA_LEFT)});

    auto edgeBottom = EdgeKeyCuda((*(iboVecFirst + tid * 3 + (topPtIdx + 1) % 3)), (*(iboVecFirst + tid * 3 + (topPtIdx + 2) % 3)));
    edgeMapInsertView.insert(tile, cuco::pair{edgeBottom, ConstructValueMapKey(tid, EDGE_TYPE_CUDA_BOTTOM)});

    auto edgeRight = EdgeKeyCuda((*(iboVecFirst + tid * 3 + (topPtIdx + 2) % 3)), (*(iboVecFirst + tid * 3 + (topPtIdx))));
    edgeMapInsertView.insert(tile, cuco::pair{edgeRight, ConstructValueMapKey(tid, EDGE_TYPE_CUDA_RIGHT)});

    edgeKeysSet.insert(tile, edgeLeft);
    edgeKeysSet.insert(tile, edgeBottom);
    edgeKeysSet.insert(tile, edgeRight);

    // auto edge01 = EdgeKeyCuda((*(iboVecFirst + tid * 3 + 0)), (*(iboVecFirst + tid * 3 + 1)));
    // edgeMapInsertView.insert(tile, cuco::pair{edge01, (int32_t)(tid)});

    // auto edge12 = EdgeKeyCuda((*(iboVecFirst + tid * 3 + 1)), (*(iboVecFirst + tid * 3 + 2)));
    // edgeMapInsertView.insert(tile, cuco::pair{edge12, (int32_t)(tid)});

    // auto edge02 = EdgeKeyCuda((*(iboVecFirst + tid * 3 + 0)), (*(iboVecFirst + tid * 3 + 2)));
    // edgeMapInsertView.insert(tile, cuco::pair{edge02, (int32_t)(tid)});

    // edgeKeysSet.insert(tile, edge01);
    // edgeKeysSet.insert(tile, edge12);
    // edgeKeysSet.insert(tile, edge02);
  }

  template <typename Int32It, typename floatIt>
  __global__ void insert_triangle_info_quad(Int32It iboDeviceFirst,
    TriangleInfoDevice* triangleInfoDeviceInput,
    floatIt faceNormalDeviceFirst,
    floatIt vboDeviceFirst,
    uint32_t numFaces)
  {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < numFaces) {
      update_triangle_info_quad(triangleInfoDeviceInput, iboDeviceFirst, vboDeviceFirst, faceNormalDeviceFirst, tid);
      tid += gridDim.x * blockDim.x;
    }
  }

  template <typename Int32It, typename floatIt>
  __global__ void insert_triangle_info_triangle(Int32It iboDeviceFirst,
    TriangleInfoDevice* triangleInfoDeviceInput,
    floatIt faceNormalDeviceFirst,
    floatIt vboDeviceFirst,
    uint32_t numFaces)
  {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < numFaces) {
      update_triangle_info_triangle(triangleInfoDeviceInput, iboDeviceFirst, vboDeviceFirst, faceNormalDeviceFirst, tid);
      tid += gridDim.x * blockDim.x;
    }
  }

template <uint32_t tile_size, typename edgeMap, typename SetRef, typename TriArrIt>
__global__ void insert_edge_map_quad(edgeMap edgeMapInsertView,
  SetRef edgeKeysSet,
  uint32_t numFaces,
  TriArrIt triArrBegin)
{
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  // std::size_t counter = 0;
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());

  while (tid < numFaces) {
    update_edge_map_quad(tile, edgeMapInsertView, edgeKeysSet, tid, triArrBegin);
    tid += gridDim.x * blockDim.x;
  }
}

template <uint32_t tile_size, typename edgeMap, typename SetRef, typename InputIt, typename TriArrIt>
__global__ void insert_edge_map_triangle(edgeMap edgeMapInsertView,
  SetRef edgeKeysSet,
  InputIt iboVecFirst,
  uint32_t numFaces,
  TriArrIt triArrBegin)
{
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  // std::size_t counter = 0;
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());

  while (tid < numFaces) {
    update_edge_map_triangle(iboVecFirst, tile, edgeMapInsertView, edgeKeysSet, tid, triArrBegin);
    tid += gridDim.x * blockDim.x;
  }
}


  template <uint32_t tile_size, typename vertMap, typename TriArrIt>
  __global__ void insert_vert_map_quad(vertMap vertMapInsertView, TriArrIt triArrFirst, uint32_t numFaces)
  {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    // std::size_t counter = 0;
    auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());

    while (tid < numFaces) {
      update_vert_map_quad(vertMapInsertView, tile, triArrFirst, tid);
      tid += gridDim.x * blockDim.x;
    }
  }

  template <uint32_t tile_size, typename vertMap, typename InputIt>
  __global__ void insert_vert_map_triangle(vertMap vertMapInsertView, InputIt iboDeviceFirst, uint32_t numFaces)
  {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    // std::size_t counter = 0;
    auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());

    while (tid < numFaces) {
      update_vert_map_triangle(vertMapInsertView, tile, iboDeviceFirst, tid);
      tid += gridDim.x * blockDim.x;
    }
  }


template <uint32_t tile_size, typename EdgeKeyIter, typename EdgeMapViewType, typename KeyEqual>
__global__ void validate_edge_map_kernel(EdgeKeyIter first, uint32_t keyNum, EdgeMapViewType edgeMap, KeyEqual equalFunc, uint32_t* zeroNbrCnt, uint32_t* oneNbrCnt, uint32_t* twoNbrCnt, uint32_t* moreThanTwoNbrCnt, uint32_t* maxEdgeNbrNum)
{
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  std::size_t thread_num_matches = 0;

  if (tid < keyNum) {
    auto key = *(first + tid);

    thread_num_matches = edgeMap.count(tile, key, equalFunc);

    if (thread_num_matches == 0) {
      atomicAdd(zeroNbrCnt, 1);
      // printf("edge cannot be find: %d, %d\n", static_cast<EdgeKeyCuda>(*(first + threadIdx.x + blockIdx.x * blockDim.x)).m_e0, static_cast<EdgeKeyCuda>(*(first + threadIdx.x + blockIdx.x * blockDim.x)).m_e1);
    }

    if (thread_num_matches == 1) {
      atomicAdd(oneNbrCnt, 1);
    }

    if (thread_num_matches == 2) {
      atomicAdd(twoNbrCnt, 1);
    }

    if (thread_num_matches > 2) {
      atomicAdd(moreThanTwoNbrCnt, 1);
    }

    atomicMax(maxEdgeNbrNum, thread_num_matches);

    // tid += gridDim.x * blockDim.x;
  }

 
}


template <uint32_t tile_size, typename VertKeyIter, typename VertMapViewType>
__global__ void validate_vert_map_kernel(VertKeyIter iboDeviceFirst, uint32_t keyNum, VertMapViewType vertMap, uint32_t* maxVertNbrNum, uint32_t* cannotFindNum)
{
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  std::size_t thread_num_matches = 0;

  if (tid < keyNum) {
    auto key = *(iboDeviceFirst + tid);

    thread_num_matches = vertMap.count(tile, key);
    atomicMax(maxVertNbrNum, thread_num_matches);

    if (thread_num_matches == 0) {
      atomicAdd(cannotFindNum, 1);
    }

    // no loop stride
    // tid += gridDim.x * blockDim.x;
  }

}

// template<typename CtrIt, typename RawCtrIt>
// __global__ void read_atomic_values(CtrIt cntArrBegin, RawCtrIt rawArrBegin, uint32_t numOfCtr)
// {
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;

//   if (idx < numOfCtr) {
//     *(rawArrBegin + idx) = (*(cntArrBegin + idx)).load(cuda::memory_order_relaxed);
//   }
// }

__global__ void my_initialize_atomic(my_atomic_ctr_type* atms, uint32_t numOfAtms);
// {
//   auto tid = threadIdx.x + blockIdx.x * blockDim.x;

//   if (tid < numOfAtms) {
//     atms[tid].store(0, cuda::memory_order_relaxed);
//   }
// }

template<uint32_t block_size,
         uint32_t flushing_cg_size,
         uint32_t probing_cg_size,
         uint32_t buffer_size,
         typename InputIt,
         typename OutputIt,
         typename CtrPtr,
         typename viewT,
         typename KeyEqual>
__global__ void custom_retrieve(InputIt first,
                                int64_t keyNum,
                                OutputIt outBufArrBegin,
                                CtrPtr* cntArrBegin,
                                viewT view,
                                KeyEqual equalFunc,
                                uint32_t maxNbrCnt)
{
  using pair_type = typename viewT::value_type;
  constexpr uint32_t num_flushing_cgs = block_size / flushing_cg_size;
  const uint32_t flushing_cg_id       = threadIdx.x / flushing_cg_size;

  auto flushing_cg          = cg::tiled_partition<flushing_cg_size>(cg::this_thread_block());
  auto probing_cg           = cg::tiled_partition<probing_cg_size>(cg::this_thread_block());
  // int64_t const loop_stride = gridDim.x * block_size / probing_cg_size;
  int64_t idx               = (block_size * blockIdx.x + threadIdx.x) / probing_cg_size;

  __shared__ pair_type output_buffer[num_flushing_cgs][buffer_size];
  // TODO: replace this with shared memory cuda::atomic variables once the dynamiic initialization
  // warning issue is solved __shared__ atomicT counter[num_flushing_cgs][buffer_size];
  __shared__ uint32_t flushing_cg_counter[num_flushing_cgs];

  if (flushing_cg.thread_rank() == 0) { flushing_cg_counter[flushing_cg_id] = 0; }

  if (idx < keyNum) {
      auto key = *(first + idx);
      auto outBufferIter = outBufArrBegin + idx * maxNbrCnt;
      // auto curKeyAtomCnter = thrust::raw_pointer_cast(&(*(cntArrBegin + idx)));
      auto curKeyAtomCnter = (cntArrBegin + idx);

      auto active_flushing_cg = cg::binary_partition<flushing_cg_size>(flushing_cg, 1);

      view.retrieve<buffer_size>(active_flushing_cg,
                                   probing_cg,
                                   key,
                                   &flushing_cg_counter[flushing_cg_id],
                                   output_buffer[flushing_cg_id],
                                   curKeyAtomCnter,
                                   outBufferIter,
                                   equalFunc);

      if (flushing_cg_counter > 0) {
        view.flush_output_buffer(flushing_cg,
                                flushing_cg_counter[flushing_cg_id],
                                output_buffer[flushing_cg_id],
                                curKeyAtomCnter,
                                outBufferIter);
      }
  } 
}



class PTexMesh {
 public:
  PTexMesh() {};

  ~PTexMesh() {};

  using edgeMapKeyType   = EdgeKeyCuda;
  using edgeMapValueType = int;

  using vertMapKeyType   = int;
  using vertMapValueType = int;

  // using my_atomic_ctr_type     = cuda::atomic<std::size_t, cuda::thread_scope_device>;
  // using my_counter_allocator_type = typename std::allocator_traits<cuco::cuda_allocator<char>>::rebind_alloc<my_atomic_ctr_type>;

  // struct my_counter_deleter {
  //   my_counter_deleter(my_counter_allocator_type& a) : allocator{a} {}

  //   my_counter_deleter(my_counter_deleter const&) = default;

  //   void operator()(my_atomic_ctr_type* ptr) { allocator.deallocate(ptr, 1); }

  //   my_counter_allocator_type& allocator;
  // };

  struct EdgeHashCompareTBB {
    size_t hash(const EdgeKey& edge) const
    {
      size_t h = 0;
      h = std::hash<unsigned int>()(edge.m_v1) ^ std::hash<unsigned int>()(edge.m_v2);
      return h;
    }

    bool equal(const EdgeKey& e1, const EdgeKey& e2) const
    {
      return (((e1.m_v1 == e2.m_v1) && (e1.m_v2 == e2.m_v2)) || ((e1.m_v1 == e2.m_v2) && (e1.m_v2 == e2.m_v1)));
    }
  };

  typedef concurrent_hash_map<EdgeKey, concurrent_vector<int>, EdgeHashCompareTBB> EdgeMapConcurrent;
  typedef concurrent_hash_map<uint32_t, concurrent_vector<uint32_t>> VertNeighborMapConcurrent;

    // key: edge, value: triangle idx
  std::unordered_map<EdgeKey, std::vector<int>, HashEdgeKey> m_edgeMap; //save triangles that have a specific edge
  std::unordered_map<uint32_t, std::vector<uint32_t>> m_vertNeighborMap; // index in ibo, value is triangle id (start from 0)

  EdgeMapConcurrent m_edgeMapConcurrent;
  VertNeighborMapConcurrent m_vertNbrMapConcurrent;

  struct ApplyCalcEachTriangleJianHengLiu
  {
    EdgeMapConcurrent& m_edgeMap;
    VertNeighborMapConcurrent& m_vertNbrMap;
    MeshData& m_originalMesh;
    std::vector<TriangleInfo>& m_triangleArr;
    std::vector<uint32_t>& m_reduceCnt;

    ApplyCalcEachTriangleJianHengLiu(EdgeMapConcurrent& edgeMap_,
                                    VertNeighborMapConcurrent& vertNbrMap_,
                                    MeshData& originalMesh_,
                                    std::vector<TriangleInfo>& triangleArr_,
                                    std::vector<uint32_t>& reduceCnt_) : m_edgeMap(edgeMap_), m_vertNbrMap(vertNbrMap_), m_originalMesh(originalMesh_), m_triangleArr(triangleArr_), m_reduceCnt(reduceCnt_) {}

    template<typename Map, typename Key, typename Val>
    uint32_t MapInsertPushBack(Map& hashMap, Key key, Val data) const
    {
      typename Map::accessor access;
      uint32_t ret = (uint32_t)(hashMap.insert(access, key));
      access->second.push_back(data);
      access.release();
      return ret;
    }

    void operator()(const blocked_range<uint32_t> range) const
    {
      // for (uint32_t i = range.begin(); i != range.end(); i++) {
      //   TriangleInfo oneTriangle1;
      //   uint32_t errCnt = 0;

      //   oneTriangle1.vert[0] = m_originalMesh.vbo[m_originalMesh.iboInt32[i * 3 + 0]].head(3);
      //   oneTriangle1.vert[1] = m_originalMesh.vbo[m_originalMesh.iboInt32[i * 3 + 1]].head(3);
      //   oneTriangle1.vert[2] = m_originalMesh.vbo[m_originalMesh.iboInt32[i * 3 + 2]].head(3);

      //   oneTriangle1.norm = m_originalMesh.m_faceNormal[i];

      //   oneTriangle1.vertIdx[0] = m_originalMesh.iboInt32[i * 3 + 0];
      //   oneTriangle1.vertIdx[1] = m_originalMesh.iboInt32[i * 3 + 1];
      //   oneTriangle1.vertIdx[2] = m_originalMesh.iboInt32[i * 3 + 2];

      //   m_triangleArr[i] = oneTriangle1;

      //   EdgeKey e01(m_originalMesh.iboInt32[i * 3 + 0], m_originalMesh.iboInt32[i * 3 + 1]);
      //   errCnt += MapInsertPushBack(m_edgeMap, e01, i);
        
      //   EdgeKey e12(m_originalMesh.iboInt32[i * 3 + 1], m_originalMesh.iboInt32[i * 3 + 2]);
      //   errCnt += MapInsertPushBack(m_edgeMap, e12, i);

      //   EdgeKey e02(m_originalMesh.iboInt32[i * 3 + 0], m_originalMesh.iboInt32[i * 3 + 2]);
      //   errCnt += MapInsertPushBack(m_edgeMap, e02, i);

      //   errCnt += MapInsertPushBack(m_vertNbrMap, m_originalMesh.iboInt32[i * 3 + 0], i);
      //   errCnt += MapInsertPushBack(m_vertNbrMap, m_originalMesh.iboInt32[i * 3 + 1], i);
      //   errCnt += MapInsertPushBack(m_vertNbrMap, m_originalMesh.iboInt32[i * 3 + 2], i);

      //   m_reduceCnt[i] = errCnt;
      // }
    }


  };

  // struct ApplyCalcEachTriangleBlenderOut
  // {
  //   EdgeMapConcurrent& m_edgeMap;
  //   VertNeighborMapConcurrent& m_vertNbrMap;
  //   MeshData& m_originalMesh;
  //   std::vector<TriangleInfo>& m_triangleArr;
  //   std::vector<uint32_t>& m_reduceCnt;

  //   ApplyCalcEachTriangleBlenderOut(EdgeMapConcurrent& edgeMap_,
  //                                   VertNeighborMapConcurrent& vertNbrMap_,
  //                                   MeshData& originalMesh_,
  //                                   std::vector<TriangleInfo>& triangleArr_,
  //                                   std::vector<uint32_t>& reduceCnt_) : m_edgeMap(edgeMap_), m_vertNbrMap(vertNbrMap_), m_originalMesh(originalMesh_), m_triangleArr(triangleArr_), m_reduceCnt(reduceCnt_) {}

  //   template<typename Map, typename Key, typename Val>
  //   uint32_t MapInsertPushBack(Map& hashMap, Key key, Val data) const
  //   {
  //     typename Map::accessor access;
  //     uint32_t ret = (uint32_t)(hashMap.insert(access, key));
  //     access->second.push_back(data);
  //     access.release();
  //     return ret;
  //   }

  //   void operator()(const blocked_range<uint32_t> range) const
  //   {
  //     for (uint32_t i = range.begin(); i != range.end(); i++) {
  //       TriangleInfo oneTriangle1;
  //       uint32_t errCnt = 0;

  //       oneTriangle1.vert[0] = m_originalMesh.vbo[m_originalMesh.ibo[i * 3 + 0]].head(3);
  //       oneTriangle1.vert[1] = m_originalMesh.vbo[m_originalMesh.ibo[i * 3 + 1]].head(3);
  //       oneTriangle1.vert[2] = m_originalMesh.vbo[m_originalMesh.ibo[i * 3 + 2]].head(3);

  //       oneTriangle1.norm = m_originalMesh.m_faceNormal[i];

  //       oneTriangle1.vertIdx[0] = m_originalMesh.ibo[i * 3 + 0];
  //       oneTriangle1.vertIdx[1] = m_originalMesh.ibo[i * 3 + 1];
  //       oneTriangle1.vertIdx[2] = m_originalMesh.ibo[i * 3 + 2];

  //       m_triangleArr[i] = oneTriangle1;

  //       EdgeKey e01(m_originalMesh.ibo[i * 3 + 0], m_originalMesh.ibo[i * 3 + 1]);
  //       errCnt += MapInsertPushBack(m_edgeMap, e01, i);
        
  //       EdgeKey e12(m_originalMesh.ibo[i * 3 + 1], m_originalMesh.ibo[i * 3 + 2]);
  //       errCnt += MapInsertPushBack(m_edgeMap, e12, i);

  //       EdgeKey e02(m_originalMesh.ibo[i * 3 + 0], m_originalMesh.ibo[i * 3 + 2]);
  //       errCnt += MapInsertPushBack(m_edgeMap, e02, i);

  //       errCnt += MapInsertPushBack(m_vertNbrMap, m_originalMesh.ibo[i * 3 + 0], i);
  //       errCnt += MapInsertPushBack(m_vertNbrMap, m_originalMesh.ibo[i * 3 + 1], i);
  //       errCnt += MapInsertPushBack(m_vertNbrMap, m_originalMesh.ibo[i * 3 + 2], i);

  //       m_reduceCnt[i] = errCnt;
  //     }
  //   }
  // };

  // struct ApplyCalcEachTriangleImMesh
  // {
  //   EdgeMapConcurrent& m_edgeMap;
  //   VertNeighborMapConcurrent& m_vertNbrMap;
  //   MeshData& m_originalMesh;
  //   std::vector<TriangleInfo>& m_triangleArr;
  //   std::vector<uint32_t>& m_reduceErrCnt;

  //   ApplyCalcEachTriangleImMesh(EdgeMapConcurrent& edgeMap_,
  //                               VertNeighborMapConcurrent& vertNbrMap_,
  //                               MeshData& originalMesh_,
  //                               std::vector<TriangleInfo>& triangleArr_,
  //                               std::vector<uint32_t>& reduceErrCnt_): m_edgeMap(edgeMap_), m_vertNbrMap(vertNbrMap_), m_originalMesh(originalMesh_), m_triangleArr(triangleArr_), m_reduceErrCnt(reduceErrCnt_) {}

  //   template<typename Map, typename Key, typename Val>
  //   uint32_t MapInsertPushBack(Map& hashMap, Key key, Val data) const
  //   {
  //     typename Map::accessor access;
  //     uint32_t ret = (uint32_t)(hashMap.insert(access, key));
  //     access->second.push_back(data);
  //     access.release();
  //     return ret;
  //   }

  //   void operator()(const blocked_range<uint32_t> range) const
  //   {
  //     for (uint32_t i = range.begin(); i != range.end(); i++) {
  //       uint32_t errCnt = 0;
  //       TriangleInfo oneTriangle1;

  //       oneTriangle1.vert[0] = m_originalMesh.vbo[m_originalMesh.ibo[i * 3 + 0]].head(3);
  //       oneTriangle1.vert[1] = m_originalMesh.vbo[m_originalMesh.ibo[i * 3 + 1]].head(3);
  //       oneTriangle1.vert[2] = m_originalMesh.vbo[m_originalMesh.ibo[i * 3 + 2]].head(3);

  //       oneTriangle1.norm = m_originalMesh.m_faceNormal[i];

  //       oneTriangle1.vertIdx[0] = m_originalMesh.ibo[i * 3 + 0];
  //       oneTriangle1.vertIdx[1] = m_originalMesh.ibo[i * 3 + 1];
  //       oneTriangle1.vertIdx[2] = m_originalMesh.ibo[i * 3 + 2];

  //       m_triangleArr[i] = oneTriangle1;

  //       EdgeKey e01(m_originalMesh.ibo[i * 3 + 0], m_originalMesh.ibo[i * 3 + 1]);
  //       errCnt += MapInsertPushBack(m_edgeMap, e01, i);
        
  //       EdgeKey e12(m_originalMesh.ibo[i * 3 + 1], m_originalMesh.ibo[i * 3 + 2]);
  //       errCnt += MapInsertPushBack(m_edgeMap, e12, i);

  //       EdgeKey e02(m_originalMesh.ibo[i * 3 + 0], m_originalMesh.ibo[i * 3 + 2]);
  //       errCnt += MapInsertPushBack(m_edgeMap, e02, i);

  //       errCnt += MapInsertPushBack(m_vertNbrMap, m_originalMesh.ibo[i * 3 + 0], i);
  //       errCnt += MapInsertPushBack(m_vertNbrMap, m_originalMesh.ibo[i * 3 + 1], i);
  //       errCnt += MapInsertPushBack(m_vertNbrMap, m_originalMesh.ibo[i * 3 + 2], i);

  //       m_reduceErrCnt[i] = errCnt;
  //     }
  //   }

  // };

  // struct ApplyCalcEachTriangleReplica
  // {
  //   EdgeMapConcurrent& m_edgeMap;
  //   VertNeighborMapConcurrent& m_vertNbrMap;
  //   MeshData& m_originalMesh;
  //   std::vector<TriangleInfo>& m_triangleArr;

  //   ApplyCalcEachTriangleReplica(EdgeMapConcurrent& edgeMap_,
  //                                VertNeighborMapConcurrent& vertNbrMap_,
  //                                MeshData& originalMesh_,
  //                                std::vector<TriangleInfo>& triangleArr_) : m_edgeMap(edgeMap_), m_vertNbrMap(vertNbrMap_), m_originalMesh(originalMesh_), m_triangleArr(triangleArr_) {}

  //   template<typename Map, typename Key, typename Val>
  //   void MapInsertPushBack(Map& hashMap, Key key, Val data) const
  //   {
  //     typename Map::accessor access;
  //     hashMap.insert(access, key);
  //     access->second.push_back(data);
  //     access.release();
  //   }

  //   void operator()(const blocked_range<uint32_t> range) const
  //   {
  //       for (uint32_t i = range.begin(); i != range.end(); i++) {
  //         TriangleInfo oneTriangle1;
  //         TriangleInfo oneTriangle2;
  //         oneTriangle1.vert[0] = m_originalMesh.vbo[m_originalMesh.ibo[i * 4 + 0]].head(3);
  //         oneTriangle1.vert[1] = m_originalMesh.vbo[m_originalMesh.ibo[i * 4 + 1]].head(3);
  //         oneTriangle1.vert[2] = m_originalMesh.vbo[m_originalMesh.ibo[i * 4 + 2]].head(3);
  //         oneTriangle1.norm = m_originalMesh.m_faceNormal[2 * i + 0];

  //         oneTriangle1.vertIdx[0] = m_originalMesh.ibo[i * 4 + 0];
  //         oneTriangle1.vertIdx[1] = m_originalMesh.ibo[i * 4 + 1];
  //         oneTriangle1.vertIdx[2] = m_originalMesh.ibo[i * 4 + 2];

  //         oneTriangle2.vert[0] = m_originalMesh.vbo[m_originalMesh.ibo[i * 4 + 0]].head(3);
  //         oneTriangle2.vert[1] = m_originalMesh.vbo[m_originalMesh.ibo[i * 4 + 2]].head(3);
  //         oneTriangle2.vert[2] = m_originalMesh.vbo[m_originalMesh.ibo[i * 4 + 3]].head(3);
  //         oneTriangle2.norm = m_originalMesh.m_faceNormal[2 * i + 1];
  //         oneTriangle2.vertIdx[0] = m_originalMesh.ibo[i * 4 + 0];
  //         oneTriangle2.vertIdx[1] = m_originalMesh.ibo[i * 4 + 2];
  //         oneTriangle2.vertIdx[2] = m_originalMesh.ibo[i * 4 + 3];

  //         m_triangleArr[i * 2 + 0] = oneTriangle1;
  //         m_triangleArr[i * 2 + 1] = oneTriangle2;

  //         // triangle 1
  //         EdgeKey e01(m_originalMesh.ibo[i * 4 + 0], m_originalMesh.ibo[i * 4 + 1]);
  //         MapInsertPushBack(m_edgeMap, e01, i * 2 + 0);

  //         EdgeKey e12(m_originalMesh.ibo[i * 4 + 1], m_originalMesh.ibo[i * 4 + 2]);
  //         MapInsertPushBack(m_edgeMap, e12, i * 2 + 0);

  //         EdgeKey e02(m_originalMesh.ibo[i * 4 + 0], m_originalMesh.ibo[i * 4 + 2]);
  //         MapInsertPushBack(m_edgeMap, e02, i * 2 + 0);

  //         // triangle 2
  //         MapInsertPushBack(m_edgeMap, e02, i * 2 + 1);
          
  //         EdgeKey e23(m_originalMesh.ibo[i * 4 + 2], m_originalMesh.ibo[i * 4 + 3]);
  //         MapInsertPushBack(m_edgeMap, e23, i * 2 + 1);

  //         EdgeKey e03(m_originalMesh.ibo[i * 4 + 0], m_originalMesh.ibo[i * 4 + 3]);
  //         MapInsertPushBack(m_edgeMap, e03, i * 2 + 1);

  //         MapInsertPushBack(m_vertNbrMap, m_originalMesh.ibo[i * 4 + 0], (i * 2 + 0));
  //         MapInsertPushBack(m_vertNbrMap, m_originalMesh.ibo[i * 4 + 0], (i * 2 + 1));

  //         MapInsertPushBack(m_vertNbrMap, m_originalMesh.ibo[i * 4 + 1], (i * 2 + 0));

  //         MapInsertPushBack(m_vertNbrMap, m_originalMesh.ibo[i * 4 + 2], (i * 2 + 0));
  //         MapInsertPushBack(m_vertNbrMap, m_originalMesh.ibo[i * 4 + 2], (i * 2 + 1));

  //         MapInsertPushBack(m_vertNbrMap, m_originalMesh.ibo[i * 4 + 3], (i * 2 + 1));
  //         // EdgeMapConcurrent::const_accessor conAccess;
  //         // if (m_edgeMap.find(conAccess, e01) == false) {
  //         //   conAccess.release();
  //         //   EdgeMapConcurrent::accessor acces;
  //         //   m_edgeMap.insert(acces, e01);
  //         //   acces->second.push_back(i * 2 + 0);
  //         //   acces.release();

  //         //   PushIntoMap(m_edgeMap, e01, i * 2 + 0);
  //         // } else {
  //         //   EdgeMapConcurrent::accessor acces;
  //         //   m_edgeMap.find(acces, e01);
  //         //   acces->second.push_back(i * 2 + 0);
  //         //   acces.release();
  //         // }
  //       }
  //   }

  // };
  
  void ValidateEdgeTriangleRelation();

  void LoadMeshDataReplica(const std::string& meshFile, float splitSize);
  void LoadMeshDataImMesh(const std::string& meshFile, float splitSize);
  void LoadMeshDataJianHengLiu(const std::string& meshFile, float splitSize);

  void CalcEachTriangleReplica(std::vector<TriangleInfoDevice>& triangleArr);
  void CalcEachTriangleImMesh(std::vector<TriangleInfoDevice>& triangleArr);
  void CalcEachTriangleBlenderOut(std::vector<TriangleInfo>& triangleArr);
  void CalcEachTriangleImMeshDouble(std::vector<TriangleInfo>& triangleArr);
  void LoadMeshDataBlenderOut(const std::string& meshFile, float splitSize);
  void CalcEachTriangleJianHengLiu(std::vector<TriangleInfoDevice>& triangleArr);
  bool IsFaceNormalValid();
  void PrintProgressBar(int progress, int total);

  // void LoadMeshData(const std::string& meshFile, float splitSize);

  std::vector<MeshData> SplitMesh(const MeshData& mesh, const float splitSize);
  void PLYParse(const std::string& filename);
  void PLYParseReplica(const std::string& filename);
  void PLYParseImMesh(const std::string& filename);
  void PLYParseBlenderOut(const std::string& filename);
  void PLYParseDouble(const std::string& filename);
  void PLYParseJianHengLiu(const std::string& filename);
  void PrintHashMapAndTriangleArr(std::vector<TriangleInfo>& triangleArr);

  unsigned long GetTimeMS();

  // void CalcEachTriangleCuda(std::vector<TriangleInfoDevice>& triangleArr, uint32_t numFaces, MeshType meshType);
  // void PrintHashMapAndTriangleArrCuda(std::vector<TriangleInfoDevice>& triangleArr);

  void GetCudaLastErr();


  MeshData m_originalMesh;
  std::vector<MeshData> m_splitMeshData;

  // template<uint32_t block_size,
  //          uint32_t flushing_cg_size,
  //          uint32_t probing_cg_size,
  //          uint32_t buffer_size
  //          typename InputIt,
  //          typename OutputIt,
  //          typename ViewT,
  //          typename KeyEqual>
  // __device__ void read_edge_map_values(InputIt allEdgeKeysFirst, uint32_t numKey, OutputIt resultFirst, ViewT edgeMapView, KeyEqual equalFunc)
  // {
  //   constexpr uint32_t num_flushing_cgs = block_size / flushing_cg_size;
  //   const uint32_t flushing_cg_id       = threadIdx.x / flushing_cg_size;

  //   auto flushing_cg          = cg::tiled_partition<flushing_cg_size>(cg::this_thread_block());
  //   auto probing_cg           = cg::tiled_partition<probing_cg_size>(cg::this_thread_block());

  //   int64_t const loop_stride = gridDim.x * block_size / probing_cg_size;
  //   int64_t idx               = (block_size * blockIdx.x + threadIdx.x) / probing_cg_size;
  // }


  template <typename VertMap>
  void RetrieveVertMapCuda(VertMap& vertNbrMultiMapDevice, thrust::device_vector<int32_t>& validateVertKeys, thrust::device_vector<cuco::pair<int, int>>& outVertBufDevice, std::vector<size_t>& vertKeyValueCounters, uint32_t maxNbrCnt)
  {
    printf("in RetrieveVertMapCuda vert keyNum %d, maxNbrCnt: %d\n", validateVertKeys.size(), maxNbrCnt);

    auto constexpr block_size = 128;
    auto const grid_size = (validateVertKeys.size() + block_size - 1) / (block_size);

    // constexpr auto buffer_size = vertNbrMultiMapDevice.uses_vector_load() ? (vertNbrMultiMapDevice.warp_size() * 10u) : (vertNbrMultiMapDevice.cg_size() * 10u);

    constexpr auto buffer_size = 3;

    auto view = vertNbrMultiMapDevice.get_device_view();

    my_atomic_ctr_type* my_d_counters;
    checkCudaErrors(cudaMalloc(&my_d_counters, validateVertKeys.size() * sizeof(my_atomic_ctr_type)));

    my_initialize_atomic<<<grid_size, block_size>>>(my_d_counters, validateVertKeys.size());
    GetCudaLastErr();
    checkCudaErrors(cudaDeviceSynchronize());

    assert(vertNbrMultiMapDevice.cg_size() == 1);

    custom_retrieve<block_size, 1, vertNbrMultiMapDevice.cg_size(), buffer_size> <<<grid_size, block_size>>>
        (validateVertKeys.begin(),
         validateVertKeys.size(),
         outVertBufDevice.begin(),
         my_d_counters,
         view,
         thrust::equal_to<int>{},
         maxNbrCnt);

    GetCudaLastErr();
    checkCudaErrors(cudaDeviceSynchronize());

    // std::vector<size_t> vertKeyValueCounters(validateVertKeys.size());
    for (uint32_t i = 0; i < validateVertKeys.size(); i++) {
      checkCudaErrors(cudaMemcpy(&(vertKeyValueCounters[i]), my_d_counters + i, sizeof(my_atomic_ctr_type), cudaMemcpyDeviceToHost));
    }

    uint32_t countersSum = 0;
    for (auto it: vertKeyValueCounters) {
      countersSum += it;
    }

    std::cout << "---- RetrieveVertMapCuda custom_retrieve load counter sum : << " << countersSum  << std::endl;

    checkCudaErrors(cudaFree(my_d_counters));

  }

  template <typename EdgeMap>
  void RetrieveEdgeMapCuda(EdgeMap& edgeNbrMultiMapDevice, thrust::device_vector<edgeMapKeyType>& allEdgeKeys, uint32_t keyNum, thrust::device_vector<cuco::pair<EdgeKeyCuda, int>>& outEdgeBufDevice, std::vector<size_t>& edgeKeyValueCounters, uint32_t maxNbrCnt)
  {
    printf("in RetrieveEdgeMapCuda edge keyNum %d, maxNbrCnt: %d\n", keyNum, maxNbrCnt);
    // std::vector<int> result(allEdgeKeys.size() * maxNbrCnt);
    // std::fill(result.begin(), result.end(), -1);

    // thrust::device_vector<int> resultDev;
    // resultDev = result;

    auto constexpr block_size = 128;
    auto const grid_size = (keyNum + block_size - 1) / (block_size);

    // constexpr auto buffer_size = edgeNbrMultiMapDevice.uses_vector_load() ? (edgeNbrMultiMapDevice.warp_size() * 3u) : (edgeNbrMultiMapDevice.cg_size() * 3u);
    constexpr auto buffer_size = 3;

    auto view = edgeNbrMultiMapDevice.get_device_view();
    // auto const flushing_cg_size = [&]() {
    //   if constexpr (edgeNbrMultiMapDevice.uses_vector_load()) { return edgeNbrMultiMapDevice.warp_size(); }
    //   return edgeNbrMultiMapDevice.cg_size();
    // }();
    
    auto myAlloc = cuco::cuda_allocator<char>{};

    my_counter_allocator_type my_counter_allocator{myAlloc};

    my_counter_deleter delete_counter_{my_counter_allocator};

    std::unique_ptr<my_atomic_ctr_type, my_counter_deleter> my_d_counter_{my_counter_allocator.allocate(1), delete_counter_};

    // std::vector<std::unique_ptr<my_atomic_ctr_type, my_counter_deleter>> my_d_counters;


    // my_d_counters.reserve(allEdgeKeys.size());

    // for (uint32_t cnt = 0; cnt < allEdgeKeys.size(); cnt++) {
    //   my_d_counters.emplace_back(my_counter_allocator.allocate(1), delete_counter_);
    // }

    // for (uint32_t cnt = 0; cnt < allEdgeKeys.size(); cnt++) {
    //   checkCudaErrors(cudaMemset(my_d_counters[cnt].get(), 0, sizeof(my_atomic_ctr_type)));
    // }

    // thrust::device_vector<my_atomic_ctr_type> my_d_counters;
    // my_d_counters.resize(10); // error!

    my_atomic_ctr_type* my_d_counters;
    checkCudaErrors(cudaMalloc(&my_d_counters, keyNum * sizeof(my_atomic_ctr_type)));

    my_initialize_atomic<<<grid_size, block_size>>>(my_d_counters, keyNum);
    GetCudaLastErr();
    checkCudaErrors(cudaDeviceSynchronize());

    assert(edgeNbrMultiMapDevice.cg_size() == 1);

    custom_retrieve<block_size, 1, edgeNbrMultiMapDevice.cg_size(), buffer_size><<<grid_size, block_size>>>(allEdgeKeys.begin(),
                                keyNum,
                                outEdgeBufDevice.begin(),
                                my_d_counters,
                                view,
                                EdgeKeyEqual{},
                                maxNbrCnt);

    GetCudaLastErr();
    checkCudaErrors(cudaDeviceSynchronize());

    // std::vector<size_t> edgeKeyValueCounters(allEdgeKeys.size());

    for (uint32_t i = 0; i < keyNum; i++) {
      checkCudaErrors(cudaMemcpy(&(edgeKeyValueCounters[i]), my_d_counters + i, sizeof(my_atomic_ctr_type), cudaMemcpyDeviceToHost));
    }

    uint32_t countersSum = 0;
    for (auto it: edgeKeyValueCounters) {
      countersSum += it;
    }

    std::cout << "---- custom_retrieve load counter sum : << " << countersSum  << std::endl;

    checkCudaErrors(cudaFree(my_d_counters));

    // read_atomic_values<<<grid_size, block_size>>>(my_d_counters.begin(), edgeKeyValueCounters.begin(), my_d_counters.size());

    


    // here is device address, because use cuco allocator
    // std::vector<std::unique_ptr<cuda::atomic<std::size_t, cuda::thread_scope_device>, edgeNbrMultiMapDevice.delete_counter_>> test;

    // cuda::atomic<std::size_t, cuda::thread_scope_device>* ptr = edgeNbrMultiMapDevice.counter_allocator_.allocate(1);
    
    // test.emplace_back(ptr, edgeNbrMultiMapDevice.delete_counter_);
    // auto ii = std::make_unique<edgeNbrMultiMapDevice.atomic_ctr_type, edgeNbrMultiMapDevice.counter_deleter>();


    // read_edge_map_values<block_size, flushing_cg_size, edgeNbrMultiMapDevice.cg_size(), buffer_size><<<grid_size, block_size, 0, 0>>>(allEdgeKeys.begin(), keyNum, resultDev.begin(), view, EdgeKeyEqual{});
    // for (auto it = m_edgeMapConcurrent.begin(); it != m_edgeMapConcurrent.end(); it++) {
    //   printf("edge %u %u info ", it->first.m_v1, it->first.m_v2);
    //   for (auto it2: it->second) {
    //     printf(" %d ", it2);
    //   }

    //   printf("\n");
    // }
  }


  template <typename EdgeMap, typename VertMap, typename EdgeKeySet>
  void PrintHashMapAndTriangleArrCuda(thrust::host_vector<TriangleInfoDevice>& triangleArr, EdgeMap& edgeNbrMultiMapDevice, VertMap& vertNbrMultiMapDevice, EdgeKeySet& edgeKeysSet, uint32_t numFaces)
  {
    unsigned long startTime;
    unsigned long endTime;
    printf(" printf all triangleArr \n");
    for (auto it : triangleArr) {
      printf("vert (%f, %f, %f), (%f, %f, %f), (%f, %f, %f), norm (%f, %f, %f), vertIdx (%u, %u, %u)\n",
              it.vert[0], it.vert[1], it.vert[2],
              it.vert[3], it.vert[4], it.vert[5],
              it.vert[6], it.vert[7], it.vert[8],
              it.norm[0], it.norm[1], it.norm[2],
              it.vertIdx[0], it.vertIdx[1], it.vertIdx[2]);
    }

    printf("111edge map cuda size : %u\n", edgeNbrMultiMapDevice.get_size());
    printf("111vert map cuda size : %u\n", vertNbrMultiMapDevice.get_size());

    thrust::device_vector<edgeMapKeyType> allEdgeKeys(numFaces * 10);
    auto const allEdgeKeysEnd = edgeKeysSet.retrieve_all(allEdgeKeys.begin());
    auto const keyNum             = std::distance(allEdgeKeys.begin(), allEdgeKeysEnd);

    std::cout << "----- allEdgeKeys in set number: " << keyNum << std::endl;

    auto edgeMapReadOnlyView = edgeNbrMultiMapDevice.get_device_view();

    thrust::device_vector<uint32_t> zeroEdgeNbrCnt(1);
    thrust::device_vector<uint32_t> oneEdgeNbrCnt(1);
    thrust::device_vector<uint32_t> twoEdgeNbrCnt(1);
    thrust::device_vector<uint32_t> moreThanTwoEdgeNbrCnt(1);
    thrust::device_vector<uint32_t> maxEdgeNbrNum(1);

    auto constexpr block_size_2 = 128;
    auto constexpr stride_2     = 1;
    auto const grid_size_2 = (keyNum + stride_2 * block_size_2 - 1) / (stride_2 * block_size_2);

    printf("block_size_2 %d, grid_size_2 : %d\n", block_size_2, grid_size_2);

    startTime = GetTimeMS();
    validate_edge_map_kernel<edgeNbrMultiMapDevice.cg_size()><<<grid_size_2, block_size_2>>>(allEdgeKeys.begin(), keyNum, edgeMapReadOnlyView, EdgeKeyEqual{}, zeroEdgeNbrCnt.data().get(), oneEdgeNbrCnt.data().get(), twoEdgeNbrCnt.data().get(), moreThanTwoEdgeNbrCnt.data().get(), maxEdgeNbrNum.data().get());

    GetCudaLastErr();
    checkCudaErrors(cudaDeviceSynchronize());
    endTime = GetTimeMS();

    std::cout << "key type: zero nbr: " << zeroEdgeNbrCnt[0] << ", one nbr: " << oneEdgeNbrCnt[0] <<  ", two nbr: " <<  twoEdgeNbrCnt[0] << ", morn than 2 nbr: "<< moreThanTwoEdgeNbrCnt[0] << " max br cnt: " << maxEdgeNbrNum[0] << " time: " << (double)(endTime - startTime) << " ms" << std::endl;


    std::unordered_set<int32_t> uniqueIdxs(m_originalMesh.m_iboVec.begin(), m_originalMesh.m_iboVec.end());
    std::vector<int32_t> uniqueKeys;
    uniqueKeys.assign(uniqueIdxs.begin(), uniqueIdxs.end());

    printf("ibo all size: %d, unique idx size : %d\n", m_originalMesh.m_iboVec.size(), uniqueIdxs.size());

    auto constexpr block_size_3 = 128;
    auto constexpr stride_3     = 1;
    auto const grid_size_3 = (uniqueIdxs.size() + stride_3 * block_size_3 - 1) / (stride_3 * block_size_3);
    thrust::device_vector<int32_t> validateVertKeys;
    validateVertKeys = uniqueKeys;
    thrust::device_vector<uint32_t> maxVertNbrNum(1);
    thrust::device_vector<uint32_t> cannotFindNum(1);

    auto vertMapReadOnlyView = vertNbrMultiMapDevice.get_device_view();


    startTime = GetTimeMS();
    validate_vert_map_kernel<vertNbrMultiMapDevice.cg_size()><<<grid_size_3, block_size_3>>>(validateVertKeys.begin(), uniqueIdxs.size(), vertMapReadOnlyView, maxVertNbrNum.data().get(), cannotFindNum.data().get());

    GetCudaLastErr();
    checkCudaErrors(cudaDeviceSynchronize());
    endTime = GetTimeMS();

    printf("block_size_3 %d, grid_size_3 : %d\n", block_size_3, grid_size_3);
    std::cout << "max vert nbr num :  " << maxVertNbrNum[0] << " cannot find num:  " <<  cannotFindNum[0] << " time : " << (double)(endTime - startTime) << " ms" << std::endl;


    thrust::host_vector<uint32_t> maxVertNbrNumHost(1);
    thrust::copy(maxVertNbrNum.begin(), maxVertNbrNum.end(), maxVertNbrNumHost.begin());

    thrust::host_vector<uint32_t> maxEdgeNbrHost(1);
    thrust::copy(maxEdgeNbrNum.begin(), maxEdgeNbrNum.end(), maxEdgeNbrHost.begin());

    
    thrust::device_vector<cuco::pair<EdgeKeyCuda, int>> outEdgeBufDevice;
    outEdgeBufDevice.resize(keyNum * maxEdgeNbrHost[0]);
    std::vector<size_t> edgeKeyValueCounters(keyNum);
    RetrieveEdgeMapCuda(edgeNbrMultiMapDevice, allEdgeKeys, keyNum, outEdgeBufDevice, edgeKeyValueCounters, maxEdgeNbrHost[0]);


    thrust::device_vector<cuco::pair<int, int>> outVertBufDevice;
    outVertBufDevice.resize(validateVertKeys.size() * maxVertNbrNumHost[0]);
    std::vector<size_t> vertKeyValueCounters(validateVertKeys.size());
  
    RetrieveVertMapCuda(vertNbrMultiMapDevice, validateVertKeys, outVertBufDevice, vertKeyValueCounters, maxVertNbrNumHost[0]);

    thrust::host_vector<cuco::pair<EdgeKeyCuda, int>> outEdgeBufHost = outEdgeBufDevice;
    thrust::host_vector<cuco::pair<int, int>> outVertBufHost = outVertBufDevice;

    // printf("m_edgeMap size : %u\n", m_edgeMap.size());
    // printf("m_vertNeighborMap size : %u\n", m_vertNeighborMap.size());

    // for (auto it = m_edgeMapConcurrent.begin(); it != m_edgeMapConcurrent.end(); it++) {
    //   printf("edge %u %u info ", it->first.m_v1, it->first.m_v2);
    //   for (auto it2: it->second) {
    //     printf(" %d ", it2);
    //   }

    //   printf("\n");
    // }
    
    thrust::host_vector<edgeMapKeyType> allEdgeKeysHost = allEdgeKeys;
    for (uint32_t ii = 0; ii < keyNum; ii++) {
      printf("edge %u %u info ", allEdgeKeysHost[ii].m_e0, allEdgeKeysHost[ii].m_e1);

      for (uint32_t jj = 0; jj < edgeKeyValueCounters[ii]; jj++) {
        printf(" %d ", (*(outEdgeBufHost.begin() + ii * maxEdgeNbrHost[0] + jj)).second);
      }
      printf("\n");
    }

    // for (auto it : m_vertNbrMapConcurrent) {
    //   printf("vertId  %u  Nbr texId ", it.first);
    //   for (auto it2: it.second) {
    //     printf(" %u ", it2);
    //   }

    //   printf("\n");
    // }

    thrust::host_vector<int32_t> validateVertKeysHost = validateVertKeys;
    for (uint32_t ii = 0; ii < validateVertKeysHost.size(); ii++) {
        printf("vertId  %u  Nbr texId ", validateVertKeysHost[ii]);

        for (uint32_t jj = 0; jj < vertKeyValueCounters[ii]; jj++) {
            printf(" %u ", (*(outVertBufHost.begin() + ii * maxVertNbrNumHost[0] + jj)).second);
        }

        printf("\n");
    }
  }


  template <typename EdgeMap, typename VertMap, typename EdgeKeySet>
  void CalcEachTriangleCuda(thrust::device_vector<TriangleInfoDevice>& triangleArr, uint32_t numFaces, MeshType meshType, EdgeMap& edgeNbrMultiMapDevice, VertMap& vertNbrMultiMapDevice, EdgeKeySet& edgeKeysSet)
  {
    unsigned long startTime = {0};
    unsigned long endTime = {0};

    std::cout << "triangle faces: " << numFaces << std::endl;

    thrust::device_vector<int32_t> iboDevice = m_originalMesh.m_iboVec;
    thrust::device_vector<float> faceNormalDevice = m_originalMesh.m_faceNormalVec;
    thrust::device_vector<float> vboDevice = m_originalMesh.m_vboVec;

    std::cout << "triangleArr size : " << triangleArr.size() << std::endl;
    printf("iboDevice size: %d, faceNormalDevice: %d, vboDevice: %d, numFaces : %d\n", iboDevice.size(), faceNormalDevice.size(), vboDevice.size(), numFaces);

    assert(edgeNbrMultiMapDevice.cg_size() == vertNbrMultiMapDevice.cg_size());

    thrust::device_vector<int> num_inserted(1);

    auto edgeMapInsertView = edgeNbrMultiMapDevice.get_device_mutable_view();
    auto vertMapInsertView = vertNbrMultiMapDevice.get_device_mutable_view();

    

    switch (meshType) {
      case MESH_TYPE_M2_MAPPING:
      case MESH_TYPE_IM_MESH:
      case MESH_TYPE_BLENDER_OUT:
      {
        std::cout << "into MESH_TYPE_M2_MAPPING MESH_TYPE_IM_MESH kernel \n";
        auto constexpr block_size = 128;
        auto constexpr stride     = 1;
        auto const grid_size = (edgeNbrMultiMapDevice.cg_size() * numFaces + stride * block_size - 1) / (stride *block_size);
        printf("grid_size, block_size : %d, %d\n", grid_size, block_size);

        insert_triangle_info_triangle <<<grid_size, block_size>>> (iboDevice.begin(),
            (TriangleInfoDevice *)thrust::raw_pointer_cast(triangleArr.data()),
            faceNormalDevice.begin(),
            vboDevice.begin(),
            numFaces);
          
        GetCudaLastErr();
        checkCudaErrors(cudaDeviceSynchronize());

        endTime = GetTimeMS();
        std::cout << "insert_triangle_info_triangle kernel run time : " << (double)(endTime - startTime) << "ms" << std::endl;

        startTime = GetTimeMS();
        insert_edge_map_triangle <edgeNbrMultiMapDevice.cg_size()> <<<grid_size, block_size>>> (edgeMapInsertView,
            edgeKeysSet.ref(cuco::insert),
            iboDevice.begin(),
            numFaces,
            triangleArr.begin());
        
        GetCudaLastErr();
        checkCudaErrors(cudaDeviceSynchronize());

        endTime = GetTimeMS();
        std::cout << "insert_edge_map_triangle kernel run time : " << (double)(endTime - startTime) << "ms" << std::endl;
        
        startTime = GetTimeMS();
        insert_vert_map_triangle <edgeNbrMultiMapDevice.cg_size()> <<<grid_size, block_size>>> (vertMapInsertView, iboDevice.begin(), numFaces);

        GetCudaLastErr();
        checkCudaErrors(cudaDeviceSynchronize());

        endTime = GetTimeMS();
        std::cout << "insert_vert_map_triangle kernel run time : " << (double)(endTime - startTime) << "ms" << std::endl;
      }
        break;

      case MESH_TYPE_REPLICA:
      {
        std::cout << "into MESH_TYPE_REPLICA kernel \n";

        auto constexpr block_size = 32;
        auto const grid_size = (numFaces + block_size - 1) / (block_size);
        printf("MESH_TYPE_REPLICA grid_size, block_size : %d, %d\n", grid_size, block_size);

        startTime = GetTimeMS();
        insert_triangle_info_quad <<<grid_size, block_size>>> (iboDevice.begin(),
            (TriangleInfoDevice *)thrust::raw_pointer_cast(triangleArr.data()),
            faceNormalDevice.begin(),
            vboDevice.begin(),
            numFaces);

        GetCudaLastErr();
        checkCudaErrors(cudaDeviceSynchronize());

        endTime = GetTimeMS();
        std::cout << "insert_triangle_info_quad kernel run time : " << (double)(endTime - startTime) << "ms" << std::endl;

        startTime = GetTimeMS();

        auto constexpr block_size2 = 32;
        auto const grid_size2 = (numFaces * 2 + block_size2 - 1) / (block_size2);
        printf("MESH_TYPE_REPLICA grid_size2, block_size2 : %d, %d\n", grid_size2, block_size2);

        insert_edge_map_quad<edgeNbrMultiMapDevice.cg_size()> <<<grid_size2, block_size2>>> (edgeMapInsertView,
            edgeKeysSet.ref(cuco::insert),
            numFaces * 2,
            triangleArr.begin());

        GetCudaLastErr();
        checkCudaErrors(cudaDeviceSynchronize());

        endTime = GetTimeMS();
        std::cout << "insert_edge_map_quad kernel run time : " << (double)(endTime - startTime) << "ms" << std::endl;

        startTime = GetTimeMS();
        insert_vert_map_quad<edgeNbrMultiMapDevice.cg_size()> <<<grid_size2, block_size2>>> (vertMapInsertView, triangleArr.begin(), numFaces * 2);

        GetCudaLastErr();
        checkCudaErrors(cudaDeviceSynchronize());

        endTime = GetTimeMS();
        std::cout << "insert_vert_map_quad kernel run time : " << (double)(endTime - startTime) << "ms" << std::endl;

        // custom_insert_quad<edgeNbrMultiMapDevice.cg_size()> <<<1024, 1024>>>(
        //   edgeMapInsertView,
        //   vertMapInsertView,
        //   edgeKeysSet.ref(cuco::insert),
        //   iboDevice.begin(),
        //   (TriangleInfoDevice *)thrust::raw_pointer_cast(triangleInfoDevice.data()),
        //   faceNormalDevice.begin(),
        //   vboDevice.begin(),
        //   num_inserted.data().get(),
        //   numFaces);

        //   GetCudaLastErr();
        //   checkCudaErrors(cudaDeviceSynchronize());
      }
        break;
      
      default:
        break;
    }


    // startTime = GetTimeMS();
    // thrust::copy(triangleInfoDevice.begin(), triangleInfoDevice.end(), triangleArr.begin());
    // endTime = GetTimeMS();

    // std::cout << "triangleInfoDevice copy time : " << (double)(endTime - startTime) << "ms" << std::endl;

    // edgeMapKeyType
    // thrust::device_vector<edgeMapKeyType> allEdgeKeys(numFaces * 5);
    // auto const allEdgeKeysEnd = edgeKeysSet.retrieve_all(allEdgeKeys.begin());
    // auto const keyNum             = std::distance(allEdgeKeys.begin(), allEdgeKeysEnd);

    // std::cout << "----- allEdgeKeys in set number: " << keyNum << std::endl;

    // auto edgeMapReadOnlyView = edgeNbrMultiMapDevice.get_device_view();

    // thrust::device_vector<uint32_t> zeroEdgeNbrCnt(1);
    // thrust::device_vector<uint32_t> oneEdgeNbrCnt(1);
    // thrust::device_vector<uint32_t> twoEdgeNbrCnt(1);
    // thrust::device_vector<uint32_t> moreThanTwoEdgeNbrCnt(1);
    // thrust::device_vector<uint32_t> maxEdgeNbrNum(1);

    // auto constexpr block_size_2 = 128;
    // auto constexpr stride_2     = 1;
    // auto const grid_size_2 = (keyNum + stride_2 * block_size_2 - 1) / (stride_2 * block_size_2);

    // printf("block_size_2 %d, grid_size_2 : %d\n", block_size_2, grid_size_2);

    // validate_edge_map_kernel<edgeNbrMultiMapDevice.cg_size()><<<grid_size_2, block_size_2>>>(allEdgeKeys.begin(), keyNum, edgeMapReadOnlyView, EdgeKeyEqual{}, zeroEdgeNbrCnt.data().get(), oneEdgeNbrCnt.data().get(), twoEdgeNbrCnt.data().get(), moreThanTwoEdgeNbrCnt.data().get(), maxEdgeNbrNum.data().get());

    // std::cout << "key type: zero nbr: " << zeroEdgeNbrCnt[0] << ", one nbr: " << oneEdgeNbrCnt[0] <<  ", two nbr: " <<  twoEdgeNbrCnt[0] << ", morn than 2 nbr: "<< moreThanTwoEdgeNbrCnt[0] << " max br cnt: " << maxEdgeNbrNum[0] << std::endl;


    // std::unordered_set<int32_t> uniqueIdxs(m_originalMesh.m_iboVec.begin(), m_originalMesh.m_iboVec.end());
    // std::vector<int32_t> uniqueKeys;
    // uniqueKeys.assign(uniqueIdxs.begin(), uniqueIdxs.end());

    // printf("ibo all size: %d, unique idx size : %d\n", m_originalMesh.m_iboVec.size(), uniqueIdxs.size());

    // auto constexpr block_size_3 = 128;
    // auto constexpr stride_3     = 1;
    // auto const grid_size_3 = (uniqueIdxs.size() + stride_3 * block_size_3 - 1) / (stride_3 * block_size_3);
    // thrust::device_vector<int32_t> validateVertKeys;
    // validateVertKeys = uniqueKeys;
    // thrust::device_vector<uint32_t> maxVertNbrNum(1);
    // thrust::device_vector<uint32_t> cannotFindNum(1);

    // auto vertMapReadOnlyView = vertNbrMultiMapDevice.get_device_view();
    // validate_vert_map_kernel<vertNbrMultiMapDevice.cg_size()><<<grid_size_3, block_size_3>>>(validateVertKeys.begin(), uniqueIdxs.size(), vertMapReadOnlyView, maxVertNbrNum.data().get(), cannotFindNum.data().get());

    // printf("block_size_3 %d, grid_size_3 : %d\n", block_size_3, grid_size_3);
    // std::cout << "max vert nbr num :  " << maxVertNbrNum[0] << " cannot find num:  " <<  cannotFindNum[0] << std::endl;

  }


  void LoadMeshData(const std::string& meshFile, float splitSize, MeshType meshType);
  uint32_t GetNumFaces(MeshType meshType);

  template <typename EdgeMap, typename VertMap>
  void CalcEachTriangle(EdgeMap& edgeMap, VertMap& vertMap, thrust::device_vector<TriangleInfoDevice>& triangleArr, MeshType meshType)
  {
    switch (meshType) {
        case MESH_TYPE_IM_MESH:
            break;
        
        case MESH_TYPE_REPLICA:
            CalcEachTriangleReplica(edgeMap, vertMap, triangleArr);
            return ;

        case MESH_TYPE_BLENDER_OUT:
            CalcEachTriangleBlenderOut(edgeMap, vertMap, triangleArr);
            return;

        case MESH_TYPE_M2_MAPPING:
            CalcEachTriangleJianHengLiu(edgeMap, vertMap, triangleArr);
            return;

        default:
            break;
    }

    exit(0);
  }



  template<typename EdgeMap, typename VertMap>
  void CalcEachTriangleReplica(EdgeMap& edgeMap, VertMap& vertMap, thrust::device_vector<TriangleInfoDevice>& triangleArr)
  {
      std::cout << "start CalcEachTriangleReplica\n";
      size_t numFaces = m_originalMesh.m_iboVec.size() / 4;

      triangleArr.resize(numFaces * 2);

      printf("m_originalMesh.ibo size :%u, m_originalMesh.vbo size :%u ", m_originalMesh.m_iboVec.size(), m_originalMesh.m_vboVec.size());

      unsigned long startTime = GetTimeMS();

      // parallel_for(blocked_range<uint32_t>(0, numFaces), ApplyCalcEachTriangleReplica(m_edgeMapConcurrent, m_vertNbrMapConcurrent, m_originalMesh, triangleArr));

      edgeMapKeyType  emptyEdgeKeySentinel{-1, -1};
      edgeMapValueType constexpr emptyEdgeValueSentinel = -1;

      vertMapKeyType emptyVertKeySentinel = -1;
      vertMapValueType constexpr emptyVertValueSentinel = -1;

      using edgeSetProbe = cuco::linear_probing<1, EdgeKeyCudaHasher>;

      // auto edgeNbrMultiMapDevice = cuco::static_multimap<edgeMapKeyType, edgeMapValueType, cuda::thread_scope_device, cuco::cuda_allocator<char>, edgeMapProbe>{numFaces * 6,
      //                                     cuco::empty_key{emptyEdgeKeySentinel},
      //                                     cuco::empty_value{emptyEdgeValueSentinel}};
      
      // auto vertNbrMultiMapDevice = cuco::static_multimap<vertMapKeyType, vertMapValueType, cuda::thread_scope_device, cuco::cuda_allocator<char>, vertMapProbe>{numFaces * 7,
      //                                     cuco::empty_key{emptyVertKeySentinel},
      //                                     cuco::empty_value{emptyVertValueSentinel}};

      auto edgeKeysSet = cuco::static_set{cuco::extent<std::size_t>{numFaces * 5},
                                  cuco::empty_key{emptyEdgeKeySentinel},
                                  EdgeKeyEqual{},
                                  edgeSetProbe{}};

      CalcEachTriangleCuda(triangleArr, numFaces, MESH_TYPE_REPLICA, edgeMap, vertMap, edgeKeysSet);
      unsigned long endTime = GetTimeMS();

      std::cout << "------parallel_for CalcEachTriangleReplica " << (double)(endTime - startTime) << " ms" << std::endl;
    
      // for (size_t cnt = 0; cnt < numFaces; cnt++) {
      //   // PrintProgressBar(cnt, numFaces);
      //   TriangleInfo oneTriangle1;
      //   TriangleInfo oneTriangle2;
      //   oneTriangle1.vert[0] = (m_originalMesh.vbo[m_originalMesh.ibo[cnt * 4 + 0]].head(3));
      //   oneTriangle1.vert[1] = (m_originalMesh.vbo[m_originalMesh.ibo[cnt * 4 + 1]].head(3));
      //   oneTriangle1.vert[2] = (m_originalMesh.vbo[m_originalMesh.ibo[cnt * 4 + 2]].head(3));
      //   oneTriangle1.norm = m_originalMesh.m_faceNormal[2 * cnt + 0];
      //   oneTriangle1.vertIdx[0] = m_originalMesh.ibo[cnt * 4 + 0];
      //   oneTriangle1.vertIdx[1] = m_originalMesh.ibo[cnt * 4 + 1];
      //   oneTriangle1.vertIdx[2] = m_originalMesh.ibo[cnt * 4 + 2];
        

      //   oneTriangle2.vert[0] = (m_originalMesh.vbo[m_originalMesh.ibo[cnt * 4 + 0]].head(3));
      //   oneTriangle2.vert[1] = (m_originalMesh.vbo[m_originalMesh.ibo[cnt * 4 + 2]].head(3));
      //   oneTriangle2.vert[2] = (m_originalMesh.vbo[m_originalMesh.ibo[cnt * 4 + 3]].head(3));
      //   oneTriangle2.norm = m_originalMesh.m_faceNormal[2 * cnt + 1];
      //   oneTriangle2.vertIdx[0] = m_originalMesh.ibo[cnt * 4 + 0];
      //   oneTriangle2.vertIdx[1] = m_originalMesh.ibo[cnt * 4 + 2];
      //   oneTriangle2.vertIdx[2] = m_originalMesh.ibo[cnt * 4 + 3];
        
      //   // triangleArr.push_back(oneTriangle1);
      //   // triangleArr.push_back(oneTriangle2);
      //   triangleArr[2 * cnt + 0] = oneTriangle1;
      //   triangleArr[2 * cnt + 1] = oneTriangle2;

      //   // triangle 1
      //   EdgeKey e01(m_originalMesh.ibo[cnt * 4 + 0], m_originalMesh.ibo[cnt * 4 + 1]);
      //   m_edgeMap[e01].push_back(cnt * 2 + 0);

      //   EdgeKey e12(m_originalMesh.ibo[cnt * 4 + 1], m_originalMesh.ibo[cnt * 4 + 2]);
      //   m_edgeMap[e12].push_back(cnt * 2 + 0);

      //   EdgeKey e02(m_originalMesh.ibo[cnt * 4 + 0], m_originalMesh.ibo[cnt * 4 + 2]);
      //   m_edgeMap[e02].push_back(cnt * 2 + 0);

      //   // triangle 2
      //   // EdgeKey e02(m_originalMesh.ibo[cnt * 4 + 0], m_originalMesh.ibo[cnt * 4 + 2]);
      //   m_edgeMap[e02].push_back(cnt * 2 + 1);

      //   EdgeKey e23(m_originalMesh.ibo[cnt * 4 + 2], m_originalMesh.ibo[cnt * 4 + 3]);
      //   m_edgeMap[e23].push_back(cnt * 2 + 1);

      //   EdgeKey e03(m_originalMesh.ibo[cnt * 4 + 0], m_originalMesh.ibo[cnt * 4 + 3]);
      //   m_edgeMap[e03].push_back(cnt * 2 + 1);

      //   m_vertNeighborMap[m_originalMesh.ibo[cnt * 4 + 0]].push_back(cnt * 2 + 0);
      //   m_vertNeighborMap[m_originalMesh.ibo[cnt * 4 + 0]].push_back(cnt * 2 + 1);

      //   m_vertNeighborMap[m_originalMesh.ibo[cnt * 4 + 1]].push_back(cnt * 2 + 0);

      //   m_vertNeighborMap[m_originalMesh.ibo[cnt * 4 + 2]].push_back(cnt * 2 + 0);
      //   m_vertNeighborMap[m_originalMesh.ibo[cnt * 4 + 2]].push_back(cnt * 2 + 1);

      //   m_vertNeighborMap[m_originalMesh.ibo[cnt * 4 + 3]].push_back(cnt * 2 + 1);
      // }

      // PrintHashMapAndTriangleArr();

      // thrust::host_vector<TriangleInfoDevice> triangleArrHost = triangleArr;
      // PrintHashMapAndTriangleArrCuda(triangleArrHost, edgeMap, vertMap, edgeKeysSet, numFaces);

      startTime = GetTimeMS();

      ValidateEdgeTriangleRelation();

      endTime = GetTimeMS();

      std::cout << "------ ValidateEdgeTriangleRelation " << (double)(endTime - startTime) << " ms" << std::endl;

      std::cout << "num of quad mesh in ibo : " << numFaces << std::endl;
      std::cout << "num of triangles out : " << triangleArr.size() << std::endl;

      // exit(0);
  }

  template<typename EdgeMap, typename VertMap>
  void CalcEachTriangleBlenderOut(EdgeMap& edgeMap, VertMap& vertMap, thrust::device_vector<TriangleInfoDevice>& triangleArr)
  {
      uint32_t numFaces = m_originalMesh.m_iboVec.size() / 3;

      triangleArr.resize(numFaces);

      printf("m_originalMesh.ibo size :%u, m_originalMesh.vbo size :%u ", m_originalMesh.m_iboVec.size(), m_originalMesh.m_vboVec.size());
      
      std::vector<uint32_t> reduceCnt(numFaces);

      unsigned long startTime = GetTimeMS();

      edgeMapKeyType  emptyEdgeKeySentinel{-1, -1};
      edgeMapValueType constexpr emptyEdgeValueSentinel = -1;

      vertMapKeyType emptyVertKeySentinel = -1;
      vertMapValueType constexpr emptyVertValueSentinel = -1;

      using edgeSetProbe = cuco::linear_probing<1, EdgeKeyCudaHasher>;

      auto edgeKeysSet = cuco::static_set{cuco::extent<std::size_t>{numFaces * 5},
                                  cuco::empty_key{emptyEdgeKeySentinel},
                                  EdgeKeyEqual{},
                                  edgeSetProbe{}};
      
      CalcEachTriangleCuda(triangleArr, numFaces, MESH_TYPE_BLENDER_OUT, edgeMap, vertMap, edgeKeysSet);
      unsigned long endTime = GetTimeMS();
    

      // parallel_for(blocked_range<uint32_t>(0, triangleArr.size()), ApplyCalcEachTriangleBlenderOut(m_edgeMapConcurrent, m_vertNbrMapConcurrent, m_originalMesh, triangleArr, reduceCnt));

      endTime = GetTimeMS();
      std::cout << "------ parallel_for ApplyCalcEachTriangleBlenderOut time : " << (double)(endTime - startTime) << " ms" << std::endl;

      ValidateEdgeTriangleRelation();

      // PrintHashMapAndTriangleArr(triangleArr);


      std::cout << "CalcEachTriangleBlenderOut num of mesh in ibo : " << numFaces << std::endl;
      std::cout << "CalcEachTriangleBlenderOut num of triangles out : " << triangleArr.size() << std::endl;

      // exit(0);
  }

  template<typename EdgeMap, typename VertMap>
  void CalcEachTriangleJianHengLiu(EdgeMap& edgeMap, VertMap& vertMap, thrust::device_vector<TriangleInfoDevice>& triangleArr)
  {
    uint32_t numFaces = m_originalMesh.m_iboVec.size() / 3;

    triangleArr.resize(numFaces);
      
    // std::vector<uint32_t> reduceErrCnt(numFaces);

    unsigned long startTime = GetTimeMS();

    edgeMapKeyType  emptyEdgeKeySentinel{-1, -1};
    edgeMapValueType constexpr emptyEdgeValueSentinel = -1;

    vertMapKeyType emptyVertKeySentinel = -1;
    vertMapValueType constexpr emptyVertValueSentinel = -1;

    using edgeSetProbe = cuco::linear_probing<1, EdgeKeyCudaHasher>;

    auto edgeKeysSet = cuco::static_set{cuco::extent<std::size_t>{numFaces * 5},
                                  cuco::empty_key{emptyEdgeKeySentinel},
                                  EdgeKeyEqual{},
                                  edgeSetProbe{}};

    // parallel_for(blocked_range<uint32_t>(0, triangleArr.size()), ApplyCalcEachTriangleJianHengLiu(m_edgeMapConcurrent, m_vertNbrMapConcurrent, m_originalMesh, triangleArr, reduceErrCnt));

    CalcEachTriangleCuda(triangleArr, numFaces, MESH_TYPE_M2_MAPPING, edgeMap, vertMap, edgeKeysSet);
  
    unsigned long endTime = GetTimeMS();
    std::cout << "------ parallel_for CalcEachTriangleCuda time : " << (double)(endTime - startTime) << "ms" << std::endl;

    ValidateEdgeTriangleRelation();

    // PrintHashMapAndTriangleArr(triangleArr);


    std::cout << "CalcEachTriangleJianHengLiu num of mesh in ibo : " << numFaces << std::endl;
    std::cout << "CalcEachTriangleJianHengLiu num of triangles out : " << triangleArr.size() << std::endl;

  }

  double GetAccurateLength(const glm::dvec3 v);
};
