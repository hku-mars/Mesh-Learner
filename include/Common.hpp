#ifndef __IMMESH_RENDER_COMMON_HPP__
#define __IMMESH_RENDER_COMMON_HPP__

#include <memory>
#include <string>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include "linmath.h"
#include <cuco/static_map.cuh>


#define MESH_HANDLE_THREAD_NUM (32)
#define MAX_VERT_NBR_NUM_EACH (32) // 5bits per dir
#define OFFSET_PER_PIX_CHANNEL (3)
#define SH_TEXTURE_CHANNEL_NUM (3)

#define VERT_NUMER_BIT_NUM (5)

#define MAX_MIP_MAP_LVL (2)

#define MAX_SH_NUMBER (64)

#define MAX_LVL1_SH_NUMBER (16)

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


typedef struct TriangleInfoDevice {
    float vert[9]; // counter clock wise
    float norm[3];
    uint32_t vertIdx[3];

    // __host__ __device__ 
    // TriangleInfoDevice& operator=(const TriangleInfoDevice& other) {
    //   vert[0] = other.vert[0]; vert[1] = other.vert[1]; vert[2] = other.vert[2];
    //   vert[3] = other.vert[3]; vert[4] = other.vert[4]; vert[5] = other.vert[5];
    //   vert[6] = other.vert[6]; vert[7] = other.vert[7]; vert[8] = other.vert[8];

    //   norm[0] = other.norm[0]; norm[1] = other.norm[1]; norm[2] = other.norm[2];

    //   vertIdx[0] = other.vertIdx[0]; vertIdx[1] = other.vertIdx[1]; vertIdx[2] = other.vertIdx[2];
    // }
} TriangleInfoDevice;

typedef struct EdgeKeyCuda {
  int32_t m_e0{-1};
  int32_t m_e1{-1};

  __host__ __device__ EdgeKeyCuda(int32_t e0_, int32_t e1_): m_e0{e0_}, m_e1{e1_} {};
  __host__ __device__ EdgeKeyCuda() {};

  __device__ bool operator==(EdgeKeyCuda const& other) const
  {
    return (((m_e0 == other.m_e0) && (m_e1 == other.m_e1)) || ((m_e0 == other.m_e1) && (m_e1 == other.m_e0)));
  }
} EdgeKeyCuda;

typedef struct EdgeKeyEqual {
  __device__ bool operator()(EdgeKeyCuda const& lhs, EdgeKeyCuda const& rhs) const noexcept
  {
    return lhs == rhs;
  }
} EdgeKeyEqual;

typedef struct EdgeKeyCudaHasher {
  cuco::default_hash_function<uint32_t> hasher;
  __device__ uint32_t operator()(EdgeKeyCuda const& input) const noexcept
  {
    return hasher(input.m_e0) ^ hasher(input.m_e1);
  }
} EdgeKeyCudaHasher;

// typedef enum EdgeTypeEdgeMap {
//     EDGE_LEFT_EDGE_MAP = 0,
//     EDGE_BOTTOM_EDGE_MAP,
//     EDGE_RIGHT_EDGE_MAP,
//     EDGE_MAX_EDGE_MAP
// } EdgeTypeEdgeMap;

typedef enum EdgeTypeCUDA
{
    EDGE_TYPE_CUDA_LEFT = 0,
    EDGE_TYPE_CUDA_RIGHT,
    EDGE_TYPE_CUDA_BOTTOM,
    EDGE_TYPE_CUDA_MAX,
} EdgeTypeCUDA;

typedef enum VertNbrType {
    VERT_NBR_TYPE_RIGHT = 0,
    VERT_NBR_TYPE_LEFT,
    VERT_NBR_TYPE_UP,
    VERT_NBR_TYPE_NUM
} VertNbrType;

typedef struct EdgeNbrInfo {
  uint32_t texId;
  EdgeTypeCUDA edgeType;
} EdgeNbrInfo;

__device__ int32_t ConstructValueMapKey(uint32_t tid, EdgeTypeCUDA edgeType)
{
  int32_t high30bits = (tid << 2);

  return (high30bits | edgeType);
}

typedef union AlignedPixLocation {
    struct {
        uint16_t x;
        uint16_t y;
    } data;
    uint32_t aligner; // Ensures 4-byte alignment
} AlignedPixLocation;

typedef struct PixInfo {
    AlignedPixLocation pixLoc;
    float depth;

    PixInfo(uint16_t x_, uint16_t y_, float depth_) {
      pixLoc.data.x = x_;
      pixLoc.data.y = y_;
      depth = depth_;
    }
} PixInfo;

typedef struct PixLocation {
  uint32_t x = {0};
  uint32_t y = {0};

  __host__ __device__ PixLocation() {};
  __host__ __device__ PixLocation(uint32_t x_, uint32_t y_): x{x_}, y{y_} {};
} PixLocation;

typedef struct MeshEdgeNbrInfo {
    uint32_t        nbrTexId[3] = {0};  // L R B
    uint32_t        offsetInHash[3] = {0};
    EdgeTypeCUDA nbrEdgeType[3] = {EDGE_TYPE_CUDA_MAX};
} MeshEdgeNbrInfo;

typedef struct MeshVertNbrInfo {
    // uint8_t topNbrNum = {0};
    // uint8_t leftNbrNum = {0};
    // uint8_t rightNbrNum = {0};

    uint8_t  nbrNum[VERT_NBR_TYPE_NUM] = {0}; // R, L, U
    uint32_t nbrTexIds[VERT_NBR_TYPE_NUM][MAX_VERT_NBR_NUM_EACH] = {0}; // R, L, U
    uint32_t nbrTexOffsetsInHash[VERT_NBR_TYPE_NUM][MAX_VERT_NBR_NUM_EACH] = {0}; // R, L, U
    // uint32_t topNbrTexIds[MAX_VERT_NBR_NUM_EACH] = {0};
    // uint32_t leftNbrTexIds[MAX_VERT_NBR_NUM_EACH] = {0};
    // uint32_t rightNbrTexIds[MAX_VERT_NBR_NUM_EACH] = {0};

    // uint32_t topNbrOffsetInHash[MAX_VERT_NBR_NUM_EACH] = {0};
    // uint32_t leftNbrOffsetInHash[MAX_VERT_NBR_NUM_EACH] = {0};
    // uint32_t rightNbrOffsetInHash[MAX_VERT_NBR_NUM_EACH] = {0};

} MeshVertNbrInfo;

typedef struct TexPixInfo
{
    uint32_t texId;
    uint32_t offset;
    uint32_t pixNum;
} TexPixInfo;

typedef struct DataAndMutexOffset
{
    int32_t dataOffset;
    int32_t mutexOffset;

    __host__ __device__ DataAndMutexOffset() {};
    __host__ __device__ DataAndMutexOffset(int32_t DataOffset_, int32_t mutexOffset_): dataOffset(DataOffset_), mutexOffset(mutexOffset_)  {};

} DataAndMutexOffset;

typedef struct TexIdDummyKey {
    uint32_t texId;
    uint32_t dummy;

    __device__ bool operator==(TexIdDummyKey const& other) const { return texId == other.texId; }
} TexIdDummyKey;

typedef struct TexIdDummyKeyEqual {
  __device__ bool operator()(const TexIdDummyKey& lhs, const TexIdDummyKey& rhs)
  {
    return lhs.texId == rhs.texId;
  }
} TexIdDummyKeyEqual;

typedef struct TexIdDummyKeyHash {
  // __host__ __device__ hash_key_pair() : hash_key_pair{0} {}
  // __host__ __device__ hash_key_pair(uint32_t offset) : offset_(offset) {}
  __device__ uint32_t operator()(TexIdDummyKey k) const { return hasher(k.texId); };

  cuco::default_hash_function<uint32_t> hasher;
  // uint32_t offset_;
} TexIdDummyKeyHash;


struct AssignValue {
    __host__ __device__ DataAndMutexOffset operator()(const thrust::tuple<TexPixInfo, uint32_t>& t) const {
        int32_t dataOffset = (int32_t)(((TexPixInfo)(thrust::get<0>(t))).offset);
        int32_t mutexOffset = (int32_t)(thrust::get<1>(t));
        return DataAndMutexOffset{dataOffset, mutexOffset};
    }
};

typedef union CurTexAlignedWH {
    struct {
        uint16_t curTexW;
        uint16_t curTexH;
    } data;
    uint32_t aligner; // Ensures 4-byte alignment
} CurTexAlignedWH;

typedef struct CurTexWH {
    uint32_t texW;
    uint32_t texH;

    CurTexWH(uint32_t w, uint32_t h) {
      texW = w;
      texH = h;
    }

    bool operator==(const CurTexWH& other) const {
        return ((other.texW == texW) && (other.texH == texH));
    }    
} CurTexWH;

struct HashCurTexWH {
    size_t operator()(const CurTexWH& curTexWH) const {
        return std::hash<uint32_t>()(curTexWH.texW) ^ std::hash<uint32_t>()(curTexWH.texH);
    }
};

typedef union TexAlignedPosWH {
  struct {
      uint16_t posW;
      uint16_t posH;
  } data;
  uint32_t aligner; // Ensures 4-byte alignment
} TexAlignedPosWH;

typedef struct JsonData {
    uint32_t index;
    uint32_t texW;
    uint32_t texH;
    uint32_t perTexOffset;
    float shDensity;

    JsonData() : texW(0), texH(0), shDensity(0.0f), perTexOffset(0), index(0) {}
} JsonData;

// Structure to hold index and data for each element
typedef struct ElementData {
    uint32_t magic{0xDEADBEEF}; // Magic number for identification
    uint32_t index;
    uint32_t perTexOffset;
    std::vector<uint8_t> data;
} ElementData;

void MatConvert(mat4x4& outMat, std::vector<float> inMat);

#endif