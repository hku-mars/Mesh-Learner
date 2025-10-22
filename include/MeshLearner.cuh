#ifndef __MESH_LEARNER_CUH_
#define __MESH_LEARNER_CUH_

#include <opencv2/stitching.hpp>

#include <GL/glew.h>

#include "include/linmath.h"

#include <stdlib.h>
#include <stdio.h>

// #include "kernel.cuh"
#include <Eigen/Core>
#include <Eigen/Dense>

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/api/include/torch/optim/schedulers/reduce_on_plateau_scheduler.h>

#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include "third_party/tensorboardCpp/record/recorder.h"

#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>

#define PER_TRIANGLE_OFFSET_CUDA (15)
#define PER_POINT_OFFSET_CUDA (5)

#include "include/my_common.hpp"

#include <unordered_map>

#include "oneapi/tbb/concurrent_hash_map.h"
#include "oneapi/tbb/concurrent_unordered_set.h"
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include <oneapi/tbb/concurrent_vector.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <future>

#include "cuda/forwardHashAcc.cuh"
#include "cuda/backwardHashAcc.cuh"


namespace MeshLearner {

enum VertType
{
    VERT_TYPE_LEFT = 0,
    VERT_TYPE_RIGHT,
    VERT_TYPE_TOP,
    VERT_TYPE_MAX,
};

struct CameraConfig {
    float fx;
    float fy;
    float cx;
    float cy;

    float scaleFactor{2.0f};
};

typedef enum TexResizeDir
{
    TEX_UP_RESOLUTION = 0,
    TEX_DOWN_RESOLUTION,
    TEX_OPT_MAX,
} TexResizeDir;


typedef struct ResizeInfo {
    bool needResize{false};

    float depthAvg = 0.0f;    
    // float psnrAvg = 0.f;
    // float psnrAvgLast = 0.f;
    // float psnrVar = 0.f;
    // float psnrVarLast = 0.f;

    
    float l1LossAvg = 0.f;
    float l1LossAvgLast = 0.f;
    float l1LossVar = 0.f;
    float l1LossVarLast = 0.f;
    
    uint32_t badCount{0};

    TexResizeDir lastResizeDir{TEX_OPT_MAX};
    
    float curDensity{0.f};
    float prevDensity{0.f};

    uint32_t resizeCnt{0};
    // bool needReturn{false};
    uint32_t returnCnt{0};

    bool isConverge{false};
    bool isProbe{false};

    float probeDensity{0.0f};
    TexResizeDir probeDir{TEX_OPT_MAX};

    float triangleBotDistance;
    float triangleHeight;

    CurTexAlignedWH curTexWH;
    CurTexAlignedWH LastTexWH;
} ResizeInfo;

typedef struct HashValue {
    // int64_t texId;
    // at::Tensor shTexValue;
    // at::Tensor adamStateExpAvg;
    // at::Tensor adamStateExpAvgSq;
    
    float* shTexValue;

    float* adamStateExpAvg;
    float* adamStateExpAvgSq;
    uint32_t perTexOffset{0};
    // std::vector<float> edgeCoeffs;

    float bottomYCoef;
    float topYCoef;
    float* worldPoseMap;
    TexAlignedPosWH* validShWHMap;
    TexAlignedPosWH* validShWHMapNew;
    uint32_t validShNum{0};
    uint32_t validShNumNew{0};

    uint32_t neighborTexId[3] = {0}; // L R Bottom
    EdgeTypeCUDA neighborEdgeType[3] = {EDGE_TYPE_CUDA_MAX};
    float cornerArea[2] = {0}; // Up is a constant as long as texture reso is invariant, Left 2, Right 2, bottom height is constant
    std::vector<uint32_t> topVertNbr;
    std::vector<uint32_t> leftVertNbr;
    std::vector<uint32_t> rightVertNbr;

    float texInWorldInfo[6] = {0}; // u dir, v dir

    std::vector<float> depth;
    std::vector<float> psnr;
    std::vector<float> l1Loss;
    
    ResizeInfo resizeInfo;
    float norm[3];

    // float depthAvg = 0.0f;    
    // float psnrAvg = 0.f;
    // float psnrAvgLast = 0.f;
    // float psnrVar = 0.f;
    // float psnrVarLast = 0.f;

    
    // float l1LossAvg = 0.f;
    // float l1LossAvgLast = 0.f;
    // float l1LossVar = 0.f;
    // float l1LossVarLast = 0.f;
    
    // uint32_t badCount{0};

    // TexResizeDir lastResizeDir{TEX_OPT_MAX};
    
    // float curDensity{0.f};
    // float prevDensity{0.f};

    // uint32_t resizeCnt{0};
    // // bool needReturn{false};
    // uint32_t returnCnt{0};

    // bool isConverge{false};
    // bool isProbe{false};

    // float probeDensity{0.0f};
    // TexResizeDir probeDir{TEX_OPT_MAX};

    // float triangleBotDistance;
    // float triangleHeight;
} HashValue;

struct CopyWHsFromGlbHashMap {
    __host__ CurTexAlignedWH operator()(const HashValue& src)
    {
        CurTexAlignedWH dst;
        dst.data.curTexW = src.resizeInfo.curTexWH.data.curTexW;
        dst.data.curTexH = src.resizeInfo.curTexWH.data.curTexH;

        return dst;
    }
};


typedef struct HashValPreDifinedDevice {
    uint32_t neighborTexId[3] = {0};
    EdgeTypeCUDA edgeType[3] = {EDGE_TYPE_CUDA_MAX};
    float triangleBotDistance;
    float triangleHeight;

    uint32_t curTexW;
    uint32_t curTexH;

    float cornerArea[2] = {0}; 

    uint32_t topVertNbrNum{0};
    uint32_t topPtIdx;

    uint32_t leftVertNbrNum{0};
    uint32_t leftPtIdx;

    uint32_t rightVertNbrNum{0};
    uint32_t rightPtIdx;

    float norm[3];
} HashValPreDifinedDevice;

struct CopyWHsFromHashValPreDifinedDeviceFunc {
    __host__ __device__ CurTexAlignedWH operator()(const HashValPreDifinedDevice& src)
    {
        CurTexAlignedWH dst;
        dst.data.curTexW = (uint16_t)src.curTexW;
        dst.data.curTexH = (uint16_t)src.curTexH;
        return dst;
    }
};

typedef struct VertNbrMemLayout {
    uint32_t offset;
    std::vector<uint32_t> topVertNbrValid;
    std::vector<uint32_t> leftVertNbrValid;
    std::vector<uint32_t> rightVertNbrValid;
} VertNbrMemLayout;

typedef struct VertNbrMemInfoTmp {
    uint32_t offset;

    uint32_t topPtIdx;
    uint32_t topVertNum;

    uint32_t leftPtIdx;
    uint32_t leftVertNum;

    uint32_t rightPtIdx;
    uint32_t rightVertNum;
} VertNbrMemInfoTmp;

typedef struct TexScaleFactor {
    float distPerTexelW = {0};
    float distPerTexelH = {0};
} TexScaleFactor;

enum SceneType
{
    SCENE_TYPE_LIVO2 = 0,
    SCENE_TYPE_REPLICA,
    SCENE_TYPE_BLENDER_OUT,
    SCENE_TYPE_MAX,
};

#define PER_TRIANGLE_OFFSET (15)
#define PER_POINT_OFFSET (5)

struct GtData
{
    Eigen::Matrix4f viewMat;
    std::vector<float> camWorldPos;
    at::Tensor img;

    GtData() {};
    ~GtData() {};

    GtData (const GtData& src)
    {
        img = src.img;
        camWorldPos.assign(src.camWorldPos.begin(), src.camWorldPos.end());
        viewMat = src.viewMat;
    }

    GtData& operator=(const GtData& src)
    {
        img = src.img;
        camWorldPos.assign(src.camWorldPos.begin(), src.camWorldPos.end());
        viewMat = src.viewMat;
        return *this;
    }


};

using my_impl_type = cuco::detail::open_addressing_impl<uint32_t,
                                                 cuco::pair<uint32_t, uint64_t>,
                                                 cuco::extent<std::size_t>,
                                                 cuda::thread_scope_device,
                                                 thrust::equal_to<uint32_t>,
                                                 cuco::linear_probing<1, cuco::default_hash_function<uint32_t>>,
                                                 cuco::cuda_allocator<cuco::pair<uint32_t, uint64_t>>,
                                                 cuco::storage<1>>;


template <uint32_t tile_size, typename Map>
__global__ void StatisticTexAndPixRelationCuda(Map map, cudaTextureObject_t texScaleFactorDepthTexId, uint32_t imgW, uint32_t imgH, float* outImg, unsigned char* maskImg, uint32_t* addMapCountPtr)
{
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int offsetPerPixelChannel = 3; // because rendered img that calc loss is RGB 3 channel
    unsigned int offsetPerPixelLine = imgW * offsetPerPixelChannel; // RGB

    auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());

    if (x >= imgW || y >= imgH) {
        return;
    }

    // don't care order, reset all pixel
    *(maskImg + (imgH - y - 1) * imgW + x) = 0;
    *(outImg + (imgH - y - 1) * offsetPerPixelLine + x * offsetPerPixelChannel) = 0.f;
    *(outImg + (imgH - y - 1) * offsetPerPixelLine + x * offsetPerPixelChannel + 1) = 0.f;
    *(outImg + (imgH - y - 1) * offsetPerPixelLine + x * offsetPerPixelChannel + 2) = 0.f;

    float4 scaleFactorDepthTexId = tex2D<float4>(texScaleFactorDepthTexId, (float)x, (float)(imgH - y - 1));

    uint32_t texId = static_cast<uint32_t>(scaleFactorDepthTexId.w);

    if (texId <= 0) {
        // printf("exit tid X:%d, Y:%d, texId: %d\n", x, y, texId);
        return;
    }

    // printf("into tid X:%d, Y:%d, texId: %d, imgWH: %d, %d\n", x, y, texId, imgW, imgH);

    PixLocation loc(x, y);
    TexIdDummyKey tmp;
    tmp.texId = texId;
    map.insert(tile, cuco::pair{tmp, loc});

    atomicAdd(addMapCountPtr, 1);
}

__device__ uint32_t FindLongestEdgeCuda(float* aPtr, float* bPtr, float* cPtr)
{
    float epsilon = 1e-6;

    glm::vec3 a(aPtr[0], aPtr[1], aPtr[2]);
    glm::vec3 b(bPtr[0], bPtr[1], bPtr[2]);
    glm::vec3 c(cPtr[0], cPtr[1], cPtr[2]);

    glm::vec3 distAB = a - b;
    glm::vec3 distBC = b - c;
    glm::vec3 distAC = a - c;

    
    if (fabsf(glm::length(distAB) - glm::length(distBC)) <= epsilon &&
        fabsf(glm::length(distBC) - glm::length(distAC)) <= epsilon) {
        return 0;
    } else if ((glm::length(distAB) - glm::length(distBC)) > epsilon &&
               (glm::length(distAB) - glm::length(distAC)) > epsilon) {
        return 2; 
    } else if ((glm::length(distBC) - glm::length(distAB)) > epsilon &&
               (glm::length(distBC) - glm::length(distAC)) > epsilon) {
        return 0;
    } else {
        return 1;
    }
}

__device__ void CalcWorldPotisionCuda(float* writeAddr, glm::vec3 startPosition, glm::vec3 bottomDirStep, glm::vec3 upDirStep, uint8_t curW, uint8_t curH)
{
    glm::vec3 curShPosition = startPosition + (float)curW * bottomDirStep + (float)curH * upDirStep;

    *(writeAddr + 0) = curShPosition.x;
    *(writeAddr + 1) = curShPosition.y;
    *(writeAddr + 2) = curShPosition.z;
}

__device__ bool IsPointInsideTriangleCuda(double curPtX, double curPtY, glm::dvec3 vecL2R, glm::dvec3 vecR2U, glm::dvec3 vecU2L, glm::dvec2 topUV, glm::dvec2 leftUV, glm::dvec2 rightUV)
{
    glm::dvec3 curPt(curPtX, curPtY, 0.);

    glm::dvec3 topUV3f(topUV.x, topUV.y, 0.);
    glm::dvec3 leftUV3f(leftUV.x, leftUV.y, 0.);
    glm::dvec3 rightUV3f(rightUV.x, rightUV.y, 0.);

    glm::dvec3 vecL2Pt = curPt - leftUV3f;
    glm::dvec3 vecR2Pt = curPt - rightUV3f;
    glm::dvec3 vecU2Pt = curPt - topUV3f;

    if (((glm::cross(vecL2R, vecL2Pt)).z > 1e-5) &&
        ((glm::cross(vecR2U, vecR2Pt)).z > 1e-5) &&
        ((glm::cross(vecU2L, vecU2Pt)).z > 1e-5)) {
        return true;
    } else if (fabs((glm::cross(vecL2R, vecL2Pt)).z * (glm::cross(vecR2U, vecR2Pt)).z * (glm::cross(vecU2L, vecU2Pt)).z) <= 1e-5) {
        return true;
    } else {
        return false;
    }
}

template <typename PreDefHashValIt>
__device__ void UpdateCornerAreaCuda(PreDefHashValIt preDefHashBegin, float leftCornerShX, float rightCornerShX, unsigned int curTriangleIdx)
{
    (thrust::raw_reference_cast(*(preDefHashBegin + curTriangleIdx))).cornerArea[0] = leftCornerShX;
    (thrust::raw_reference_cast(*(preDefHashBegin + curTriangleIdx))).cornerArea[1] = rightCornerShX;
}

__device__ void UpdateTexInWorldInfo(float* texInWorldInfo, unsigned int curTriangleIdx, glm::dvec3 bottomLeftPt, glm::dvec3 bottomRightPt, glm::dvec3 topPt, glm::dvec3 foot)
{
    glm::dvec3 bottomUniVecL2R = glm::normalize(bottomRightPt - bottomLeftPt);
    glm::dvec3 uniVecUp = glm::normalize(topPt - foot);

    // U
    *(texInWorldInfo + curTriangleIdx * 6 + 0) = (float)(bottomUniVecL2R.x);
    *(texInWorldInfo + curTriangleIdx * 6 + 1) = (float)(bottomUniVecL2R.y);
    *(texInWorldInfo + curTriangleIdx * 6 + 2) = (float)(bottomUniVecL2R.z);

    // V
    *(texInWorldInfo + curTriangleIdx * 6 + 3) = (float)(uniVecUp.x);
    *(texInWorldInfo + curTriangleIdx * 6 + 4) = (float)(uniVecUp.y);
    *(texInWorldInfo + curTriangleIdx * 6 + 5) = (float)(uniVecUp.z);

    }

template <typename ScaleFactorIt, typename PreDefHashValIt>
__device__ void calc_in_texture_info_cuda(glm::vec3 topPt, glm::vec3 bottomLeft, glm::vec3 bottomRight, glm::dvec2 topUV, glm::dvec2 leftUV, glm::dvec2 rightUV, double k, unsigned int curTriangleIdx, ScaleFactorIt& scaleFactorBegin, PreDefHashValIt preDefHashBegin, float* texInWorldInfo, float shDensity, uint32_t maxWH)
{
    // double texelW = 1.f / (double)texW;
    // double texelH = 1.f / (double)texH;

    glm::dvec3 foot(k * ((double)bottomRight.x - (double)bottomLeft.x) + (double)bottomLeft.x, k * ((double)bottomRight.y - (double)bottomLeft.y) + (double)bottomLeft.y, k * ((double)bottomRight.z - (double)bottomLeft.z) + (double)bottomLeft.z);

    glm::dvec3 bottomRightDouble(bottomRight.x, bottomRight.y, bottomRight.z);
    glm::dvec3 bottomLeftDouble(bottomLeft.x, bottomLeft.y, bottomLeft.z);
    double bottomDistance = glm::length(bottomRightDouble - bottomLeftDouble);

    glm::dvec3 topPtDouble(topPt.x, topPt.y, topPt.z);
    double triangleRealHeight = glm::length(topPtDouble - foot);

    // (thrust::raw_reference_cast(*(scaleFactorBegin + curTriangleIdx))).distPerTexelW = (float)(bottomDistance / (double)((double)texW - (double)texPadding * 2.0));
    // (thrust::raw_reference_cast(*(scaleFactorBegin + curTriangleIdx))).distPerTexelH = (float)(triangleRealHeight / (double)((double)texH - (double)texPadding * 2.0));

    uint32_t tmpW = (uint32_t)(glm::ceil(bottomDistance / (double)shDensity));
    uint32_t tmpH = (uint32_t)(glm::ceil(triangleRealHeight / (double)shDensity));

    if ((isnan(bottomDistance) == true) || (isinf(bottomDistance) == true) ||
        (isnan(triangleRealHeight) == true) || (isinf(triangleRealHeight) == true)) {
            tmpH = 1;
            tmpW = 1;
    } else {
        if (tmpW <= 0) {
            tmpW = 1;
        }

        if (tmpH <= 0) {
            tmpH = 1;
        }
    }

    if (tmpH >= maxWH) {
        tmpH = maxWH;
    }

    if (tmpW >= maxWH) {
        tmpW = maxWH;
    }

    // if (curTriangleIdx == 873443) {
    //     printf("tmpW : %d, tmpH: %d\n", tmpW, tmpH);
    //     printf("bottomDistance : %f, triangleRealHeight: %f\n", (float)bottomDistance, (float)triangleRealHeight);
    //     printf("texResolution: %f\n", (float)texResolution);
    // }
    

    (thrust::raw_reference_cast(*(preDefHashBegin + curTriangleIdx))).curTexW = tmpW;
    (thrust::raw_reference_cast(*(preDefHashBegin + curTriangleIdx))).curTexH = tmpH;

    (thrust::raw_reference_cast(*(preDefHashBegin + curTriangleIdx))).triangleBotDistance = bottomDistance;
    (thrust::raw_reference_cast(*(preDefHashBegin + curTriangleIdx))).triangleHeight = triangleRealHeight;

    // uint8_t curTexW = (uint8_t)(glm::ceil(bottomDistance / (double)texResolution));
    // uint8_t curTexH = (uint8_t)(glm::ceil(triangleRealHeight / (double)texResolution));

    // double texelW = 1.f / (double)curTexW;
    // double texelH = 1.f / (double)curTexH;

    UpdateTexInWorldInfo(texInWorldInfo, curTriangleIdx, bottomLeftDouble, bottomRightDouble, topPtDouble, foot);

    // glm::dvec3 bottomDirStep = (bottomRightDouble - bottomLeftDouble) / (double)((double)curTexW - (double)texPadding * 2.0);
    // glm::dvec3 upDirStep = (topPtDouble - foot) / (double)((double)curTexH - (double)texPadding * 2.0);

    // glm::dvec3 startPosition = bottomLeftDouble + 0.5 * bottomDirStep + 0.5 * upDirStep;
    // double startH = texelH / 2.0;
    // double startW = texelW / 2.0;

    // glm::dvec3 topUV3d(topUV.x, topUV.y, 0.);
    // glm::dvec3 leftUV3d(leftUV.x, leftUV.y, 0.);
    // glm::dvec3 rightUV3d(rightUV.x, rightUV.y, 0.);

    // glm::dvec3 vecL2R = rightUV3d - leftUV3d;
    // glm::dvec3 vecR2U = topUV3d - rightUV3d;
    // glm::dvec3 vecU2L = leftUV3d - topUV3d;

    // double minLeftX = 999.; double maxRightX = -999.;

    // start from left down corner tex coord.
    // for (unsigned char curH = 0; curH < curTexH; curH++) {
    //     for (unsigned char curW = 0; curW < curTexW; curW++) {
    //         if (IsPointInsideTriangleCuda(startW + texelW * (double)curW, startH + texelH * (double)curH, vecL2R, vecR2U, vecU2L, topUV, leftUV, rightUV) == true) {

    //             CalcWorldPotisionCuda((worldPoseMap + curTriangleIdx * poseMapOffset + curH * curTexW * 3 + curW * 3), startPosition, bottomDirStep, upDirStep, curW, curH);

    //             if (curH == 0) {
    //                 if ((startW + texelW * (float)curW) <= minLeftX) {
    //                     minLeftX = (startW + texelW * (float)curW);
    //                 }

    //                 if ((startW + texelW * (float)curW) >= maxRightX) {
    //                     maxRightX = (startW + texelW * (float)curW);
    //                 }
    //             }
    //         }
    //     }
    // }

    // UpdateCornerAreaCuda(preDefHashBegin, minLeftX, maxRightX, curTriangleIdx);
}

__device__ void AssignVertexBufImplCuda(float* outBufPtr, glm::vec3 p, float texU, float texV, uint32_t curTriIdx, uint8_t curPoint)
{
    *(outBufPtr + curTriIdx * PER_TRIANGLE_OFFSET_CUDA + curPoint * PER_POINT_OFFSET_CUDA + 0) = p.x;
    *(outBufPtr + curTriIdx * PER_TRIANGLE_OFFSET_CUDA + curPoint * PER_POINT_OFFSET_CUDA + 1) = p.y;
    *(outBufPtr + curTriIdx * PER_TRIANGLE_OFFSET_CUDA + curPoint * PER_POINT_OFFSET_CUDA + 2) = p.z;

    *(outBufPtr + curTriIdx * PER_TRIANGLE_OFFSET_CUDA + curPoint * PER_POINT_OFFSET_CUDA + 3) = texU;
    *(outBufPtr + curTriIdx * PER_TRIANGLE_OFFSET_CUDA + curPoint * PER_POINT_OFFSET_CUDA + 4) = texV;
}

__device__ double AccurateLength(const glm::dvec3 v) {
	// return sqrtl(v.x * v.x + v.y * v.y + v.z * v.z);
    return hypot(hypot((double)v.x, (double)v.y), (double)v.z);
}

__device__ uint8_t IfNeedFlipCuda(glm::vec3 topPt, glm::vec3 assumedBottomLeft, glm::vec3 assumedBottomRight, glm::vec3 readedNorm, float& deltaLen)
{
    glm::dvec3 topPtDouble(topPt.x, topPt.y, topPt.z);
    glm::dvec3 assumedBottomLeftDouble(assumedBottomLeft.x, assumedBottomLeft.y, assumedBottomLeft.z);
    glm::dvec3 assumedBottomRightDouble(assumedBottomRight.x, assumedBottomRight.y, assumedBottomRight.z);

    glm::dvec3 vec1Double = glm::normalize(assumedBottomLeftDouble - topPtDouble);
    glm::dvec3 vec2Double = glm::normalize(assumedBottomRightDouble - assumedBottomLeftDouble);

    // glm::vec3 vec1 = glm::normalize(assumedBottomLeft - topPt);
    // glm::vec3 vec2 = glm::normalize(assumedBottomRight - assumedBottomLeft);

    // glm::vec3 assumedNorm = glm::normalize(glm::cross(vec1, vec2));
    // glm::vec3 delta = assumedNorm - readedNorm;
    

    // deltaLen = glm::length(delta);

    // glm::vec3 vec1 = assumedBottomLeft - topPt;
    // glm::vec3 vec2 = assumedBottomRight - assumedBottomLeft;

    // glm::vec3 crossed = glm::cross(vec1, vec2);
    glm::dvec3 crossedDouble = glm::cross(vec1Double, vec2Double);
    double realLen = AccurateLength(crossedDouble);

    // Eigen::Vector3f assumedNorm = vec1.cross(vec2).normalized();
    // glm::vec3 assumedNorm = glm::normalize(glm::cross(vec1, vec2));
    // glm::vec3 assumedNorm(crossed.x / realLen, crossed.y / realLen, crossed.z / realLen);
    glm::dvec3 assumedNormDouble(crossedDouble.x / realLen, crossedDouble.y / realLen, crossedDouble.z / realLen);

    glm::vec3 assumedNorm((float)assumedNormDouble.x, (float)assumedNormDouble.y, (float)assumedNormDouble.z);

    glm::vec3 delta = assumedNorm - readedNorm;

    deltaLen = glm::length(delta);

    // if ((threadIdx.x + blockIdx.x * blockDim.x) == 20) {
    //     printf("20 glm deltaLen : %f\n", deltaLen);
    //     // printf("vec1 : %f, %f, %f\n", vec1.x, vec1.y, vec1.z);
    //     // printf("vec2 : %f, %f, %f\n", vec2.x, vec2.y, vec2.z);
    //     printf("vec1Double : %lf, %lf, %lf\n", vec1Double.x, vec1Double.y, vec1Double.z);
    //     printf("vec2Double : %lf, %lf, %lf\n", vec2Double.x, vec2Double.y, vec2Double.z);
    //     printf("assumedNorm : %f, %f, %f\n", assumedNorm.x, assumedNorm.y, assumedNorm.z);
    //     printf("assumedNormDouble : %lf, %lf, %lf\n", assumedNormDouble.x, assumedNormDouble.y, assumedNormDouble.z);
    //     printf("readedNorm :%f, %f, %f \n", readedNorm.x, readedNorm.y, readedNorm.z);
    // }

    if (isnan(deltaLen) == true) {
        return 0;
    }

    if (glm::length(delta) <= 1e-2) {
        return 0;
    } else {
        return 1;
    }

    // if (glm::epsilonEqual(assumedNorm, readedNorm, 1e-2f) == glm::bvec3(true, true, true)) {
    //     return 0;
    // } else {
    //     return 1;
    // }
}

__device__ uint8_t IfNeedFlipCudaEigen(glm::vec3 topPt, glm::vec3 assumedBottomLeft, glm::vec3 assumedBottomRight, glm::vec3 readedNorm, float& deltaLen)
{
    Eigen::Vector3f topPtEigen(topPt.x, topPt.y, topPt.z);
    Eigen::Vector3f assumedBottomLeftEigen(assumedBottomLeft.x, assumedBottomLeft.y, assumedBottomLeft.z);
    Eigen::Vector3f assumedBottomRightEigen(assumedBottomRight.x, assumedBottomRight.y, assumedBottomRight.z);
    Eigen::Vector3f vec1 = (assumedBottomLeftEigen - topPtEigen).normalized();
    Eigen::Vector3f vec2 = (assumedBottomRightEigen - assumedBottomLeftEigen).normalized();

    // glm::vec3 crossed = glm::cross(vec1, vec2);
    // glm::dvec3 crossedDouble = glm::cross(vec1Double, vec2Double);
    // double realLen = AccurateLength(crossedDouble);

    Eigen::Vector3f assumedNorm = vec1.cross(vec2).normalized();
    Eigen::Vector3f readedNormEigen(readedNorm.x, readedNorm.y, readedNorm.z);

    Eigen::Vector3f delta = assumedNorm - readedNormEigen;

    deltaLen = delta.norm();

    if ((threadIdx.x + blockIdx.x * blockDim.x) == 1177213) {
        printf("1177213 eigen deltaLen : %f\n", deltaLen);
        printf("vec1 : %f, %f, %f\n", vec1(0), vec1(1), vec1(2));
        printf("vec2 : %f, %f, %f\n", vec2(0), vec2(1), vec2(2));
        printf("assumedNorm : %f, %f, %f\n", assumedNorm(0), assumedNorm(1), assumedNorm(2));
        printf("readedNorm :%f, %f, %f \n", readedNorm.x, readedNorm.y, readedNorm.z);
        printf("readedNormEigen :%f, %f, %f \n", readedNormEigen(0), readedNormEigen(1), readedNormEigen(2));
        
    }

    if (deltaLen <= 1e-2) {
        return 0;
    } else {
        return 1;
    }

    // if (glm::epsilonEqual(assumedNorm, readedNorm, 1e-2f) == glm::bvec3(true, true, true)) {
    //     return 0;
    // } else {
    //     return 1;
    // }
}

__device__ void CalcTexUVCuda(double k, glm::dvec2& topPtUV, glm::dvec2& bottomLeftPtUV, glm::dvec2& bottomRightPtUV)
{
    // double texelW = 1.0 / (double)texW;
    // double texelH = 1.0 / (double)texH;

    // double texCoordStartPosW = texelW * (double)texPadding;
    // double texCoordEndPosW = 1.0 - texelW * (double)texPadding;

    double texCoordStartPosW = 0.0;
    double texCoordEndPosW = 1.0;

    // double texCoordStartPosH = texelH * (double)texPadding;
    // double texCoordEndPosH = 1.0 - texelH * (double)texPadding;

    double texCoordStartPosH = 0.0;
    double texCoordEndPosH = 1.0;

    topPtUV.x = texCoordStartPosW + (texCoordEndPosW - texCoordStartPosW) * k;
    topPtUV.y = texCoordEndPosH;

    bottomLeftPtUV.x = texCoordStartPosW;
    bottomLeftPtUV.y = texCoordStartPosH;

    bottomRightPtUV.x = texCoordEndPosW;
    bottomRightPtUV.y = texCoordStartPosH;
}


__device__ bool CalcSpaceLineKCuda(glm::vec3 topPt, glm::vec3 assumedBottomLeft, glm::vec3 assumedBottomRight, glm::vec3 readedNorm, uint8_t& flip, double& k, float& deltaLen)
{
    flip = IfNeedFlipCuda(topPt, assumedBottomLeft, assumedBottomRight, readedNorm, deltaLen);
    // flip = IfNeedFlipCudaEigen(topPt, assumedBottomLeft, assumedBottomRight, readedNorm, deltaLen);
    
    glm::dvec3 bottomLeft(0., 0., 0.);
    glm::dvec3 bottomRight(0., 0., 0.);

    if (flip == 1) {
        bottomLeft.x = (double)assumedBottomRight.x;
        bottomLeft.y = (double)assumedBottomRight.y;
        bottomLeft.z = (double)assumedBottomRight.z;

        bottomRight.x = (double)assumedBottomLeft.x;
        bottomRight.y = (double)assumedBottomLeft.y;
        bottomRight.z = (double)assumedBottomLeft.z;
    } else {
        bottomLeft.x = (double)assumedBottomLeft.x;
        bottomLeft.y = (double)assumedBottomLeft.y;
        bottomLeft.z = (double)assumedBottomLeft.z;

        bottomRight.x = (double)assumedBottomRight.x;
        bottomRight.y = (double)assumedBottomRight.y;
        bottomRight.z = (double)assumedBottomRight.z;
    }

    k = - ((bottomLeft.x - (double)topPt.x) * (bottomRight.x - bottomLeft.x) + (bottomLeft.y - (double)topPt.y) * (bottomRight.y - bottomLeft.y) + (bottomLeft.z - (double)topPt.z) * (bottomRight.z - bottomLeft.z)) / \
         ((bottomRight.x - bottomLeft.x) * (bottomRight.x - bottomLeft.x) + (bottomRight.y - bottomLeft.y) * (bottomRight.y - bottomLeft.y) + (bottomRight.z - bottomLeft.z) * (bottomRight.z - bottomLeft.z));
    
    if (k < 0) {
        printf("k is negative, invalid, topPt : %f, %f, %f, assumedBottomLeft: %f, %f, %f, assumedBottomRight: %f, %f, %f, readedNorm: %f, %f, %f,  flip : %d\n", (double)topPt.x, (double)topPt.y, (double)topPt.z,
                      assumedBottomLeft.x, assumedBottomLeft.y, assumedBottomLeft.z,
                      assumedBottomRight.x, assumedBottomRight.y, assumedBottomRight.z,
                      readedNorm.x, readedNorm.y, readedNorm.z,
                      flip);
        return false;
    }

    return true;
}

template <typename EdgeMap, typename TileType, typename KeyEqual>
__device__ uint32_t GetEdgeMapCount(EdgeMap edgeMap, TileType& tile, EdgeKeyCuda edge, KeyEqual equalFunc)
{
    return edgeMap.count(tile, edge, equalFunc);
}

__device__ void DecodeEdgeMapValue(int32_t value, uint32_t& triIdx, EdgeTypeCUDA& edgeType)
{
    triIdx = ((uint32_t)(value >> 2));
    edgeType = (EdgeTypeCUDA)(value & 0x3);
}

template <typename TileType, typename EdgeMap, typename KeyEqual, typename ProbeCgType, typename FlushCgType, typename PreDefHashValIt, typename AtomicT>
__device__ void UpdateOneNeighborCuda(ProbeCgType& probCg, FlushCgType& flushCg, TileType& tile, EdgeKeyCuda edge, PreDefHashValIt& preDefHashBegin, EdgeMap edgeMap, KeyEqual equalFunc, uint32_t curTriIdx, EdgeTypeCUDA type, AtomicT* atomicCnter)
{
    using pair_type = typename EdgeMap::value_type;
    uint32_t constexpr bufSize = 2;
    pair_type outBufFlush[bufSize];
    pair_type outBuf[bufSize];
    uint32_t flushCnt = 0;
    // cuda::atomic<size_t> atomicCnt{0};

    uint32_t findCnt = 0;
    findCnt = GetEdgeMapCount(edgeMap, tile, edge, equalFunc);

    // findCnt = edgeMap.count(tile, edge, equalFunc);
    if (findCnt == 2) {
        edgeMap.retrieve<bufSize>(flushCg, probCg, edge, &flushCnt, outBufFlush, atomicCnter, outBuf, equalFunc);
        uint32_t triIdx0 = 0;
        uint32_t triIdx1 = 0;
        EdgeTypeCUDA edgeType0 = EDGE_TYPE_CUDA_MAX;
        EdgeTypeCUDA edgeType1 = EDGE_TYPE_CUDA_MAX;

        DecodeEdgeMapValue(outBuf[0].second, triIdx0, edgeType0);
        DecodeEdgeMapValue(outBuf[1].second, triIdx1, edgeType1);
        

        if (flushCnt > 0) {
            edgeMap.flush_output_buffer(flushCg, flushCnt, outBufFlush, atomicCnter, outBuf);
        }

        if ((triIdx0 != curTriIdx) && (triIdx1 == curTriIdx)) {
            // DecodeEdgeMapValue(outBuf[0].second, texId, edgeType);
            // (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).neighborTexId[type] = (outBuf[0].second + 1);
            (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).neighborTexId[type] = triIdx0 + 1;
            (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).edgeType[type] = edgeType0;
        } else if ((triIdx0 == curTriIdx) && (triIdx1 != curTriIdx)) {
            // DecodeEdgeMapValue(outBuf[1].second, texId, edgeType);
            // (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).neighborTexId[type] = (outBuf[1].second + 1);
            (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).neighborTexId[type] = triIdx1 + 1;
            (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).edgeType[type] = edgeType1;
        } else {
            printf("decode value two triangle idx are not as expected %d %d %d\n", triIdx0, triIdx1, curTriIdx);
        }
    } else {
        printf("invalid edge, cnt: %d, edge: %d %d\n", findCnt, edge.m_e0, edge.m_e1);
    }

    // else if (findCnt == 1) {
    //     edgeMap.retrieve<bufSize>(flushCg, probCg, edge, &flushCnt, outBufFlush, &atomicCnt, outBuf, equalFunc);

    //     if (flushCnt > 0) {
    //         edgeMap.flush_output_buffer(flushCg, flushCnt, outBufFlush, &atomicCnt, outBuf);
    //     }

    //     if (outBuf[0].second != curTriIdx) {
    //         printf("one edge is not same %d\n", outBuf[0].second);
    //     }
    // }
}

template <typename TileType, typename EdgeMap, typename KeyEqual, typename PreDefHashValIt, typename ProbeCgType, typename FlushCgType, typename AtomicT>
__device__ void UpdateEdgeNbrInfoCuda(TileType& tile, PreDefHashValIt preDefHashBegin, EdgeMap edgeMap, uint32_t topPtIdx, uint32_t leftBotPtIdx, uint32_t rightBotPtIdx, uint32_t curTriIdx, ProbeCgType& probCg, FlushCgType& flushCg, KeyEqual equalFunc, AtomicT* atomicCnter)
{
    EdgeKeyCuda leftEdge(topPtIdx, leftBotPtIdx);
    EdgeKeyCuda rightEdge(topPtIdx, rightBotPtIdx);
    EdgeKeyCuda botEdge(leftBotPtIdx, rightBotPtIdx);

    UpdateOneNeighborCuda(probCg, flushCg, tile, leftEdge, preDefHashBegin, edgeMap, equalFunc, curTriIdx, EDGE_TYPE_CUDA_LEFT, atomicCnter);

    UpdateOneNeighborCuda(probCg, flushCg, tile, rightEdge, preDefHashBegin, edgeMap, equalFunc, curTriIdx, EDGE_TYPE_CUDA_RIGHT, atomicCnter + 1);

    UpdateOneNeighborCuda(probCg, flushCg, tile, botEdge, preDefHashBegin, edgeMap, equalFunc, curTriIdx, EDGE_TYPE_CUDA_BOTTOM, atomicCnter + 2);
}

template <typename EdgeMap, typename VertMap, typename ScaleFactorIt, typename TileType, typename PreDefHashValIt, typename ProbeCgType, typename FlushCgType, typename KeyEqual, typename AtomicT>
__device__ void assign_vertex_buf_cuda(TileType& tile, PreDefHashValIt preDefHashBegin, ScaleFactorIt scaleFactorBegin, float* outBufPtr, EdgeMap edgeMap, VertMap vertMap, glm::vec3* vertArr, glm::vec3 norm, uint8_t triangleTypeIdx, uint32_t curTriIdx, uint32_t* vertIndicies, uint32_t& invalidKCnt, ProbeCgType& probCg, FlushCgType& flushCg, KeyEqual equalFunc, uint8_t& flip, AtomicT* atomicCnter, float& deltaLen, float* texInWorldInfo, float shDensity, uint32_t maxWH)
{
    // glm::vec3 pa(paPtr[0], paPtr[1], paPtr[2]);
    // glm::vec3 pb(pbPtr[0], pbPtr[1], pbPtr[2]);
    // glm::vec3 pc(pcPtr[0], pcPtr[1], pcPtr[2]);
    // glm::vec3 norm(normPtr[0], normPtr[1], normPtr[2]);

    // float pertTexelOffsetW = 1.f / texW;
    // float pertTexelOffsetH = 1.f / texH;

    glm::dvec2 topPtUV(0., 0.);
    glm::dvec2 bottomLeftPtUV(0., 0.);
    glm::dvec2 bottomRightPtUV(0., 0.);
    double k;

    if (CalcSpaceLineKCuda(*(vertArr + triangleTypeIdx),
                           *(vertArr + ((triangleTypeIdx + 1) % 3)),
                           *(vertArr + ((triangleTypeIdx + 2) % 3)),
                           norm, flip, k, deltaLen) == false) {
        invalidKCnt++;

        (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).curTexW = 1;
        (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).curTexH = 1;

        (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).norm[0] = 0.0f;
        (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).norm[1] = 0.0f;
        (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).norm[2] = 0.0f;

        return;
    }

    (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).norm[0] = norm.x;
    (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).norm[1] = norm.y;
    (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).norm[2] = norm.z;

    CalcTexUVCuda(k, topPtUV, bottomLeftPtUV, bottomRightPtUV);

    AssignVertexBufImplCuda(outBufPtr, *(vertArr + triangleTypeIdx), (float)topPtUV.x, (float)topPtUV.y, curTriIdx, 0);
    AssignVertexBufImplCuda(outBufPtr, *(vertArr + ((triangleTypeIdx + 1 + flip) % 3)), (float)bottomLeftPtUV.x, (float)bottomLeftPtUV.y, curTriIdx, 1);
    AssignVertexBufImplCuda(outBufPtr, *(vertArr + ((triangleTypeIdx + 2 - flip) % 3)), (float)bottomRightPtUV.x, (float)bottomRightPtUV.y, curTriIdx, 2);

    calc_in_texture_info_cuda(*(vertArr + triangleTypeIdx),
                          *(vertArr + ((triangleTypeIdx + 1 + flip) % 3)),
                          *(vertArr + ((triangleTypeIdx + 2 - flip) % 3)),
                          topPtUV, bottomLeftPtUV, bottomRightPtUV, k, curTriIdx, scaleFactorBegin, preDefHashBegin, texInWorldInfo, shDensity, maxWH);
    
    UpdateEdgeNbrInfoCuda(tile, preDefHashBegin, edgeMap,
        vertIndicies[triangleTypeIdx],
        vertIndicies[(triangleTypeIdx + 1 + flip) % 3],
        vertIndicies[(triangleTypeIdx + 2 - flip) % 3],
        curTriIdx, probCg, flushCg, equalFunc, atomicCnter);
    
    CountVertNbrNum(vertIndicies[triangleTypeIdx],
                    vertIndicies[(triangleTypeIdx + 1 + flip) % 3],
                    vertIndicies[(triangleTypeIdx + 2 - flip) % 3],
                    curTriIdx, vertMap, tile, probCg, flushCg, preDefHashBegin);


    // if (triangleTypeIdx == 0) {
    //     if (CalcSpaceLineKCuda(pa, pb, pc, norm, flip, k, flipCnt) == false) {
    //         invalidKCnt++;
    //         return;
    //     }

    //     CalcTexUVCuda(k, topPtUV, bottomLeftPtUV, bottomRightPtUV, texW, texH, texPadding);

    //     AssignVertexBufImplCuda(outBufPtr, pa, topPtUV.x, topPtUV.y, curTriIdx, 0);

    //     if (flip == false) {
    //         AssignVertexBufImplCuda(outBufPtr, pb, bottomLeftPtUV.x, bottomLeftPtUV.y, curTriIdx, 1);
    //         AssignVertexBufImplCuda(outBufPtr, pc, bottomRightPtUV.x, bottomRightPtUV.y, curTriIdx, 2);

    //         calc_in_texture_info_cuda(pa, pb, pc, topPtUV, bottomLeftPtUV, bottomRightPtUV, k, curTriIdx, texW, texH, texPadding, scaleFactorBegin, validShMap, shMapOffset, worldPoseMap, poseMapOffset, preDefHashBegin);

    //         // UpdateEdgeNbrInfoCuda(tile, preDefHashBegin, edgeMap, vertIndicies[0], vertIndicies[1], vertIndicies[2], curTriIdx, probCg, flushCg, equalFunc);

    //         // CountVertNbrNum(vertIndicies[0], vertIndicies[1], vertIndicies[2], curTriIdx, vertMap, equalFunc, tile, probCg, flushCg, preDefHashBegin);
    //     } else {
    //         AssignVertexBufImplCuda(outBufPtr, pc, bottomLeftPtUV.x, bottomLeftPtUV.y, curTriIdx, 1);
    //         AssignVertexBufImplCuda(outBufPtr, pb, bottomRightPtUV.x, bottomRightPtUV.y, curTriIdx, 2);

    //         calc_in_texture_info_cuda(pa, pc, pb, topPtUV, bottomLeftPtUV, bottomRightPtUV, k, curTriIdx, texW, texH, texPadding, scaleFactorBegin, validShMap, shMapOffset, worldPoseMap, poseMapOffset, preDefHashBegin);

    //         // UpdateEdgeNbrInfoCuda(tile, preDefHashBegin, edgeMap, vertIndicies[0], vertIndicies[2], vertIndicies[1], curTriIdx, probCg, flushCg, equalFunc);

    //         // CountVertNbrNum(vertIndicies[0], vertIndicies[2], vertIndicies[1], curTriIdx, vertMap, equalFunc, tile, probCg, flushCg, preDefHashBegin);
    //     }
    // } else if (triangleTypeIdx == 1) {
    //     if (CalcSpaceLineKCuda(pb, pc, pa, norm, flip, k, flipCnt) == false) {
    //         invalidKCnt++;
    //         return;
    //     }

    //     CalcTexUVCuda(k, topPtUV, bottomLeftPtUV, bottomRightPtUV, texW, texH, texPadding);

    //     AssignVertexBufImplCuda(outBufPtr, pb, topPtUV.x, topPtUV.y, curTriIdx, 0);

    //     if (flip == false) {
    //         AssignVertexBufImplCuda(outBufPtr, pc, bottomLeftPtUV.x, bottomLeftPtUV.y, curTriIdx, 1);
    //         AssignVertexBufImplCuda(outBufPtr, pa, bottomRightPtUV.x, bottomRightPtUV.y, curTriIdx, 2);

    //         calc_in_texture_info_cuda(pb, pc, pa, topPtUV, bottomLeftPtUV, bottomRightPtUV, k, curTriIdx, texW, texH, texPadding, scaleFactorBegin, validShMap, shMapOffset, worldPoseMap, poseMapOffset, preDefHashBegin);

    //         // UpdateEdgeNbrInfoCuda(tile, preDefHashBegin, edgeMap, vertIndicies[1], vertIndicies[2], vertIndicies[0], curTriIdx, probCg, flushCg, equalFunc);

    //         // CountVertNbrNum(vertIndicies[1], vertIndicies[2], vertIndicies[0], curTriIdx, vertMap, equalFunc, tile, probCg, flushCg, preDefHashBegin);
    //     } else {
    //         AssignVertexBufImplCuda(outBufPtr, pa, bottomLeftPtUV.x, bottomLeftPtUV.y, curTriIdx, 1);
    //         AssignVertexBufImplCuda(outBufPtr, pc, bottomRightPtUV.x, bottomRightPtUV.y, curTriIdx, 2);

    //         calc_in_texture_info_cuda(pb, pa, pc, topPtUV, bottomLeftPtUV, bottomRightPtUV, k, curTriIdx, texW, texH, texPadding, scaleFactorBegin, validShMap, shMapOffset, worldPoseMap, poseMapOffset, preDefHashBegin);

    //         // UpdateEdgeNbrInfoCuda(tile, preDefHashBegin, edgeMap, vertIndicies[1], vertIndicies[0], vertIndicies[2], curTriIdx, probCg, flushCg, equalFunc);

    //         // CountVertNbrNum(vertIndicies[1], vertIndicies[0], vertIndicies[2], curTriIdx, vertMap, equalFunc, tile, probCg, flushCg, preDefHashBegin);
    //     }
    // } else if (triangleTypeIdx == 2) {
    //     if (CalcSpaceLineKCuda(pc, pa, pb, norm, flip, k, flipCnt) == false) {
    //         invalidKCnt++;
    //         return;
    //     }

    //     CalcTexUVCuda(k, topPtUV, bottomLeftPtUV, bottomRightPtUV, texW, texH, texPadding);
    //     AssignVertexBufImplCuda(outBufPtr, pc, topPtUV.x, topPtUV.y, curTriIdx, 0);
        
    //     if (flip == false) {
    //         AssignVertexBufImplCuda(outBufPtr, pa, bottomLeftPtUV.x, bottomLeftPtUV.y, curTriIdx, 1);
    //         AssignVertexBufImplCuda(outBufPtr, pb, bottomRightPtUV.x, bottomRightPtUV.y, curTriIdx, 2);

    //         calc_in_texture_info_cuda(pc, pa, pb, topPtUV, bottomLeftPtUV, bottomRightPtUV, k, curTriIdx, texW, texH, texPadding, scaleFactorBegin, validShMap, shMapOffset, worldPoseMap, poseMapOffset, preDefHashBegin);

    //         // UpdateEdgeNbrInfoCuda(tile, preDefHashBegin, edgeMap, vertIndicies[2], vertIndicies[0], vertIndicies[1], curTriIdx, probCg, flushCg, equalFunc);

    //         // CountVertNbrNum(vertIndicies[2], vertIndicies[0], vertIndicies[1], curTriIdx, vertMap, equalFunc, tile, probCg, flushCg, preDefHashBegin);
    //     } else {
    //         AssignVertexBufImplCuda(outBufPtr, pb, bottomLeftPtUV.x, bottomLeftPtUV.y, curTriIdx, 1);
    //         AssignVertexBufImplCuda(outBufPtr, pa, bottomRightPtUV.x, bottomRightPtUV.y, curTriIdx, 2);

    //         calc_in_texture_info_cuda(pc, pb, pa, topPtUV, bottomLeftPtUV, bottomRightPtUV, k, curTriIdx, texW, texH, texPadding, scaleFactorBegin, validShMap, shMapOffset, worldPoseMap, poseMapOffset, preDefHashBegin);

    //         // UpdateEdgeNbrInfoCuda(tile, preDefHashBegin, edgeMap, vertIndicies[2], vertIndicies[1], vertIndicies[0], curTriIdx, probCg, flushCg, equalFunc);

    //         // CountVertNbrNum(vertIndicies[2], vertIndicies[1], vertIndicies[0], curTriIdx, vertMap, equalFunc, tile, probCg, flushCg, preDefHashBegin);
    //     }
    // } else {
    //     printf("invalid idx for AssignVertexBuf \n");
    // }
}

template <typename VertMap, typename TileType>
__device__ uint32_t GetVertMapCount(VertMap vertMap, uint32_t curPtIdx, TileType& tile)
{
    return vertMap.count(tile, curPtIdx);
}

template <typename VertMap, typename TileType, typename ProbeCgType, typename FlushCgType>
__device__ uint32_t FindValidVertNbrNum(VertMap vertMap, uint32_t curPtIdx, uint32_t curTriIdx, TileType& tile, ProbeCgType& probCg, FlushCgType& flushCg)
{
    // using pair_type = typename VertMap::value_type;
    // uint32_t constexpr bufSize = 10;
    // pair_type outBufFlush[bufSize];
    // pair_type outBuf[bufSize];
    // uint32_t flushCnt = 0;
    // cuda::atomic<size_t> atomicCnt{0};
    size_t findCnt = 0;

    findCnt = vertMap.count(tile, curPtIdx);
    // findCnt = GetVertMapCount(vertMap, curPtIdx, tile);
    if (findCnt <= 0) {
        return 0;
    }

    // vertMap.retrieve<bufSize>(flushCg, probCg, curPtIdx, &flushCnt, outBufFlush, &atomicCnt, outBuf, thrust::equal_to<int>{});

    // if (flushCnt > 0) {
    //     vertMap.flush_output_buffer(flushCg, flushCnt, outBufFlush, &atomicCnt, outBuf);
    // }

    return findCnt;

    // for (uint32_t ii = 0; ii < atomicCnt.load(); ii++) {
    //     if (curTriIdx == outBuf[ii].second) {
    //         return findCnt;
    //     }
    // }
    // return 0;
}



template <typename VertMap, typename TileType, typename ProbeCgType, typename FlushCgType, typename PreDefHashValIt>
__device__ void CountVertNbrNum(uint32_t topPtIdx, uint32_t leftPtIdx, uint32_t rightPtIdx, uint32_t curTriIdx, VertMap vertMap, TileType& tile, ProbeCgType& probCg, FlushCgType& flushCg, PreDefHashValIt preDefHashBegin)
{
    (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).topVertNbrNum = FindValidVertNbrNum(vertMap, topPtIdx, curTriIdx, tile, probCg, flushCg);
    (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).topPtIdx = topPtIdx;

    (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).leftVertNbrNum = FindValidVertNbrNum(vertMap, leftPtIdx, curTriIdx, tile, probCg, flushCg);
    (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).leftPtIdx = leftPtIdx;

    (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).rightVertNbrNum = FindValidVertNbrNum(vertMap, rightPtIdx, curTriIdx, tile, probCg, flushCg);
    (thrust::raw_reference_cast(*(preDefHashBegin + curTriIdx))).rightPtIdx = rightPtIdx;
}

template <uint32_t tile_size, uint32_t prob_size, uint32_t flush_size, typename EdgeMap, typename VertMap, typename TriArrIt, typename ScaleFactorIt, typename PreDefHashValIt, typename KeyEqual, typename AtomicT, typename DeltaPoolIt>
__global__ void assign_predefined_and_calc_alloc_size_eval(
    EdgeMap edgeMap,
    VertMap vertMap,
    TriArrIt triArrBegin,
    uint32_t totalNum,
    ScaleFactorIt scaleFactorBegin,
    PreDefHashValIt preDefHashBegin,
    float* outBufPtr,
    KeyEqual equalFunc,
    uint32_t* idxCntReducePool,
    uint32_t* invalidKCntReducePool,
    uint8_t* flipCountReducePool,
    DeltaPoolIt deltaLenReducePool,
    AtomicT* atomicCnter,
    float* texInWorldInfo,
    float* shDensityBegin,
    uint32_t maxWH
)
{
    auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
    auto prob = cg::tiled_partition<prob_size>(cg::this_thread_block());
    auto flush = cg::tiled_partition<flush_size>(cg::this_thread_block());

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < totalNum) {
        auto atmCtr = atomicCnter + tid * 3;
        TriangleInfoDevice curTri = *(triArrBegin + tid);
        uint8_t idx = FindLongestEdgeCuda(&(curTri.vert[0]), &(curTri.vert[3]), &(curTri.vert[6]));

        uint32_t invalidKCnt = 0;
        uint8_t flip = 0;
        float deltaLen = -0.2f;
        glm::vec3 vertArr[3];
        glm::vec3 norm(curTri.norm[0], curTri.norm[1], curTri.norm[2]);
        vertArr[0] = glm::vec3(curTri.vert[0], curTri.vert[1], curTri.vert[2]);
        vertArr[1] = glm::vec3(curTri.vert[3], curTri.vert[4], curTri.vert[5]);
        vertArr[2] = glm::vec3(curTri.vert[6], curTri.vert[7], curTri.vert[8]);

        float shDensity = *(shDensityBegin + tid);
        assign_vertex_buf_cuda(tile, preDefHashBegin, scaleFactorBegin, outBufPtr, edgeMap, vertMap, vertArr, norm, idx, tid, &(curTri.vertIdx[0]), invalidKCnt, prob, flush, equalFunc, flip, atmCtr, deltaLen, texInWorldInfo, shDensity, maxWH);

        idxCntReducePool[tid * 3 + idx] = 1;
        invalidKCntReducePool[tid] = invalidKCnt;
        flipCountReducePool[tid] = flip;
        *(deltaLenReducePool + tid) = deltaLen;
    }
}


template <uint32_t tile_size, uint32_t prob_size, uint32_t flush_size, typename EdgeMap, typename VertMap, typename TriArrIt, typename ScaleFactorIt, typename PreDefHashValIt, typename KeyEqual, typename AtomicT, typename DeltaPoolIt>
__global__ void assign_predefined_and_calc_alloc_size(
    EdgeMap edgeMap,
    VertMap vertMap,
    TriArrIt triArrBegin,
    uint32_t totalNum,
    ScaleFactorIt scaleFactorBegin,
    PreDefHashValIt preDefHashBegin,
    float* outBufPtr,
    KeyEqual equalFunc,
    uint32_t* idxCntReducePool,
    uint32_t* invalidKCntReducePool,
    uint8_t* flipCountReducePool,
    DeltaPoolIt deltaLenReducePool,
    AtomicT* atomicCnter,
    float* texInWorldInfo,
    float shDensity,
    uint32_t maxWH
)
{
    auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
    auto prob = cg::tiled_partition<prob_size>(cg::this_thread_block());
    auto flush = cg::tiled_partition<flush_size>(cg::this_thread_block());

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < totalNum) {
        auto atmCtr = atomicCnter + tid * 3;
        TriangleInfoDevice curTri = *(triArrBegin + tid);
        uint8_t idx = FindLongestEdgeCuda(&(curTri.vert[0]), &(curTri.vert[3]), &(curTri.vert[6]));

        uint32_t invalidKCnt = 0;
        uint8_t flip = 0;
        float deltaLen = -0.2f;
        glm::vec3 vertArr[3];
        glm::vec3 norm(curTri.norm[0], curTri.norm[1], curTri.norm[2]);
        vertArr[0] = glm::vec3(curTri.vert[0], curTri.vert[1], curTri.vert[2]);
        vertArr[1] = glm::vec3(curTri.vert[3], curTri.vert[4], curTri.vert[5]);
        vertArr[2] = glm::vec3(curTri.vert[6], curTri.vert[7], curTri.vert[8]);
        assign_vertex_buf_cuda(tile, preDefHashBegin, scaleFactorBegin, outBufPtr, edgeMap, vertMap, vertArr, norm, idx, tid, &(curTri.vertIdx[0]), invalidKCnt, prob, flush, equalFunc, flip, atmCtr, deltaLen, texInWorldInfo, shDensity, maxWH);

        idxCntReducePool[tid * 3 + idx] = 1;
        invalidKCntReducePool[tid] = invalidKCnt;
        flipCountReducePool[tid] = flip;
        *(deltaLenReducePool + tid) = deltaLen;
    }
}

template <typename VertMap, typename ProbeCgType, typename FlushCgType, typename OutBufIt, typename KeyEqual, typename atomicT>
__device__ void ReadEdgeNbrTest(VertMap vertMap, uint32_t curPtIdx, OutBufIt outBufCurPose, uint32_t nbrNum, ProbeCgType& probCg, FlushCgType& flushCg, KeyEqual equalFunc, atomicT* curKeyAtomCnter)
{
    using pair_type = typename VertMap::value_type;
    uint32_t constexpr bufSize = 2;
    pair_type outBufFlush[bufSize];
    uint32_t flushCnt = 0;
    // cuda::atomic<size_t> atomicCnt{0};
    EdgeKeyCuda testKey(curPtIdx, curPtIdx+1);

    vertMap.retrieve<bufSize>(flushCg, probCg, testKey, &flushCnt, outBufFlush, curKeyAtomCnter, outBufCurPose, equalFunc);

    if (flushCnt > 0) {
        vertMap.flush_output_buffer(flushCg, flushCnt, outBufFlush, curKeyAtomCnter, outBufCurPose);
    }

    // if (nbrNum != atomicCnt.load()) {
    //     printf("nbrNum is not same with read get %d, %d\n", nbrNum, atomicCnt.load());
    // }
}



template <uint32_t bufferSize, typename VertMap, typename ProbeCgType, typename FlushCgType, typename OutBufIt, typename KeyEqual, typename atomicT, typename PairType>
__device__ void ReadVrtNbr(VertMap vertMap, uint32_t curPtIdx, OutBufIt outBufCurPose, uint32_t nbrNum, ProbeCgType& probCg, FlushCgType& flushCg, KeyEqual equalFunc, atomicT* curKeyAtomCnter, uint32_t* flushCounter, PairType* flushBuffer)
{
    // __shared__ atomicT atomicCnt;
    // volatile atomicT* ptr = curKeyAtomCnter;
    vertMap.retrieve<bufferSize>(flushCg, probCg, curPtIdx, flushCounter, flushBuffer, curKeyAtomCnter, outBufCurPose, equalFunc);

    if (*flushCounter > 0) {
        vertMap.flush_output_buffer(flushCg, *flushCounter, flushBuffer, curKeyAtomCnter, outBufCurPose);
    }

    // if (nbrNum != (*curKeyAtomCnter).load()) {
    //     printf("nbrNum is not same with read get %d, %d\n", nbrNum, atomicCnt.load());
    // }
}


template <uint32_t block_size,
          uint32_t prob_size,
          uint32_t flush_cg_size,
          uint32_t buffer_size,
          typename VertMap,
          typename VertInfoIt,
          typename OutBufIt,
          typename KeyEqual,
          typename atomicT>
__global__ void vert_map_nbr_write_back(uint32_t totalNum, OutBufIt outBufBegin, VertMap vertMap, VertInfoIt vertInfoBegin, KeyEqual equalFunc, atomicT* cntArrBegin)
{
    using pair_type = typename VertMap::value_type;
    constexpr uint32_t num_flushing_cgs = block_size / flush_cg_size;
    const uint32_t flushing_cg_id       = threadIdx.x / flush_cg_size;

    auto prob = cg::tiled_partition<prob_size>(cg::this_thread_block());
    auto flushing_cg = cg::tiled_partition<flush_cg_size>(cg::this_thread_block());

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ pair_type output_buffer[num_flushing_cgs][buffer_size];
    __shared__ uint32_t flushing_cg_counter[num_flushing_cgs];

    if (flushing_cg.thread_rank() == 0) { flushing_cg_counter[flushing_cg_id] = 0; }

    // ReadEdgeNbrTest
    if (tid < totalNum) {
        auto curKeyAtomCnter = cntArrBegin + (tid * 3);
        ReadVrtNbr<buffer_size>(vertMap,
            (thrust::raw_reference_cast(*(vertInfoBegin + tid))).topPtIdx,
            outBufBegin + (thrust::raw_reference_cast(*(vertInfoBegin + tid))).offset,
            (thrust::raw_reference_cast(*(vertInfoBegin + tid))).topVertNum, prob, flushing_cg, equalFunc, curKeyAtomCnter,
            &flushing_cg_counter[flushing_cg_id],
            output_buffer[flushing_cg_id]);

        flushing_cg_counter[flushing_cg_id] = 0;
        curKeyAtomCnter++;

        ReadVrtNbr<buffer_size>(vertMap,
            (thrust::raw_reference_cast(*(vertInfoBegin + tid))).leftPtIdx,
            outBufBegin + (thrust::raw_reference_cast(*(vertInfoBegin + tid))).offset +
                (thrust::raw_reference_cast(*(vertInfoBegin + tid))).topVertNum,
            (thrust::raw_reference_cast(*(vertInfoBegin + tid))).leftVertNum, prob, flushing_cg, equalFunc,
            curKeyAtomCnter,
            &flushing_cg_counter[flushing_cg_id],
            output_buffer[flushing_cg_id]);

        flushing_cg_counter[flushing_cg_id] = 0;
        curKeyAtomCnter++;

        ReadVrtNbr<buffer_size>(vertMap,
            (thrust::raw_reference_cast(*(vertInfoBegin + tid))).rightPtIdx,
            outBufBegin + (thrust::raw_reference_cast(*(vertInfoBegin + tid))).offset +
                (thrust::raw_reference_cast(*(vertInfoBegin + tid))).topVertNum +
                (thrust::raw_reference_cast(*(vertInfoBegin + tid))).leftVertNum,
            (thrust::raw_reference_cast(*(vertInfoBegin + tid))).rightVertNum, prob, flushing_cg, equalFunc,
            curKeyAtomCnter,
            &flushing_cg_counter[flushing_cg_id],
            output_buffer[flushing_cg_id]);
    }

}


//each thread dedicate to one texId assignment
template <typename TexInfoIt, typename CompactPixLocBufIt, typename RawDataIt>
__global__ void fill_compact_buffer_per_tex_id(TexInfoIt texInfoBegin,
    uint32_t texNum,
    CompactPixLocBufIt compactPixLocBufBegin,
    RawDataIt rawDataIt,
    uint32_t rawDataLen)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= texNum) {
        return;
    }

    uint32_t curTexId = thrust::raw_reference_cast(*(texInfoBegin + tid)).texId;
    uint32_t curOffset = thrust::raw_reference_cast(*(texInfoBegin + tid)).offset;
    uint32_t writePos = 0;

    for (uint32_t i = 0; i < rawDataLen; i++) {
        if (thrust::raw_reference_cast(*(rawDataIt + i)).first != curTexId) {
            continue;
        }

        thrust::raw_reference_cast(*(compactPixLocBufBegin + curOffset + writePos)).x = thrust::raw_reference_cast(*(rawDataIt + i)).second.x;
        thrust::raw_reference_cast(*(compactPixLocBufBegin + curOffset + writePos)).y = thrust::raw_reference_cast(*(rawDataIt + i)).second.y;

        writePos++;
    }
}



// https://no5-aaron-wu.github.io/2021/11/30/CUDA-5-mutexLock/
template <typename TexId2DataAndMutexMap, typename CompactPixLocBufIt, typename RawDataIt, typename AtomicT>
__global__ void fill_visible_pix_location_compact_buffer(TexId2DataAndMutexMap texId2DataAndMutexMap,
    uint32_t* mutexArr,
    uint32_t* counterArr,
    CompactPixLocBufIt compactPixLocBufBegin,
    RawDataIt rawDataIt,
    uint32_t rawDataNum,
    AtomicT* atomicCnter,
    uint32_t texIdNum,
    uint32_t* addCounter)
{
    using MyPairType = typename TexId2DataAndMutexMap::value_type;
    uint32_t constexpr bufSize = 2;
    MyPairType outBufFlush[bufSize];
    MyPairType outBuf[bufSize];
    uint32_t flushCnt = 0;

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    DataAndMutexOffset dataAndMutexOffset;
    PixLocation curPixLoc;
    uint32_t texId;

    if (tid >= rawDataNum) {
        return;
    }

    auto tile = cg::tiled_partition<1>(cg::this_thread_block());
    auto prob = cg::tiled_partition<1>(cg::this_thread_block());
    auto flush = cg::tiled_partition<1>(cg::this_thread_block());

    texId = (thrust::raw_reference_cast(*(rawDataIt + tid))).first.texId;
    curPixLoc = (thrust::raw_reference_cast(*(rawDataIt + tid))).second;


    TexIdDummyKey keyDummy;
    keyDummy.texId = texId;
    auto count = texId2DataAndMutexMap.count(tile, keyDummy, TexIdDummyKeyEqual{});
    if (count <= 0) {
        return;
    }

    texId2DataAndMutexMap.retrieve<bufSize>(flush, prob, keyDummy, &flushCnt, outBufFlush, (atomicCnter + tid), outBuf, TexIdDummyKeyEqual{});

    if (flushCnt > 0) {
        texId2DataAndMutexMap.flush_output_buffer(flush, flushCnt, outBufFlush, (atomicCnter + tid), outBuf);
    }

    assert(count == 1);

    dataAndMutexOffset.dataOffset = outBuf[0].second.dataOffset;
    dataAndMutexOffset.mutexOffset = outBuf[0].second.mutexOffset;

    // auto found = texId2DataAndMutexMap.find(texId);
    // if (found != texId2DataAndMutexMap.end()) {
    //     dataAndMutexOffset = found->second;
    // } else {
    //     printf("k-Er %d\n", texId);
    //     return;
    // }
    
    if (dataAndMutexOffset.mutexOffset >= texIdNum) {
        printf(" invalid mutex offset : %d, texId: %d\n", dataAndMutexOffset.mutexOffset, texId);
    }

    if (dataAndMutexOffset.dataOffset >= rawDataNum) {
        printf(" invalid dataOffset offset : %d, texId: %d\n", dataAndMutexOffset.dataOffset, texId);
    }



    bool blocked = true;
    while (blocked) {
        if (0 == atomicCAS(mutexArr + dataAndMutexOffset.mutexOffset, 0, 1)) {
            // (thrust::raw_reference_cast(*(compactPixLocBufBegin + dataAndMutexOffset.dataOffset + (*(counterArr + dataAndMutexOffset.mutexOffset))))).x = curPixLoc.x;
            // (thrust::raw_reference_cast(*(compactPixLocBufBegin + dataAndMutexOffset.dataOffset + (*(counterArr + dataAndMutexOffset.mutexOffset))))).y = curPixLoc.y;

            atomicAdd(&((thrust::raw_reference_cast(*(compactPixLocBufBegin + dataAndMutexOffset.dataOffset + (*(counterArr + dataAndMutexOffset.mutexOffset))))).x), curPixLoc.x);
            atomicAdd(&((thrust::raw_reference_cast(*(compactPixLocBufBegin + dataAndMutexOffset.dataOffset + (*(counterArr + dataAndMutexOffset.mutexOffset))))).y), curPixLoc.y);
            // __threadfence();

            atomicAdd((counterArr + dataAndMutexOffset.mutexOffset), 1);
            atomicExch(mutexArr + dataAndMutexOffset.mutexOffset, 0);

            atomicAdd(addCounter, 1);
            blocked = false;
        }
    }
}

template <typename MutableView, typename KeyIt, typename ValueIt>
__global__ void insert_tex_2_data_n_mutex_offset(MutableView map, KeyIt keyBegin, ValueIt valueBegin, uint32_t totalNum)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto tile = cg::tiled_partition<1>(cg::this_thread_block());

    
    if (tid < totalNum) {
        // tmp.dataOffset = thrust::raw_reference_cast(*(valueBegin + tid)).dataOffset;
        // tmp.mutexOffset = thrust::raw_reference_cast(*(valueBegin + tid)).mutexOffset;

        // uint32_t texIdTmp = thrust::raw_reference_cast(*(keyBegin + tid));

        TexIdDummyKey keyDummy;
        keyDummy.texId = thrust::raw_reference_cast(*(keyBegin + tid));


        DataAndMutexOffset tmp(thrust::raw_reference_cast(*(valueBegin + tid)).dataOffset,
                               thrust::raw_reference_cast(*(valueBegin + tid)).mutexOffset);

        map.insert(tile, cuco::pair{keyDummy, tmp});
    }
}

template <typename TexInfoIt, typename FillIt>
__global__ void fill_data_n_mutex_offset(TexInfoIt texInfoBegin, FillIt targetBegin, uint32_t totalNum)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < totalNum) {
        thrust::raw_reference_cast(*(targetBegin + tid)).dataOffset = thrust::raw_reference_cast(*(texInfoBegin + tid)).offset;

        thrust::raw_reference_cast(*(targetBegin + tid)).mutexOffset = tid;
    }
}


__device__ void CalcValidShNumCuda(glm::vec3 topPt, glm::vec3 bottomLeft, glm::vec3 bottomRight, glm::dvec2 topUV, glm::dvec2 leftUV, glm::dvec2 rightUV, double k, unsigned int curTriangleIdx, CurTexAlignedWH* preDefHashBegin, uint32_t* validShNumAll)
{
    glm::dvec3 foot(k * ((double)bottomRight.x - (double)bottomLeft.x) + (double)bottomLeft.x, k * ((double)bottomRight.y - (double)bottomLeft.y) + (double)bottomLeft.y, k * ((double)bottomRight.z - (double)bottomLeft.z) + (double)bottomLeft.z);

    glm::dvec3 bottomRightDouble(bottomRight.x, bottomRight.y, bottomRight.z);
    glm::dvec3 bottomLeftDouble(bottomLeft.x, bottomLeft.y, bottomLeft.z);
    double bottomDistance = glm::length(bottomRightDouble - bottomLeftDouble);

    glm::dvec3 topPtDouble(topPt.x, topPt.y, topPt.z);

    uint32_t curTexW = (uint32_t)((preDefHashBegin + curTriangleIdx)->data.curTexW);
    uint32_t curTexH = (uint32_t)((preDefHashBegin + curTriangleIdx)->data.curTexH);

    // if (curTexW > 20 || curTexH > 20) {
    //     printf("CalcValidShNumCuda meet invalid WH! %d, %d\n", curTexW, curTexH);
    // }

    double texelW = 1.0 / (double)curTexW;
    double texelH = 1.0 / (double)curTexH;

    // glm::dvec3 bottomDirStep = (bottomRightDouble - bottomLeftDouble) / (double)((double)curTexW);
    // glm::dvec3 upDirStep = (topPtDouble - foot) / (double)((double)curTexH);

    // glm::dvec3 startPosition = bottomLeftDouble + 0.5 * bottomDirStep + 0.5 * upDirStep;
    double startH = texelH / 2.0;
    double startW = texelW / 2.0;

    glm::dvec3 topUV3d(topUV.x, topUV.y, 0.);
    glm::dvec3 leftUV3d(leftUV.x, leftUV.y, 0.);
    glm::dvec3 rightUV3d(rightUV.x, rightUV.y, 0.);

    glm::dvec3 vecL2R = rightUV3d - leftUV3d;
    glm::dvec3 vecR2U = topUV3d - rightUV3d;
    glm::dvec3 vecU2L = leftUV3d - topUV3d;

    uint32_t validNum = 0;

    for (unsigned char curH = 0; curH < curTexH; curH++) {
        for (unsigned char curW = 0; curW < curTexW; curW++) {
            if (IsPointInsideTriangleCuda(startW + texelW * (double)curW, startH + texelH * (double)curH, vecL2R, vecR2U, vecU2L, topUV, leftUV, rightUV) == false) {
                continue;
            }

            validNum++;
        }
    }

    *(validShNumAll + curTriangleIdx) = validNum;
}

template <typename PreDefHashValIt>
__device__ void CalcCorerAreaPosMapValidMap(glm::vec3 topPt, glm::vec3 bottomLeft, glm::vec3 bottomRight,
    glm::dvec2 topUV, glm::dvec2 leftUV, glm::dvec2 rightUV,
    double k, unsigned int curTriangleIdx,
    // uint32_t* shPoseMapCompactBufOffsets, uint32_t* shValidMapCompactBufOffsets,
    PreDefHashValIt preDefHashBegin, float* shPoseMapCompactBuf, uint32_t* shValidCpctOffsets, TexAlignedPosWH* validShWHMap)
//uint32_t* shValidMapCompactBuf
{
    glm::dvec3 foot(k * ((double)bottomRight.x - (double)bottomLeft.x) + (double)bottomLeft.x, k * ((double)bottomRight.y - (double)bottomLeft.y) + (double)bottomLeft.y, k * ((double)bottomRight.z - (double)bottomLeft.z) + (double)bottomLeft.z);

    glm::dvec3 bottomRightDouble(bottomRight.x, bottomRight.y, bottomRight.z);
    glm::dvec3 bottomLeftDouble(bottomLeft.x, bottomLeft.y, bottomLeft.z);
    double bottomDistance = glm::length(bottomRightDouble - bottomLeftDouble);

    glm::dvec3 topPtDouble(topPt.x, topPt.y, topPt.z);
    double triangleRealHeight = glm::length(topPtDouble - foot);

    // uint32_t curTexW = (curTexWH + curTriangleIdx)->data.curTexW;
    // uint32_t curTexH = (curTexWH + curTriangleIdx)->data.curTexH;

    uint32_t curTexW = thrust::raw_reference_cast(*(preDefHashBegin + curTriangleIdx)).curTexW;
    uint32_t curTexH = thrust::raw_reference_cast(*(preDefHashBegin + curTriangleIdx)).curTexH;



    double texelW = 1.0 / (double)curTexW;
    double texelH = 1.0 / (double)curTexH;

    glm::dvec3 bottomDirStep = (bottomRightDouble - bottomLeftDouble) / (double)((double)curTexW);
    glm::dvec3 upDirStep = (topPtDouble - foot) / (double)((double)curTexH);

    glm::dvec3 startPosition = bottomLeftDouble + 0.5 * bottomDirStep + 0.5 * upDirStep;
    double startH = texelH / 2.0;
    double startW = texelW / 2.0;

    glm::dvec3 topUV3d(topUV.x, topUV.y, 0.);
    glm::dvec3 leftUV3d(leftUV.x, leftUV.y, 0.);
    glm::dvec3 rightUV3d(rightUV.x, rightUV.y, 0.);

    glm::dvec3 vecL2R = rightUV3d - leftUV3d;
    glm::dvec3 vecR2U = topUV3d - rightUV3d;
    glm::dvec3 vecU2L = leftUV3d - topUV3d;

    double minLeftX = 999.; double maxRightX = -999.;

    uint32_t curWrtPos = 0;

    // start from left down corner tex coord.
    for (unsigned char curH = 0; curH < curTexH; curH++) {
        for (unsigned char curW = 0; curW < curTexW; curW++) {
            if (IsPointInsideTriangleCuda(startW + texelW * (double)curW, startH + texelH * (double)curH, vecL2R, vecR2U, vecU2L, topUV, leftUV, rightUV) == true) {
                // *(shValidMapCompactBuf + (*(shValidMapCompactBufOffsets + curTriangleIdx)) + curH * curTexW + curW) = 1;

                (validShWHMap + (*(shValidCpctOffsets + curTriangleIdx)) + curWrtPos)->data.posW = curW;
                (validShWHMap + (*(shValidCpctOffsets + curTriangleIdx)) + curWrtPos)->data.posH = curH;

                // curH * curTexW * 3 + curW * 3
                CalcWorldPotisionCuda((shPoseMapCompactBuf + 3 * ((*(shValidCpctOffsets + curTriangleIdx)) + curWrtPos)), startPosition, bottomDirStep, upDirStep, curW, curH);

                if (curH == 0) {
                    if ((startW + texelW * (float)curW) <= minLeftX) {
                        minLeftX = (startW + texelW * (float)curW);
                    }

                    if ((startW + texelW * (float)curW) >= maxRightX) {
                        maxRightX = (startW + texelW * (float)curW);
                    }
                }

                curWrtPos++;
            }
        }
    }

    UpdateCornerAreaCuda(preDefHashBegin, minLeftX, maxRightX, curTriangleIdx);
}

__device__ void calc_valid_sh_num_each(glm::vec3* vertArr, glm::vec3 norm, uint8_t triangleTypeIdx, uint8_t& flip, uint32_t& invalidKCnt, CurTexAlignedWH* preDefHashBegin, uint32_t curTriIdx, uint32_t* validShNumAll)
{
    glm::dvec2 topPtUV(0., 0.);
    glm::dvec2 bottomLeftPtUV(0., 0.);
    glm::dvec2 bottomRightPtUV(0., 0.);
    double k;
    float deltaLen;

    if (CalcSpaceLineKCuda(*(vertArr + triangleTypeIdx),
                           *(vertArr + ((triangleTypeIdx + 1) % 3)),
                           *(vertArr + ((triangleTypeIdx + 2) % 3)),
                           norm, flip, k, deltaLen) == false) {
        invalidKCnt++;
            return;
    }

    CalcTexUVCuda(k, topPtUV, bottomLeftPtUV, bottomRightPtUV);

    CalcValidShNumCuda(*(vertArr + triangleTypeIdx),
    *(vertArr + ((triangleTypeIdx + 1 + flip) % 3)),
    *(vertArr + ((triangleTypeIdx + 2 - flip) % 3)),
    topPtUV, bottomLeftPtUV, bottomRightPtUV, k, curTriIdx,
    preDefHashBegin, validShNumAll);
}

template <typename PreDefHashValIt>
__device__ void do_assign_corerArea_posMap_validMap(glm::vec3* vertArr, glm::vec3 norm, uint8_t triangleTypeIdx, uint8_t& flip, uint32_t& invalidKCnt, PreDefHashValIt preDefHashBegin, uint32_t curTriIdx, float* shPoseMapCompactBuf, uint32_t* shValidCpctOffsets, TexAlignedPosWH* validShWHMap)
// uint32_t* shValidMapCompactBuf
{
    glm::dvec2 topPtUV(0., 0.);
    glm::dvec2 bottomLeftPtUV(0., 0.);
    glm::dvec2 bottomRightPtUV(0., 0.);
    double k;
    float deltaLen;

    if (CalcSpaceLineKCuda(*(vertArr + triangleTypeIdx),
                           *(vertArr + ((triangleTypeIdx + 1) % 3)),
                           *(vertArr + ((triangleTypeIdx + 2) % 3)),
                           norm, flip, k, deltaLen) == false) {
        invalidKCnt++;
            return;
    }

    CalcTexUVCuda(k, topPtUV, bottomLeftPtUV, bottomRightPtUV);

    CalcCorerAreaPosMapValidMap(*(vertArr + triangleTypeIdx),
                          *(vertArr + ((triangleTypeIdx + 1 + flip) % 3)),
                          *(vertArr + ((triangleTypeIdx + 2 - flip) % 3)),
                          topPtUV, bottomLeftPtUV, bottomRightPtUV, k, curTriIdx,
                          preDefHashBegin, shPoseMapCompactBuf, shValidCpctOffsets, validShWHMap);
}

template <typename TriArrIt, typename PreDefHashValIt>
__global__ void reassign_corerArea_posMap_validMap(uint32_t* idxNeedToReassign,
    TriArrIt triArrBegin,
    PreDefHashValIt preDefHashBegin, 
    // uint32_t* shPoseMapOffsets, uint32_t* shValidMapOffsets,
    float* shPoseMapCompactBuf,
    uint32_t* shValidCpctOffsets,
    TexAlignedPosWH* validShWHMap,
    uint32_t totalNum)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < totalNum) {
        uint32_t idx = *(idxNeedToReassign + tid);
        TriangleInfoDevice curTri = *(triArrBegin + idx);

        uint8_t triIdx = FindLongestEdgeCuda(&(curTri.vert[0]), &(curTri.vert[3]), &(curTri.vert[6]));
        uint8_t flip = 0;
        uint32_t invalidKCnt = 0;

        glm::vec3 vertArr[3];
        vertArr[0] = glm::vec3(curTri.vert[0], curTri.vert[1], curTri.vert[2]);
        vertArr[1] = glm::vec3(curTri.vert[3], curTri.vert[4], curTri.vert[5]);
        vertArr[2] = glm::vec3(curTri.vert[6], curTri.vert[7], curTri.vert[8]);
        glm::vec3 norm(curTri.norm[0], curTri.norm[1], curTri.norm[2]);
        do_assign_corerArea_posMap_validMap(vertArr, norm, triIdx, flip, invalidKCnt, preDefHashBegin, tid, shPoseMapCompactBuf, shValidCpctOffsets, validShWHMap);
    }
}


template <typename TriArrIt>
__global__ void calc_valid_sh_number(TriArrIt triArrBegin, CurTexAlignedWH* preDefHashBegin, uint32_t totalNum, uint32_t* validShNumAll)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < totalNum) {
        TriangleInfoDevice curTri = *(triArrBegin + tid);
        uint8_t idx = FindLongestEdgeCuda(&(curTri.vert[0]), &(curTri.vert[3]), &(curTri.vert[6]));
        uint8_t flip = 0;
        uint32_t invalidKCnt = 0;

        glm::vec3 vertArr[3];
        vertArr[0] = glm::vec3(curTri.vert[0], curTri.vert[1], curTri.vert[2]);
        vertArr[1] = glm::vec3(curTri.vert[3], curTri.vert[4], curTri.vert[5]);
        vertArr[2] = glm::vec3(curTri.vert[6], curTri.vert[7], curTri.vert[8]);
        glm::vec3 norm(curTri.norm[0], curTri.norm[1], curTri.norm[2]);

        calc_valid_sh_num_each(vertArr, norm, idx, flip, invalidKCnt, preDefHashBegin, tid, validShNumAll);
    }
}


template <typename TriArrIt, typename PreDefHashValIt>
__global__ void assign_corerArea_posMap_validMap(TriArrIt triArrBegin,
    // uint32_t* shPoseMapCompactBufOffsets, uint32_t* shValidMapCompactBufOffsets,
    PreDefHashValIt preDefHashBegin, float* shPoseMapCompactBuf, uint32_t* shValidCpctOffsets, TexAlignedPosWH* validShWHMap, uint32_t totalNum)
    // uint32_t* shValidMapCompactBuf, uint32_t totalNum)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < totalNum) {
        TriangleInfoDevice curTri = *(triArrBegin + tid);
        uint8_t idx = FindLongestEdgeCuda(&(curTri.vert[0]), &(curTri.vert[3]), &(curTri.vert[6]));
        uint8_t flip = 0;
        uint32_t invalidKCnt = 0;

        glm::vec3 vertArr[3];
        vertArr[0] = glm::vec3(curTri.vert[0], curTri.vert[1], curTri.vert[2]);
        vertArr[1] = glm::vec3(curTri.vert[3], curTri.vert[4], curTri.vert[5]);
        vertArr[2] = glm::vec3(curTri.vert[6], curTri.vert[7], curTri.vert[8]);
        glm::vec3 norm(curTri.norm[0], curTri.norm[1], curTri.norm[2]);
        // do_assign_corerArea_posMap_validMap(vertArr, norm, idx, flip, invalidKCnt, shPoseMapCompactBufOffsets, shValidMapCompactBufOffsets, preDefHashBegin, tid, shPoseMapCompactBuf, shValidMapCompactBuf);

        do_assign_corerArea_posMap_validMap(vertArr, norm, idx, flip, invalidKCnt, preDefHashBegin, tid, shPoseMapCompactBuf, shValidCpctOffsets, validShWHMap);
    }
}


template <typename TriArrIt>
__global__ void sift_invalid_triangles_cuda(uint32_t rawTriNum, TriArrIt triArrBegin, uint32_t* triIsValidPool)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < rawTriNum) {
        TriangleInfoDevice curTri = *(triArrBegin + tid);
        uint8_t idx = FindLongestEdgeCuda(&(curTri.vert[0]), &(curTri.vert[3]), &(curTri.vert[6]));
        glm::vec3 vertArr[3];
        glm::vec3 norm(curTri.norm[0], curTri.norm[1], curTri.norm[2]);
        vertArr[0] = glm::vec3(curTri.vert[0], curTri.vert[1], curTri.vert[2]);
        vertArr[1] = glm::vec3(curTri.vert[3], curTri.vert[4], curTri.vert[5]);
        vertArr[2] = glm::vec3(curTri.vert[6], curTri.vert[7], curTri.vert[8]);

        uint8_t noUse;
        double noUse2;
        float noUse3;

        if (CalcSpaceLineKCuda(vertArr[idx], vertArr[((idx + 1) % 3)], vertArr[((idx + 2) % 3)],
            norm, noUse, noUse2, noUse3) == true) {
            *(triIsValidPool + tid) = 1;
        }
    }
}

typedef enum ThreadState {
    READY_FOR_RENDER,      // 可以用于渲染
    READY_FOR_TRAIN_PREP,         // 渲染完毕，可以进行中处理
    READY_FOR_TRAIN,        // 中处理完毕，可以进行后处理
} ThreadState;

typedef concurrent_hash_map<uint32_t, concurrent_vector<PixLocation>> texId2PixLocMapConcurrent_t;
typedef concurrent_unordered_set<uint32_t> uniqueTexIdsConcurrent_t;

typedef struct ThreadBuffer {
    GLuint fboId;

    GLuint texRGB;
    GLuint texLerpCoeffAndTexCoord;
    GLuint texViewDirFrag2CamAndNull;
    GLuint texOriShLUandRU;
    GLuint texOriShLDandRD;
    GLuint texScaleFacDepthTexId;
    GLuint texWorldPoseAndNull;
    GLuint texEllipCoeffsAndLodLvl;

    float* texScaleFacDepthTexIdCPU{nullptr};
    float* viewDirFrag2CamAndNullCPU{nullptr};
    float* oriShLUandRUCPU{nullptr};
    float* oriShLDandRDCPU{nullptr};
    float* lerpCoeffAndTexCoordCPU{nullptr};
    float* texWorldPoseAndNullCPU{nullptr};
    float* ellipCoeffsAndLodLvlCPU{nullptr};

    float* texScaleFacDepthTexIdGPU{nullptr};
    float* viewDirFrag2CamAndNullGPU{nullptr};
    float* oriShLUandRUGPU{nullptr};
    float* oriShLDandRDGPU{nullptr};
    float* lerpCoeffAndTexCoordGPU{nullptr};
    float* texWorldPoseAndNullGPU{nullptr};
    float* ellipCoeffsAndLodLvlGPU{nullptr};

    GLuint texDepthRBO;

    at::Tensor depthPunishmentMap;

    thrust::device_vector<TexPixInfo> visibleTexIdsInfo;
    thrust::device_vector<PixLocation> visibleTexPixLocCompact;

    std::vector<uint32_t> curTexIdsAll;

    std::unique_ptr<cuco::static_map<uint32_t, uint64_t>> curTrainHashMap;

    at::Tensor shTexturesCpct;
    float* shTexturesCpctHead{nullptr};

    at::Tensor adamExpAvg;
    float* adamExpAvgHead{nullptr};

    at::Tensor adamExpAvgSq;
    float* adamExpAvgSqHead{nullptr};

    at::Tensor shPosMapCpct;
    float* shPosMapCpctHead{nullptr};
    // thrust::device_vector<uint32_t> curPosMapMemLayoutDevice;

    at::Tensor validShWHMapCpct;
    int32_t* validShWHMapCpctHead{nullptr};

    at::Tensor validShNumsAll;
    int32_t* validShNumsAllHead{nullptr};
    // int32_t* shValidMapCpctHead{nullptr};

    at::Tensor edgeNbrs;
    int32_t* edgeNbrsHead{nullptr};

    at::Tensor vertNbrs;
    int32_t* vertNbrsHead{nullptr};

    at::Tensor cornerAreaInfo;
    float* cornerAreaInfoHead{nullptr};

    at::Tensor texInWorldInfo;
    float* texInWorldInfoHead{nullptr};

    at::Tensor texWHs;
    uint32_t* texWHsHead{nullptr};

    at::Tensor botYCoeffs;
    float* botYCoeffsHead{nullptr};

    at::Tensor topYCoeffs;
    float* topYCoeffsHead{nullptr};

    at::Tensor meshDensities;
    float* meshDensitiesHead{nullptr};

    at::Tensor meshNormals;
    float* meshNormalsHead{nullptr};

    std::vector<uint32_t> vertNbrCpctMemLayout;
    // std::vector<uint32_t> shValidMapMemLayout;
    std::vector<uint32_t> validShWHMapLayout;
    std::vector<uint32_t> posMapMemLayout;
    std::vector<uint32_t> shTexMemLayout;

    thrust::device_vector<uint32_t> posMapMemLayoutDevice;
    thrust::device_vector<uint32_t> validShWHMapLayoutDevice;
    thrust::device_vector<uint32_t> shTexMemLayoutDevice;

    uint32_t poseIdx;
    uint32_t curTrainCount{0};

    cuco::static_map_ref<uint32_t,
                             uint64_t,
                             cuda::thread_scope_device,
                             thrust::equal_to<uint32_t>,
                             cuco::linear_probing<1, cuco::default_hash_function<uint32_t>>,
                             my_impl_type::storage_ref_type,
                             cuco::op::find_tag
                             > devHashFindRefFind;

    ThreadState state{ThreadState::READY_FOR_RENDER};

    bool needResetOptimizor{false};
    std::mutex bufMutex;

    cudaStream_t curBufStream;
    cudaStream_t curBufStream2;

    uint32_t dummyCount{0};

    uniqueTexIdsConcurrent_t m_uniqueTexIds[2];
    texId2PixLocMapConcurrent_t m_texId2PixLocMap[2];

    uint32_t curValidIdx{0};
    bool isFirstTime{true};


    // fixed element size
    uint32_t m_cpyBufCurElemNum{0};
    
    uint32_t* m_validShNumsAllCpyBuf{nullptr};
    uint32_t* m_edgeNbrsCpyBuf{nullptr};
    uint32_t* m_texWHsCpyBuf{nullptr};

    float* m_cornerAreaInfoCpyBuf{nullptr};
    float* m_texInWorldInfoCpyBuf{nullptr};
    float* m_botYCoeffsCpyBuf{nullptr};
    float* m_topYCoeffsCpyBuf{nullptr};
    float* m_meshDensitiesCpyBuf{nullptr};
    float* m_meshNormalsCpyBuf{nullptr};
    
} ThreadBuffer;

typedef struct EvalBuffer
{
    ThreadBuffer thrdBuf;

    GLuint gtResoFBO;
    GLuint texGtResoRGB;
    GLuint texGtResoLerpCoeffAndTexCoord;
    GLuint texGtResoViewDirFrag2CamAndNull;
    GLuint texGtResoShLUandRU;
    GLuint texGtResoShLDandRD;
    GLuint texGtResoScaleFacDepthTexId;
    GLuint texGtResoEdgeLR;
    GLuint texGtResoWorldPoseAndNull;
    GLuint texGTResoEllipCoeffsAndLodLvl;

    GLuint gtResoDepthBuf;

    mat4x4 perspectiveMatGtReso;

    float* depthAndTexIdBufferGtReso{nullptr}; // depth: channel 2, texId channel 3
    
} EvalBuffer;

struct FileConfig
{
    // std::string cameraConfigPath;
    std::string cameraConfigGtPath;
    std::string algoConfigPath;
    SceneType sceneType;
    MeshType meshType;
};

// both tensor images, pix loc's origin are LU
template <typename TexIdIt, typename TexId2PixInfoIt>
__global__ void statistic_each_mesh_info(uint32_t meshNum, TexIdIt texIdBegin, TexId2PixInfoIt texId2PixInfoBegin, uint32_t TexId2PixInfoSize, float* gtImg, float* renderImg, float* psnrPoolBegin, float* l1PoolBegin, float* depthPoolBegin, uint32_t imgW, uint32_t imgH)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int offsetPerPixelLine = imgW * OFFSET_PER_PIX_CHANNEL;

    if (tid < meshNum) {
        uint32_t curTexId = *(texIdBegin + tid);

        float depthAvg = 0.0f;
        uint32_t findCnt = 0;
        float mse = 0.0f;
        float l1Diff = 0.0f;

        for (uint32_t i = 0; i < TexId2PixInfoSize; i++) {
            if (thrust::raw_reference_cast(*(texId2PixInfoBegin + i)).first != curTexId) {
                continue;
            }

            depthAvg = ((float)(depthAvg * (float)findCnt + thrust::raw_reference_cast(*(texId2PixInfoBegin + i)).second.depth) / (float)(findCnt + 1));

            // this xy, left up is origin. in render img and GT img, left down ls origin
            uint32_t x = thrust::raw_reference_cast(*(texId2PixInfoBegin + i)).second.pixLoc.data.x;
            uint32_t y = thrust::raw_reference_cast(*(texId2PixInfoBegin + i)).second.pixLoc.data.y;

            float pixR = *(renderImg + (y) * offsetPerPixelLine + x * OFFSET_PER_PIX_CHANNEL + 0);
            float pixG = *(renderImg + (y) * offsetPerPixelLine + x * OFFSET_PER_PIX_CHANNEL + 1);
            float pixB = *(renderImg + (y) * offsetPerPixelLine + x * OFFSET_PER_PIX_CHANNEL + 2);

            float pixR_GT = *(gtImg + (y) * offsetPerPixelLine + x * OFFSET_PER_PIX_CHANNEL + 0);
            float pixG_GT = *(gtImg + (y) * offsetPerPixelLine + x * OFFSET_PER_PIX_CHANNEL + 1);
            float pixB_GT = *(gtImg + (y) * offsetPerPixelLine + x * OFFSET_PER_PIX_CHANNEL + 2);

            mse += ((pixR - pixR_GT) * (pixR - pixR_GT));
            mse += ((pixG - pixG_GT) * (pixG - pixG_GT));
            mse += ((pixB - pixB_GT) * (pixB - pixB_GT));

            l1Diff += fabsf(pixR - pixR_GT);
            l1Diff += fabsf(pixG - pixG_GT);
            l1Diff += fabsf(pixB - pixB_GT);

            findCnt += 1;
        }

        mse = mse / ((float)findCnt * 3.0f);
        float l1loss = (l1Diff / ((float)findCnt * 3.f));

        float psnr = 10.f * log10f(((255.0f * 255.0f) / (mse)));
        
        *(psnrPoolBegin + tid) = psnr;
        *(l1PoolBegin + tid) = l1loss;
        *(depthPoolBegin + tid) = depthAvg;
    }
}

struct AlgoConfig
{
    unsigned int shOrder;

    unsigned int gtW;
    unsigned int gtH;

    unsigned int renderW;
    unsigned int renderH;
    unsigned int perPoseTrainCnt;

    std::string saveDir;
    std::string experimentName;
    std::string tensorboardDir;
    std::string gtPosePath;
    std::string gtImageDir;
    std::string meshPath;

    double highOrderSHLrMultiplier;
    std::string testSequencePath;
    std::string testGtimgPath;

    std::string evalSequencePath;
    std::string evalImgPath;

    unsigned int printPixX;
    unsigned int printPixY;
    unsigned int printGradTexId;
    unsigned int schedulerPatience;
    double invDistFactorEdge;
    double invDistFactorCorner;
    double gaussianCoeffEWA;

    uint32_t resizeStartStep;
    double shDensity;
    double shDensityMax;
    uint32_t maxTexWH;

    unsigned int texResizeInterval;
    uint32_t testInterval;

    double varThresholdPSNR;
    double resizeDepthThreshold;
    double densityUpdateStep;
    double densityUpdateStepInner;

    double L1lossThreshold;
    uint32_t resizePatience;
    uint32_t resizeReturnCntMax;
    double resizeConvergeThreshold;
    double resizeTriangleEdgeMinLen;

    uint32_t trainPoseCount;

    uint32_t isRenderingMode{0};
    std::string savedJsonPath;
    std::string savedBinPath;
    std::string RenderingPosePath;
    uint32_t ConvergedSaveInterval{10000};
};

__device__ __inline__ bool IsBetterCuda(ResizeInfo* resizeInfoBegin, uint32_t tid)
{
    if ((resizeInfoBegin + tid)->isProbe == true) {
        if ((resizeInfoBegin + tid)->probeDir == TEX_UP_RESOLUTION) {
            if ((((resizeInfoBegin + tid)->l1LossAvgLast) >= ((resizeInfoBegin + tid)->l1LossAvg)) ||
            (((resizeInfoBegin + tid)->l1LossVarLast) >= ((resizeInfoBegin + tid)->l1LossVar))) {
                return true;
            } else {
                return false;
            }
        } else {
            if ((((resizeInfoBegin + tid)->l1LossAvgLast) >= ((resizeInfoBegin + tid)->l1LossAvg)) &&
            (((resizeInfoBegin + tid)->l1LossVarLast) >= ((resizeInfoBegin + tid)->l1LossVar))) {
                return true;
            } else {
                return false;
            }
        }
    } else {
        if ((resizeInfoBegin + tid)->lastResizeDir == TEX_UP_RESOLUTION) {
            if ((((resizeInfoBegin + tid)->l1LossAvgLast) >= ((resizeInfoBegin + tid)->l1LossAvg)) ||
            (((resizeInfoBegin + tid)->l1LossVarLast) >= ((resizeInfoBegin + tid)->l1LossVar))) {
                return true;
            } else {
                return false;
            }
        } else {
            if ((((resizeInfoBegin + tid)->l1LossAvgLast) >= ((resizeInfoBegin + tid)->l1LossAvg)) &&
            (((resizeInfoBegin + tid)->l1LossVarLast) >= ((resizeInfoBegin + tid)->l1LossVar))) {
                return true;
            } else {
                return false;
            }
        }
    }
    // if ((((resizeInfoBegin + tid)->l1LossAvgLast) >= ((resizeInfoBegin + tid)->l1LossAvg)) ||
    //     (((resizeInfoBegin + tid)->l1LossVarLast) >= ((resizeInfoBegin + tid)->l1LossVar))) {
    //     return true;
    // } else {
    //     return false;
    // }
}


__device__ bool DoResizeCuda(TexResizeDir nextDir, float densityResizeTo,
    ResizeInfo* resizeInfoBegin,
    uint32_t tid, uint32_t maxTexWH,
    float densityUpdateStepInner,
    float resizeTriangleEdgeMinLen,
    float shDensityMax)
{
    assert((nextDir == TEX_UP_RESOLUTION) || (nextDir == TEX_DOWN_RESOLUTION));

    uint32_t needPrtIdx = 0;
    // if (meshIdx == needPrtIdx) {
    //     printf("idx : %d, nextDir: %d, input densityResizeTo: %f\n", needPrtIdx, nextDir, densityResizeTo);
    // }

    // prevent WH too large
    if (densityResizeTo <= shDensityMax) {
        printf("densityResizeTo invalid ! : %f\n", densityResizeTo);
        return true;
    }

    if (((resizeInfoBegin + tid)->curTexWH.data.curTexW >= maxTexWH) ||
        ((resizeInfoBegin + tid)->curTexWH.data.curTexH >= maxTexWH)) {
        printf("densityResizeTo: %f, WH invalid ! : cur : %d, %d, max: %d\n", densityResizeTo,
            (uint32_t)(resizeInfoBegin + tid)->curTexWH.data.curTexW,
            (uint32_t)(resizeInfoBegin + tid)->curTexWH.data.curTexH,
            maxTexWH);
        return true;
    }

    if (isnan((resizeInfoBegin + tid)->triangleBotDistance) == true ||
        isnan((resizeInfoBegin + tid)->triangleHeight) == true ||
        isinf((resizeInfoBegin + tid)->triangleBotDistance) == true ||
        isinf((resizeInfoBegin + tid)->triangleHeight) == true ||
        ((resizeInfoBegin + tid)->triangleBotDistance <= resizeTriangleEdgeMinLen) ||
        ((resizeInfoBegin + tid)->triangleHeight <= resizeTriangleEdgeMinLen)) {
        // curMeshResizeInfo.curTexWH.data.curTexW = 1;
        // curMeshResizeInfo.curTexWH.data.curTexH = 1;

        (resizeInfoBegin + tid)->curTexWH.data.curTexW = 1;
        (resizeInfoBegin + tid)->curTexWH.data.curTexH = 1;

        // curMeshResizeInfo.needResize = true;
        (resizeInfoBegin + tid)->needResize = true;
        // ShTextureRealloc(meshIdx);
        // tensor update later
        return true;
    }
    
    uint32_t resizedW = ceil((double)((resizeInfoBegin + tid)->triangleBotDistance) / (double)(densityResizeTo));
    uint32_t resizedH = ceil((double)((resizeInfoBegin + tid)->triangleHeight) / (double)(densityResizeTo));

    while ((resizedW == (resizeInfoBegin + tid)->curTexWH.data.curTexW) &&
           (resizedH == (resizeInfoBegin + tid)->curTexWH.data.curTexH)) {

            if (((resizeInfoBegin + tid)->curTexWH.data.curTexW == 1) &&
                ((resizeInfoBegin + tid)->curTexWH.data.curTexH == 1) &&
                (nextDir == TEX_DOWN_RESOLUTION)) {
                break;
            }


            if (nextDir == TEX_UP_RESOLUTION) {
                densityResizeTo -= densityUpdateStepInner;
            } else {
                densityResizeTo += densityUpdateStepInner;
            }

            resizedW = ceil((double)((resizeInfoBegin + tid)->triangleBotDistance) / (double)(densityResizeTo));
            resizedH = ceil((double)((resizeInfoBegin + tid)->triangleHeight) / (double)(densityResizeTo));

            if (tid == needPrtIdx) {
                printf("another resized WH: %d, %d, density: %f\n", resizedW, resizedH, densityResizeTo);
            }
            if (densityResizeTo <= shDensityMax) {
                break;
            }
    }

    if ((resizedW > maxTexWH) || (resizedH > maxTexWH)) {
        (resizeInfoBegin + tid)->needResize = false;
        return true;
    }


    if ((resizedW <= 0) || (resizedH <= 0) ||
        (isnan(resizedW) == true) || (isnan(resizedH) == true) ||
        (isinf(resizedW) == true) || (isinf(resizedH) == true)) {
        // curMeshResizeInfo.curTexWH.data.curTexW = 1;
        // curMeshResizeInfo.curTexWH.data.curTexH = 1;

        (resizeInfoBegin + tid)->curTexWH.data.curTexW = 1;
        (resizeInfoBegin + tid)->curTexWH.data.curTexH = 1;

        // curMeshResizeInfo.needResize = true;
        (resizeInfoBegin + tid)->needResize = true;
        // tensor update later

        // ShTextureRealloc(meshIdx);

        return true;
    }

    // if (meshIdx == needPrtIdx) {
    //     printf("idx : %d, nextDir: %d, densityResizeTo: %f, prevDen: %f\n", needPrtIdx, nextDir, densityResizeTo, m_shTexHashMap[meshIdx].curDensity);
    //     printf("idx : %d, resizedWH: %d, %d, prevWH: %d, %d\n", needPrtIdx, resizedW, resizedH, m_shTexHashMap[meshIdx].curTexWH.data.curTexW, m_shTexHashMap[meshIdx].curTexWH.data.curTexH);
    // }
    
    // curMeshResizeInfo.LastTexWH.aligner = curMeshResizeInfo.curTexWH.aligner;
    // (thrust::raw_reference_cast(*(outBufBegin + tid))).LastTexWH.aligner = (resizeInfoBegin + tid)->curTexWH.aligner;
    (resizeInfoBegin + tid)->LastTexWH.aligner = (resizeInfoBegin + tid)->curTexWH.aligner;

    // curMeshResizeInfo.curTexWH.data.curTexW = resizedW;
    // curMeshResizeInfo.curTexWH.data.curTexH = resizedH;

    // (thrust::raw_reference_cast(*(outBufBegin + tid))).curTexWH.data.curTexW = resizedW;
    // (thrust::raw_reference_cast(*(outBufBegin + tid))).curTexWH.data.curTexH = resizedH;

    (resizeInfoBegin + tid)->curTexWH.data.curTexW = resizedW;
    (resizeInfoBegin + tid)->curTexWH.data.curTexH = resizedH;

    // if (m_shTexHashMap[meshIdx].needReturn == false) {
    //     // m_shTexHashMap[meshIdx].prevDensity = m_shTexHashMap[meshIdx].curDensity;
    // } else {
    //     m_shTexHashMap[meshIdx].returnCnt++;
    // }

    if ((resizeInfoBegin + tid)->isProbe == true) {
        // curMeshResizeInfo.resizeInfo.probeDensity = densityResizeTo;
        (resizeInfoBegin + tid)->probeDensity = densityResizeTo;

        // curMeshResizeInfo.resizeInfo.probeDir = nextDir;
        (resizeInfoBegin + tid)->probeDir = nextDir;
    } else {
        // curMeshResizeInfo.resizeInfo.prevDensity = curMeshResizeInfo.resizeInfo.curDensity;
        (resizeInfoBegin + tid)->prevDensity = (resizeInfoBegin + tid)->curDensity;

        // curMeshResizeInfo.resizeInfo.curDensity = densityResizeTo;
        (resizeInfoBegin + tid)->curDensity = densityResizeTo;

        // curMeshResizeInfo.resizeInfo.l1LossAvgLast = curMeshResizeInfo.resizeInfo.l1LossAvg;
        (resizeInfoBegin + tid)->l1LossAvgLast = (resizeInfoBegin + tid)->l1LossAvg;
        (resizeInfoBegin + tid)->l1LossVarLast = (resizeInfoBegin + tid)->l1LossVar;

        // curMeshResizeInfo.resizeInfo.psnrVarLast = curMeshResizeInfo.resizeInfo.psnrVar;
        // (resizeInfoBegin + tid)->psnrVarLast = (resizeInfoBegin + tid)->psnrVar;

        // curMeshResizeInfo.resizeInfo.lastResizeDir = nextDir;
        (resizeInfoBegin + tid)->lastResizeDir = nextDir;
    }
    

    // curMeshResizeInfo.resizeInfo.needResize = true;
    (resizeInfoBegin + tid)->needResize = true;
    // tensor update later

    // ShTextureRealloc(meshIdx);
    // printf("DoResizeCuda final return\n");
    return false;
}

__global__ void resize_each_texture_cuda(uint32_t totalNum,
    ResizeInfo* resizeInfoBegin,
    float l1lossThreshold, float depthThreshold, float densityUpdateStep, float densityUpdateStepInner, uint32_t resizePatience, uint32_t maxTexWH, uint32_t resizeReturnCntMax, float resizeTriangleEdgeMinLen, float shDensityMax)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < totalNum) {
        // ResizeInfo curMeshResizeInfo = *(resizeInfoBegin + tid);

        if ((resizeInfoBegin + tid)->isConverge == true) {
            // printf("isConverge !!!\n");
            return;
        }

        // if ((resizeInfoBegin + tid)->l1LossAvg < l1lossThreshold) {
        //     return;
        // }

        bool meshNeedResize = true;
        TexResizeDir nextDir = TEX_OPT_MAX;
        float densityResizeTo = 0.0f;

        if ((resizeInfoBegin + tid)->resizeCnt == 0) { // first time based on depth
            if ((resizeInfoBegin + tid)->depthAvg >= depthThreshold) {
                nextDir = TEX_DOWN_RESOLUTION;
                densityResizeTo = ((resizeInfoBegin + tid)->curDensity + densityUpdateStep);
            } else {
                nextDir = TEX_UP_RESOLUTION;
                densityResizeTo = ((resizeInfoBegin + tid)->curDensity - densityUpdateStep);
            }

            if (DoResizeCuda(nextDir, densityResizeTo,
                resizeInfoBegin,
                // outBufBegin,
                tid, maxTexWH,
                densityUpdateStepInner,
                resizeTriangleEdgeMinLen,
                shDensityMax) == true) {
                // curMeshResizeInfo.isConverge = true;
                (resizeInfoBegin + tid)->isConverge = true;
            }
            // curMeshResizeInfo.resizeCnt++;
            (resizeInfoBegin + tid)->resizeCnt = ((resizeInfoBegin + tid)->resizeCnt + 1);

            return;
        }

#if 0
        nextDir = (resizeInfoBegin + tid)->lastResizeDir;

        if (nextDir == TEX_DOWN_RESOLUTION) {
            densityResizeTo = ((resizeInfoBegin + tid)->curDensity + densityUpdateStep);
        } else if (nextDir == TEX_UP_RESOLUTION) {
            densityResizeTo = ((resizeInfoBegin + tid)->curDensity - densityUpdateStep);
        } else {
            printf("invalid TexResizeDir!!! : %d\n", nextDir);
            assert(0);
        }

        (resizeInfoBegin + tid)->badCount = 0;
#else
        if (IsBetterCuda(resizeInfoBegin, tid) == true) {
            if ((resizeInfoBegin + tid)->isProbe == true) {
                // curMeshResizeInfo.isProbe = false;
                (resizeInfoBegin + tid)->isProbe = false;

                nextDir = (resizeInfoBegin + tid)->probeDir;
                if (nextDir == TEX_DOWN_RESOLUTION) {
                    densityResizeTo = ((resizeInfoBegin + tid)->probeDensity + densityUpdateStep);
                } else if (nextDir == TEX_UP_RESOLUTION) {
                    densityResizeTo = ((resizeInfoBegin + tid)->probeDensity - densityUpdateStep);
                } else {
                    printf("invalid TexResizeDir!!! : %d\n", nextDir);
                    assert(0);
                }
            } else {
                nextDir = (resizeInfoBegin + tid)->lastResizeDir;

                if (nextDir == TEX_DOWN_RESOLUTION) {
                    densityResizeTo = ((resizeInfoBegin + tid)->curDensity + densityUpdateStep);
                } else if (nextDir == TEX_UP_RESOLUTION) {
                    densityResizeTo = ((resizeInfoBegin + tid)->curDensity - densityUpdateStep);
                } else {
                    printf("invalid TexResizeDir!!! : %d\n", nextDir);
                    assert(0);
                }
            }
            // curMeshResizeInfo.badCount = 0;
            (resizeInfoBegin + tid)->badCount = 0;
            // if (i == needPrtIdx) {
            //     printf(" is Better, nextDir: %d, densityResizeTo: %f, curDensity: %f, step: %f\n", nextDir, densityResizeTo, m_shTexHashMap[i].curDensity, m_densityUpdateStep);
            // }
        } else {
            if ((resizeInfoBegin + tid)->isProbe == true) {
                nextDir = (TexResizeDir)(1 - (uint32_t)(resizeInfoBegin + tid)->lastResizeDir);
                // curMeshResizeInfo.badCount = 0;
                (resizeInfoBegin + tid)->badCount = 0;

                // curMeshResizeInfo.returnCnt++;
                (resizeInfoBegin + tid)->returnCnt = ((resizeInfoBegin + tid)->returnCnt + 1);

                // curMeshResizeInfo.isProbe = false;
                (resizeInfoBegin + tid)->isProbe = false;

                if ((resizeInfoBegin + tid)->returnCnt >= resizeReturnCntMax) {
                    // curMeshResizeInfo.isConverge = true;
                    (resizeInfoBegin + tid)->isConverge = true;
                    // fallback to the backward resolution
                }

                if (nextDir == TEX_DOWN_RESOLUTION) {
                    densityResizeTo = (resizeInfoBegin + tid)->prevDensity + densityUpdateStep;
                } else if (nextDir == TEX_UP_RESOLUTION) {
                    densityResizeTo = (resizeInfoBegin + tid)->prevDensity - densityUpdateStep;
                } else {
                    printf("invalid TexResizeDir!!! : %d\n", nextDir);
                    assert(0);
                }
            } else {
                // curMeshResizeInfo.badCount++;
                (resizeInfoBegin + tid)->badCount = ((resizeInfoBegin + tid)->badCount + 1);
                if ((resizeInfoBegin + tid)->badCount > resizePatience) {
                    if ((resizeInfoBegin + tid)->isProbe == false) {
                        // curMeshResizeInfo.isProbe = true;
                        (resizeInfoBegin + tid)->isProbe = true;
                    } else {
                        assert(0);
                    }
                    
                    nextDir = (resizeInfoBegin + tid)->lastResizeDir;
                    if (nextDir == TEX_DOWN_RESOLUTION) {
                        densityResizeTo = (resizeInfoBegin + tid)->curDensity + densityUpdateStep;
                    } else if (nextDir == TEX_UP_RESOLUTION) {
                        densityResizeTo = (resizeInfoBegin + tid)->curDensity - densityUpdateStep;
                    } else {
                        printf("invalid TexResizeDir!!! : %d\n", nextDir);
                        assert(0);
                    }

                    // curMeshResizeInfo.badCount = 0;
                    (resizeInfoBegin + tid)->badCount = 0;
                    

                    // if (i == needPrtIdx) {
                    //     printf(" need reverse, nextDir: %d, densityResizeTo: %f, curDensity: %f, step: %f\n", nextDir, densityResizeTo, curMeshResizeInfo.curDensity, m_densityUpdateStep);
                    // }
                } else {
                    meshNeedResize = false;
                }
            }
        }
#endif     

        if (meshNeedResize == false) {
            return;
        }

        if (DoResizeCuda(nextDir, densityResizeTo, resizeInfoBegin, tid, maxTexWH,
            densityUpdateStepInner, resizeTriangleEdgeMinLen, shDensityMax) == true) {
            // curMeshResizeInfo.isConverge = true;
            (resizeInfoBegin + tid)->isConverge = true;
        }

        // curMeshResizeInfo.resizeCnt++;
        (resizeInfoBegin + tid)->resizeCnt = ((resizeInfoBegin + tid)->resizeCnt + 1);
    }
}

typedef struct BufferObjects
{
    GLuint vao;

    GLuint vertVbo;
    GLuint meshVbo;

    uint32_t meshNumOfThisKind{0};
} BufferObjects;


__device__ void CalcLerpAndTexCoords(float texCoordX, float texCoordY, uint32_t srcW, uint32_t srcH, glm::vec2& coordLU, glm::vec2& coordRU, glm::vec2& coordLD, glm::vec2& coordRD, glm::vec2& coeff)
{
    glm::vec2 texCoordTmp(texCoordX, texCoordY);

    float srcTexelSizeW = 1.0 / (float)srcW;
    float srcTexelSizeH = 1.0 / (float)srcH;

    float fractW = glm::fract(texCoordX * (float)srcW);
    float fractH = glm::fract(texCoordY * (float)srcH);

    // glm::vec2 coordLU;
    // glm::vec2 coordRU;
    // glm::vec2 coordLD;
    // glm::vec2 coordRD;

    // glm::vec2 coeff;

    if (fractW >= 0.5) {
        if (fractH >= 0.5) {
            coordLU = texCoordTmp + glm::vec2(0.0, srcTexelSizeH);
            coordRU = texCoordTmp + glm::vec2(srcTexelSizeW, srcTexelSizeH);
            coordLD = texCoordTmp;
            coordRD = texCoordTmp + glm::vec2(srcTexelSizeW, 0.0);
            
            coeff.x = fractW - (0.5);
            coeff.y = (1.5) - fractH;
        } else {
            coordLU = texCoordTmp;
            coordRU = texCoordTmp + glm::vec2(srcTexelSizeW, 0.0);
            coordLD = texCoordTmp + glm::vec2(0.0, -srcTexelSizeH);
            coordRD = texCoordTmp + glm::vec2(srcTexelSizeW, -srcTexelSizeH);
            
            coeff.x = fractW - (0.5);
            coeff.y = (0.5) - fractH;
        }
    } else {
        if (fractH >= 0.5) {
            coordLU = texCoordTmp + glm::vec2(-srcTexelSizeW, srcTexelSizeH);
            coordRU = texCoordTmp + glm::vec2(0.0, srcTexelSizeH);
            coordLD = texCoordTmp + glm::vec2(-srcTexelSizeW, 0.0);
            coordRD = texCoordTmp;

            coeff.x = (0.5) + fractW;
            coeff.y = (1.5) - fractH;
        } else {
            coordLU = texCoordTmp + glm::vec2(-srcTexelSizeW, 0.0);
            coordRU = texCoordTmp;
            coordLD = texCoordTmp + glm::vec2(-srcTexelSizeW, -srcTexelSizeH);
            coordRD = texCoordTmp + glm::vec2(0.0, -srcTexelSizeH);

            coeff.x = (0.5) + fractW;
            coeff.y = (0.5) - fractH;
        }
    }

    

}

__device__ void RectifyTexCoords(glm::vec2& texCoord)
{
    if (texCoord.x < 0.f) {
        texCoord.x = 0.f;
    }

    if (texCoord.y < 0.f) {
        texCoord.y = 0.f;
    }

    if (texCoord.x > 1.f) {
        texCoord.x = 1.f;
    }

    if (texCoord.y > 1.f) {
        texCoord.y = 1.f;
    }
}

__device__ __inline__ float ReadTexel(float* curTexStartAddr, uint32_t texW, uint32_t texH, uint32_t curLayer, uint32_t curChannel, glm::vec2 texCoord, uint32_t perLineOffset, uint32_t perLayerOffset)
{
    return (*(curTexStartAddr + perLayerOffset * curLayer + (uint32_t)(glm::floor(texH * texCoord.y)) * perLineOffset + (uint32_t)(glm::floor(texW * texCoord.x)) * 3 + curChannel));
}

__device__ __inline__ float DoBiLerp(float dataLU, float dataRU, float dataLD, float dataRD, glm::vec2 coeff)
{
    return (((1.f - coeff.x) * (1.f - coeff.y) * dataLU) +
            ((coeff.x) * (1.f - coeff.y) * dataRU) +
            ((1.f - coeff.x) * (coeff.y) * dataLD) +
            ((coeff.x) * (coeff.y) * dataRD));
}

__global__ void resample_each_texture(float* oriShTexCpct, uint32_t curOriCpctStartOffset, uint32_t* oriShTexOffsets, uint32_t* oriShTexWs, uint32_t* oriShTexHs, float* rszSHTexCpct, uint32_t curRszCpctStartOffset, uint32_t* resizedShTexOffsets, uint32_t* resizedShTexWs, uint32_t* resizedShTexHs, uint32_t totalNum, uint32_t shLayerNum)
{
    // auto blockId = blockIdx.x;
    // auto grp = cg::tiled_partition<MESH_HANDLE_THREAD_NUM>(cg::this_thread_block());
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < totalNum) {
        uint32_t targetW = (*(resizedShTexWs + tid));
        uint32_t targetH = (*(resizedShTexHs + tid));

        uint32_t srcW = (*(oriShTexWs + tid));
        uint32_t srcH = (*(oriShTexHs + tid));

        float targetTexelW = 1.f / (float)targetW;
        float targetTexelH = 1.f / (float)targetH;

        float startW = (targetTexelW / 2.f);
        float startH = (targetTexelH / 2.f);

        uint32_t perLineOffsetSrc = 3 * srcW;
        uint32_t perLayerOffsetSrc = perLineOffsetSrc * srcH;

        uint32_t perLineOffsetDst = 3 * targetW;
        uint32_t perLayerOffsetDst = perLineOffsetDst * targetH;

        for (uint32_t curH = 0; curH < targetH; curH++) {
            for (uint32_t curW = 0; curW < targetW; curW++) {
                float curTexCoordW = startW + (float)curW * targetTexelW;
                float curTexCoordH = startH + (float)curH * targetTexelH;

                glm::vec2 coordLU;
                glm::vec2 coordRU;
                glm::vec2 coordLD;
                glm::vec2 coordRD;
                glm::vec2 coeff;
                CalcLerpAndTexCoords(curTexCoordW, curTexCoordH, srcW, srcH, coordLU, coordRU, coordLD, coordRD, coeff);

                RectifyTexCoords(coordLU);
                RectifyTexCoords(coordRU);
                RectifyTexCoords(coordLD);
                RectifyTexCoords(coordRD);

                for (uint32_t curLayer = 0; curLayer < shLayerNum; curLayer++) {
                    for (uint32_t c = 0; c < 3; c++) { // channel RGB
                        float dataLU = ReadTexel((oriShTexCpct + oriShTexOffsets[tid] - curOriCpctStartOffset), (srcW), (srcH), curLayer, c, coordLU, perLineOffsetSrc, perLayerOffsetSrc);
                        float dataRU = ReadTexel((oriShTexCpct + oriShTexOffsets[tid] - curOriCpctStartOffset), (srcW), (srcH), curLayer, c, coordRU, perLineOffsetSrc, perLayerOffsetSrc);
                        float dataLD = ReadTexel((oriShTexCpct + oriShTexOffsets[tid] - curOriCpctStartOffset), (srcW), (srcH), curLayer, c, coordLD, perLineOffsetSrc, perLayerOffsetSrc);
                        float dataRD = ReadTexel((oriShTexCpct + oriShTexOffsets[tid] - curOriCpctStartOffset), (srcW), (srcH), curLayer, c, coordRD, perLineOffsetSrc, perLayerOffsetSrc);

                        *(rszSHTexCpct + resizedShTexOffsets[tid] - curRszCpctStartOffset + curLayer * perLayerOffsetDst + curH * perLineOffsetDst + curW * 3 + c) = DoBiLerp(dataLU, dataRU, dataLD, dataRD, coeff);
                    }
                }
            }
        }
    }
}

typedef enum TrainState
{
    STATE_NOT_RESIZED = 0,
    STATE_UNDER_RESIZED,
    STATE_CONVERGE_RESIZED,
    STATE_NUM,
} TrainState;


__global__ void custom_maxpool_2x2(uint32_t srcW, uint32_t srcH, uint32_t dstW, uint32_t dstH, uint8_t* srcBuf, uint8_t* dstBuf)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= dstW) || (y >= dstH)) {
        return;
    }

    uint8_t dataLU = *(srcBuf + y * 2 * srcW + x * 2);
    uint8_t dataRU = *(srcBuf + y * 2 * srcW + x * 2 + 1);
    uint8_t dataLD = *(srcBuf + (y * 2 + 1) * srcW + x * 2);
    uint8_t dataRD = *(srcBuf + (y * 2 + 1) * srcW + x * 2 + 1);

    *(dstBuf + y * dstW + x) = (dataLU * dataRU * dataLD * dataRD);
}

struct ConstructCompactTex2PixBufferMultiThrd {
    std::mutex& glbLock;
    std::vector<TexPixInfo>& visibleTexIdsInfo;
    std::vector<PixLocation>& visibleTexPixLocCompact;
    std::vector<uint32_t>& visibleTexIdsVec;
    texId2PixLocMapConcurrent_t& texId2PixLocMap;
    uint32_t& wrtIdx;
    uint32_t& wrtOffset;

    ConstructCompactTex2PixBufferMultiThrd(std::mutex& glbLock_,
        std::vector<TexPixInfo>& visibleTexIdsInfo_,
        std::vector<PixLocation>& visibleTexPixLocCompact_,
        std::vector<uint32_t>& visibleTexIdsVec_,
        texId2PixLocMapConcurrent_t& texId2PixLocMap_,
        uint32_t& wrtIdx_,
        uint32_t& wrtOffset_):
            glbLock(glbLock_),
            visibleTexIdsInfo(visibleTexIdsInfo_),
            visibleTexPixLocCompact(visibleTexPixLocCompact_),
            visibleTexIdsVec(visibleTexIdsVec_),
            texId2PixLocMap(texId2PixLocMap_),
            wrtIdx(wrtIdx_),
            wrtOffset(wrtOffset_) {}

    void operator()(const blocked_range<uint32_t> range) const
    {
        for (uint32_t i = range.begin(); i != range.end(); i++) {
            typename texId2PixLocMapConcurrent_t::const_accessor access;
            auto ret = texId2PixLocMap.find(access, visibleTexIdsVec[i]);
            uint32_t findNum = access->second.size();

            uint32_t curIdx = 0;
            uint32_t curOffset = 0;

            std::unique_lock<std::mutex> lock(glbLock);

            curIdx = wrtIdx;
            curOffset = wrtOffset;

            wrtIdx += 1;
            wrtOffset += findNum;

            lock.unlock();

            for (uint32_t ii = 0; ii < findNum; ii++) {
                visibleTexPixLocCompact[curOffset + ii].x = access->second[ii].x;
                visibleTexPixLocCompact[curOffset + ii].y = access->second[ii].y;
            }

            access.release();

            if (findNum <= 0) {
                printf("cannot find texId : %d-------------\n", visibleTexIdsVec[i]);
                exit(0);
            }

            visibleTexIdsInfo[curIdx].offset = curOffset;
            visibleTexIdsInfo[curIdx].pixNum = findNum;
            visibleTexIdsInfo[curIdx].texId = visibleTexIdsVec[i];
        }
    }
};

struct AddInvisibleTexIdMultiThrd {
    std::vector<HashValue>& glbHashMap;
    uniqueTexIdsConcurrent_t& extraTexIds;
    std::vector<uint32_t>& visibleTexIds;
    uniqueTexIdsConcurrent_t& visibleTexIdsHash;

    AddInvisibleTexIdMultiThrd(std::vector<HashValue>& glbHashMap_, uniqueTexIdsConcurrent_t& extraTexIds_, std::vector<uint32_t>& visibleTexIds_, uniqueTexIdsConcurrent_t& visibleTexIdsHash_): glbHashMap(glbHashMap_), extraTexIds(extraTexIds_), visibleTexIds(visibleTexIds_), visibleTexIdsHash(visibleTexIdsHash_) {}
    
    void operator()(const blocked_range<uint32_t> range) const
    {
        for (uint32_t i = range.begin(); i != range.end(); i++) {
            for (auto it2: glbHashMap[visibleTexIds[i] - 1].topVertNbr) {
                if (visibleTexIdsHash.contains(it2) == true) {
                    continue;
                }

                extraTexIds.insert(it2);
            }

            for (auto it2: glbHashMap[visibleTexIds[i] - 1].leftVertNbr) {
                if (visibleTexIdsHash.contains(it2) == true) {
                    continue;
                }

                extraTexIds.insert(it2);
            }

            for (auto it2: glbHashMap[visibleTexIds[i] - 1].rightVertNbr) {
                if (visibleTexIdsHash.contains(it2) == true) {
                    continue;
                }

                extraTexIds.insert(it2);
            }
        }
    }
};

class ProgressBar {
private:
    int total;
    int barWidth;
    std::chrono::steady_clock::time_point startTime;
    
public:
    ProgressBar(int total, int barWidth = 50) 
        : total(total), barWidth(barWidth), startTime(std::chrono::steady_clock::now()) {}
    
    void update(int current) {
        float progress = static_cast<float>(current) / total;
        int pos = static_cast<int>(barWidth * progress);
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
        
        std::cout << "\r[";
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) 
                  << (progress * 100.0) << "% (" 
                  << current << "/" << total << ")";
        
        if (current > 0 && elapsed > 0) {
            int eta = static_cast<int>((elapsed * (total - current)) / current);
            std::cout << " ETA: " << eta << "s";
            std::cout << " Speed: " << std::fixed << std::setprecision(2) 
                      << (static_cast<float>(current) / elapsed) << " it/s";
        }
        
        std::cout << std::flush;
    }
    
    void finish() {
        std::cout << std::endl;
    }
};

class ImMeshRenderer {
public:
    ImMeshRenderer() {};
    ~ImMeshRenderer() {};

    using edgeMapKeyType   = EdgeKeyCuda;
    using edgeMapValueType = int;

    using vertMapKeyType   = int;
    using vertMapValueType = int;

    edgeMapKeyType  emptyEdgeKeySentinel{-1, -1};
    edgeMapValueType emptyEdgeValueSentinel = -1;

    vertMapKeyType emptyVertKeySentinel = -1;
    vertMapValueType emptyVertValueSentinel = -1;

    using edgeMapProbe = cuco::legacy::linear_probing<1, EdgeKeyCudaHasher>;
    using vertMapProbe = cuco::legacy::linear_probing<1, cuco::default_hash_function<vertMapKeyType>>;
    using edgeSetProbe = cuco::linear_probing<1, EdgeKeyCudaHasher>;
    using testSetProbe = cuco::double_hashing<1, cuco::default_hash_function<int>>;

    // Struct to hold both multimaps for easier passing between functions
    struct MeshMultimaps {
        std::unique_ptr<cuco::static_multimap<edgeMapKeyType, edgeMapValueType, cuda::thread_scope_device, cuco::cuda_allocator<char>, edgeMapProbe>> edgeNbrMultiMapCuda;
        std::unique_ptr<cuco::static_multimap<vertMapKeyType, vertMapValueType, cuda::thread_scope_device, cuco::cuda_allocator<char>, vertMapProbe>> vertNbrMultiMapCuda;
        
        MeshMultimaps() = default;
        
        // Move constructor and assignment
        MeshMultimaps(MeshMultimaps&& other) noexcept 
            : edgeNbrMultiMapCuda(std::move(other.edgeNbrMultiMapCuda))
            , vertNbrMultiMapCuda(std::move(other.vertNbrMultiMapCuda)) {}
            
        MeshMultimaps& operator=(MeshMultimaps&& other) noexcept {
            if (this != &other) {
                edgeNbrMultiMapCuda = std::move(other.edgeNbrMultiMapCuda);
                vertNbrMultiMapCuda = std::move(other.vertNbrMultiMapCuda);
            }
            return *this;
        }
        
        // Delete copy constructor and assignment to prevent accidental copying
        MeshMultimaps(const MeshMultimaps&) = delete;
        MeshMultimaps& operator=(const MeshMultimaps&) = delete;
    };

    void SetConfig(float fx, float fy, float cx, float cy, unsigned int order, float highOrderSHLrMultiplier, unsigned int printPixX, unsigned int printPixY, unsigned int printGradTexId, unsigned int schedulerPatience, float invDistFactorEdge, float invDistFactorCorner, float gaussianCoeffEWA, float shDensity, unsigned int gtW, unsigned int gtH, uint32_t texResizeInterval, CameraConfig camGtConfig, float varThresholdPSNR, float resizeDepthThreshold, float densityUpdateStep, float densityUpdateStepInner, uint32_t renderW, uint32_t renderH, uint32_t testInterval, float L1lossThreshold, uint32_t resizePatience, uint32_t trainPoseCount, uint32_t maxTexWH, uint32_t resizeStartStep, uint32_t resizeReturnCntMax, float resizeConvergeThreshold, float resizeTriangleEdgeMinLen, float shDensityMax) {
        m_fxRender = fx;
        m_fyRender = fy;
        m_cxRender = cx;
        m_cyRender = cy;

        m_fxGt = camGtConfig.fx;
        m_fyGt = camGtConfig.fy;
        m_cxGt = camGtConfig.cx;
        m_cyGt = camGtConfig.cy;

        m_resizeStartStep = resizeStartStep;
        m_gtW = gtW;
        m_gtH = gtH;

        m_width = renderW;
        m_height = renderH;

        m_shOrder = order;

        m_highOrderSHLrMultiplier = highOrderSHLrMultiplier;

        m_printPixX = printPixX;
        m_printPixY = printPixY;
        m_printGradTexId = printGradTexId;

        m_schedulerPatience = schedulerPatience;

        m_invDistFactorEdge = invDistFactorEdge;
        m_invDistFactorCorner = invDistFactorCorner;

        m_shLayerNum = int(pow((m_shOrder + 1), 2));

        m_gaussianCoeffEWA = gaussianCoeffEWA;

    	m_trainPoseCount = trainPoseCount;
        m_initialShDensity = shDensity;
        m_shDensityMax = shDensityMax;

        m_texResizeInterval = texResizeInterval;

        m_varThresholdPSNR = varThresholdPSNR;
        m_resizeDepthThreshold = resizeDepthThreshold;
        m_L1lossThreshold = L1lossThreshold;

        m_densityUpdateStep = densityUpdateStep;
        m_densityUpdateStepInner = densityUpdateStepInner;

        m_testInterval = testInterval;

        m_resizePatience = resizePatience;
        m_resizeReturnCntMax = resizeReturnCntMax;
        m_resizeConvergeThreshold = resizeConvergeThreshold;
        m_resizeTriangleEdgeMinLen = resizeTriangleEdgeMinLen;

        m_maxTexWH = maxTexWH;
    };

    void WindowInit();
    void VertexInit();
    void CudaInit();
    void ShaderInit();
    void ImgSaveFromOPGL(std::string dir);
    // void VertexShaderInit();
    // void FragShaderInit();
    void MatConvert(mat4x4& outMat, std::vector<float> inMat);
    void MapIntermediateTextures();
    void InitRenderedResultTensor();
    void ImgSaveCUDA(std::string dir, std::string name, bool needAddCnt = true);
    void UnmapIntermediateTextures();
    
    // Data saving and loading functions for m_shTexHashMap
    bool LoadHashMapData(const std::string& binFilename, const std::string& jsonFilename);
    bool LoadHashMapDataByTimestamp(const std::string& timestamp);
    void ClearHashMapData();  // Function to properly cleanup allocated memory
    
    // Global profiling control functions
    static void EnableProfilingPrints(bool enable);
    static bool IsProfilingEnabled();
    void ShTextureTensorInitCUDADummy();
    void InitVerts(std::string path, MeshType meshType);
    unsigned char FindMaxInArray();
    bool IsCurPixInsideFrame(int curH, int curW, int width, int height);
    void DepthPunishment(std::string dir);
    void GenDepthPunishmentMap();
    void VisualizePunishmentMap(std::string dir);
    float Depth2Coefficient(unsigned char depth);

    uint32_t CalcCValidBitNum(unsigned char validBits);

    void AddInvisibleTexId(uniqueTexIdsConcurrent_t& uniqueTexIds);

    void InitDeviceSpace(std::vector<float*>& worldPoseMapDevPtr, std::vector<float*>& validShMapDevPtr);

    void HashValuePredifinedHandle(thrust::host_vector<HashValPreDifinedDevice>& hashValPredifinedHost, std::vector<VertNbrMemInfoTmp>& vertNbrMemInfoTmp, uint32_t& compactVertBufSize, thrust::host_vector<TexScaleFactor>& scaleFactorHost, float* texInWorldInfoHostPtr, bool isRenderingMode);

    void MallocedDataWriteToHashMap(std::vector<VertNbrMemInfoTmp>& vertNbrMemInfoTmp, thrust::host_vector<cuco::pair<int, int>>& vertNbrGetHost);

    void ResetImg();

    void WriteCompactBuffersToGlobalHash(thrust::device_vector<float>& shPoseMapBufDev,
        thrust::device_vector<uint32_t>& shValidCpctOffsets, std::vector<uint32_t>& shValidCpctNumbersHost,
        thrust::device_vector<TexAlignedPosWH>& validShWHMap,
        thrust::host_vector<HashValPreDifinedDevice>& hashValPredifinedHost);

    GLuint GenDummyTexture(uint32_t texW, uint32_t texH);

    uint32_t m_gtW{0}, m_gtH{0};

    // std::vector<GLuint> m_dummyTextures;
    // std::unordered_map<CurTexWH, uint32_t, HashCurTexWH> m_dummyTexEntryMap;

    std::unordered_map<CurTexWH, GLuint, HashCurTexWH> m_texWH2dummyTex;
    std::unordered_map<GLuint, std::unordered_set<uint32_t>> m_dummyOpenglTexId2TexId;
    std::unordered_map<GLuint, BufferObjects> m_dummyTex2BufferObjs;
    std::unordered_map<GLuint, BufferObjects> m_dummyTex2BufferObjsInfer;
    std::unordered_set<GLuint> m_OpenglDummyTexId;

    void CopyCurTrainEssentials(uint32_t thrdBufIdx);
    void CopyCurInferEssentials(ThreadBuffer& buf);
    void PrepOptimizer(uint32_t thrdBufIdx, std::shared_ptr<torch::optim::Adam> optimizer);
    void RegisterCudaTexturesInfer();
    void RegisterCudaTextures(uint32_t thrdBufIdx);
    void TrainedWriteBackPipeline(uint32_t thrdBufIdx);
    void DoDrawInfer(mat4x4 viewMatrix, std::vector<float> camPos);
    void Infer(std::string dir, std::string prefix, unsigned int curTrainLayerNum, unsigned int cnt);
    void GenHashMaps(uniqueTexIdsConcurrent_t& uniqueTexIds, texId2PixLocMapConcurrent_t& texId2PixLocMap, ThreadBuffer& buf);
    void ConstructCompactTex2PixBufferPipeLine(texId2PixLocMapConcurrent_t& texId2PixLocMap, uniqueTexIdsConcurrent_t& uniqueTexIds, ThreadBuffer& buf);
    void ConstructDeviceShHashAllNbrPipeLine(ThreadBuffer& buf);
    void ConstructCustomHashValueAllNbrPipeLine(std::vector<uint64_t>& insertVals, std::vector<uint32_t>& curMemLayoutVec, ThreadBuffer& buf);
    void InitCurBufTrainTensors(uint32_t thrdBufIdx);
    void DoRender(uint32_t bufIdx, uint32_t poseIdx);
    void InitInferBuffer(ThreadBuffer& buf);
    void RealDepth2PixelValPipeLine(std::set<int>& uniqueDepthVal, ThreadBuffer& buf);
    void PixelVal2RealDepthPipeLine(ThreadBuffer& buf);
    void DepthPunishmentPipeLine(ThreadBuffer& buf);
    void GenDepthPunishmentMapPipeLine(ThreadBuffer& buf);
    void InitCurBufInferTensors(ThreadBuffer& buf);
    void InferPreparePipeline(ThreadBuffer& buf);

    void ShaderInit(bool isInfer);
    void DrawPrepare(SceneType sceneType, bool isInfer);

    void ConstructCpctBuffersForCuda(ThreadBuffer& buf, bool isInfer);

    // void UnRegisterCudaTextures();

    void RegisterTexForEval();

    void ResizeShTextures();

    void StactisticEachMesh();

    void DrawPrepareGtReso(SceneType sceneType);

    void CalcEachMeshStatus();

    void ResizeEachTexture();

    void print_progress_bar(int progress, int total);

    void ResizeDataPrepare(thrust::device_vector<ResizeInfo>& eachMeshResizeInfoDev);

    void ResizeEachTextureCuda();

    void ResizeResultWriteBack(thrust::device_vector<ResizeInfo>& eachMeshResizeInfoOutDev);

    void AllMeshRealloc();

    std::string m_taskStartTime;

    uint32_t m_testInterval{100};
    std::vector<float> m_vertBufData;
    // std::vector<float> m_vertBufDataInstanced;
    // std::vector<float> m_meshBufDataInstanced;

    GLuint m_renderThreadVBO, m_renderThrdDummyVBO;
    // GLuint m_renderThrdVertAttriInstancedVBO;
    // GLuint m_renderThrdMeshAttriInstancedVBO;

    GLuint m_trainThreadVBO, m_trainThreadVAO;
    GLuint m_trainThrdVertAttriInstancedVBO;
    GLuint m_trainThrdMeshAttriInstancedVBO;
    int m_cudaDevice{0};

    std::unique_ptr<nlptk::Recorder> m_recorderPtr{nullptr};

    AlgoConfig m_algoConfig;
    FileConfig m_fileConfig;

    static const int NUM_BUFFERS = 4;
    ThreadBuffer m_thrdBufs[NUM_BUFFERS];
    ThreadBuffer m_inferBuffer;
    EvalBuffer m_evalBuffer;

    // std::mutex m_renderMtx, m_trainPrepMtx, m_trainMtx;
    // std::condition_variable m_renderCv, m_midCv, m_postCv;
    uint32_t m_renderThrdCurIdx{0}, m_trainPrepThrdCurIdx{0}, m_trainThrdCurIdx{0};

    std::atomic<bool> m_threadRunning{true};

    std::atomic<bool> m_renderThrdInited{false};
    std::atomic<bool> m_trainPrepThrdInited{false};

    void InitThreadBuffers();
    void ThreadStart();

    void TrainThreadLoop();
    void TrainPrepareThreadLoop();
    void RenderThreadLoop();

    void DoDraw(mat4x4 viewMatrix, std::vector<float> camPos, ThreadBuffer& buf);

    void ReassignWriteToGlobalHash(std::vector<uint32_t>& idxNeedToUpdate,
        std::vector<uint32_t>& shPosMapOffsets,
        std::vector<uint32_t>& shValidMapOffsets,
        thrust::device_vector<float>& shPoseMapBufDev,
        thrust::device_vector<TexAlignedPosWH>& shValidMapBufDev,
        thrust::device_vector<HashValPreDifinedDevice>& hashValPredifinedDev);

    void RecalcAndFillShTextureInfo();
    void NewDummyTextures();
    void ShTextureRealloc(uint32_t meshIdx);
    void UpdateMeshStatisticInfoOnce(std::vector<float>& curMeshesPsnr, std::vector<float>& curMeshesL1Loss, std::vector<float>& curMeshesDepth, std::vector<uint32_t>& uniqueTexIds);

    void DoDrawGtReso(mat4x4 viewMatrix, std::vector<float> camPos);
    void RenderFrame(uint32_t poseIdx);
    void GtResolutionTextureInit();
    void RegisterTexForTrain();
    void RegisterTexForResize();
    void ResizeCleanUp();
    void ResizeEachTextureTargetMode();

    void EvalBufferInit();
    void RenderFrameEval(uint32_t poseIdx);


    bool IsBetter(uint32_t meshIdx);

    void EGLInit();
    bool IsConverge(uint32_t trainCnt);

    void SaveTrainTextures(ThreadBuffer& buf);

    void MoveTex2GPU(ThreadBuffer& buf);

    void ReassignBuffeObjs();

    void ReassignBuffeObjsInfer();

    bool m_isSubTrainMode{false};

    void CopyCurTrainEssentialsFixed(uint32_t thrdBufIdx);

    void ResampleToResizedBuffer(std::vector<uint32_t>& idxNeedToUpdate, std::vector<uint32_t>& oriShTexOffsets, std::vector<uint32_t>& resizedShTexOffsets, std::vector<uint32_t> oriShTexWs, std::vector<uint32_t> oriShTexHs, std::vector<uint32_t> resizedShTexWs, std::vector<uint32_t> resizedShTexHs, std::vector<uint32_t> dividePositions);

    void __inline__ FillValidSHs(float* dstStartAddr, float* curValidShData, uint32_t curTexLayerNumAll, TexAlignedPosWH* curTexValidShWH, uint32_t curValidShNum, uint32_t curTexW, uint32_t curTexH);

    void __inline__ ExtractValidShs(float* dstValidShDataStartAddr, float* srcAllShs, uint32_t curTexLayerNumAll, TexAlignedPosWH* curTexValidShWH, uint32_t curTexValidShNum, uint32_t curTexW, uint32_t curTexH);

    void RecalcValidShNums();

    void InitTrainThrdCpyBuffer(uint32_t thrdBufIdx);

    int32_t ReleaseResourse(uint32_t curThrdIdx, uint32_t curBufIdx);

    void InitTrainPrepThrdCpyBufferDynamic(uint32_t posMapBufSize, uint32_t validShWHMapSize, uint32_t vertNbrsSize);
    void InitTrainPrepThrdCpyBufferDynamicInfer(uint32_t posMapBufSize, uint32_t validShWHMapSize, uint32_t vertNbrsSize);

    void InitThrdFixedCpyBuffer(ThreadBuffer& buf, uint32_t elementNum);

    void LoadMesh(std::string path, MeshType meshType, MeshMultimaps& multimaps);

    bool LoadShTexturesFromBin();

    bool CheckCudaResultsWithJsonData(thrust::device_vector<HashValPreDifinedDevice>& hashValPreDefined);

    bool ReadValidationSequence(SceneType scene, std::string posePath, mat4x4** viewMatrix, std::vector<std::vector<float>>& camPosWorld);

    void SaveImg(std::string dir, std::string name, at::Tensor img);

    void Rendering(std::string outDir);

    uint32_t m_trainThrdCurTrainCount{0};

    std::vector<JsonData> m_readJsonDataVec;

    cudaStream_t m_shTexturesCpctCpyStrm;
    cudaStream_t m_adamExpAvgCpyStrm;
    cudaStream_t m_adamExpAvgSqCpyStrm;

    float* m_shTexturesCpctCpyBuf{nullptr};
    uint32_t m_shTexturesCpctCpyBufBytes{0};

    float* m_adamExpAvgCpyBuf{nullptr};
    uint32_t m_adamExpAvgCpyBufBytes{0};

    float* m_adamExpAvgSqCpyBuf{nullptr};
    uint32_t m_adamExpAvgSqCpyBufBytes{0};

    cudaStream_t m_trainPrepThrdCpyStrm1;
    cudaStream_t m_trainPrepThrdCpyStrm2;

    float* m_shPosMapCpctCpyBuf{nullptr};
    uint32_t m_shPosMapCpctCpyBufSize{0};

    float* m_validShWHMapCpctCpyBuf{nullptr};
    uint32_t m_validShWHMapCpctCpyBufSize{0};

    float* m_vertNbrsCpyBuf{nullptr};
    uint32_t m_vertNbrsCpyBufSize{0};

    float* m_shPosMapCpctCpyBufInfer{nullptr};
    uint32_t m_shPosMapCpctCpyBufSizeInfer{0};

    float* m_validShWHMapCpctCpyBufInfer{nullptr};
    uint32_t m_validShWHMapCpctCpyBufSizeInfer{0};

    float* m_vertNbrsCpyBufInfer{nullptr};
    uint32_t m_vertNbrsCpyBufSizeInfer{0};

    uint32_t m_curResizeCnt{0};
    uint32_t m_resizeMaxCnt{300};
    TrainState m_curTrainState{STATE_NOT_RESIZED};
    uint32_t m_resizeStartStep{0};

    uint32_t m_maxTexWH{20};

    mat4x4* m_viewMatrixEval = nullptr;
    std::vector<std::vector<float>> m_camPosWorldEval;
    std::vector<at::Tensor> m_gtImagesEval;

    mat4x4* m_viewMatrixInfer = nullptr;
    std::vector<std::vector<float>> m_camPosWorldInfer;
    std::vector<at::Tensor> m_gtImagesInfer;

    mat4x4* m_viewMatrixRenderingMode = nullptr;
    std::vector<std::vector<float>> m_camPosWorldRenderingMode;

    thrust::device_vector<TriangleInfoDevice> m_triangleArr;

    uint32_t m_resizePatience{0};
    uint32_t m_resizeReturnCntMax{0};
    float m_resizeConvergeThreshold{0.0f};
    float m_resizeTriangleEdgeMinLen{0.0f};

    float m_L1lossThreshold{0.0f};
    float m_densityUpdateStepInner{0.0f};
    float m_densityUpdateStep{0.0f};
    float m_varThresholdPSNR{0.0f};
    float m_resizeDepthThreshold{0.0f};
    
    uint32_t m_texResizeInterval{0};

    std::vector<std::vector<float>> m_camPosWorld;
    std::vector<at::Tensor> m_gtImages;
    mat4x4* m_viewMatrix = nullptr;

    // std::vector<std::vector<float>> m_camPosWorldSub;
    // std::vector<at::Tensor> m_gtImagesSub;

    std::vector<float> m_poseConfidenceVec;
    
    uint32_t m_trainPoseCount{0};

    float m_gaussianCoeffEWA{0.0f};
    float m_initialShDensity{0.0f};
    float m_shDensityMax{0.0f};

    // std::vector<uint32_t> m_curPosMapMemLayout;
    // std::vector<uint32_t> m_curValidMapMemLayout;

    std::vector<uint32_t> m_curPosMapMemLayout;
    std::vector<uint32_t> m_curValidMapMemLayout;
    std::vector<uint32_t> m_curShTexMemLayout;

    thrust::device_vector<uint32_t> m_curPosMapMemLayoutDevice;
    thrust::device_vector<uint32_t> m_curValidMapMemLayoutDevice;
    thrust::device_vector<uint32_t> m_curShTexMemLayoutDevice;

    using MyValueType = PixLocation;
    using MyProbe = cuco::legacy::linear_probing<1, TexIdDummyKeyHash>;

    std::unique_ptr<cuco::static_multimap<TexIdDummyKey, MyValueType, cuda::thread_scope_device, cuco::cuda_allocator<char>, MyProbe>> m_texIdToPixLocationMapPtr{nullptr}; 

    // thrust::device_vector<TexPixInfo> m_visibleTexIdsInfo;
    thrust::device_vector<PixLocation> m_visibleTexPixLocCompact;

    thrust::device_vector<cuco::pair<TexIdDummyKey, PixLocation>> m_recieveBufferDevice;
    
    unsigned char m_vertNbrAddrOffsetBit{27};
    unsigned char m_inTensorOffsetBit{22};
    float m_invDistFactorEdge{2.0f};
    float m_invDistFactorCorner{1.0f};

    // float m_bottomYCoef, m_topYCoeff;
    uint32_t m_printPixX{0}, m_printPixY{0}, m_printGradTexId{0}, m_schedulerPatience{1};
    uint32_t m_renderScaleOPGL{0};
    uint32_t m_flipCnt{0};
    bool m_needInfer{false};

    float m_highOrderSHLrMultiplier{1.0};

    // at::Tensor m_depthPunishmentMap;
    float m_near{7.f};
    float m_depthMagnifyCoeff{2.f};

    // unsigned char* m_mappedDepthMap{nullptr};
    // unsigned char* m_dilatedMappedDepthMap{nullptr};
    // unsigned char* m_dilatedDepthMap{nullptr};
    // unsigned char* m_dilateAccessedMap{nullptr};

    std::vector<uint32_t> m_curTexIdVec;
    // std::vector<uint32_t> m_curVisibileTexId;
    // at::Tensor m_curTrainShTex;
    at::Tensor m_curTrainShTexCpct;
    
    at::Tensor m_curTrainAdamExpAvg;
    at::Tensor m_curTrainAdamExpAvgSq;
    // at::Tensor m_curTrainShPosMap;
    // at::Tensor m_curTrainShValidMap;
    at::Tensor m_curTrainTexNbrInfo;
    at::Tensor m_curTrainCornerAreaInfo;
    at::Tensor m_curTrainVertNbrBuf;
    at::Tensor m_curTrainTexInWorldInfo;
    at::Tensor m_curTrainTexWH;
    at::Tensor m_curTrainShValidMapCpct;
    at::Tensor m_curTrainShPosMapCpct;
    at::Tensor m_curTrainBotYCoeff;
    at::Tensor m_curTrainTopYCoeff;

    // at::Tensor m_curInferShTex;
    at::Tensor m_curInferShTexCpct;
    // at::Tensor m_curInferShPosMap;
    // at::Tensor m_curInferShValidMap;
    at::Tensor m_curInferTexNbrInfo;
    at::Tensor m_curInferCornerAreaInfo;
    at::Tensor m_curInferVertNbrBuf;
    at::Tensor m_curInferTexInWorldInfo;
    at::Tensor m_curInferTexWH;
    at::Tensor m_curInferShValidMapCpct;
    at::Tensor m_curInferShPosMapCpct;
    at::Tensor m_curInferBotYCoeff;
    at::Tensor m_curInferTopYCoeff;

    unsigned int* m_devErrCntPtr{nullptr};
    unsigned int* m_devErrCntBackwardPtr{nullptr};
    cuco::static_map_ref<uint32_t,
                             uint64_t,
                             cuda::thread_scope_device,
                             thrust::equal_to<uint32_t>,
                             cuco::linear_probing<1, cuco::default_hash_function<uint32_t>>,
                             my_impl_type::storage_ref_type,
                             cuco::op::find_tag
                             > m_devHashFindRefFind;


    // HashValue* m_shTexHashMap{nullptr};
    std::vector<HashValue> m_shTexHashMap;

    thrust::host_vector<ResizeInfo> m_resizeInfoForCuda;

    float* m_ViewDirAndTexIdBufferPinned{nullptr};
    float* m_texIncludeTexIdBuffer{nullptr};

    float* m_texIncludeTexIdBufferInfer{nullptr};
    float* m_texIncludeDepthBuffer{nullptr};

    // cuco::static_map<unsigned int, unsigned int> m_curTrainHashMap;
    // cuco::static_map<uint32_t, uint64_t> m_curTrainHashMap;
    // cuco::static_map<uint32_t, uint64_t> m_curInferHashMap;

    GLuint m_FBO;
    GLuint m_texFBO, m_depthBufFBO, m_LerpCoeffAndTexCoordFBO, m_ViewDirFrag2CamAndNullFBO, m_oriShLUandRUFBO, m_oriShLDandRDFBO, m_texScaleFacDepthTexId, m_texEdgeLR, m_texWorldPoseAndNull;
    GLuint m_texInvDistCoeffs;
    GLuint m_ellipCoeffsAndLodLvl;
    GLuint m_dFdxy;
    at::Tensor m_renderedPixels;
    at::Tensor m_renderedPixelsCUDA;
    at::Tensor m_renderedPixelsMask;
    at::Tensor m_shTensorAllMerge;
    at::Tensor m_shTensorAllMergeCUDA;

    // GLuint m_dummyTexture;

public:
    void InversePiexls();
    int gpuGetMaxGflopsDeviceId();

    void* m_handle{nullptr};
    void* (*eglGetCurrentContext)(void);
    unsigned int (*eglInitialize)(void*, int32_t*, int32_t*);
    unsigned int (*eglChooseConfig)(void*, const int32_t*, void**, int32_t, int32_t*);
    void (*(*eglGetProcAddress)(const char*))();
    void* (*eglCreatePbufferSurface)(void*, void*, const int32_t*);
    unsigned int (*eglBindAPI)(unsigned int);
    void* (*eglCreateContext)(void*, void*, void*, const int32_t*);
    unsigned int (*eglMakeCurrent)(void*, void*, void*, void*);
    unsigned int (*eglTerminate)(void*);

    EGLDisplay m_eglMainThrdDsp;
    EGLDisplay m_eglRenderThrdDsp;
    EGLDisplay m_eglTrainPrepThrdDsp;
    EGLDisplay m_eglTrainThrdDsp;

    EGLContext m_mainThrdCtx{nullptr};
    EGLContext m_renderThreadCtx{nullptr};
    EGLContext m_trainPrepThreadCtx{nullptr};
    EGLContext m_trainThreadCtx{nullptr};



    GLuint m_width, m_height, m_channelNum{4}, m_shTexChannelNum{3};
    // int m_offsetPerTexture{0}, m_offsetPerLayer{0}, m_offsetPerLine{0};
    GLuint m_VBO{0}, m_VAO{0}, m_EBO{0};

    GLuint m_vertexShader{0}, m_fragmentShader{0}, m_program{0};

    GLuint m_vertexShaderInfer{0}, m_fragmentShaderInfer{0}, m_programInfer{0};

    mat4x4 m_perspectiveMat, m_viewMat, m_modelMat;
    GLint m_modelLocation, m_viewLocation, m_projLocation;
    GLint m_modelLocationInfer, m_viewLocationInfer, m_projLocationInfer;
    // GLint m_drawLineBoolLocation;
    // GLint m_drawLineBoolLocationInfer;

    GLint m_camWorldPosLocation;
    GLint m_camWorldPosLocationInfer;
    
    // include RGB. for example, 2 order SH is 3 * 9 = 27, 1 order 3 * 4 = 12
    GLint m_shLayerNumLocationInfer;

    // GLint m_texIdxLocation;
    // GLint m_texIdxLocationInfer;

    // GLint m_texWLocation;
    // GLint m_texWLocationInfer;
    
    // GLint m_texHLocation;
    // GLint m_texHLocationInfer;

    GLint m_texScaleFacWLocation;
    GLint m_texScaleFacWLocationInfer;

    GLint m_texScaleFacHLocation;
    GLint m_texScaleFacHLocationInfer;

    GLint m_texEdgeLrLocation;
    GLint m_texEdgeBotLocation;
    GLint m_texInvDistFactorLocation;
    GLint m_texInvDistFactorLocationInfer;

    std::vector<TexScaleFactor> m_scaleFactor;

    // camera model
    float m_fxRender{0.f};
    float m_fyRender{0.f};
    float m_cxRender{0.f};
    float m_cyRender{0.f};

    float m_fxGt{0.f};
    float m_fyGt{0.f};
    float m_cxGt{0.f};
    float m_cyGt{0.f};

    // SH texture config
    unsigned int m_texNum{0};
    GLuint* m_texSH{nullptr};
    unsigned int m_shOrder{0}, m_shLayerNum{0};
    // unsigned int m_texW{0}, m_texH{0};

    const float m_SH0 = 0.28209479177387814;
    const float m_SH1 = 0.4886025119029199;
    const float m_SH2[5] = {
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    };

    const float m_SH3[7] = {
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    };

    /* arrangement in GPU texture: 0.0, 0.5, 0.0,
                                   0.5, 0.5, 0.5,
                                   0.5, 0.5, 0.5
    from bottom to up row, from left to right element
    */

    GLfloat *m_shCoeffArrTest{nullptr};
    GLfloat *m_shCoeffArrTestNull{nullptr};
    unsigned int m_shCoeffArrTestLen;    

    static unsigned int savedImgCnt;


    // ---------- cuda --------------
    cudaGraphicsResource_t m_cuResRenderPixel;
    cudaGraphicsResource_t* m_cuResShTextures;
    // std::vector<cudaGraphicsResource_t>  m_cuResShTexturesVec;

    void* m_ptrCuRenderPixel{nullptr};
    void* m_ptrCuRenderTexId{nullptr};
    void* m_ptrCuLerpCoeffAndTexCoord{nullptr};
    void* m_ptrCuViewDirFrag2Cam{nullptr};
    void* m_ptrCuShTextures[12]{nullptr};


    // cudaArray_t m_cuArrRenderPixel;
    // cudaArray_t m_cuArrRenderTexId;
    // cudaArray_t m_cuArrLerpCoeffAndTexCoord;
    // cudaArray_t m_cuArrViewDirFrag2Cam;
    // cudaArray_t m_cuArrShTextures[12];
    
    cudaSurfaceObject_t m_surfaceObj;
    void* m_ptrCudaImtermBuf{nullptr};
    std::vector<std::vector<void *>> m_shTexArrLayers;

    ImMeshRenderer(const ImMeshRenderer&);
	ImMeshRenderer& operator=(const ImMeshRenderer&);
    static ImMeshRenderer& GetInstance()
    {
        static ImMeshRenderer instance;
        return instance;
    }


    // cudaGraphicsResource_t g_cuResLerpCoeffAndTexCoord;
    // cudaGraphicsResource_t g_cuResViewDirFrag2CamAndNull;
    // cudaGraphicsResource_t g_cuResOriShLUandRUTexCoord;
    // cudaGraphicsResource_t g_cuResOriShLDandRDTexCoord;
    // cudaGraphicsResource_t g_cuResTexScaleFactorDepthTexId;
    // cudaGraphicsResource_t g_cuResTexEdgeLR;
    // cudaGraphicsResource_t g_cuResWorldPoseAndNull;
    // cudaGraphicsResource_t g_cuResTexInvDistCoeffs;
    // cudaGraphicsResource_t g_cuResEllipCoeffsAndLodLvl;

    // // cudaArray_t g_cuArrayTexId;
    // cudaArray_t g_cuArrayLerpCoeffAndTexCoord;
    // cudaArray_t g_cuArrayViewDirFrag2CamAndNull;
    // cudaArray_t g_cuArrayOriShLUandRUTexCoord;
    // cudaArray_t g_cuArrayOriShLDandRDTexCoord;
    // cudaArray_t g_cuArrayTexScaleFactorDepthTexId;
    // cudaArray_t g_cuArrayTexEdgeLR;
    // cudaArray_t g_cuArrayWorldPoseAndNull;
    // cudaArray_t g_cuArrayTexInvDistCoeffs;
    // cudaArray_t g_cuArrayEllipCoeffsAndLodLvl;

    // // cudaResourceDesc g_resDescTexId;
    // cudaResourceDesc g_resDescLerpCoeffAndTexCoord;
    // cudaResourceDesc g_resDescViewDirFrag2CamAndNull;
    // cudaResourceDesc g_resDescOriShLUandRUTexCoord;
    // cudaResourceDesc g_resDescOriShLDandRDTexCoord;
    // cudaResourceDesc g_resDescTexScaleFactorDepthTexId;
    // cudaResourceDesc g_resDescTexEdgeLR;
    // cudaResourceDesc g_resDescWorldPoseAndNull;
    // cudaResourceDesc g_resDescTexInvDistCoeffs;
    // cudaResourceDesc g_resDescEllipCoeffsAndLodLvl;

    // // cudaTextureDesc g_texDescTexId;
    // cudaTextureDesc g_texDescLerpCoeffAndTexCoord;
    // cudaTextureDesc g_texDescViewDirFrag2CamAndNull;
    // cudaTextureDesc g_texDescOriShLUandRUTexCoord;
    // cudaTextureDesc g_texDescOriShLDandRDTexCoord;
    // cudaTextureDesc g_texDescTexScaleFactorDepthTexId;
    // cudaTextureDesc g_texDescTexEdgeLR;
    // cudaTextureDesc g_texDescWorldPoseAndNull;
    // cudaTextureDesc g_texDescTexInvDistCoeffs;
    // cudaTextureDesc g_texDescEllipCoeffsAndLodLvl;

    // // cudaTextureObject_t g_texObjTexId;
    // cudaTextureObject_t g_texObjLerpCoeffAndTexCoord;
    // cudaTextureObject_t g_texObjViewDirFrag2CamAndNull;
    // cudaTextureObject_t g_texObjOriShLUandRUTexCoord;
    // cudaTextureObject_t g_texObjOriShLDandRDTexCoord;
    // cudaTextureObject_t g_texObjTexScaleFactorDepthTexId;
    // cudaTextureObject_t g_texObjTexEdgeLR;
    // cudaTextureObject_t g_texObjWorldPoseAndNull;
    // cudaTextureObject_t g_texObjTexInvDistCoeffs;
    // cudaTextureObject_t g_texObjEllipCoeffsAndLodLvl;
};  


}

#endif
