#ifndef __BACKWARD_HASH_ACC_CUH__
#define __BACKWARD_HASH_ACC_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_texture_types.h>
#include <cuda_fp16.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include "../include/my_common.hpp"

namespace cg = cooperative_groups;

__device__ void calcDerivatives(glm::vec3 dL_dcolor, glm::vec3 viewDir, glm::vec3* dColor_dshs, unsigned char shOrders, float highOrderSHLrMultiplier)
{
    // float SH_C0 = 0.28209479177387814f;
    // float SH_C1 = 0.4886025119029199f;
    // float SH_C2[] = {
    //     1.0925484305920792f,
    //     -1.0925484305920792f,
    //     0.31539156525252005f,
    //     -1.0925484305920792f,
    //     0.5462742152960396f
    // };

    float xx = viewDir.x * viewDir.x;
    float yy = viewDir.y * viewDir.y;
    float zz = viewDir.z * viewDir.z;
    float xy = viewDir.x * viewDir.y;
    float yz = viewDir.y * viewDir.z;
    float xz = viewDir.x * viewDir.z;

    dColor_dshs[0] = dL_dcolor * SH_C0;

    switch (shOrders) {
        case 2:
            dColor_dshs[4] = dL_dcolor * SH_C2[0] * xy * highOrderSHLrMultiplier;
            dColor_dshs[5] = dL_dcolor * SH_C2[1] * yz * highOrderSHLrMultiplier;
            dColor_dshs[6] = dL_dcolor * SH_C2[2] * (2.f * zz - xx - yy) * highOrderSHLrMultiplier;
            dColor_dshs[7] = dL_dcolor * SH_C2[3] * xz * highOrderSHLrMultiplier;
            dColor_dshs[8] = dL_dcolor * SH_C2[4] * (xx - yy) * highOrderSHLrMultiplier;


        case 1:
            dColor_dshs[1] = dL_dcolor * -SH_C1 * viewDir.y * highOrderSHLrMultiplier;
            dColor_dshs[2] = dL_dcolor * SH_C1 * viewDir.z * highOrderSHLrMultiplier;
            dColor_dshs[3] = dL_dcolor * -SH_C1 * viewDir.x * highOrderSHLrMultiplier;
        default:
            break;
    }
}

__device__ void UpdateGradShared(uint32_t texOffsets, glm::vec3 grad, float* dL_dshs, uint32_t* shTexturesMemLayout, unsigned char layer, unsigned int offsetPerSh, int32_t innieOffset)
{
    // atomicAdd((dL_dshs + (*(shTexturesMemLayout + texOffsets)) + offsetPerLayer * layer + offsetPerLine * offsetH + offsetW * SH_TEXTURE_CHANNEL_NUM + 0), grad.x);

    // atomicAdd((dL_dshs + (*(shTexturesMemLayout + texOffsets)) + offsetPerLayer * layer + offsetPerLine * offsetH + offsetW * SH_TEXTURE_CHANNEL_NUM + 1), grad.y);

    // atomicAdd((dL_dshs + (*(shTexturesMemLayout + texOffsets)) + offsetPerLayer * layer + offsetPerLine * offsetH + offsetW * SH_TEXTURE_CHANNEL_NUM + 2), grad.z);

    atomicAdd((dL_dshs + (*(shTexturesMemLayout + texOffsets)) + innieOffset * offsetPerSh + layer * SH_TEXTURE_CHANNEL_NUM + 0), grad.x);

    atomicAdd((dL_dshs + (*(shTexturesMemLayout + texOffsets)) + innieOffset * offsetPerSh + layer * SH_TEXTURE_CHANNEL_NUM + 1), grad.y);

    atomicAdd((dL_dshs + (*(shTexturesMemLayout + texOffsets)) + innieOffset * offsetPerSh + layer * SH_TEXTURE_CHANNEL_NUM + 2), grad.z);
}

__device__ void TraverseEachShGradShared(uint32_t* texOffsets, float* dL_dsh, uint32_t* shTexturesMemLayout, CurTexAlignedWH* curTexAllWH, const glm::vec3 dL_dinterpSh, uint32_t* texIdOfShs, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint8_t shNumber, uint8_t layer, TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll, uint32_t shTextureLayerNumAll)
{
    float coeffSum = 0.f;

    for (uint8_t i = 0; i < shNumber; i++) {
        if (isnan((*(lerpCoeffsOfShs + i))) == 1 ||
            isinf((*(lerpCoeffsOfShs + i))) == 1) {
            continue;
        }

        coeffSum += (*(lerpCoeffsOfShs + i));
    }

    for (uint8_t i = 0; i < shNumber; i++) {
        if (isnan((*(lerpCoeffsOfShs + i))) == 1 ||
            isinf((*(lerpCoeffsOfShs + i))) == 1) {
            continue;
        }

        glm::vec3 dL_dcurSh = dL_dinterpSh * (*(lerpCoeffsOfShs + i)) / coeffSum;
        // unsigned int offsetH = (unsigned int)floorf(texH * (*(offsetOfH + i)));
        // unsigned int offsetW = (unsigned int)floorf(texW * (*(offsetOfW + i)));
        if (isnan(dL_dcurSh.x) || isnan(dL_dcurSh.y) || isnan(dL_dcurSh.z)) {
            printf("dL_dinterpSh : %f, %f, %f\n", dL_dinterpSh.x, dL_dinterpSh.y, dL_dinterpSh.z);
            printf("coeffSum : %f, lerpCoeffsOfShs:\n", coeffSum);
            for (uint8_t ii = 0; ii < shNumber; ii++) {
                uint32_t readW, readH;
                ReadFromCompactWH(offsetOfWH, ii, readW, readH);
                printf("val : %f , texId : %d, (W, H) : %d, %d\n", (*(lerpCoeffsOfShs + ii)), (*(texIdOfShs + ii)), readW, readH);
            }
            printf("cur X : %d, Y : %d\n", (blockIdx.x * blockDim.x + threadIdx.x), (blockIdx.y * blockDim.y + threadIdx.y));
            printf("\n");
        }

        assert(isnan(dL_dcurSh.x) == 0);
        assert(isnan(dL_dcurSh.y) == 0);
        assert(isnan(dL_dcurSh.z) == 0);

        uint32_t readW, readH;
        ReadFromCompactWH(offsetOfWH, i, readW, readH);
        TexAlignedPosWH* curValidMapStartAddr = (validShWHMap + (*(validShWHMapMemLayout + texOffsets[i])));
        uint32_t curValidShNum = (*(validShNumsAll + texOffsets[i]));
        int32_t innieOffset = FindOffsetInValidShMap(curValidMapStartAddr, curValidShNum, readW, readH);

        uint32_t offsetPerSh = SH_TEXTURE_CHANNEL_NUM * shTextureLayerNumAll;
        // uint32_t offsetPerLine = (uint32_t)((curTexAllWH + texOffsets[i])->data.curTexW) * SH_TEXTURE_CHANNEL_NUM;
        // uint32_t offsetPerLayer = (uint32_t)((curTexAllWH + texOffsets[i])->data.curTexH) * offsetPerLine;

        // UpdateGradShared(*(texOffsets + i), dL_dcurSh, dL_dsh, shTexturesMemLayout, (*(texIdOfShs + i)), offsetPerLine, offsetPerLayer, layer, readH, readW);
        UpdateGradShared(*(texOffsets + i), dL_dcurSh, dL_dsh, shTexturesMemLayout, layer, offsetPerSh, innieOffset);


        // if ((needPrintX == (blockIdx.x * blockDim.x + threadIdx.x)) &&
        //     (needPrintY == (blockIdx.y * blockDim.y + threadIdx.y))) {
        //     printf("update grad * 100000 : %f, %f, %f\n", dL_dcurSh.x * 100000, dL_dcurSh.y * 100000, dL_dcurSh.z * 100000);
        //     printf("update texid : %u,  W H : %d, %d\n", (*(texIdOfShs + i)), (*(offsetOfW + i)), (*(offsetOfH + i)));
        // }
    }
}

__device__ void GradientsFillShared(uint32_t* texOffsets, float* dL_dsh, uint32_t* shTexturesMemLayout, CurTexAlignedWH* curTexAllWH, glm::vec3 viewDir, glm::vec3 dL_dcolor, unsigned char shOrders, float highOrderSHLrMultiplier, uint8_t shLayerNum, uint32_t* texIdOfShs, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint8_t shNumber, TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll)
{
    glm::vec3 dColor_dshs[9];
    calcDerivatives(dL_dcolor, viewDir, dColor_dshs, shOrders, highOrderSHLrMultiplier);

    for (unsigned char layer = 0; layer < shLayerNum; layer++) {
        TraverseEachShGradShared(texOffsets, dL_dsh, shTexturesMemLayout, curTexAllWH, dColor_dshs[layer], texIdOfShs, offsetOfWH, lerpCoeffsOfShs, shNumber, layer, validShWHMap, validShWHMapMemLayout, validShNumsAll, shLayerNum);
    }
}

// __device__ void CalcGrad(float* dL_dcolor, float* dL_dsh, uint32_t* shTexturesMemLayout,
//     MeshEdgeNbrInfo* meshEdgeNbrInfo, MeshVertNbrInfo* meshVertNbrInfo,
//     uint32_t selfTexId, uint32_t selfOffsetInHash,
//     uint32_t pixX, uint32_t pixY,
//     uint32_t imgW, uint32_t imgH,
//     uint32_t selfTexW, uint32_t selfTexH,
//     cudaTextureObject_t texViewDirFrag2CamAndNull,
//     cudaTextureObject_t texLerpCoeffAndTexCoord,
//     cudaTextureObject_t texCoordShLUandRU,
//     cudaTextureObject_t texCoordShLDandRD,
//     cudaTextureObject_t texObjWorldPoseAndNull,
//     cudaTextureObject_t texEllipseAndLodLvl,
//     cudaTextureObject_t texTexScaleFactorDepthTexId,
//     float2 cornerAreaInfoShared,
//     uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
//     float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
//     float* botYCoefCurBuf,
//     float* topYCoefCurBuf,
//     float invDistFactorEdge, float invDistFactorCorner,
//     unsigned char shLayerNum, unsigned char shOrder,
//     float highOrderSHLrMultiplier, float gaussianCoeffEWA, float* texInWorldInfo,
//     CurTexAlignedWH* curTexAllWH, float curTexDensity,
//     uint32_t needPrintX, uint32_t needPrintY)
// {
//     // unsigned int offsetPerPixelLine = imgW * OFFSET_PER_PIX_CHANNEL;
//     unsigned int offsetPerPixelLine = imgW;
//     unsigned int offsetPerPixelChannel = imgW * imgH;

//     float4 viewDirAndNull = tex2D<float4>(texViewDirFrag2CamAndNull, (float)pixX, (float)(imgH - pixY - 1));

//     float4 lerpAndTexCoord = tex2D<float4>(texLerpCoeffAndTexCoord, (float)pixX, (float)(imgH - pixY - 1));
//     float2 oriLerpCoeff = make_float2(lerpAndTexCoord.x, lerpAndTexCoord.y);
//     float2 oriTexCoord = make_float2(lerpAndTexCoord.z, lerpAndTexCoord.w);

//     float4 oriShLUandRU = tex2D<float4>(texCoordShLUandRU, (float)pixX, (float)(imgH - pixY - 1));
//     float2 oriLU = make_float2(oriShLUandRU.x, oriShLUandRU.y);
//     float2 oriRU = make_float2(oriShLUandRU.z, oriShLUandRU.w);
//     float4 oriShLDandRD = tex2D<float4>(texCoordShLDandRD, (float)pixX, (float)(imgH - pixY - 1));
//     float2 oriLD = make_float2(oriShLDandRD.x, oriShLDandRD.y);
//     float2 oriRD = make_float2(oriShLDandRD.z, oriShLDandRD.w);

//     float4 worldPoseAndNull = tex2D<float4>(texObjWorldPoseAndNull, (float)pixX, (float)(imgH - pixY - 1));
//     glm::vec3 worldPosition(worldPoseAndNull.x, worldPoseAndNull.y, worldPoseAndNull.z);

//     float4 ellipseAndLodLvl = tex2D<float4>(texEllipseAndLodLvl, (float)pixX, (float)(imgH - pixY - 1));

//     // float4 texScaleFactorDepthTexId = tex2D<float4>(texTexScaleFactorDepthTexId, (float)pixX, (float)(imgH - pixY - 1));
//     // float2 texScaleFactor = make_float2(texScaleFactorDepthTexId.x, texScaleFactorDepthTexId.y);

//     uint32_t texIdOfShs[MAX_SH_NUMBER];
//     uint32_t texOffsets[MAX_SH_NUMBER];
//     // uint8_t offsetOfW[MAX_SH_NUMBER];
//     // uint8_t offsetOfH[MAX_SH_NUMBER];
//     uint32_t offsetOfWH[MAX_SH_NUMBER]; // High16: W, Low16: H
//     float lerpCoeffsOfShs[MAX_SH_NUMBER];
//     uint8_t shNumber = 0;    

//     for (uint8_t i = 0; i < MAX_SH_NUMBER; i++) {
//         texIdOfShs[i] = 0;
//         offsetOfWH[i] = 0;
//         lerpCoeffsOfShs[i] = 0.f;
//     }

//     if (((uint32_t)(ellipseAndLodLvl.w)) >= 1) {
//         CalcEWA(meshVertNbrInfo, meshEdgeNbrInfo, selfTexId, selfOffsetInHash, ellipseAndLodLvl, shNumber, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, texOffsets, (uint8_t)selfTexW, (uint8_t)selfTexH, oriTexCoord, gaussianCoeffEWA, texInWorldInfo, worldPosition, shPositionMapCpct, shPosMapMemLayout, cornerAreaInfoShared, botYCoefCurBuf, topYCoefCurBuf, oriLU, oriRU, oriLD, oriRD, shValidMapCpct, shValidMapMemLayout, curTexAllWH, false, curTexDensity);
//     } else {
//         CalcLerpCoeffsShared(meshEdgeNbrInfo, meshVertNbrInfo, selfTexId, selfOffsetInHash,
//             oriLU, oriRU, oriLD, oriRD,
//             cornerAreaInfoShared,
//             shValidMapCpct, shValidMapMemLayout,
//             shPositionMapCpct, shPosMapMemLayout,
//             worldPosition,
//             oriTexCoord, oriLerpCoeff,
//             botYCoefCurBuf, topYCoefCurBuf,
//             selfTexW, selfTexH,
//             shNumber, texIdOfShs,
//             offsetOfWH, lerpCoeffsOfShs, texOffsets,
//             invDistFactorEdge, invDistFactorCorner, curTexAllWH,
//             false);
//     }


//     // glm::vec3 dL_dcolorTmp(*(dL_dcolor + (imgH - pixY - 1) * offsetPerPixelLine + pixX * OFFSET_PER_PIX_CHANNEL),
//     //                        *(dL_dcolor + (imgH - pixY - 1) * offsetPerPixelLine + pixX * OFFSET_PER_PIX_CHANNEL + 1),
//     //                        *(dL_dcolor + (imgH - pixY - 1) * offsetPerPixelLine + pixX * OFFSET_PER_PIX_CHANNEL + 2));

//     glm::vec3 dL_dcolorTmp(*(dL_dcolor + 0 * offsetPerPixelChannel + (imgH - pixY - 1) * offsetPerPixelLine + pixX),
//                            *(dL_dcolor + 1 * offsetPerPixelChannel + (imgH - pixY - 1) * offsetPerPixelLine + pixX),
//                            *(dL_dcolor + 2 * offsetPerPixelChannel + (imgH - pixY - 1) * offsetPerPixelLine + pixX));

//     glm::vec3 viewDir(viewDirAndNull.x, viewDirAndNull.y, viewDirAndNull.z);

//     GradientsFillShared(texOffsets, dL_dsh, shTexturesMemLayout, curTexAllWH, viewDir, dL_dcolorTmp, shOrder, highOrderSHLrMultiplier, shLayerNum, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, shNumber);
// }


__device__ void CalcGradTexPtr(float* dL_dcolor, float* dL_dsh, uint32_t* shTexturesMemLayout,
    MeshEdgeNbrInfo* meshEdgeNbrInfo, MeshVertNbrInfo* meshVertNbrInfo,
    uint32_t selfTexId, uint32_t selfOffsetInHash,
    uint32_t pixX, uint32_t pixY,
    uint32_t imgW, uint32_t imgH,
    uint32_t selfTexW, uint32_t selfTexH,
    float* texViewDirFrag2CamAndNull,
    float* texLerpCoeffAndTexCoord,
    float* texCoordShLUandRU,
    float* texCoordShLDandRD,
    float* texObjWorldPoseAndNull,
    float* texEllipseAndLodLvl,
    float* texTexScaleFactorDepthTexId,
    float2 cornerAreaInfoShared,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
    float* botYCoefCurBuf,
    float* topYCoefCurBuf,
    float invDistFactorEdge, float invDistFactorCorner,
    unsigned char shLayerNum, unsigned char shOrder,
    float highOrderSHLrMultiplier, float gaussianCoeffEWA, float* texInWorldInfo,
    CurTexAlignedWH* curTexAllWH, float curTexDensity, float* curTexNormal, float* curTexNormals,
    uint32_t needPrintX, uint32_t needPrintY)
{
    // unsigned int offsetPerPixelLine = imgW * OFFSET_PER_PIX_CHANNEL;
    unsigned int offsetPerPixelLine = imgW;
    unsigned int offsetPerPixelChannel = imgW * imgH;

    uint32_t offsetPerTexLine = imgW * 4;

    // float4 viewDirAndNull = tex2D<float4>(texViewDirFrag2CamAndNull, (float)pixX, (float)(imgH - pixY - 1));
    float4 viewDirAndNull = ReadTexPtr(texViewDirFrag2CamAndNull, pixX, (imgH - pixY - 1), offsetPerTexLine);

    // float4 lerpAndTexCoord = tex2D<float4>(texLerpCoeffAndTexCoord, (float)pixX, (float)(imgH - pixY - 1));
    float4 lerpAndTexCoord = ReadTexPtr(texLerpCoeffAndTexCoord, pixX, (imgH - pixY - 1), offsetPerTexLine);
    float2 oriLerpCoeff = make_float2(lerpAndTexCoord.x, lerpAndTexCoord.y);
    float2 oriTexCoord = make_float2(lerpAndTexCoord.z, lerpAndTexCoord.w);

    // float4 oriShLUandRU = tex2D<float4>(texCoordShLUandRU, (float)pixX, (float)(imgH - pixY - 1));
    float4 oriShLUandRU = ReadTexPtr(texCoordShLUandRU, pixX, (imgH - pixY - 1), offsetPerTexLine);
    float2 oriLU = make_float2(oriShLUandRU.x, oriShLUandRU.y);
    float2 oriRU = make_float2(oriShLUandRU.z, oriShLUandRU.w);

    // float4 oriShLDandRD = tex2D<float4>(texCoordShLDandRD, (float)pixX, (float)(imgH - pixY - 1));
    float4 oriShLDandRD = ReadTexPtr(texCoordShLDandRD, pixX, (imgH - pixY - 1), offsetPerTexLine);
    float2 oriLD = make_float2(oriShLDandRD.x, oriShLDandRD.y);
    float2 oriRD = make_float2(oriShLDandRD.z, oriShLDandRD.w);

    // float4 worldPoseAndNull = tex2D<float4>(texObjWorldPoseAndNull, (float)pixX, (float)(imgH - pixY - 1));
    float4 worldPoseAndNull = ReadTexPtr(texObjWorldPoseAndNull, pixX, (imgH - pixY - 1), offsetPerTexLine);
    glm::vec3 worldPosition(worldPoseAndNull.x, worldPoseAndNull.y, worldPoseAndNull.z);

    // float4 ellipseAndLodLvl = tex2D<float4>(texEllipseAndLodLvl, (float)pixX, (float)(imgH - pixY - 1));
    float4 ellipseAndLodLvl = ReadTexPtr(texEllipseAndLodLvl, pixX, (imgH - pixY - 1), offsetPerTexLine);

    // float4 texScaleFactorDepthTexId = tex2D<float4>(texTexScaleFactorDepthTexId, (float)pixX, (float)(imgH - pixY - 1));
    // float2 texScaleFactor = make_float2(texScaleFactorDepthTexId.x, texScaleFactorDepthTexId.y);

    float curDepth = (ReadTexPtr(texTexScaleFactorDepthTexId, pixX, (imgH - pixY - 1), offsetPerTexLine)).z;

    uint32_t texIdOfShs[MAX_SH_NUMBER];
    uint32_t texOffsets[MAX_SH_NUMBER];
    // uint8_t offsetOfW[MAX_SH_NUMBER];
    // uint8_t offsetOfH[MAX_SH_NUMBER];
    uint32_t offsetOfWH[MAX_SH_NUMBER]; // High16: W, Low16: H
    float lerpCoeffsOfShs[MAX_SH_NUMBER];
    uint8_t shNumber = 0;    

    for (uint8_t i = 0; i < MAX_SH_NUMBER; i++) {
        texIdOfShs[i] = 0;
        offsetOfWH[i] = 0;
        lerpCoeffsOfShs[i] = 0.f;
    }

    if ((((uint32_t)(ellipseAndLodLvl.w)) >= 1) && ((curDepth - 10.f) > 1e-5f)) {
        CalcEWA(meshVertNbrInfo, meshEdgeNbrInfo, selfTexId, selfOffsetInHash, ellipseAndLodLvl, shNumber, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, texOffsets, (uint8_t)selfTexW, (uint8_t)selfTexH, oriTexCoord, gaussianCoeffEWA, texInWorldInfo, worldPosition, shPositionMapCpct, shPosMapMemLayout, cornerAreaInfoShared, botYCoefCurBuf, topYCoefCurBuf, oriLU, oriRU, oriLD, oriRD, validShWHMap, validShWHMapMemLayout, validShNumsAll, curTexAllWH, false, curTexDensity, curTexNormal, curTexNormals);
    } else
    {
        CalcLerpCoeffsShared(meshEdgeNbrInfo, meshVertNbrInfo, selfTexId, selfOffsetInHash,
            oriLU, oriRU, oriLD, oriRD,
            cornerAreaInfoShared,
            // shValidMapCpct, shValidMapMemLayout,
            validShWHMap, validShWHMapMemLayout, validShNumsAll,
            shPositionMapCpct, shPosMapMemLayout,
            worldPosition,
            oriTexCoord, oriLerpCoeff,
            botYCoefCurBuf, topYCoefCurBuf,
            selfTexW, selfTexH,
            shNumber, texIdOfShs,
            offsetOfWH, lerpCoeffsOfShs, texOffsets,
            invDistFactorEdge, invDistFactorCorner, curTexAllWH, curTexNormal, curTexNormals,
            false);
    }


    // glm::vec3 dL_dcolorTmp(*(dL_dcolor + (imgH - pixY - 1) * offsetPerPixelLine + pixX * OFFSET_PER_PIX_CHANNEL),
    //                        *(dL_dcolor + (imgH - pixY - 1) * offsetPerPixelLine + pixX * OFFSET_PER_PIX_CHANNEL + 1),
    //                        *(dL_dcolor + (imgH - pixY - 1) * offsetPerPixelLine + pixX * OFFSET_PER_PIX_CHANNEL + 2));

    glm::vec3 dL_dcolorTmp(*(dL_dcolor + 0 * offsetPerPixelChannel + (imgH - pixY - 1) * offsetPerPixelLine + pixX),
                           *(dL_dcolor + 1 * offsetPerPixelChannel + (imgH - pixY - 1) * offsetPerPixelLine + pixX),
                           *(dL_dcolor + 2 * offsetPerPixelChannel + (imgH - pixY - 1) * offsetPerPixelLine + pixX));

    glm::vec3 viewDir(viewDirAndNull.x, viewDirAndNull.y, viewDirAndNull.z);

    GradientsFillShared(texOffsets, dL_dsh, shTexturesMemLayout, curTexAllWH, viewDir, dL_dcolorTmp, shOrder, highOrderSHLrMultiplier, shLayerNum, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, shNumber, validShWHMap, validShWHMapMemLayout, validShNumsAll);
}

// template <typename Map>
// __global__ void MeshLearnerbackwardCUDA(uint32_t texNum,
//     float* dL_dcolor, float* dL_dsh, uint32_t* shTexturesMemLayout,
//     Map texHash, unsigned int* errCnt,
//     TexPixInfo* visibleTexIdInfo, PixLocation* compactPixLocBuf,
//     uint32_t imgW, uint32_t imgH,
//     cudaTextureObject_t texViewDirFrag2CamAndNull,
//     cudaTextureObject_t texLerpCoeffAndTexCoord,
//     cudaTextureObject_t texCoordShLUandRU,
//     cudaTextureObject_t texCoordShLDandRD,
//     cudaTextureObject_t texObjWorldPoseAndNull,
//     cudaTextureObject_t texEllipseAndLodLvl,
//     cudaTextureObject_t texTexScaleFactorDepthTexId,
//     float* cornerAreaInfo,
//     uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
//     float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
//     float* botYCoefCurBuf,
//     float* topYCoefCurBuf,
//     float invDistFactorEdge, float invDistFactorCorner,
//     unsigned char shLayerNum, unsigned char shOrder,
//     uint32_t* edgeNbrInfo, uint32_t* vertNbrInfo,
//     float highOrderSHLrMultiplier, float gaussianCoeffEWA, float* texInWorldInfo, CurTexAlignedWH* curTexAllWH, float* curTexDensities,
//     uint32_t needPrintX, uint32_t needPrintY)
// {
//     auto blockId = blockIdx.x;
//     auto grp = cg::tiled_partition<MESH_HANDLE_THREAD_NUM>(cg::this_thread_block());

//     __shared__ PixLocation pixLocations[MESH_HANDLE_THREAD_NUM];
//     __shared__ MeshEdgeNbrInfo meshEdgeNbrInfo;
//     __shared__ MeshVertNbrInfo meshVertNbrInfo;
//     __shared__ float2 cornerAreaInfoShared;

//     uint32_t numOfPixToHandle = 0;
//     // uint32_t totalPixNum = 0;
//     uint32_t readPtrPos = 0;
//     bool isValidThd = false;
//     uint32_t curTexId = 0;
//     uint32_t curTexOffset = 0;
//     uint32_t curTexW = 0;
//     uint32_t curTexH = 0;
//     float curTexDensity = 0.0f;

//     TexPixInfo tmp;

//     if (blockId < texNum) {
//         isValidThd = true;
//     }

//     grp.sync();

//     if (grp.thread_rank() == 0 && isValidThd) {
//         tmp = (*(visibleTexIdInfo + blockId));
//         curTexId = tmp.texId;

//         ReadMeshData(texHash, errCnt, blockId, edgeNbrInfo, vertNbrInfo, curTexId, meshEdgeNbrInfo, meshVertNbrInfo, curTexOffset, cornerAreaInfo, cornerAreaInfoShared, curTexAllWH, curTexW, curTexH, curTexDensities, curTexDensity);
//         readPtrPos = ReadPixLocation(compactPixLocBuf, pixLocations, numOfPixToHandle, tmp.offset, tmp.pixNum, readPtrPos);

//         // totalPixNum = tmp.pixNum;
//     }

//     grp.sync();

//     // totalPixNum = grp.shfl(totalPixNum, 0);
//     numOfPixToHandle = grp.shfl(numOfPixToHandle, 0);
//     // readPtrPos = grp.shfl(readPtrPos, 0);
//     curTexId = grp.shfl(curTexId, 0);
//     curTexOffset = grp.shfl(curTexOffset, 0);
//     curTexW = grp.shfl(curTexW, 0);
//     curTexH = grp.shfl(curTexH, 0);
//     curTexDensity = grp.shfl(curTexDensity, 0);

//     while (numOfPixToHandle > 0) {
//         uint32_t idxInGrp = grp.thread_rank();

//         while (grp.any(idxInGrp < numOfPixToHandle)) {
//             if ((idxInGrp < numOfPixToHandle) && isValidThd) {
//                 PixLocation curLoc = pixLocations[idxInGrp];

//                 CalcGrad(dL_dcolor, dL_dsh, shTexturesMemLayout,
//                     &meshEdgeNbrInfo, &meshVertNbrInfo,
//                     curTexId, curTexOffset,
//                     curLoc.x, curLoc.y,
//                     imgW, imgH,
//                     curTexW, curTexH,
//                     texViewDirFrag2CamAndNull,
//                     texLerpCoeffAndTexCoord,
//                     texCoordShLUandRU,
//                     texCoordShLDandRD,
//                     texObjWorldPoseAndNull,
//                     texEllipseAndLodLvl,
//                     texTexScaleFactorDepthTexId,
//                     cornerAreaInfoShared,
//                     shValidMapCpct, shValidMapMemLayout,
//                     shPositionMapCpct, shPosMapMemLayout,
//                     botYCoefCurBuf,
//                     topYCoefCurBuf,
//                     invDistFactorEdge, invDistFactorCorner,
//                     shLayerNum, shOrder,
//                     highOrderSHLrMultiplier,
//                     gaussianCoeffEWA, texInWorldInfo, curTexAllWH, curTexDensity,
//                     needPrintX, needPrintY);
//             }

//             idxInGrp += MESH_HANDLE_THREAD_NUM;
//         }

//         grp.sync();

//         if (grp.thread_rank() == 0 && isValidThd) {
//             readPtrPos = ReadPixLocation(compactPixLocBuf, pixLocations, numOfPixToHandle, tmp.offset, tmp.pixNum, readPtrPos);
//         }

//         grp.sync();

//         // readPtrPos = grp.shfl(readPtrPos, 0);
//         numOfPixToHandle = grp.shfl(numOfPixToHandle, 0);
//     }
// }

template <typename Map>
__global__ void MeshLearnerbackwardTexPtrCUDA(uint32_t texNum,
    float* dL_dcolor, float* dL_dsh, uint32_t* shTexturesMemLayout,
    Map texHash, unsigned int* errCnt,
    TexPixInfo* visibleTexIdInfo, PixLocation* compactPixLocBuf,
    uint32_t imgW, uint32_t imgH,
    float* texViewDirFrag2CamAndNull,
    float* texLerpCoeffAndTexCoord,
    float* texCoordShLUandRU,
    float* texCoordShLDandRD,
    float* texObjWorldPoseAndNull,
    float* texEllipseAndLodLvl,
    float* texTexScaleFactorDepthTexId,
    float* cornerAreaInfo,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
    float* botYCoefCurBuf,
    float* topYCoefCurBuf,
    float invDistFactorEdge, float invDistFactorCorner,
    unsigned char shLayerNum, unsigned char shOrder,
    uint32_t* edgeNbrInfo, uint32_t* vertNbrInfo,
    float highOrderSHLrMultiplier, float gaussianCoeffEWA, float* texInWorldInfo, CurTexAlignedWH* curTexAllWH, float* curTexDensities, float* curTexNormals,
    uint32_t needPrintX, uint32_t needPrintY)
{
    auto blockId = blockIdx.x;
    auto grp = cg::tiled_partition<MESH_HANDLE_THREAD_NUM>(cg::this_thread_block());

    __shared__ PixLocation pixLocations[MESH_HANDLE_THREAD_NUM];
    __shared__ MeshEdgeNbrInfo meshEdgeNbrInfo;
    __shared__ MeshVertNbrInfo meshVertNbrInfo;
    __shared__ float2 cornerAreaInfoShared;

    uint32_t numOfPixToHandle = 0;
    // uint32_t totalPixNum = 0;
    uint32_t readPtrPos = 0;
    bool isValidThd = false;
    uint32_t curTexId = 0;
    uint32_t curTexOffset = 0;
    uint32_t curTexW = 0;
    uint32_t curTexH = 0;
    float curTexDensity = 0.0f;
    float curTexNormal[3];

    TexPixInfo tmp;

    if (blockId < texNum) {
        isValidThd = true;
    }

    grp.sync();

    if (grp.thread_rank() == 0 && isValidThd) {
        tmp = (*(visibleTexIdInfo + blockId));
        curTexId = tmp.texId;

        ReadMeshData(texHash, errCnt, blockId, edgeNbrInfo, vertNbrInfo, curTexId, meshEdgeNbrInfo, meshVertNbrInfo, curTexOffset, cornerAreaInfo, cornerAreaInfoShared, curTexAllWH, curTexW, curTexH, curTexDensities, curTexDensity, curTexNormals, curTexNormal);
        readPtrPos = ReadPixLocation(compactPixLocBuf, pixLocations, numOfPixToHandle, tmp.offset, tmp.pixNum, readPtrPos);

        // totalPixNum = tmp.pixNum;
    }

    grp.sync();

    // totalPixNum = grp.shfl(totalPixNum, 0);
    numOfPixToHandle = grp.shfl(numOfPixToHandle, 0);
    // readPtrPos = grp.shfl(readPtrPos, 0);
    curTexId = grp.shfl(curTexId, 0);
    curTexOffset = grp.shfl(curTexOffset, 0);
    curTexW = grp.shfl(curTexW, 0);
    curTexH = grp.shfl(curTexH, 0);
    curTexDensity = grp.shfl(curTexDensity, 0);
    curTexNormal[0] = grp.shfl(curTexNormal[0], 0);
    curTexNormal[1] = grp.shfl(curTexNormal[1], 0);
    curTexNormal[2] = grp.shfl(curTexNormal[2], 0);

    while (numOfPixToHandle > 0) {
        uint32_t idxInGrp = grp.thread_rank();

        while (grp.any(idxInGrp < numOfPixToHandle)) {
            if ((idxInGrp < numOfPixToHandle) && isValidThd) {
                PixLocation curLoc = pixLocations[idxInGrp];

                CalcGradTexPtr(dL_dcolor, dL_dsh, shTexturesMemLayout,
                    &meshEdgeNbrInfo, &meshVertNbrInfo,
                    curTexId, curTexOffset,
                    curLoc.x, curLoc.y,
                    imgW, imgH,
                    curTexW, curTexH,
                    texViewDirFrag2CamAndNull,
                    texLerpCoeffAndTexCoord,
                    texCoordShLUandRU,
                    texCoordShLDandRD,
                    texObjWorldPoseAndNull,
                    texEllipseAndLodLvl,
                    texTexScaleFactorDepthTexId,
                    cornerAreaInfoShared,
                    // shValidMapCpct, shValidMapMemLayout,
                    validShWHMap, validShWHMapMemLayout, validShNumsAll,
                    shPositionMapCpct, shPosMapMemLayout,
                    botYCoefCurBuf,
                    topYCoefCurBuf,
                    invDistFactorEdge, invDistFactorCorner,
                    shLayerNum, shOrder,
                    highOrderSHLrMultiplier,
                    gaussianCoeffEWA, texInWorldInfo, curTexAllWH, curTexDensity, curTexNormal, curTexNormals,
                    needPrintX, needPrintY);
            }

            idxInGrp += MESH_HANDLE_THREAD_NUM;
        }

        grp.sync();

        if (grp.thread_rank() == 0 && isValidThd) {
            readPtrPos = ReadPixLocation(compactPixLocBuf, pixLocations, numOfPixToHandle, tmp.offset, tmp.pixNum, readPtrPos);
        }

        grp.sync();

        // readPtrPos = grp.shfl(readPtrPos, 0);
        numOfPixToHandle = grp.shfl(numOfPixToHandle, 0);
    }
}

// template <typename Map>
// void MeshLearnerBackward(uint32_t texNum,
//     float* dL_dcolor, float* dL_dsh, uint32_t* shTexturesMemLayout,
//     Map texHash, unsigned int* errCnt,
//     TexPixInfo* visibleTexIdInfo, PixLocation* compactPixLocBuf,
//     uint32_t imgW, uint32_t imgH,
//     cudaTextureObject_t texViewDirFrag2CamAndNull,
//     cudaTextureObject_t texLerpCoeffAndTexCoord,
//     cudaTextureObject_t texCoordShLUandRU,
//     cudaTextureObject_t texCoordShLDandRD,
//     cudaTextureObject_t texObjWorldPoseAndNull,
//     cudaTextureObject_t texEllipseAndLodLvl,
//     cudaTextureObject_t texTexScaleFactorDepthTexId,
//     float* cornerAreaInfo,
//     uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
//     float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
//     float* botYCoefCurBuf,
//     float* topYCoefCurBuf,
//     float invDistFactorEdge, float invDistFactorCorner,
//     unsigned char shLayerNum, unsigned char shOrder,
//     uint32_t* edgeNbrInfo, uint32_t* vertNbrInfo,
//     float highOrderSHLrMultiplier, float gaussianCoeffEWA, float* texInWorldInfo,
//     CurTexAlignedWH* curTexAllWH, float* curTexDensities,
//     uint32_t needPrintX, uint32_t needPrintY)
// {
//     uint32_t constexpr blockSize = MESH_HANDLE_THREAD_NUM;
//     uint32_t gridSize = texNum;

//     MeshLearnerbackwardCUDA <<<gridSize, blockSize>>> (texNum,
//         dL_dcolor, dL_dsh, shTexturesMemLayout,
//         texHash, errCnt,
//         visibleTexIdInfo, compactPixLocBuf,
//         imgW, imgH,
//         texViewDirFrag2CamAndNull,
//         texLerpCoeffAndTexCoord,
//         texCoordShLUandRU,
//         texCoordShLDandRD,
//         texObjWorldPoseAndNull,
//         texEllipseAndLodLvl,
//         texTexScaleFactorDepthTexId,
//         cornerAreaInfo,
//         shValidMapCpct, shValidMapMemLayout,
//         shPositionMapCpct, shPosMapMemLayout,
//         botYCoefCurBuf,
//         topYCoefCurBuf,
//         invDistFactorEdge, invDistFactorCorner,
//         shLayerNum, shOrder,
//         edgeNbrInfo, vertNbrInfo,
//         highOrderSHLrMultiplier, gaussianCoeffEWA, texInWorldInfo,
//         curTexAllWH, curTexDensities,
//         needPrintX, needPrintY);
// }

template <typename Map>
void MeshLearnerBackwardTexPtr(uint32_t texNum,
    float* dL_dcolor, float* dL_dsh, uint32_t* shTexturesMemLayout,
    Map texHash, unsigned int* errCnt,
    TexPixInfo* visibleTexIdInfo, PixLocation* compactPixLocBuf,
    uint32_t imgW, uint32_t imgH,
    float* texViewDirFrag2CamAndNull,
    float* texLerpCoeffAndTexCoord,
    float* texCoordShLUandRU,
    float* texCoordShLDandRD,
    float* texObjWorldPoseAndNull,
    float* texEllipseAndLodLvl,
    float* texTexScaleFactorDepthTexId,
    float* cornerAreaInfo,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
    float* botYCoefCurBuf,
    float* topYCoefCurBuf,
    float invDistFactorEdge, float invDistFactorCorner,
    unsigned char shLayerNum, unsigned char shOrder,
    uint32_t* edgeNbrInfo, uint32_t* vertNbrInfo,
    float highOrderSHLrMultiplier, float gaussianCoeffEWA, float* texInWorldInfo,
    CurTexAlignedWH* curTexAllWH, float* curTexDensities, float* curTexNormals,
    uint32_t needPrintX, uint32_t needPrintY)
{
    uint32_t constexpr blockSize = MESH_HANDLE_THREAD_NUM;
    uint32_t gridSize = texNum;

    MeshLearnerbackwardTexPtrCUDA <<<gridSize, blockSize>>> (texNum,
        dL_dcolor, dL_dsh, shTexturesMemLayout,
        texHash, errCnt,
        visibleTexIdInfo, compactPixLocBuf,
        imgW, imgH,
        texViewDirFrag2CamAndNull,
        texLerpCoeffAndTexCoord,
        texCoordShLUandRU,
        texCoordShLDandRD,
        texObjWorldPoseAndNull,
        texEllipseAndLodLvl,
        texTexScaleFactorDepthTexId,
        cornerAreaInfo,
        // shValidMapCpct, shValidMapMemLayout,
        validShWHMap, validShWHMapMemLayout, validShNumsAll,
        shPositionMapCpct, shPosMapMemLayout,
        botYCoefCurBuf,
        topYCoefCurBuf,
        invDistFactorEdge, invDistFactorCorner,
        shLayerNum, shOrder,
        edgeNbrInfo, vertNbrInfo,
        highOrderSHLrMultiplier, gaussianCoeffEWA, texInWorldInfo,
        curTexAllWH, curTexDensities, curTexNormals,
        needPrintX, needPrintY);
}


#endif