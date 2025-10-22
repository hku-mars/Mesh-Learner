#ifndef __FORWARD_HASH_ACC_CUH__
#define __FORWARD_HASH_ACC_CUH__

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

// #include "common.cuh"
#include "../include/my_common.hpp"

namespace cg = cooperative_groups;

__device__ __constant__ float SH_C0 = 0.28209479177387814f;
__device__ __constant__ float SH_C1 = 0.4886025119029199f;
__device__ __constant__ float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};

// __device__ int32_t FindOffsetInValidShMap(TexAlignedPosWH* validMapStartAddr, uint32_t validShNumThisTex, uint32_t targetW, uint32_t targetH, bool prt=false)
__device__ int32_t FindOffsetInValidShMap(TexAlignedPosWH* validMapStartAddr, uint32_t validShNumThisTex, uint32_t targetW, uint32_t targetH)
{
    for (int32_t i = 0; i < validShNumThisTex; i++) {
        uint32_t curW = ((TexAlignedPosWH*)(validMapStartAddr + i))->data.posW;
        uint32_t curH = ((TexAlignedPosWH*)(validMapStartAddr + i))->data.posH;

        // if (prt == true) {
        //     printf("curW: %d, curH: %d, target WH: %d, %d, tot valid sh num: %d\n",
        //         (uint32_t)curW, (uint32_t)curH, targetW, targetH, validShNumThisTex);
        // }

        //ealry quit
        if (curH != targetH) {
            continue;
        }

        if (curW != targetW) {
            continue;
        }

        return i;
    }

    return -1;
}

// cos(15deg) = 0.9659
__device__ bool IsInSamePlane(float* curNormal, float* nbrNormal, float thrs=0.9659f)
{
    glm::dvec3 curN((*(curNormal + 0)), (*(curNormal + 1)), (*(curNormal + 2)));
    glm::dvec3 nbrN((*(nbrNormal + 0)), (*(nbrNormal + 1)), (*(nbrNormal + 2)));

    float result = (float)(glm::dot(curN, nbrN));
    if (result < thrs) {
        return false;
    } else {
        return true;
    }
}

__forceinline__ __device__ bool step(float edge, float x)
{
    float eps = 1e-8;

    return (bool)(x - edge >= eps);
}

template <typename Map>
__device__ uint64_t HashFindVal(Map HashFindRef, uint32_t key, unsigned int* errCnt)
{
    // printf("------------hash key is : %d\n", key);
    assert(key != 0);
    auto found = HashFindRef.find(key);
    if (found != HashFindRef.end()) {
        return (uint64_t)found->second;
    } else {
        atomicAdd(errCnt, 1);
        printf("k-Er %u\n", key);
        return (uint64_t)0;
    }
}

template <typename Map>
__device__ uint32_t GetTexOffsetFromHash(Map HashFindRef, uint32_t key, unsigned int* errCnt)
{
    uint64_t rawData = HashFindVal(HashFindRef, key, errCnt);
    return (rawData >> 42); // high 22 bits for offset in Tensor
}

template <typename Map>
__device__ void GetVertNbrInfoFromHash(Map HashFindRef, uint32_t* baseAddr, uint32_t key, unsigned int* errCnt, uint32_t* texIdOfShs, uint8_t& shNumber, VertNbrType type)
{
    uint64_t rawData = HashFindVal(HashFindRef, key, errCnt);

    uint64_t tmp = (rawData >> 15);
    uint32_t offset = (uint32_t)(tmp & 0x7FFFFFF);

    uint32_t neighborNum = ((rawData >> ((VERT_NUMER_BIT_NUM) * type)) & 0x1f);
    uint8_t calcOffsetTimes = (VERT_NBR_TYPE_NUM - 1 - type);

    uint32_t readOffset = 0;

    for (uint8_t cnt = 0; cnt < calcOffsetTimes; cnt++) {
        uint8_t nbrNumTmp = (rawData >> (VERT_NUMER_BIT_NUM) * (VERT_NBR_TYPE_NUM - 1 - cnt)) & 0x1f;
        readOffset += nbrNumTmp;
    }

    // if ((offset + readOffset) % VEC_LOAD_ELEMENT_NUM == 0) {
    //     uint8_t writePos = 0;
    //     uint8_t vecLoadTime = neighborNum / VEC_LOAD_ELEMENT_NUM;
    //     for (uint8_t cnt = 0; cnt < vecLoadTime; cnt++) {
    //         uint2 tmp = *(reinterpret_cast<uint2 *>(baseAddr + offset + readOffset + cnt * VEC_LOAD_ELEMENT_NUM));
    //         // assert(tmp.x != 0); assert(tmp.y != 0); assert(tmp.z != 0); assert(tmp.w != 0);

    //         for (uint8_t ii = 0; ii < VEC_LOAD_ELEMENT_NUM; ii++) {
    //             assert((reinterpret_cast<uint*>(&tmp))[ii] != 0);
    //             *(texIdOfShs + writePos) = (reinterpret_cast<uint*>(&tmp))[ii];
    //             writePos++;
    //         }
    //     }

    //     for (uint8_t cnt = vecLoadTime * VEC_LOAD_ELEMENT_NUM; cnt < neighborNum; cnt++) {
    //         assert((*(baseAddr + offset + readOffset + cnt)) != 0);
    //         *(texIdOfShs + writePos) = *(baseAddr + offset + readOffset + cnt);
    //         writePos++;
    //     }
    // } else {
    //     for (uint8_t cnt = 0; cnt < neighborNum; cnt++) {
    //         assert((*(baseAddr + offset + readOffset + cnt)) != 0);
    //         *(texIdOfShs + cnt) = *(baseAddr + offset + readOffset + cnt);
    //     }
    // }

    for (uint8_t cnt = 0; cnt < neighborNum; cnt++) {
        assert((*(baseAddr + offset + readOffset + cnt)) != 0);
        *(texIdOfShs + cnt) = *(baseAddr + offset + readOffset + cnt);
    }

    shNumber = neighborNum;
}




__device__ uint32_t DecodeW(uint32_t compactWH)
{
    return (uint32_t)(compactWH >> 16);
}

__device__ uint32_t DecodeH(uint32_t compactWH)
{
    return (uint32_t)(compactWH & 0xffff);
}

__device__ void Write2CompactW(uint32_t& data, uint32_t toWrite)
{
    data = data & 0x0000ffff;
    data = data | (toWrite << 16);
}

__device__ void Write2CompactH(uint32_t& data, uint32_t toWrite)
{
    data = data & 0xffff0000;
    data = data | (toWrite & 0x0000ffff);
}

__device__ void Write2CompactWH(uint32_t* offsetOfWH, uint32_t idx, uint32_t toWriteW, uint32_t toWriteH)
{
    uint32_t toWrite = (toWriteW << 16) | (toWriteH & 0x0000ffff);
    *(offsetOfWH + idx) = toWrite;
}

__device__ void ReadFromCompactWH(uint32_t* offsetOfWH, uint32_t idx, uint32_t& readW, uint32_t& readH)
{
    uint32_t raw = *(offsetOfWH + idx);
    readW = (raw >> 16);
    readH = (raw & 0x0000ffff);
}


__device__ VertNbrType IsInCornerShared(float2 texCoord, float2 cornerAreaInfoShared, float* botYCoefCurBuf, float* topYCoefCurBuf, uint32_t texOffset)
{
    float leftCornerShX = cornerAreaInfoShared.x;
    float rightCornerShX = cornerAreaInfoShared.y;

    if (texCoord.y >= *(topYCoefCurBuf + texOffset)) {
        return VERT_NBR_TYPE_UP;
    } else if (texCoord.y <= *(botYCoefCurBuf + texOffset)) {
        if (texCoord.x <= leftCornerShX) {
            return VERT_NBR_TYPE_LEFT;
        }

        if (texCoord.x >= rightCornerShX) {
            return VERT_NBR_TYPE_RIGHT;
        }
    } else {
        return VERT_NBR_TYPE_NUM;
    }

    return VERT_NBR_TYPE_NUM;
}

__device__ bool IsInExludeListShared(uint8_t curW, uint8_t curH, uint32_t texId, uint8_t* excludeWs, uint8_t* excludeHs, uint32_t* excludeTexIds, uint8_t excludeNum)
{
    if (excludeNum <= 0) {
        return false;
    }

    for (uint8_t i = 0; i < excludeNum; i++) {
        if (texId != (*(excludeTexIds + i))) {
            continue;
        }
        if ((curW == *(excludeWs + i)) && (curH == *(excludeHs + i))) {
            return true;
        }
    }

    return false;
}

__device__ bool IsInExludeList(uint8_t curW, uint8_t curH, uint32_t texId, uint8_t* excludeWs, uint8_t* excludeHs, uint32_t* excludeTexIds, uint8_t excludeNum)
{
    if (excludeNum <= 0) {
        return false;
    }

    for (uint8_t i = 0; i < excludeNum; i++) {
        if (texId != (*(excludeTexIds + i))) {
            continue;
        }
        if ((curW == *(excludeWs + i)) && (curH == *(excludeHs + i))) {
            return true;
        }
    }

    return false;
}

__device__ bool FindNearestShShared(glm::vec3 worldPosition,
    uint32_t nbrTexId, uint32_t nbrTexIdOffset,
    float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    uint8_t texW, uint8_t texH,
    uint8_t& minW, uint8_t& minH,
    float& minDistance, uint8_t* excludeWs, uint8_t* excludeHs, uint32_t* excludeTexIds, uint8_t excludeNum)
{
    uint32_t shPositionMapOffsetPerLine = texW * 3;
    bool findAtleastOne = false;

    TexAlignedPosWH* curValidWhMapStartAddr = (validShWHMap + (*(validShWHMapMemLayout + nbrTexIdOffset)));
    uint32_t curValidShNum = (*(validShNumsAll + nbrTexIdOffset));

    for (uint8_t curH = 0; curH < texH; curH++) {
        for (uint8_t curW = 0; curW < texW; curW++) {
            // if ( (*(shValidMapCpct + (*(shValidMapMemLayout + nbrTexIdOffset)) + curH * texW + curW)) == 0) {
            //     // if(((blockIdx.x * blockDim.x + threadIdx.x) == needPrintX) &&
            //     //     ((blockIdx.y * blockDim.y + threadIdx.y) == needPrintY)) {
            //     //     printf("find nearest continue, texId: %d, W:%d, H:%d\n", nbrTexId, curW, curH);
            //     // }
            //     continue;
            // }

            // :)
            int32_t innieOffset = FindOffsetInValidShMap(curValidWhMapStartAddr, curValidShNum, curW, curH);
            if (innieOffset == -1) {
                continue;
            }

            if (IsInExludeList(curW, curH, nbrTexId, excludeWs, excludeHs, excludeTexIds, excludeNum) == true) {
                continue;
            }

            glm::vec3 curShWorldPose;
            // curShWorldPose.x = *(shPositionMapCpct + (*(shPosMapMemLayout + nbrTexIdOffset)) + curH * shPositionMapOffsetPerLine + curW * 3 + 0);
            // curShWorldPose.y = *(shPositionMapCpct + (*(shPosMapMemLayout + nbrTexIdOffset)) + curH * shPositionMapOffsetPerLine + curW * 3 + 1);
            // curShWorldPose.z = *(shPositionMapCpct + (*(shPosMapMemLayout + nbrTexIdOffset)) + curH * shPositionMapOffsetPerLine + curW * 3 + 2);

            curShWorldPose.x = *(shPositionMapCpct + (*(shPosMapMemLayout + nbrTexIdOffset)) + 3 * innieOffset + 0);
            curShWorldPose.y = *(shPositionMapCpct + (*(shPosMapMemLayout + nbrTexIdOffset)) + 3 * innieOffset + 1);
            curShWorldPose.z = *(shPositionMapCpct + (*(shPosMapMemLayout + nbrTexIdOffset)) + 3 * innieOffset + 2);


            float curDist = glm::length(curShWorldPose - worldPosition);

            // if(((blockIdx.x * blockDim.x + threadIdx.x) == needPrintX) &&
            //     ((blockIdx.y * blockDim.y + threadIdx.y) == needPrintY)) {
            //         printf("curDist:%f texId: %d, W:%d, H:%d\n", curDist, nbrTexId, curW, curH);
            // }

            if (curDist <= minDistance) {
                minDistance = curDist;
                minW = curW;
                minH = curH;

                findAtleastOne = true;
            }
        }
    }

    return findAtleastOne;
}

__device__ void CalcInverseDistanceWeightsCornerShared(VertNbrType vertNbrType,
    MeshVertNbrInfo* meshVertNbrInfo,
    uint32_t selfTexId, uint32_t selfOffsetInHash,
    glm::vec3 worldPosition,
    float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    uint8_t selfTexW, uint8_t selfTexH,
    uint8_t& shNumber, uint32_t* texIdOfShs,
    uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint32_t* texIdOffsets, float invDistFactorCorner,
    CurTexAlignedWH* curTexAllWH, float* curTexNormal, float* curTexNormals)
{
    for (uint8_t cnt = 0; cnt < meshVertNbrInfo->nbrNum[vertNbrType]; cnt++) {
        float minDistance = 10000.f;
        uint8_t minW, minH;

        if (IsInSamePlane(curTexNormal, (curTexNormals + meshVertNbrInfo->nbrTexOffsetsInHash[vertNbrType][cnt] * 3)) == false) {
            continue;
        }

        if (FindNearestShShared(worldPosition,
            meshVertNbrInfo->nbrTexIds[vertNbrType][cnt],
            meshVertNbrInfo->nbrTexOffsetsInHash[vertNbrType][cnt],
            shPositionMapCpct, shPosMapMemLayout,
            // shValidMapCpct, shValidMapMemLayout,
            validShWHMap, validShWHMapMemLayout, validShNumsAll,
            (uint8_t)((curTexAllWH + meshVertNbrInfo->nbrTexOffsetsInHash[vertNbrType][cnt])->data.curTexW),
            (uint8_t)((curTexAllWH + meshVertNbrInfo->nbrTexOffsetsInHash[vertNbrType][cnt])->data.curTexH),
            minW, minH, minDistance, nullptr, nullptr, nullptr, 0) == false) {
            continue;
        }

        // *(offsetOfW + shNumber) = minW;
        // *(offsetOfH + shNumber) = minH;
        Write2CompactWH(offsetOfWH, shNumber, minW, minH);
        *(lerpCoeffsOfShs + shNumber) = (1.0f / (powf(minDistance * 1.0f, invDistFactorCorner))) + 1e-6f;
        *(texIdOfShs + shNumber) = meshVertNbrInfo->nbrTexIds[vertNbrType][cnt];
        *(texIdOffsets + shNumber) = meshVertNbrInfo->nbrTexOffsetsInHash[vertNbrType][cnt];
        shNumber++;

        // if(((blockIdx.x * blockDim.x + threadIdx.x) == needPrintX) &&
        //     ((blockIdx.y * blockDim.y + threadIdx.y) == needPrintY)) {
        //     printf("vertNbr: %d, minDist: %f, W:%d, H:%d\n", texIdOfShs[cnt], minDistance, minW, minH);
        // }
    }

    // self tex also
    float minDistance = 10000.f;
    uint8_t minW{0}, minH{0};
    if (FindNearestShShared(worldPosition,
            selfTexId,
            selfOffsetInHash,
            shPositionMapCpct, shPosMapMemLayout,
            // shValidMapCpct, shValidMapMemLayout,
            validShWHMap, validShWHMapMemLayout, validShNumsAll,
            selfTexW, selfTexH, minW, minH, minDistance, nullptr, nullptr, nullptr, 0) == false) {
        return;
    }

    // *(offsetOfW + shNumber) = minW;
    // *(offsetOfH + shNumber) = minH;
    Write2CompactWH(offsetOfWH, shNumber, minW, minH);
    *(lerpCoeffsOfShs + shNumber) = (1.0f / (powf(minDistance * 1.0f, invDistFactorCorner))) + 1e-6f;
    *(texIdOfShs + shNumber) = selfTexId;
    *(texIdOffsets + shNumber) = selfOffsetInHash;
    shNumber++;

    for (uint8_t i = 0; i < shNumber; i++) {
        assert((*(texIdOfShs + i)) > 0);
        assert((*(lerpCoeffsOfShs + i)) != 0.f);
        
        // if ((*(lerpCoeffsOfShs + i)) == 0.f) {
        //     printf("---- at X : %d, Y:%d :\n", blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
        //     for (uint8_t i = 0; i < shNumber; i++) {
        //         printf("err: texIdOfShs : %d, lerp coeff: %f\n", (*(texIdOfShs + i)), (*(lerpCoeffsOfShs + i)));
        //     }
        // }

        // assert((*(lerpCoeffsOfShs + i)) != 0.f);
    }
}

__device__ void FallBackToIDWShared(uint32_t* texOffsets, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint8_t shNumber, glm::vec3 worldPosition, float* shPositionMapCpct, uint32_t* shPosMapMemLayout, float invDistFactor, CurTexAlignedWH* curTexAllWH)
{
    for (uint8_t cnt = 0; cnt < shNumber; cnt++) {
        uint32_t shPositionMapOffsetPerLine = (curTexAllWH + texOffsets[cnt])->data.curTexW * 3;
        glm::vec3 shWorldPosition;
        uint32_t offsetW, offsetH;
        ReadFromCompactWH(offsetOfWH, cnt, offsetW, offsetH);
        shWorldPosition.x = *(shPositionMapCpct + (*(shPosMapMemLayout + texOffsets[cnt])) + offsetH * shPositionMapOffsetPerLine + offsetW * 3 + 0);
        shWorldPosition.y = *(shPositionMapCpct + (*(shPosMapMemLayout + texOffsets[cnt])) + offsetH * shPositionMapOffsetPerLine + offsetW * 3 + 1);
        shWorldPosition.z = *(shPositionMapCpct + (*(shPosMapMemLayout + texOffsets[cnt])) + offsetH * shPositionMapOffsetPerLine + offsetW * 3 + 2);

        float curDist = glm::length(shWorldPosition - worldPosition);
        
        *(lerpCoeffsOfShs + cnt) = (1.0f / (powf(curDist * 1.0f, invDistFactor))) + 1e-6f;
    }
}

__device__ void RetrieveShPositionsShared(uint32_t* texOffsets, float* shPositionMapCpct, uint32_t* shPosMapMemLayout, uint32_t* offsetOfWH, uint8_t shNumber, glm::dvec3* shPositions, CurTexAlignedWH* curTexAllWH)
{
    uint32_t texIdOffset;

    for (uint8_t idx = 0; idx < shNumber; idx++) {
        texIdOffset = *(texOffsets + idx);
        uint32_t shPositionMapOffsetPerLine = (curTexAllWH + texIdOffset)->data.curTexW * 3;
        uint32_t offsetW, offsetH;
        ReadFromCompactWH(offsetOfWH, idx, offsetW, offsetH);

        //  * 10000.0
        (*(shPositions + idx)).x = (double)(*(shPositionMapCpct + (*(shPosMapMemLayout + texIdOffset)) + offsetH * shPositionMapOffsetPerLine + offsetW * 3 + 0));

        (*(shPositions + idx)).y = (double)(*(shPositionMapCpct + (*(shPosMapMemLayout + texIdOffset)) + offsetH * shPositionMapOffsetPerLine + offsetW * 3 + 1));

        (*(shPositions + idx)).z = (double)(*(shPositionMapCpct + (*(shPosMapMemLayout + texIdOffset)) + offsetH * shPositionMapOffsetPerLine + offsetW * 3 + 2));
    }
}

__device__ void CalcMeanValueCoordShared(uint32_t* texOffsets, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint8_t shNumber, glm::vec3 worldPosition, float* shPositionMapCpct, uint32_t* shPosMapMemLayout, float invDistFactor, CurTexAlignedWH* curTexAllWH)
{
    glm::dvec3 shPositions[32];

    if (shNumber <= 2) {
        FallBackToIDWShared(texOffsets, offsetOfWH, lerpCoeffsOfShs, shNumber, worldPosition, shPositionMapCpct, shPosMapMemLayout, invDistFactor, curTexAllWH);
        return;
    }

    // glm::dvec3 worldPositionDouble((double)worldPosition.x * 10000.0, (double)worldPosition.y * 10000.0, (double)worldPosition.z * 10000.0);
    glm::dvec3 worldPositionDouble((double)worldPosition.x, (double)worldPosition.y, (double)worldPosition.z);

    RetrieveShPositionsShared(texOffsets, shPositionMapCpct, shPosMapMemLayout, offsetOfWH, shNumber, shPositions, curTexAllWH);

    for (uint8_t idx = 0; idx < shNumber; idx++) {
        uint8_t idxNext = ((idx + 1) % shNumber);
        uint8_t idxPrev = ((idx + (shNumber - 1)) % shNumber);

        glm::dvec3 siPrev = shPositions[idxPrev] - worldPositionDouble;
        glm::dvec3 si = shPositions[idx] - worldPositionDouble;
        glm::dvec3 siNext = shPositions[idxNext] - worldPositionDouble;

        float Ai = (float)(glm::length(glm::cross(si, siNext)) / 2.0);
        float AiPrev = (float)(glm::length(glm::cross(si, siPrev)) / 2.0);

        float Di = (float)(glm::dot(si, siNext));
        float DiPrev = (float)(glm::dot(si, siPrev));


        if (glm::length(si) <= (1e-2)) { // cur Pt is on a vertex
            for (uint8_t cnt = 0; cnt < shNumber; cnt++) {
                if (cnt == idx) {
                    *(lerpCoeffsOfShs + cnt) = 1.0f;
                    continue;
                }

                *(lerpCoeffsOfShs + cnt) = 0.0f;
            }

            return;
        }

        // 10000.0
        if ((Ai <= (1e-2)) && (Di <= (1e-2))) { // cur Pt is on a edge
            // assert(glm::abs(glm::length(worldPositionDouble - pvi) + glm::length(pvNext - worldPositionDouble) - glm::length(pvNext - pvi)) < 1e-5);

            float miu = (float)((glm::length(worldPositionDouble - shPositions[idx]) / glm::length(shPositions[idxNext] - shPositions[idx])));
            // float miu = (float)((glm::length(worldPositionDouble - pvi) / glm::length(pvNext - pvi)));

            for (uint8_t cnt = 0; cnt < shNumber; cnt++) {
                if (cnt == idx) {
                    *(lerpCoeffsOfShs + cnt) = 1.0f - miu;
                    continue;
                }

                if (cnt == ((idx + 1) % shNumber)) {
                    *(lerpCoeffsOfShs + cnt) = miu;
                    continue;
                }

                *(lerpCoeffsOfShs + cnt) = 0.0f;
            }

            return;
        }

        double tanAlphaIDiv2 = ((glm::length(si) * glm::length(siNext) - (double)Di) / (2.0 * (double)Ai));

        double tanAlphaIPrevDiv2 = ((glm::length(si) * glm::length(siPrev) - (double)DiPrev) / (2.0 * (double)AiPrev));

        *(lerpCoeffsOfShs + idx) = float(((tanAlphaIDiv2 + tanAlphaIPrevDiv2) / (glm::length(si))));

        assert(isnan(*(lerpCoeffsOfShs + idx)) == 0);
        assert(isinf(*(lerpCoeffsOfShs + idx)) == 0);
    }
}

__device__ unsigned char IfCurShInsideShared(float2 shTexCoord, TexAlignedPosWH* validMapStartAddr, uint32_t validShNumThisTex, uint8_t selfTexW, uint8_t selfTexH, int32_t& innieOffset)
{
    unsigned char isInsideTexSpace = step(-1e-7, shTexCoord.x) * step(shTexCoord.x, 1.f + 1e-7) * step(-1e-7, shTexCoord.y) * step(shTexCoord.y, 1.f + 1e-7);

    if (isInsideTexSpace == 0) {
        return 0;
    }

    unsigned int offsetH = (unsigned int)floorf(selfTexH * (shTexCoord.y));
    unsigned int offsetW = (unsigned int)floorf(selfTexW * (shTexCoord.x));

    innieOffset = FindOffsetInValidShMap(validMapStartAddr, validShNumThisTex, offsetW, offsetH);
    if (innieOffset == -1) {
        return 0;
    } else {
        return 1;
    }

    // return (*(shValidMapCpct + *(shValidMapMemLayout + texOffset) + offsetH * selfTexW + offsetW));
}

__device__ void CalcBiLinearWeightsShared(float2 oriLerpCoeff, const float2 oriLU, const float2 oriRU, const float2 oriLD, const float2 oriRD, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint8_t texW, uint8_t texH, uint8_t& shNumber, uint32_t* texIdOfShs, uint32_t texId, uint32_t* texOffsets, uint32_t selfTexOffset)
{
    // *(offsetOfH + shNumber) = (uint8_t)floorf(texH * (oriLU.y));
    // *(offsetOfW + shNumber) = (uint8_t)floorf(texW * (oriLU.x));
    Write2CompactWH(offsetOfWH, shNumber, (uint32_t)floorf(texW * (oriLU.x)), (uint32_t)floorf(texH * (oriLU.y)));
    *(lerpCoeffsOfShs + shNumber) = ((1.0f - oriLerpCoeff.x) * (1.0f - oriLerpCoeff.y)) + 1e-6f;
    *(texIdOfShs + shNumber) = texId;
    *(texOffsets + shNumber) = selfTexOffset;
    shNumber++;

    // *(offsetOfH + shNumber) = (uint8_t)floorf(texH * (oriRU.y));
    // *(offsetOfW + shNumber) = (uint8_t)floorf(texW * (oriRU.x));
    Write2CompactWH(offsetOfWH, shNumber, (uint32_t)floorf(texW * (oriRU.x)), (uint32_t)floorf(texH * (oriRU.y)));
    *(lerpCoeffsOfShs + shNumber) = (oriLerpCoeff.x * (1.0f - oriLerpCoeff.y)) + 1e-6f;
    *(texIdOfShs + shNumber) = texId;
    *(texOffsets + shNumber) = selfTexOffset;
    shNumber++;

    // *(offsetOfH + shNumber) = (uint8_t)floorf(texH * (oriLD.y));
    // *(offsetOfW + shNumber) = (uint8_t)floorf(texW * (oriLD.x));
    Write2CompactWH(offsetOfWH, shNumber, (uint32_t)floorf(texW * (oriLD.x)), (uint32_t)floorf(texH * (oriLD.y)));
    *(lerpCoeffsOfShs + shNumber) = ((1.0f - oriLerpCoeff.x) * oriLerpCoeff.y) + 1e-6f;
    *(texIdOfShs + shNumber) = texId;
    *(texOffsets + shNumber) = selfTexOffset;
    shNumber++;

    // *(offsetOfH + shNumber) = (uint8_t)floorf(texH * (oriRD.y));
    // *(offsetOfW + shNumber) = (uint8_t)floorf(texW * (oriRD.x));
    Write2CompactWH(offsetOfWH, shNumber, (uint32_t)floorf(texW * (oriRD.x)), (uint32_t)floorf(texH * (oriRD.y)));
    *(lerpCoeffsOfShs + shNumber) = (oriLerpCoeff.x * oriLerpCoeff.y) + 1e-6f;
    *(texIdOfShs + shNumber) = texId;
    *(texOffsets + shNumber) = selfTexOffset;
    shNumber++;

    // for (uint8_t i = 0; i < shNumber; i++) {
    //     if (((*(texIdOfShs + i)) <= 0 )|| ((*(lerpCoeffsOfShs + i)) == 0.f)) {
    //         printf("current oriLerpCoeff: %f, %f, current XY: %d, %d", oriLerpCoeff.x, oriLerpCoeff.y,
    //             (blockIdx.x * blockDim.x + threadIdx.x), (blockIdx.y * blockDim.y + threadIdx.y));
    //     }

    //     assert((*(texIdOfShs + i)) > 0);
    //     assert((*(lerpCoeffsOfShs + i)) != 0.f);
    // }
}

__device__ void RuleOutSHsShared(uint8_t* excludeW, uint8_t* excludeH, uint8_t& exludeNum, uint32_t* excludeTexIds, int8_t posW, int8_t posH, uint8_t texW, uint8_t texH,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    uint32_t texId, uint32_t texIdOffset)
{
    TexAlignedPosWH* curValidShMapStarrAddr = (validShWHMap + (*(validShWHMapMemLayout + texIdOffset)));
    uint32_t curValidShNum = (*(validShNumsAll + texIdOffset));
    if (posW >= 0 && posW < texW && posH >= 0 && posH < texH) {
        if (FindOffsetInValidShMap(curValidShMapStarrAddr, curValidShNum, posW, posH) != -1) {
            *(excludeW + exludeNum) = posW;
            *(excludeH + exludeNum) = posH;
            *(excludeTexIds + exludeNum) = texId;
            exludeNum++;
        }        

        // if ((*(shValidMapCpct + (*(shValidMapMemLayout + texIdOffset)) + posH * texW + posW)) == 1) {
        //     *(excludeW + exludeNum) = posW;
        //     *(excludeH + exludeNum) = posH;
        //     *(excludeTexIds + exludeNum) = texId;
        //     exludeNum++;
        // }
    }
}

__device__ void FindShTopNearestNShared(uint8_t needFindNum, uint32_t nbrTexIdOffset, uint32_t nbrTexId, glm::vec3 worldPosition, float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    uint8_t texW, uint8_t texH, EdgeTypeCUDA edgeType, uint8_t& shNumber, uint32_t* texIdOfShs, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint32_t* texOffsets, float invDistFactorEdge)
{
    uint8_t exludeNum = 0;
    uint8_t excludeWs[10] = {0};
    uint8_t excludeHs[10] = {0};
    uint32_t excludeTexIds[10] = {0};

    // int8_t excludeOffsetW[3][3] = {{1, 1, 0}, {-1, -1, 0}, {0, 1, -1}};
    // int8_t excludeOffsetH[3][3] = {{0, -1, -1}, {0, -1, -1}, {1, 1, 1},};

    int8_t excludeOffsetW[3][1] = {{1}, {-1}, {0}};
    int8_t excludeOffsetH[3][1] = {{-1}, {-1}, {1}};

    for (uint8_t i = 0; i < needFindNum; i++) {
        float minDistance = 10000.f;
        uint8_t minW, minH;
        if (FindNearestShShared(worldPosition,
            nbrTexId,
            nbrTexIdOffset,
            shPositionMapCpct, shPosMapMemLayout, validShWHMap, validShWHMapMemLayout, validShNumsAll, texW, texH, minW, minH, minDistance, excludeWs, excludeHs, excludeTexIds, exludeNum) == false) {
            return;
        }

        //  *(offsetOfW + shNumber) = minW;
        // *(offsetOfH + shNumber) = minH;
        Write2CompactWH(offsetOfWH, shNumber, minW, minH);
        *(lerpCoeffsOfShs + shNumber) = (1.0f / (powf(minDistance * 1.0f, invDistFactorEdge))) + 1e-6f;
        *(texIdOfShs + shNumber) = nbrTexId;
        *(texOffsets + shNumber) = nbrTexIdOffset;
        shNumber++;

        *(excludeWs + exludeNum) = minW;
        *(excludeHs + exludeNum) = minH;
        *(excludeTexIds + exludeNum) = nbrTexId;
        exludeNum++;

        if (i == 0) {
            // encourage to find SH along a edge
            for (uint8_t cnt = 0; cnt < 1; cnt++) {
                RuleOutSHsShared(excludeWs, excludeHs, exludeNum, excludeTexIds, ((int8_t)minW + excludeOffsetW[edgeType][cnt]), ((int8_t)minH + excludeOffsetH[edgeType][cnt]), texW, texH, validShWHMap, validShWHMapMemLayout, validShNumsAll, nbrTexId, nbrTexIdOffset);
            }
        }
    }
}

__device__ void CalcInverseDistanceWeightsShared(uint32_t nbrTexIdOffset, uint32_t nbrTexId, EdgeTypeCUDA nbrEdgeType,
    float* shPositionMapCpct, uint32_t* shPosMapMemLayout, glm::vec3 worldPosition, float invDistFactorEdge,
    uint8_t needFindNum,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    uint8_t texW, uint8_t texH, uint8_t& shNumber, uint32_t* texIdOfShs, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint32_t* texOffsets)
{
    if (nbrTexId == 0) {
        return;
    }

    FindShTopNearestNShared(needFindNum, nbrTexIdOffset, nbrTexId, worldPosition, shPositionMapCpct, shPosMapMemLayout, validShWHMap, validShWHMapMemLayout, validShNumsAll, texW, texH, nbrEdgeType, shNumber, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, texOffsets, invDistFactorEdge);

    for (uint8_t i = 0; i < shNumber; i++) {
        assert((*(texIdOfShs + i)) > 0);
        assert((*(lerpCoeffsOfShs + i)) != 0.f);
    }
}



__device__ void CalcInsideShInverseDistanceWeightShared(uint32_t selfTexId, uint32_t selfTexIdOffset, uint8_t isLUinside, uint8_t isRUinside, uint8_t isLDinside, uint8_t isRDinside, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, const float2 oriLU, const float2 oriRU, const float2 oriLD, const float2 oriRD, uint8_t texW, uint8_t texH, float* shPositionMapCpct, uint32_t* shPosMapMemLayout, glm::vec3 worldPosition, float invDistFactorEdge, uint8_t& shNumber, uint32_t* texIdOfShs, uint32_t* texOffsets, int32_t posMapOffsetLU, int32_t posMapOffsetRU, int32_t posMapOffsetLD, int32_t posMapOffsetRD)
{
    uint32_t shPositionMapOffsetPerLine = texW * 3;

    if (isLUinside == 1) {
        uint8_t offsetW = (uint8_t)floorf(texW * (oriLU.x));
        uint8_t offsetH = (uint8_t)floorf(texH * (oriLU.y));

        glm::vec3 shWorldPosition;
        shWorldPosition.x = *(shPositionMapCpct + (*(shPosMapMemLayout + selfTexIdOffset)) + posMapOffsetLU * 3 + 0);
        shWorldPosition.y = *(shPositionMapCpct + (*(shPosMapMemLayout + selfTexIdOffset)) + posMapOffsetLU * 3 + 1);
        shWorldPosition.z = *(shPositionMapCpct + (*(shPosMapMemLayout + selfTexIdOffset)) + posMapOffsetLU * 3 + 2);

        float curDist = glm::length(shWorldPosition - worldPosition);
        
        *(lerpCoeffsOfShs + shNumber) = (1.0f / (powf(curDist * 1.0f, invDistFactorEdge))) + 1e-6f;
        // *(offsetOfW + shNumber) = offsetW;
        // *(offsetOfH + shNumber) = offsetH;
        Write2CompactWH(offsetOfWH, shNumber, offsetW, offsetH);
        *(texIdOfShs + shNumber) = selfTexId;
        *(texOffsets + shNumber) = selfTexIdOffset;
        shNumber++;
    }

    if (isRUinside == 1) {
        uint8_t offsetW = (uint8_t)floorf(texW * (oriRU.x));
        uint8_t offsetH = (uint8_t)floorf(texH * (oriRU.y));

        glm::vec3 shWorldPosition;
        shWorldPosition.x = *(shPositionMapCpct + (*(shPosMapMemLayout + selfTexIdOffset)) + posMapOffsetRU * 3 + 0);
        shWorldPosition.y = *(shPositionMapCpct + (*(shPosMapMemLayout + selfTexIdOffset)) + posMapOffsetRU * 3 + 1);
        shWorldPosition.z = *(shPositionMapCpct + (*(shPosMapMemLayout + selfTexIdOffset)) + posMapOffsetRU * 3 + 2);

        float curDist = glm::length(shWorldPosition - worldPosition);
        
        *(lerpCoeffsOfShs + shNumber) = (1.0f / (powf(curDist * 1.0f, invDistFactorEdge))) + 1e-6f;
        // *(offsetOfW + shNumber) = offsetW;
        // *(offsetOfH + shNumber) = offsetH;
        Write2CompactWH(offsetOfWH, shNumber, offsetW, offsetH);
        *(texIdOfShs + shNumber) = selfTexId;
        *(texOffsets + shNumber) = selfTexIdOffset;
        shNumber++;
    }

    if (isLDinside == 1) {
        uint8_t offsetW = (uint8_t)floorf(texW * (oriLD.x));
        uint8_t offsetH = (uint8_t)floorf(texH * (oriLD.y));

        glm::vec3 shWorldPosition;
        shWorldPosition.x = *(shPositionMapCpct + (*(shPosMapMemLayout + selfTexIdOffset)) + posMapOffsetLD * 3 + 0);
        shWorldPosition.y = *(shPositionMapCpct + (*(shPosMapMemLayout + selfTexIdOffset)) + posMapOffsetLD * 3 + 1);
        shWorldPosition.z = *(shPositionMapCpct + (*(shPosMapMemLayout + selfTexIdOffset)) + posMapOffsetLD * 3 + 2);

        float curDist = glm::length(shWorldPosition - worldPosition);
        
        *(lerpCoeffsOfShs + shNumber) = (1.0f / (powf(curDist * 1.0f, invDistFactorEdge))) + 1e-6f;
        // *(offsetOfW + shNumber) = offsetW;
        // *(offsetOfH + shNumber) = offsetH;
        Write2CompactWH(offsetOfWH, shNumber, offsetW, offsetH);
        *(texIdOfShs + shNumber) = selfTexId;
        *(texOffsets + shNumber) = selfTexIdOffset;
        shNumber++;
    }

    if (isRDinside == 1) {
        uint8_t offsetW = (uint8_t)floorf(texW * (oriRD.x));
        uint8_t offsetH = (uint8_t)floorf(texH * (oriRD.y));

        glm::vec3 shWorldPosition;
        shWorldPosition.x = *(shPositionMapCpct + (*(shPosMapMemLayout + selfTexIdOffset)) + posMapOffsetRD * 3 + 0);
        shWorldPosition.y = *(shPositionMapCpct + (*(shPosMapMemLayout + selfTexIdOffset)) + posMapOffsetRD * 3 + 1);
        shWorldPosition.z = *(shPositionMapCpct + (*(shPosMapMemLayout + selfTexIdOffset)) + posMapOffsetRD * 3 + 2);

        float curDist = glm::length(shWorldPosition - worldPosition);
        
        *(lerpCoeffsOfShs + shNumber) = (1.0f / (powf(curDist * 1.0f, invDistFactorEdge))) + 1e-6f;
        // *(offsetOfW + shNumber) = offsetW;
        // *(offsetOfH + shNumber) = offsetH;
        Write2CompactWH(offsetOfWH, shNumber, offsetW, offsetH);
        *(texIdOfShs + shNumber) = selfTexId;
        *(texOffsets + shNumber) = selfTexIdOffset;
        shNumber++;
    }

    for (uint8_t i = 0; i < shNumber; i++) {
        assert((*(texIdOfShs + i)) > 0);
        assert((*(lerpCoeffsOfShs + i)) != 0.f);
    }

}

__device__ void CalcLerpCoeffsShared(MeshEdgeNbrInfo* meshEdgeNbrInfo,
    MeshVertNbrInfo* meshVertNbrInfo,
    uint32_t selfTexId, uint32_t selfOffsetInHash,
    const float2 oriLU, const float2 oriRU, const float2 oriLD, const float2 oriRD,
    float2 cornerAreaInfoShared,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
    glm::vec3 worldPosition,
    float2 texCoord, float2 oriLerpCoeff,
    float* botYCoefCurBuf,
    float* topYCoefCurBuf,
    uint8_t selfTexW, uint8_t selfTexH,
    uint8_t& shNumber, uint32_t* texIdOfShs,
    uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint32_t* texOffsets,
    float invDistFactorEdge, float invDistFactorCorner, CurTexAlignedWH* curTexAllWH, float* curTexNormal, float* curTexNormals,
    bool print)
{
    TexAlignedPosWH* validMapStartAddrSelf = (validShWHMap + (*(validShWHMapMemLayout + selfOffsetInHash)));
    uint32_t validShNumThisTexSelf = (*(validShNumsAll + selfOffsetInHash));

    VertNbrType vertNbrType = IsInCornerShared(texCoord, cornerAreaInfoShared, botYCoefCurBuf, topYCoefCurBuf, selfOffsetInHash);
    if (vertNbrType < VERT_NBR_TYPE_NUM) {
        if (print == true) {
            printf("into corner CalcInverseDistanceWeightsCornerShared \n");
        }
        CalcInverseDistanceWeightsCornerShared(vertNbrType,
            meshVertNbrInfo,
            selfTexId, selfOffsetInHash,
            worldPosition,
            shPositionMapCpct, shPosMapMemLayout,
            // shValidMapCpct, shValidMapMemLayout,
            validShWHMap, validShWHMapMemLayout, validShNumsAll,
            selfTexW, selfTexH,
            shNumber, texIdOfShs,
            offsetOfWH, lerpCoeffsOfShs, texOffsets, invDistFactorCorner, curTexAllWH, curTexNormal, curTexNormals);

        // CalcMeanValueCoordShared(texOffsets, offsetOfWH, lerpCoeffsOfShs, shNumber, worldPosition, shPositionMapCpct, shPosMapMemLayout, invDistFactorCorner, curTexAllWH);
    } else {
        int32_t posMapOffsetLU = 0;
        int32_t posMapOffsetRU = 0;
        int32_t posMapOffsetLD = 0;
        int32_t posMapOffsetRD = 0;

        uint8_t isLUinside = IfCurShInsideShared(oriLU, validMapStartAddrSelf, validShNumThisTexSelf, selfTexW, selfTexH, posMapOffsetLU);
        uint8_t isRUinside = IfCurShInsideShared(oriRU, validMapStartAddrSelf, validShNumThisTexSelf, selfTexW, selfTexH, posMapOffsetRU);
        uint8_t isLDinside = IfCurShInsideShared(oriLD, validMapStartAddrSelf, validShNumThisTexSelf, selfTexW, selfTexH, posMapOffsetLD);
        uint8_t isRDinside = IfCurShInsideShared(oriRD, validMapStartAddrSelf, validShNumThisTexSelf, selfTexW, selfTexH, posMapOffsetRD);

        assert(isLUinside <= 1);
        assert(isRUinside <= 1);
        assert(isLDinside <= 1);
        assert(isRDinside <= 1);


        if ((isLUinside & isRUinside & isLDinside & isRDinside) == 1) {
            if (print == true) {
                printf("into inside CalcBiLinearWeightsShared \n");
            }
            CalcBiLinearWeightsShared(oriLerpCoeff, oriLU, oriRU, oriLD, oriRD, offsetOfWH, lerpCoeffsOfShs, selfTexW, selfTexH, shNumber, texIdOfShs, selfTexId, texOffsets, selfOffsetInHash);
        } else {
            if (print == true) {
                printf("into edge CalcInsideShInverseDistanceWeightShared \n");
            }

            EdgeTypeCUDA edgeNbrType = EDGE_TYPE_CUDA_MAX;
            if (isLUinside == 0) {
                edgeNbrType = EDGE_TYPE_CUDA_LEFT;
            } else if (isRUinside == 0) {
                edgeNbrType = EDGE_TYPE_CUDA_RIGHT;
            } else if ((isLDinside == 0) && (isRDinside == 0)) {
                edgeNbrType = EDGE_TYPE_CUDA_BOTTOM;
            }

            uint8_t needFindNum = 1;

            // if (IsInSamePlane(curTexNormal, (curTexNormals + meshEdgeNbrInfo->offsetInHash[edgeNbrType] * 3)) == false) {
            //     return;
            // }

            if (meshEdgeNbrInfo->nbrTexId[edgeNbrType] != 0) {
                if (IsInSamePlane(curTexNormal, (curTexNormals + meshEdgeNbrInfo->offsetInHash[edgeNbrType] * 3)) == true) {
                    CalcInverseDistanceWeightsShared(meshEdgeNbrInfo->offsetInHash[edgeNbrType],
                        meshEdgeNbrInfo->nbrTexId[edgeNbrType],
                        edgeNbrType,
                        shPositionMapCpct, shPosMapMemLayout,
                        worldPosition, invDistFactorEdge,
                        needFindNum,
                        // shValidMapCpct, shValidMapMemLayout,
                        validShWHMap, validShWHMapMemLayout, validShNumsAll,
                        (uint8_t)((curTexAllWH + meshEdgeNbrInfo->offsetInHash[edgeNbrType])->data.curTexW),
                        (uint8_t)((curTexAllWH + meshEdgeNbrInfo->offsetInHash[edgeNbrType])->data.curTexH),
                        shNumber, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, texOffsets);
                }
            }

            CalcInsideShInverseDistanceWeightShared(selfTexId, selfOffsetInHash, isLUinside, isRUinside, isLDinside, isRDinside, offsetOfWH, lerpCoeffsOfShs, oriLU, oriRU, oriLD, oriRD, selfTexW, selfTexH, shPositionMapCpct, shPosMapMemLayout, worldPosition, invDistFactorEdge, shNumber, texIdOfShs, texOffsets, posMapOffsetLU, posMapOffsetRU, posMapOffsetLD, posMapOffsetRD);

            // CalcMeanValueCoordShared(texOffsets, offsetOfWH, lerpCoeffsOfShs, shNumber, worldPosition, shPositionMapCpct, shPosMapMemLayout, invDistFactorEdge, curTexAllWH);
        }
    }
}

__device__ uint32_t ReadPixLocation(PixLocation* compactPixLocBuf, PixLocation* pixLocations, uint32_t& validPixNum, uint32_t offset, uint32_t pixNumTotal, uint32_t readPos)
{
    if (readPos >= pixNumTotal) {
        validPixNum = 0;
        return pixNumTotal;
    }

    for (uint32_t i = 0; i < MESH_HANDLE_THREAD_NUM; i++) {
        if ((readPos + i) >= pixNumTotal) {
            validPixNum = i;
            return (readPos + i);
        }

        *(pixLocations + i) = *(compactPixLocBuf + offset + readPos + i);
    }

    validPixNum = MESH_HANDLE_THREAD_NUM;
    return (readPos + MESH_HANDLE_THREAD_NUM);
}


template <typename Map>
__device__ void ReadMeshData(Map texHash, uint32_t* errCnt, uint32_t blockId, uint32_t* edgeNbrInfo, uint32_t* vertNbrInfo, uint32_t curTexId, MeshEdgeNbrInfo& meshEdgeNbrInfo, MeshVertNbrInfo& meshVertNbrInfo, uint32_t& curTexOffset, float* cornerAreaInfo, float2& cornerAreaInfoShared, CurTexAlignedWH* curTexAllWH, uint32_t& curTexW, uint32_t& curTexH, float* curTexDensities, float& curTexDensity, float* curTexNormals, float* curTexNormal)
{
    curTexOffset = GetTexOffsetFromHash(texHash, curTexId, errCnt);

    *(curTexNormal + 0) = (*(curTexNormals + curTexOffset * 3 + 0));
    *(curTexNormal + 1) = (*(curTexNormals + curTexOffset * 3 + 1));
    *(curTexNormal + 2) = (*(curTexNormals + curTexOffset * 3 + 2));

    curTexDensity = *(curTexDensities + curTexOffset);

    curTexW = (curTexAllWH + curTexOffset)->data.curTexW;
    curTexH = (curTexAllWH + curTexOffset)->data.curTexH;

    for (uint32_t i = 0; i < VERT_NBR_TYPE_NUM; i++) {
        GetVertNbrInfoFromHash(texHash, vertNbrInfo, curTexId, errCnt, meshVertNbrInfo.nbrTexIds[i], meshVertNbrInfo.nbrNum[i], (VertNbrType)i);

        for (uint32_t ii = 0; ii < meshVertNbrInfo.nbrNum[i]; ii++) {
            meshVertNbrInfo.nbrTexOffsetsInHash[i][ii] =
                GetTexOffsetFromHash(texHash, meshVertNbrInfo.nbrTexIds[i][ii], errCnt);
        }
    }

    for (uint32_t i = 0; i < EDGE_TYPE_CUDA_MAX; i++) {
        meshEdgeNbrInfo.nbrTexId[i] = *(edgeNbrInfo + curTexOffset * 3 * 2 + i * 2 + 0);
        meshEdgeNbrInfo.nbrEdgeType[i] = (EdgeTypeCUDA)(*(edgeNbrInfo + curTexOffset * 3 * 2 + i * 2 + 1));

        if (meshEdgeNbrInfo.nbrTexId[i] == 0) {
            continue;
        }
        meshEdgeNbrInfo.offsetInHash[i] = GetTexOffsetFromHash(texHash, meshEdgeNbrInfo.nbrTexId[i], errCnt);
    }

    float2 tmp = *(reinterpret_cast<float2 *>(cornerAreaInfo + curTexOffset * 2));
    cornerAreaInfoShared.x = tmp.x;
    cornerAreaInfoShared.y = tmp.y;
}


__device__ void calcShBasisesOrder2(float viewDirX, float viewDirY, float viewDirZ, float* basises, unsigned char shOrder)
{
    // __half_raw SH_C2_half_0 = {0x3C5FU};
    // __half_raw SH_C2_half_1 = {0xBC5FU};
    // __half_raw SH_C2_half_2 = {0x350CU};
    // __half_raw SH_C2_half_3 = {0xBC5FU};
    // __half_raw SH_C2_half_4 = {0x385FU};

    // __half_raw half_2e0 = {0x4000U};

    float xx = viewDirX * viewDirX;
    float yy = viewDirY * viewDirY;
    float zz = viewDirZ * viewDirZ;
    float xy = viewDirX * viewDirY;
    float yz = viewDirY * viewDirZ;
    float xz = viewDirX * viewDirZ;

    switch (shOrder) {
        case 2:
            basises[12] = SH_C2[0] * xy;
            basises[13] = SH_C2[0] * xy;
            basises[14] = SH_C2[0] * xy;

            basises[15] = SH_C2[1] * yz;
            basises[16] = SH_C2[1] * yz;
            basises[17] = SH_C2[1] * yz;

            basises[18] = SH_C2[2] * (2.0f * zz - xx - yy);
            basises[19] = SH_C2[2] * (2.0f * zz - xx - yy);
            basises[20] = SH_C2[2] * (2.0f * zz - xx - yy);

            basises[21] = SH_C2[3] * xz;
            basises[22] = SH_C2[3] * xz;
            basises[23] = SH_C2[3] * xz;

            basises[24] = SH_C2[4] * (xx - yy);
            basises[25] = SH_C2[4] * (xx - yy);
            basises[26] = SH_C2[4] * (xx - yy);
        default:
            break;
    }
}

__device__ void calcShBasises(float viewDirX, float viewDirY, float viewDirZ, float* basises, unsigned char shOrder)
{
    // __half_raw SH_C0_half = {0x3483U};
    // __half_raw SH_C1_half = {0x37D1U};

    basises[0] = SH_C0;
    basises[1] = SH_C0;
    basises[2] = SH_C0;

    switch (shOrder) {
        case 2:
        case 1:
            basises[3] = -SH_C1 * viewDirY;
            basises[4] = -SH_C1 * viewDirY;
            basises[5] = -SH_C1 * viewDirY;

            basises[6] = SH_C1 * viewDirZ;
            basises[7] = SH_C1 * viewDirZ;
            basises[8] = SH_C1 * viewDirZ;

            basises[9] = -SH_C1 * viewDirX;
            basises[10] = -SH_C1 * viewDirX;
            basises[11] = -SH_C1 * viewDirX;
        default:
            break;
    };
}

__device__ void RetrieveOriShShared(uint32_t texOffset, float* shCoeffTexture, uint32_t* shTexturesMemLayout, unsigned char layer, unsigned int offsetPerSh, int32_t innieOffset, float* valOut)
{
    // *(valOut + 0) = (*(shCoeffTexture + (*(shTexturesMemLayout + texOffset)) + offsetPerLayer * layer + offsetPerLine * (uint32_t)offsetH + (uint32_t)offsetW * SH_TEXTURE_CHANNEL_NUM));

    // *(valOut + 1) = (*(shCoeffTexture + (*(shTexturesMemLayout + texOffset)) + offsetPerLayer * layer + offsetPerLine * (uint32_t)offsetH + (uint32_t)offsetW * SH_TEXTURE_CHANNEL_NUM + 1));

    // *(valOut + 2) = (*(shCoeffTexture + (*(shTexturesMemLayout + texOffset)) + offsetPerLayer * layer + offsetPerLine * (uint32_t)offsetH + (uint32_t)offsetW * SH_TEXTURE_CHANNEL_NUM + 2));

    *(valOut + 0) = (*(shCoeffTexture + (*(shTexturesMemLayout + texOffset)) + innieOffset * offsetPerSh + layer * SH_TEXTURE_CHANNEL_NUM + 0));

    *(valOut + 1) = (*(shCoeffTexture + (*(shTexturesMemLayout + texOffset)) + innieOffset * offsetPerSh + layer * SH_TEXTURE_CHANNEL_NUM + 1));

    *(valOut + 2) = (*(shCoeffTexture + (*(shTexturesMemLayout + texOffset)) + innieOffset * offsetPerSh + layer * SH_TEXTURE_CHANNEL_NUM + 2));
}

__device__ void TraverseAllShShared(uint32_t* texOffsets, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint8_t shNumber, float* shCoeffTexture, uint32_t* shTexturesMemLayout, unsigned char layer, float* lerpShR, float* lerpShG, float* lerpShB, CurTexAlignedWH* curTexAllWH, TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll, uint32_t shTextureLayerNumAll)
{
    float coeffSum = 0.f;
    for (uint8_t i = 0; i < shNumber; i++) {
        if (isnan((*(lerpCoeffsOfShs + i))) == 1 ||
            isinf((*(lerpCoeffsOfShs + i))) == 1) {
            continue;
        }

        coeffSum += lerpCoeffsOfShs[i];
    }

    for (uint8_t i = 0; i < shNumber; i++) {
        if (isnan((*(lerpCoeffsOfShs + i))) == 1 ||
            isinf((*(lerpCoeffsOfShs + i))) == 1) {
            continue;
        }

        float ShCoeffs[3];
        uint32_t offsetW, offsetH;

        ReadFromCompactWH(offsetOfWH, i, offsetW, offsetH);
        TexAlignedPosWH* curValidMapStartAddr = (validShWHMap + (*(validShWHMapMemLayout + texOffsets[i])));
        uint32_t curValidShNum = (*(validShNumsAll + texOffsets[i]));
        int32_t innieOffset = FindOffsetInValidShMap(curValidMapStartAddr, curValidShNum, offsetW, offsetH);

        // uint32_t offsetPerLine = (uint32_t)((curTexAllWH + texOffsets[i])->data.curTexW) * SH_TEXTURE_CHANNEL_NUM;
        // uint32_t offsetPerLayer = (uint32_t)((curTexAllWH + texOffsets[i])->data.curTexH) * offsetPerLine;

        uint32_t offsetPerSh = SH_TEXTURE_CHANNEL_NUM * shTextureLayerNumAll;

        // RetrieveOriShShared(texOffsets[i], shCoeffTexture, shTexturesMemLayout, layer, offsetPerLayer, offsetPerLine, offsetH, offsetW, ShCoeffs);

        RetrieveOriShShared(texOffsets[i], shCoeffTexture, shTexturesMemLayout, layer, offsetPerSh, innieOffset, ShCoeffs);
        // if ((needPrintX == (blockIdx.x * blockDim.x + threadIdx.x)) &&
        //     (needPrintY == (blockIdx.y * blockDim.y + threadIdx.y))) {
        //     printf("read texId : %u, W H : %u, %u, sh val: %f %f %f\n", texIdOfShs[i], offsetOfW[i], offsetOfH[i], ShCoeffs[0], ShCoeffs[1], ShCoeffs[2]);
        // }

        *lerpShR = (*lerpShR) + lerpCoeffsOfShs[i] / coeffSum * ShCoeffs[0];
        *lerpShG = (*lerpShG) + lerpCoeffsOfShs[i] / coeffSum * ShCoeffs[1];
        *lerpShB = (*lerpShB) + lerpCoeffsOfShs[i] / coeffSum * ShCoeffs[2];
    }
}

__device__ void CalcColorShared(float* outColorR, float* outColorG, float* outColorB,
    float* shCoeffTexture, uint32_t* shTexturesMemLayout, CurTexAlignedWH* curTexAllWH,
    unsigned char shTextureLayerNumAll,
    float viewDirX, float viewDirY, float viewDirZ,
    unsigned char shOrder, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint32_t* texOffsets, uint8_t shNumber,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll)
{
    float basises[27];

    calcShBasisesOrder2(viewDirX, viewDirY, viewDirZ, basises, shOrder);
    calcShBasises(viewDirX, viewDirY, viewDirZ, basises, shOrder);

    for (unsigned char cnt = 0; cnt < shTextureLayerNumAll; cnt++) { // including RGB 3 channel
        float lerpedShCoeffR{0.f}, lerpedShCoeffG{0.f}, lerpedShCoeffB{0.f};

        TraverseAllShShared(texOffsets, offsetOfWH, lerpCoeffsOfShs, shNumber, shCoeffTexture, shTexturesMemLayout, cnt, &lerpedShCoeffR, &lerpedShCoeffG, &lerpedShCoeffB, curTexAllWH, validShWHMap, validShWHMapMemLayout, validShNumsAll, shTextureLayerNumAll);

        basises[cnt * 3]     = basises[cnt * 3]     *  lerpedShCoeffR;
        basises[cnt * 3 + 1] = basises[cnt * 3 + 1] *  lerpedShCoeffG;
        basises[cnt * 3 + 2] = basises[cnt * 3 + 2] *  lerpedShCoeffB;

        *outColorR = (*outColorR) + basises[cnt * 3];
        *outColorG = (*outColorG) + basises[cnt * 3 + 1];
        *outColorB = (*outColorB) + basises[cnt * 3 + 2];
    }
}

__device__ void PrintPixInfo(uint32_t selfTexId, uint32_t selfTexIdOffset, float2 cornerAreaInfo, float2 texCoord, float* botYCoefCurBuf, float* topYCoefCurBuf, const float2 oriLU, const float2 oriRU, const float2 oriLD, const float2 oriRD, uint8_t selfTexW, uint8_t selfTexH,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    MeshEdgeNbrInfo* meshEdgeNbrInfo, MeshVertNbrInfo* meshVertNbrInfo, CurTexAlignedWH* curTexAllWH, uint32_t pixX, uint32_t pixY)
{
    // uint8_t isLUinside = IfCurShInside(texHash, oriLU, shValidMap, texId, errCnt, texW, texH);
    // uint8_t isRUinside = IfCurShInside(texHash, oriRU, shValidMap, texId, errCnt, texW, texH);
    // uint8_t isLDinside = IfCurShInside(texHash, oriLD, shValidMap, texId, errCnt, texW, texH);
    // uint8_t isRDinside = IfCurShInside(texHash, oriRD, shValidMap, texId, errCnt, texW, texH);

    printf("selfTexId: %d, pixloc: %d, %d, texCoord: %f, %f\n", selfTexId, pixX, pixY, texCoord.x, texCoord.y);

    printf("cur texWH: %d, %d\n", (curTexAllWH + selfTexIdOffset)->data.curTexW, (curTexAllWH + selfTexIdOffset)->data.curTexH);

    TexAlignedPosWH* validMapStartAddr = (validShWHMap + validShWHMapMemLayout[selfTexIdOffset]);
    uint32_t validShNumThisTex = (*(validShNumsAll + selfTexIdOffset));

    int32_t dummy;
    uint8_t isLUinside = IfCurShInsideShared(oriLU, validMapStartAddr, validShNumThisTex, selfTexW, selfTexH, dummy);
    uint8_t isRUinside = IfCurShInsideShared(oriRU, validMapStartAddr, validShNumThisTex, selfTexW, selfTexH, dummy);
    uint8_t isLDinside = IfCurShInsideShared(oriLD, validMapStartAddr, validShNumThisTex, selfTexW, selfTexH, dummy);
    uint8_t isRDinside = IfCurShInsideShared(oriRD, validMapStartAddr, validShNumThisTex, selfTexW, selfTexH, dummy);

    EdgeTypeCUDA edgeNbrType = EDGE_TYPE_CUDA_MAX;
        if (isLUinside == 0) {
            edgeNbrType = EDGE_TYPE_CUDA_LEFT;
        } else if (isRUinside == 0) {
            edgeNbrType = EDGE_TYPE_CUDA_RIGHT;
        } else if ((isLDinside == 0) && (isRDinside == 0)) {
            edgeNbrType = EDGE_TYPE_CUDA_BOTTOM;
        }

    printf("cur on edge: %d, LU : %d, RU: %d, LD: %d, RD: %d\n", (uint32_t)edgeNbrType, isLUinside, isRUinside, isLDinside, isRDinside);

    // for (uint8_t i = 0; i < (uint8_t)EDGE_TYPE_CUDA_MAX; i++) {
    //     uint32_t nbrTexId = *(texNbrInfo + selfTexIdOffset * 3 * 2 + i * 2 + 0);
    //     EdgeTypeEdgeMap nbrEdgeType = (EdgeTypeEdgeMap)(*(texNbrInfo + selfTexIdOffset * 3 * 2 + i * 2 + 1));
    //     printf("nbr type: %d, nbrTexId: %d, nbrEdgeType: %d\n", i, nbrTexId, (uint32_t)nbrEdgeType);
    // }

    for (uint8_t i = 0; i < (uint8_t)EDGE_TYPE_CUDA_MAX; i++) {
        uint32_t nbrTexId = meshEdgeNbrInfo->nbrTexId[i];
        EdgeTypeCUDA nbrEdgeType = meshEdgeNbrInfo->nbrEdgeType[i];
        printf("edge type change ! nbr type: %d, nbrTexId: %d, nbrEdgeType: %d\n", i, nbrTexId, (uint32_t)nbrEdgeType);
    }


    printf("vert nbrType %d\n", (uint32_t)IsInCornerShared(texCoord, cornerAreaInfo, botYCoefCurBuf, topYCoefCurBuf, selfTexIdOffset));

    for (uint8_t i = 0; i < (uint8_t)VERT_NBR_TYPE_NUM; i++) {

        printf("vert type : %d vert neighbors: \n", i);
        for (uint8_t ii = 0; ii < meshVertNbrInfo->nbrNum[i]; ii++) {
            printf(" %d ", meshVertNbrInfo->nbrTexIds[i][ii]);
        }

        printf("\n");

    }
}



__device__ void DoNeighborsEWA(uint32_t curTexId, uint32_t curTexOffset, glm::dvec3 ellipseCenterPos, uint8_t texW, uint8_t texH, glm::dvec3 ellipseSpaceAxisU, glm::dvec3 ellipseSpaceAxisV, float* shPositionMapCpct, uint32_t* shPosMapMemLayout, float4 ellipseAndLodLvl, uint8_t& shNumber, uint32_t* texIdOfShs, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint32_t* texOffsets, float gaussianCoeffEWA, TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll, float curTexDensity, uint8_t selfTexW, uint8_t selfTexH)
{
    uint32_t shPositionMapOffsetPerLine = texW * 3;
    uint32_t validShWhMapCurOffset = (*(validShWHMapMemLayout + curTexOffset));

    if (curTexId == 0) {
        return;
    }

    for (uint8_t curH = 0; curH < texH; curH++) {
        for (uint8_t curW = 0; curW < texW; curW++) {
            if (shNumber >= MAX_SH_NUMBER) {
                return;
            }

            int innerOffset = FindOffsetInValidShMap(validShWHMap + validShWhMapCurOffset, *(validShNumsAll + curTexOffset), curW, curH);
            if (innerOffset == -1) {
                continue;
            }
            // all texels are valid now
            // if ((*(shValidMapCpct + (*(shValidMapMemLayout + curTexOffset)) + curH * texW + curW)) == 0) {
            //     continue;
            // }

            // glm::dvec3 curShWorldPose(*(shPositionMapCpct + (*(shPosMapMemLayout + curTexOffset)) + curH * shPositionMapOffsetPerLine + curW * 3 + 0),
            //                           *(shPositionMapCpct + (*(shPosMapMemLayout + curTexOffset)) + curH * shPositionMapOffsetPerLine + curW * 3 + 1),
            //                           *(shPositionMapCpct + (*(shPosMapMemLayout + curTexOffset)) + curH * shPositionMapOffsetPerLine + curW * 3 + 2));


            glm::dvec3 curShWorldPose(*(shPositionMapCpct + (*(shPosMapMemLayout + curTexOffset)) + innerOffset * 3 + 0),
                                      *(shPositionMapCpct + (*(shPosMapMemLayout + curTexOffset)) + innerOffset * 3 + 1),
                                      *(shPositionMapCpct + (*(shPosMapMemLayout + curTexOffset)) + innerOffset * 3 + 2));

            
            glm::dvec3 worldVec = curShWorldPose - ellipseCenterPos;

            double proj2U = glm::dot(worldVec, ellipseSpaceAxisU);
            double proj2V = glm::dot(worldVec, ellipseSpaceAxisV);

            // double s = proj2U / (double)texScaleFactor.x;
            // double t = proj2V / (double)texScaleFactor.y;

            double s = (proj2U / (double)curTexDensity / (double)selfTexW);
            double t = (proj2V / (double)curTexDensity / (double)selfTexH);

            double r2 = (double)(ellipseAndLodLvl.x) * s * s + (double)(ellipseAndLodLvl.y) * s * t + (double)(ellipseAndLodLvl.z) * t * t;
            if (r2 - 1.0 > 1e-5) { // outside of ellipse
                continue;
            }

            *(texIdOfShs + shNumber) = curTexId;
            Write2CompactWH(offsetOfWH, shNumber, curW, curH);

            *(lerpCoeffsOfShs + shNumber) = exp(-1.0f * gaussianCoeffEWA * r2) - exp(-1.0f * gaussianCoeffEWA);

            // use IDW instead of gaussian kernel
            // *(lerpCoeffsOfShs + shNumber) = (1.0f / (powf(glm::length(worldVec), 0.9f)) + 1e-6f);

            *(texOffsets + shNumber) = curTexOffset;
            shNumber++;
        }
    }
}


__device__ void DoSelfEWA(float2 oriTexCoord, uint32_t selfTexId, uint32_t selfTexOffset, uint8_t texW, uint8_t texH, float4 ellipseAndLodLvl, uint8_t& shNumber, uint32_t* texIdOfShs, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint32_t* texOffsets, float gaussianCoeffEWA, TexAlignedPosWH* validMapStartAddr, uint32_t validShNumThisTex, float curTexDensity)
{
    uint8_t ellipCenterU = floor(oriTexCoord.x * texW);
    uint8_t ellipCenterV = floor(oriTexCoord.y * texH);

    for (uint8_t curH = 0; curH < texH; curH++) {
        for (uint8_t curW = 0; curW < texW; curW++) {
            if (shNumber >= MAX_SH_NUMBER) {
                return;
            }

            
            // if (FindOffsetInValidShMap(validMapStartAddr, validShNumThisTex, curW, curH, print) == -1) {
            if (FindOffsetInValidShMap(validMapStartAddr, validShNumThisTex, curW, curH) == -1) {
                continue;
            }

            double s = ((double)(curW - ellipCenterU)) / (double)texW;
            double t = ((double)(curH - ellipCenterV)) / (double)texH;

            double r2 = (double)(ellipseAndLodLvl.x) * s * s + (double)(ellipseAndLodLvl.y) * s * t + (double)(ellipseAndLodLvl.z) * t * t;
            if (r2 - 1.0 > 1e-5) { // outside of ellipse
                continue;
            }

            // float texDistance = hypotf(float(curW - ellipCenterU), float(curH - ellipCenterV));
            // if (texDistance < 1e-5) {
            //     continue;
            // }

            *(texIdOfShs + shNumber) = selfTexId;
            Write2CompactWH(offsetOfWH, shNumber, curW, curH);

            *(lerpCoeffsOfShs + shNumber) = exp(-1.0f * gaussianCoeffEWA * r2) - exp(-1.0f * gaussianCoeffEWA);

            // use IDW instead of gaussian kernel
            // *(lerpCoeffsOfShs + shNumber) = (1.0f / powf((texDistance * curTexDensity), 0.9f) + 1e-6f);

            *(texOffsets + shNumber) = selfTexOffset;
            shNumber++;
        }
    }
}

__device__ void InCornerEWA(uint32_t selfTexOffset, VertNbrType vertNbrType, MeshVertNbrInfo* meshVertNbrInfo, glm::dvec3 ellipseCenterPos, float* texInWorldInfo, float* shPositionMapCpct, uint32_t* shPosMapMemLayout, float4 ellipseAndLodLvl, uint8_t& shNumber, uint32_t* texIdOfShs, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint32_t* texOffsets, float gaussianCoeffEWA, TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll, CurTexAlignedWH* curTexAllWH, float curTexDensity, float* curTexNormal, float* curTexNormals, uint8_t selfTexW, uint8_t selfTexH)
{
    glm::dvec3 ellipseSpaceAxisU((double)(*(texInWorldInfo + selfTexOffset * 6 + 0)),
                                 (double)(*(texInWorldInfo + selfTexOffset * 6 + 1)),
                                 (double)(*(texInWorldInfo + selfTexOffset * 6 + 2)));

    glm::dvec3 ellipseSpaceAxisV((double)(*(texInWorldInfo + selfTexOffset * 6 + 3)),
                                 (double)(*(texInWorldInfo + selfTexOffset * 6 + 4)),
                                 (double)(*(texInWorldInfo + selfTexOffset * 6 + 5)));

    for (uint32_t i = 0; i < meshVertNbrInfo->nbrNum[vertNbrType]; i++) {
        if (IsInSamePlane(curTexNormal, (curTexNormals + meshVertNbrInfo->nbrTexOffsetsInHash[vertNbrType][i] * 3)) == false) {
            continue;
        }

        DoNeighborsEWA(meshVertNbrInfo->nbrTexIds[vertNbrType][i], meshVertNbrInfo->nbrTexOffsetsInHash[vertNbrType][i],
            ellipseCenterPos,
            (uint8_t)((curTexAllWH + meshVertNbrInfo->nbrTexOffsetsInHash[vertNbrType][i])->data.curTexW),
            (uint8_t)((curTexAllWH + meshVertNbrInfo->nbrTexOffsetsInHash[vertNbrType][i])->data.curTexH),
            ellipseSpaceAxisU, ellipseSpaceAxisV, shPositionMapCpct, shPosMapMemLayout, ellipseAndLodLvl, shNumber, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, texOffsets, gaussianCoeffEWA, validShWHMap, validShWHMapMemLayout, validShNumsAll, curTexDensity, selfTexW, selfTexH);
    }
}

__device__ void InEdgeEWA(uint32_t selfTexOffset, EdgeTypeCUDA edgeNbrType, MeshEdgeNbrInfo* meshEdgeNbrInfo, glm::dvec3 ellipseCenterPos, float* texInWorldInfo, float* shPositionMapCpct, uint32_t* shPosMapMemLayout, float4 ellipseAndLodLvl, uint8_t& shNumber, uint32_t* texIdOfShs, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint32_t* texOffsets, float gaussianCoeffEWA, TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll, CurTexAlignedWH* curTexAllWH, float curTexDensity, float* curTexNormal, float* curTexNormals, uint8_t selfTexW, uint8_t selfTexH)
{
    glm::dvec3 ellipseSpaceAxisU((double)(*(texInWorldInfo + selfTexOffset * 6 + 0)),
                                 (double)(*(texInWorldInfo + selfTexOffset * 6 + 1)),
                                 (double)(*(texInWorldInfo + selfTexOffset * 6 + 2)));

    glm::dvec3 ellipseSpaceAxisV((double)(*(texInWorldInfo + selfTexOffset * 6 + 3)),
                                 (double)(*(texInWorldInfo + selfTexOffset * 6 + 4)),
                                 (double)(*(texInWorldInfo + selfTexOffset * 6 + 5)));

    if (meshEdgeNbrInfo->nbrTexId[edgeNbrType] == 0) {
        return;
    }

    if (IsInSamePlane(curTexNormal, (curTexNormals + meshEdgeNbrInfo->offsetInHash[edgeNbrType] * 3)) == false) {
        return;
    }

    DoNeighborsEWA(meshEdgeNbrInfo->nbrTexId[edgeNbrType], meshEdgeNbrInfo->offsetInHash[edgeNbrType], ellipseCenterPos,
    (uint8_t)((curTexAllWH + meshEdgeNbrInfo->offsetInHash[edgeNbrType])->data.curTexW),
    (uint8_t)((curTexAllWH + meshEdgeNbrInfo->offsetInHash[edgeNbrType])->data.curTexH),
    ellipseSpaceAxisU, ellipseSpaceAxisV, shPositionMapCpct, shPosMapMemLayout, ellipseAndLodLvl, shNumber, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, texOffsets, gaussianCoeffEWA, validShWHMap, validShWHMapMemLayout, validShNumsAll, curTexDensity, selfTexW, selfTexH);
}

__device__ void CalcEWA(MeshVertNbrInfo* meshVertNbrInfo, MeshEdgeNbrInfo* meshEdgeNbrInfo, uint32_t selfTexId, uint32_t selfTexIdOffset, float4 ellipseAndLodLvl, uint8_t& shNumber, uint32_t* texIdOfShs, uint32_t* offsetOfWH, float* lerpCoeffsOfShs, uint32_t* texOffsets, uint8_t selfTexW, uint8_t selfTexH, float2 oriTexCoord, float gaussianCoeffEWA, float* texInWorldInfo, glm::vec3 worldPosition, float* shPositionMapCpct, uint32_t* shPosMapMemLayout, float2 cornerAreaInfoShared, float* botYCoefCurBuf, float* topYCoefCurBuf, const float2 oriLU, const float2 oriRU, const float2 oriLD, const float2 oriRD, TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll, CurTexAlignedWH* curTexAllWH, bool print, float curTexDensity, float* curTexNormal, float* curTexNormals)
{
    glm::dvec3 ellipseCenterPos(worldPosition.x, worldPosition.y, worldPosition.z);
    VertNbrType vertNbrType = IsInCornerShared(oriTexCoord, cornerAreaInfoShared, botYCoefCurBuf, topYCoefCurBuf, selfTexIdOffset);

    TexAlignedPosWH* validMapStartAddrSelf = (validShWHMap + (*(validShWHMapMemLayout + selfTexIdOffset)));
    uint32_t validShNumThisTexSelf = (*(validShNumsAll + selfTexIdOffset));

    if (vertNbrType < VERT_NBR_TYPE_NUM) {
        
        InCornerEWA(selfTexIdOffset, vertNbrType, meshVertNbrInfo, ellipseCenterPos, texInWorldInfo, shPositionMapCpct, shPosMapMemLayout, ellipseAndLodLvl, shNumber, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, texOffsets, gaussianCoeffEWA, validShWHMap, validShWHMapMemLayout, validShNumsAll, curTexAllWH, curTexDensity, curTexNormal, curTexNormals, selfTexW, selfTexH);

        // if (print == true) {
        //     printf("into corner EWA, get SH num: %d\n", shNumber);
            
        //     for (uint32_t i = 0; i < shNumber; i++) {
        //         uint32_t readW, readH;
        //         ReadFromCompactWH(offsetOfWH, i, readW, readH);
        //         printf("texIdOfShs : %d, texOffsets: %d, lerpCoeffsOfShs: %f, WH: %d, %d\n", texIdOfShs[i], texOffsets[i], lerpCoeffsOfShs[i], readW, readH);
        //     }
        // }
        
    } else {
        int32_t dummy;
        uint8_t isLUinside = IfCurShInsideShared(oriLU, validMapStartAddrSelf, validShNumThisTexSelf, selfTexW, selfTexH, dummy);
        uint8_t isRUinside = IfCurShInsideShared(oriRU, validMapStartAddrSelf, validShNumThisTexSelf, selfTexW, selfTexH, dummy);
        uint8_t isLDinside = IfCurShInsideShared(oriLD, validMapStartAddrSelf, validShNumThisTexSelf, selfTexW, selfTexH, dummy);
        uint8_t isRDinside = IfCurShInsideShared(oriRD, validMapStartAddrSelf, validShNumThisTexSelf, selfTexW, selfTexH, dummy);

        assert(isLUinside <= 1);
        assert(isRUinside <= 1);
        assert(isLDinside <= 1);
        assert(isRDinside <= 1);

        if ((isLUinside & isRUinside & isLDinside & isRDinside) != 1) {
            EdgeTypeCUDA edgeNbrType = EDGE_TYPE_CUDA_MAX;
            if (isLUinside == 0) {
                edgeNbrType = EDGE_TYPE_CUDA_LEFT;
            } else if (isRUinside == 0) {
                edgeNbrType = EDGE_TYPE_CUDA_RIGHT;
            } else if ((isLDinside == 0) && (isRDinside == 0)) {
                edgeNbrType = EDGE_TYPE_CUDA_BOTTOM;
            }

            // if (print == true) {
            //     printf("into edge EWA");
            // }

            InEdgeEWA(selfTexIdOffset, edgeNbrType, meshEdgeNbrInfo, ellipseCenterPos, texInWorldInfo, shPositionMapCpct, shPosMapMemLayout, ellipseAndLodLvl, shNumber, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, texOffsets, gaussianCoeffEWA, validShWHMap, validShWHMapMemLayout, validShNumsAll, curTexAllWH, curTexDensity, curTexNormal, curTexNormals, selfTexW, selfTexH);
        }

    }

    // always need to do self EWA
    DoSelfEWA(oriTexCoord, selfTexId, selfTexIdOffset, selfTexW, selfTexH, ellipseAndLodLvl, shNumber, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, texOffsets, gaussianCoeffEWA, validMapStartAddrSelf, validShNumThisTexSelf, curTexDensity);

    // if (print == true) {
    //     printf("out EWA, get SH num: %d, validSH num: %d\n", shNumber, validShNumThisTexSelf);
        
    //     for (uint32_t i = 0; i < shNumber; i++) {
    //         uint32_t readW, readH;
    //         ReadFromCompactWH(offsetOfWH, i, readW, readH);
    //         printf("texIdOfShs : %d, texOffsets: %d, lerpCoeffsOfShs: %f, WH: %d, %d\n", texIdOfShs[i], texOffsets[i], lerpCoeffsOfShs[i], readW, readH);
    //     }
    // }


    // uint32_t shPositionMapOffsetPerTex = texH * texW * 3;
    // uint32_t shPositionMapOffsetPerLine = texW * 3;
    // uint8_t ellipCenterU = floor(oriTexCoord.x * texW);
    // uint8_t ellipCenterV = floor(oriTexCoord.y * texH);

    // for (uint8_t curH = 0; curH < texH; curH++) {
    //     for (uint8_t curW = 0; curW < texW; curW++) {
    //         if (shNumber >= MAX_SH_NUMBER) {
    //             return;
    //         }

    //         float s = ((float)(curW - ellipCenterU)) / (float)texW;
    //         float t = ((float)(curH - ellipCenterV)) / (float)texH;

    //         float r2 = ellipseAndLodLvl.x * s * s + ellipseAndLodLvl.y * s * t + ellipseAndLodLvl.z * t * t;
    //         if (r2 - 1.0f > 1e-5) { // outside of ellipse
    //             continue;
    //         }

    //         *(texIdOfShs + shNumber) = selfTexId;
    //         Write2CompactWH(offsetOfWH, shNumber, curW, curH);
    //         *(lerpCoeffsOfShs + shNumber) = exp(-1.0f * gaussianCoeffEWA * r2) - exp(-1.0f * gaussianCoeffEWA);
    //         *(texOffsets + shNumber) = selfTexIdOffset;
    //         shNumber++;
    //     }
    // }

}

// __device__ void CalcRGB(float* shTextures, uint32_t* shTexturesMemLayout,
//     float* outImg, unsigned char* maskImg,
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
//     cudaTextureObject_t texEllipseAndLodlvl,
//     cudaTextureObject_t texTexScaleFactorDepthTexId,
//     float2 cornerAreaInfoShared,
//     uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
//     float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
//     float* botYCoefCurBuf,
//     float* topYCoefCurBuf,
//     float invDistFactorEdge, float invDistFactorCorner,
//     unsigned char shLayerNum, unsigned char shOrder,
//     float gaussianCoeffEWA, float* texInWorldInfo,
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

//     float4 ellipseAndLodLvl = tex2D<float4>(texEllipseAndLodlvl, (float)pixX, (float)(imgH - pixY - 1));

//     float4 texScaleFactorDepthTexId = tex2D<float4>(texTexScaleFactorDepthTexId, (float)pixX, (float)(imgH - pixY - 1));
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

//     bool print = false;
//     // if (needPrintX == pixX && needPrintY == pixY) {
//     //     print = true;
//     // }

//     if (((uint32_t)(ellipseAndLodLvl.w)) >= 1) {
//         CalcEWA(meshVertNbrInfo, meshEdgeNbrInfo, selfTexId, selfOffsetInHash, ellipseAndLodLvl, shNumber, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, texOffsets, (uint8_t)selfTexW, (uint8_t)selfTexH, oriTexCoord, gaussianCoeffEWA, texInWorldInfo, worldPosition, shPositionMapCpct, shPosMapMemLayout, cornerAreaInfoShared, botYCoefCurBuf, topYCoefCurBuf, oriLU, oriRU, oriLD, oriRD, shValidMapCpct, shValidMapMemLayout, curTexAllWH, print, curTexDensity);
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
//             print);
//     }   


//     float red{0.f}, green{0.f}, blue{0.f};

//     // if (needPrintX == pixX && needPrintY == pixY) {
//     //     PrintPixInfo(selfTexId, selfOffsetInHash, cornerAreaInfoShared, oriTexCoord, botYCoefCurBuf, topYCoefCurBuf, oriLU, oriRU, oriLD, oriRD, selfTexW, selfTexH, shValidMapCpct, shValidMapMemLayout, meshEdgeNbrInfo, meshVertNbrInfo, curTexAllWH, needPrintX, needPrintY);

//     //     printf("get SH num: %d \n", shNumber);
    
//     //     for (uint32_t i = 0; i < shNumber; i++) {
//     //         uint32_t readW, readH;
//     //         ReadFromCompactWH(offsetOfWH, i, readW, readH);
//     //         printf("texIdOfShs : %d, texOffsets: %d, lerpCoeffsOfShs: %f, WH: %d, %d\n", texIdOfShs[i], texOffsets[i], lerpCoeffsOfShs[i], readW, readH);
//     //     }
//     // }
    

//     CalcColorShared(&red, &green, &blue,
//     shTextures, shTexturesMemLayout, curTexAllWH,
//     shLayerNum,
//     viewDirAndNull.x, viewDirAndNull.y, viewDirAndNull.z,
//     shOrder, offsetOfWH, lerpCoeffsOfShs, texOffsets, shNumber);
    

//     // *(outImg + (imgH - pixY - 1) * offsetPerPixelLine + pixX * OFFSET_PER_PIX_CHANNEL) = __int2float_rn(__float2int_rn(__saturatef(__half2float(red)) * 255.0f));
//     // *(outImg + (imgH - pixY - 1) * offsetPerPixelLine + pixX * OFFSET_PER_PIX_CHANNEL + 1) = __int2float_rn(__float2int_rn(__saturatef(__half2float(green)) * 255.0f));
//     // *(outImg + (imgH - pixY - 1) * offsetPerPixelLine + pixX * OFFSET_PER_PIX_CHANNEL + 2) = __int2float_rn(__float2int_rn(__saturatef(__half2float(blue)) * 255.0f));

//     *(outImg + 0 * offsetPerPixelChannel + (imgH - pixY - 1) * offsetPerPixelLine + pixX) = __int2float_rn(__float2int_rn(__saturatef(__half2float(red)) * 255.0f));
//     *(outImg + 1 * offsetPerPixelChannel + (imgH - pixY - 1) * offsetPerPixelLine + pixX) = __int2float_rn(__float2int_rn(__saturatef(__half2float(green)) * 255.0f));
//     *(outImg + 2 * offsetPerPixelChannel + (imgH - pixY - 1) * offsetPerPixelLine + pixX) = __int2float_rn(__float2int_rn(__saturatef(__half2float(blue)) * 255.0f));



//     *(maskImg + (imgH - pixY - 1) * imgW + pixX) = 1;
// }

// xy origin is LU, tex origiin is LU
__device__ __inline__ float4 ReadTexPtr(float* texBuf, uint32_t x, uint32_t y, uint32_t offsetPerTexLine)
{
    return *((float4 *)(texBuf + y * offsetPerTexLine + x * 4));
}

__device__ void CalcRGBTexPtr(float* shTextures, uint32_t* shTexturesMemLayout,
    float* outImg, unsigned char* maskImg,
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
    float* texEllipseAndLodlvl,
    float* texTexScaleFactorDepthTexId,
    float2 cornerAreaInfoShared,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
    float* botYCoefCurBuf,
    float* topYCoefCurBuf,
    float invDistFactorEdge, float invDistFactorCorner,
    unsigned char shLayerNum, unsigned char shOrder,
    float gaussianCoeffEWA, float* texInWorldInfo,
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

    // float4 ellipseAndLodLvl = tex2D<float4>(texEllipseAndLodlvl, (float)pixX, (float)(imgH - pixY - 1));
    float4 ellipseAndLodLvl = ReadTexPtr(texEllipseAndLodlvl, pixX, (imgH - pixY - 1), offsetPerTexLine);

    // float4 texScaleFactorDepthTexId = tex2D<float4>(texTexScaleFactorDepthTexId, (float)pixX, (float)(imgH - pixY - 1));
    // float4 texScaleFactorDepthTexId = ReadTexPtr(texTexScaleFactorDepthTexId, pixX, (imgH - pixY - 1), offsetPerTexLine);
    float curDepth = (ReadTexPtr(texTexScaleFactorDepthTexId, pixX, (imgH - pixY - 1), offsetPerTexLine)).z;
    // float2 texScaleFactor = make_float2(texScaleFactorDepthTexId.x, texScaleFactorDepthTexId.y);


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

    bool print = false;
    // if (needPrintX == pixX && needPrintY == pixY) {
    //     print = true;
    // }

    if ((((uint32_t)(ellipseAndLodLvl.w)) >= 1) && ((curDepth - 10.f) > 1e-5f)) {
        // if (needPrintX == pixX && needPrintY == pixY) {
        //     printf("into EWA\n");
        //     print = true;
        // }
        CalcEWA(meshVertNbrInfo, meshEdgeNbrInfo, selfTexId, selfOffsetInHash, ellipseAndLodLvl, shNumber, texIdOfShs, offsetOfWH, lerpCoeffsOfShs, texOffsets, (uint8_t)selfTexW, (uint8_t)selfTexH, oriTexCoord, gaussianCoeffEWA, texInWorldInfo, worldPosition, shPositionMapCpct, shPosMapMemLayout, cornerAreaInfoShared, botYCoefCurBuf, topYCoefCurBuf, oriLU, oriRU, oriLD, oriRD, validShWHMap, validShWHMapMemLayout, validShNumsAll, curTexAllWH, print, curTexDensity, curTexNormal, curTexNormals);
    } else
    {
        // if (needPrintX == pixX && needPrintY == pixY) {
        //     printf("into CalcLerpCoeffsShared\n");
        //     print = true;
        // }
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
            print);
    }   


    float red{0.f}, green{0.f}, blue{0.f};

    // if (needPrintX == pixX && needPrintY == pixY) {
    //     PrintPixInfo(selfTexId, selfOffsetInHash, cornerAreaInfoShared, oriTexCoord, botYCoefCurBuf, topYCoefCurBuf, oriLU, oriRU, oriLD, oriRD, selfTexW, selfTexH,
    //         validShWHMap, validShWHMapMemLayout, validShNumsAll,
    //         meshEdgeNbrInfo, meshVertNbrInfo, curTexAllWH, needPrintX, needPrintY);

    //     printf("get SH num: %d \n", shNumber);
    
    //     for (uint32_t i = 0; i < shNumber; i++) {
    //         uint32_t readW, readH;
    //         ReadFromCompactWH(offsetOfWH, i, readW, readH);
    //         printf("texIdOfShs : %d, texOffsets: %d, lerpCoeffsOfShs: %f, WH: %d, %d\n", texIdOfShs[i], texOffsets[i], lerpCoeffsOfShs[i], readW, readH);
    //     }
    // }
    

    CalcColorShared(&red, &green, &blue,
    shTextures, shTexturesMemLayout, curTexAllWH,
    shLayerNum,
    viewDirAndNull.x, viewDirAndNull.y, viewDirAndNull.z,
    shOrder, offsetOfWH, lerpCoeffsOfShs, texOffsets, shNumber,
    validShWHMap, validShWHMapMemLayout, validShNumsAll);
    

    // *(outImg + (imgH - pixY - 1) * offsetPerPixelLine + pixX * OFFSET_PER_PIX_CHANNEL) = __int2float_rn(__float2int_rn(__saturatef(__half2float(red)) * 255.0f));
    // *(outImg + (imgH - pixY - 1) * offsetPerPixelLine + pixX * OFFSET_PER_PIX_CHANNEL + 1) = __int2float_rn(__float2int_rn(__saturatef(__half2float(green)) * 255.0f));
    // *(outImg + (imgH - pixY - 1) * offsetPerPixelLine + pixX * OFFSET_PER_PIX_CHANNEL + 2) = __int2float_rn(__float2int_rn(__saturatef(__half2float(blue)) * 255.0f));

    *(outImg + 0 * offsetPerPixelChannel + (imgH - pixY - 1) * offsetPerPixelLine + pixX) = __int2float_rn(__float2int_rn(__saturatef(__half2float(red)) * 255.0f));
    *(outImg + 1 * offsetPerPixelChannel + (imgH - pixY - 1) * offsetPerPixelLine + pixX) = __int2float_rn(__float2int_rn(__saturatef(__half2float(green)) * 255.0f));
    *(outImg + 2 * offsetPerPixelChannel + (imgH - pixY - 1) * offsetPerPixelLine + pixX) = __int2float_rn(__float2int_rn(__saturatef(__half2float(blue)) * 255.0f));



    *(maskImg + (imgH - pixY - 1) * imgW + pixX) = 1;
}


// template <typename Map>
// __global__ void MeshLearnerForwardCUDA(uint32_t texNum,
//     float* shTextures, uint32_t* shTexturesMemLayout,
//     float* outImg, unsigned char* maskImg,
//     Map texHash, unsigned int* errCnt,
//     TexPixInfo* visibleTexIdInfo, PixLocation* compactPixLocBuf,
//     uint32_t imgW, uint32_t imgH,
//     cudaTextureObject_t texViewDirFrag2CamAndNull,
//     cudaTextureObject_t texLerpCoeffAndTexCoord,
//     cudaTextureObject_t texCoordShLUandRU,
//     cudaTextureObject_t texCoordShLDandRD,
//     cudaTextureObject_t texObjWorldPoseAndNull,
//     cudaTextureObject_t texEllipseAndLodlvl,
//     cudaTextureObject_t texTexScaleFactorDepthTexId,
//     float* cornerAreaInfo,
//     uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
//     float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
//     float* botYCoefCurBuf,
//     float* topYCoefCurBuf,
//     float invDistFactorEdge, float invDistFactorCorner,
//     unsigned char shLayerNum, unsigned char shOrder,
//     uint32_t* edgeNbrInfo, uint32_t* vertNbrInfo, float gaussianCoeffEWA, float* texInWorldInfo, CurTexAlignedWH* curTexAllWH,
//     float* curTexDensities,
//     uint32_t needPrintX, uint32_t needPrintY)
// {
//     // auto tid = blockIdx.x * blockDim.x + threadIdx.x;
//     auto blockId = blockIdx.x;
//     auto grp = cg::tiled_partition<MESH_HANDLE_THREAD_NUM>(cg::this_thread_block());

//     // maybe more than 32 pixels need to be handle, read in next round
//     __shared__ PixLocation pixLocations[MESH_HANDLE_THREAD_NUM];
//     __shared__ MeshEdgeNbrInfo meshEdgeNbrInfo;
//     __shared__ MeshVertNbrInfo meshVertNbrInfo;
//     __shared__ float2 cornerAreaInfoShared;
    

//     uint32_t numOfPixToHandle = 0;
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

//         // if (curTexId == 370998) {
//         //     printf("cuda handle tex 370998 ! numOfPixToHandle: %d \n", numOfPixToHandle);
//         // }
//     }

//     grp.sync();

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

//                 // if (curTexId == 370998) {
//                 //     printf("cuda handle tex 370998 !, XY: %d, %d\n", curLoc.x, curLoc.y);
//                 // }

//                 CalcRGB(shTextures, shTexturesMemLayout,
//                     outImg, maskImg,
//                     &meshEdgeNbrInfo, &meshVertNbrInfo,
//                     curTexId, curTexOffset,
//                     curLoc.x, curLoc.y,
//                     imgW, imgH, curTexW, curTexH,
//                     texViewDirFrag2CamAndNull,
//                     texLerpCoeffAndTexCoord,
//                     texCoordShLUandRU,
//                     texCoordShLDandRD,
//                     texObjWorldPoseAndNull,
//                     texEllipseAndLodlvl,
//                     texTexScaleFactorDepthTexId,
//                     cornerAreaInfoShared,
//                     shValidMapCpct, shValidMapMemLayout,
//                     shPositionMapCpct, shPosMapMemLayout,
//                     botYCoefCurBuf,
//                     topYCoefCurBuf,
//                     invDistFactorEdge, invDistFactorCorner,
//                     shLayerNum, shOrder,
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
__global__ void MeshLearnerForwardTexPtrCUDA(uint32_t texNum,
    float* shTextures, uint32_t* shTexturesMemLayout,
    float* outImg, unsigned char* maskImg,
    Map texHash, unsigned int* errCnt,
    TexPixInfo* visibleTexIdInfo, PixLocation* compactPixLocBuf,
    uint32_t imgW, uint32_t imgH,
    float* texViewDirFrag2CamAndNull,
    float* texLerpCoeffAndTexCoord,
    float* texCoordShLUandRU,
    float* texCoordShLDandRD,
    float* texObjWorldPoseAndNull,
    float* texEllipseAndLodlvl,
    float* texTexScaleFactorDepthTexId,
    float* cornerAreaInfo,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
    float* botYCoefCurBuf,
    float* topYCoefCurBuf,
    float invDistFactorEdge, float invDistFactorCorner,
    unsigned char shLayerNum, unsigned char shOrder,
    uint32_t* edgeNbrInfo, uint32_t* vertNbrInfo, float gaussianCoeffEWA, float* texInWorldInfo, CurTexAlignedWH* curTexAllWH,
    float* curTexDensities, float* curTexNormals,
    uint32_t needPrintX, uint32_t needPrintY)
{
    // auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto blockId = blockIdx.x;
    auto grp = cg::tiled_partition<MESH_HANDLE_THREAD_NUM>(cg::this_thread_block());

    // maybe more than 32 pixels need to be handle, read in next round
    __shared__ PixLocation pixLocations[MESH_HANDLE_THREAD_NUM];
    __shared__ MeshEdgeNbrInfo meshEdgeNbrInfo;
    __shared__ MeshVertNbrInfo meshVertNbrInfo;
    __shared__ float2 cornerAreaInfoShared;
    

    uint32_t numOfPixToHandle = 0;
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

        // if (curTexId == 370998) {
        //     printf("cuda handle tex 370998 ! numOfPixToHandle: %d \n", numOfPixToHandle);
        // }
    }

    grp.sync();

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

                // if (curTexId == 370998) {
                //     printf("cuda handle tex 370998 !, XY: %d, %d\n", curLoc.x, curLoc.y);
                // }

                CalcRGBTexPtr(shTextures, shTexturesMemLayout,
                    outImg, maskImg,
                    &meshEdgeNbrInfo, &meshVertNbrInfo,
                    curTexId, curTexOffset,
                    curLoc.x, curLoc.y,
                    imgW, imgH, curTexW, curTexH,
                    texViewDirFrag2CamAndNull,
                    texLerpCoeffAndTexCoord,
                    texCoordShLUandRU,
                    texCoordShLDandRD,
                    texObjWorldPoseAndNull,
                    texEllipseAndLodlvl,
                    texTexScaleFactorDepthTexId,
                    cornerAreaInfoShared,
                    // shValidMapCpct, shValidMapMemLayout,
                    validShWHMap, validShWHMapMemLayout, validShNumsAll,
                    shPositionMapCpct, shPosMapMemLayout,
                    botYCoefCurBuf,
                    topYCoefCurBuf,
                    invDistFactorEdge, invDistFactorCorner,
                    shLayerNum, shOrder,
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
// void MeshLearnerForward(uint32_t texNum,
//     float* shTextures, uint32_t* shTexturesMemLayout,
//     float* outImg, unsigned char* maskImg,
//     Map texHash, unsigned int* errCnt,
//     TexPixInfo* visibleTexIdInfo, PixLocation* compactPixLocBuf,
//     uint32_t imgW, uint32_t imgH,
//     cudaTextureObject_t texViewDirFrag2CamAndNull,
//     cudaTextureObject_t texLerpCoeffAndTexCoord,
//     cudaTextureObject_t texCoordShLUandRU,
//     cudaTextureObject_t texCoordShLDandRD,
//     cudaTextureObject_t texObjWorldPoseAndNull,
//     cudaTextureObject_t texEllipseAndLodlvl,
//     cudaTextureObject_t texTexScaleFactorDepthTexId,
//     float* cornerAreaInfo,
//     uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
//     float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
//     float* botYCoefCurBuf,
//     float* topYCoefCurBuf,
//     float invDistFactorEdge, float invDistFactorCorner,
//     unsigned char shLayerNum, unsigned char shOrder,
//     uint32_t* edgeNbrInfo, uint32_t* vertNbrInfo, float gaussianCoeffEWA, float* texInWorldInfo, CurTexAlignedWH* curTexAllWH,
//     float* curTexDensities,
//     uint32_t needPrintX, uint32_t needPrintY)
// {
//     uint32_t constexpr blockSize = MESH_HANDLE_THREAD_NUM;
//     uint32_t gridSize = texNum;

//     MeshLearnerForwardCUDA <<<gridSize, blockSize>>> (texNum,
//         shTextures, shTexturesMemLayout,
//         outImg, maskImg,
//         texHash, errCnt,
//         visibleTexIdInfo, compactPixLocBuf,
//         imgW, imgH,
//         texViewDirFrag2CamAndNull,
//         texLerpCoeffAndTexCoord,
//         texCoordShLUandRU,
//         texCoordShLDandRD,
//         texObjWorldPoseAndNull,
//         texEllipseAndLodlvl,
//         texTexScaleFactorDepthTexId,
//         cornerAreaInfo,
//         shValidMapCpct, shValidMapMemLayout,
//         shPositionMapCpct, shPosMapMemLayout,
//         botYCoefCurBuf,
//         topYCoefCurBuf,
//         invDistFactorEdge, invDistFactorCorner,
//         shLayerNum, shOrder,
//         edgeNbrInfo, vertNbrInfo, gaussianCoeffEWA, texInWorldInfo,
//         curTexAllWH, curTexDensities,
//         needPrintX, needPrintY);
// }

template <typename Map>
void MeshLearnerForwardTexPtr(uint32_t texNum,
    float* shTextures, uint32_t* shTexturesMemLayout,
    float* outImg, unsigned char* maskImg,
    Map texHash, unsigned int* errCnt,
    TexPixInfo* visibleTexIdInfo, PixLocation* compactPixLocBuf,
    uint32_t imgW, uint32_t imgH,
    float* texViewDirFrag2CamAndNull,
    float* texLerpCoeffAndTexCoord,
    float* texCoordShLUandRU,
    float* texCoordShLDandRD,
    float* texObjWorldPoseAndNull,
    float* texEllipseAndLodlvl,
    float* texTexScaleFactorDepthTexId,
    float* cornerAreaInfo,
    // uint32_t* shValidMapCpct, uint32_t* shValidMapMemLayout,
    TexAlignedPosWH* validShWHMap, uint32_t* validShWHMapMemLayout, int32_t* validShNumsAll,
    float* shPositionMapCpct, uint32_t* shPosMapMemLayout,
    float* botYCoefCurBuf,
    float* topYCoefCurBuf,
    float invDistFactorEdge, float invDistFactorCorner,
    unsigned char shLayerNum, unsigned char shOrder,
    uint32_t* edgeNbrInfo, uint32_t* vertNbrInfo, float gaussianCoeffEWA, float* texInWorldInfo, CurTexAlignedWH* curTexAllWH,
    float* curTexDensities, float* curTexNormals,
    uint32_t needPrintX, uint32_t needPrintY)
{
    uint32_t constexpr blockSize = MESH_HANDLE_THREAD_NUM;
    uint32_t gridSize = texNum;

    MeshLearnerForwardTexPtrCUDA <<<gridSize, blockSize>>> (texNum,
        shTextures, shTexturesMemLayout,
        outImg, maskImg,
        texHash, errCnt,
        visibleTexIdInfo, compactPixLocBuf,
        imgW, imgH,
        texViewDirFrag2CamAndNull,
        texLerpCoeffAndTexCoord,
        texCoordShLUandRU,
        texCoordShLDandRD,
        texObjWorldPoseAndNull,
        texEllipseAndLodlvl,
        texTexScaleFactorDepthTexId,
        cornerAreaInfo,
        // shValidMapCpct, shValidMapMemLayout,
        validShWHMap, validShWHMapMemLayout, validShNumsAll,
        shPositionMapCpct, shPosMapMemLayout,
        botYCoefCurBuf,
        topYCoefCurBuf,
        invDistFactorEdge, invDistFactorCorner,
        shLayerNum, shOrder,
        edgeNbrInfo, vertNbrInfo, gaussianCoeffEWA, texInWorldInfo,
        curTexAllWH, curTexDensities, curTexNormals,
        needPrintX, needPrintY);
}

#endif