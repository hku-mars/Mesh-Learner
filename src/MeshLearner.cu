#include "include/linmath.h"

#include <stdlib.h>
#include <stdio.h>
#include <filesystem>
#include <csignal>

#include "third_party/rapidjson/document.h"
#include "third_party/rapidjson/filereadstream.h"
#include "third_party/ReplicaSDK/PTexLib.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <vector>
#include <sstream>
#include <time.h>
#include <Eigen/Dense>

#include "include/MeshLearner.cuh"

#include "cuda_runtime.h"   
#include "cuda_fp16.h"                          
#include "device_launch_parameters.h"
#include <driver_types.h>                           
#include <cuda_runtime_api.h>   
#include <cuda_gl_interop.h>

#include <opencv2/opencv.hpp>
#include "include/trainer.hpp"

#include "torch/csrc/api/include/torch/nn/modules/loss.h"
#include "torch/csrc/api/include/torch/nn/modules/pooling.h"
#include <random>
#include <tbb/tbb.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include "third_party/ReplicaSDK/PTexLib.h"

#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <memory>
#include <cstdio>
#include "third_party/rapidjson/writer.h"
#include "third_party/rapidjson/stringbuffer.h"

#include "include/my_common.hpp"

#include "include/stb_image.h"

#include <iomanip>
#include <cstdint>

#include <EGL/egl.h>
#include <EGL/eglext.h>


using namespace tbb;

using namespace MeshLearner;
using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;
using namespace rapidjson;

using namespace torch::nn;
namespace fs = std::filesystem;


const std::string g_renderThrdName = "RENDER_THRD";
const std::string g_trainPrepThrdName = "TRAIN_PREP_THRD";
const std::string g_trainThrdName = "TRAIN_THRD";

bool g_enableProfilingPrints = false;

#define PROFILE_PRINT(x) do { if (g_enableProfilingPrints) { x; } } while(0)

static const EGLint configAttribs[] = {
          EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
          EGL_BLUE_SIZE, 8,
          EGL_GREEN_SIZE, 8,
          EGL_RED_SIZE, 8,
          EGL_DEPTH_SIZE, 8,
          EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
          EGL_NONE
}; 

std::string GetCurrentTimeAsString() {
  std::time_t now = std::time(nullptr);
  std::tm* timeInfo = std::localtime(&now);

  std::ostringstream oss;
  oss << (timeInfo->tm_year + 1900) << '-'
      << std::setw(2) << std::setfill('0') << (timeInfo->tm_mon + 1) << '-'
      << std::setw(2) << std::setfill('0') << timeInfo->tm_mday << '_'
      << std::setw(2) << std::setfill('0') << timeInfo->tm_hour << '.'
      << std::setw(2) << std::setfill('0') << timeInfo->tm_min << '.'
      << std::setw(2) << std::setfill('0') << timeInfo->tm_sec;

  return oss.str();
}

void printMemoryUsage(string funcName)
{
    std::ifstream status_file("/proc/self/status");
    std::string line;

    while (std::getline(status_file, line)) {
        // 查找 VmRSS 字段
        if (line.find("VmRSS") != std::string::npos) {
            std::cout << " before " <<  funcName <<  ": Memory usage: " << line << std::endl;
            break;
        }
    }
}


torch::Tensor MyTrainer::forward(AutogradContext *ctx, at::Tensor curTrainShTex, unsigned int* errCnt, int shLayerNum, int shOrders, int imgW, int imgH)
{
    ctx->saved_data["sh_tex_size"] = curTrainShTex.sizes().data()[0];
    ctx->saved_data["sh_layer_num"] = shLayerNum;
    ctx->saved_data["sh_orders"] = shOrders;
    ctx->saved_data["img_h"] = imgH;
    ctx->saved_data["img_w"] = imgW;


    ImMeshRenderer& render = ImMeshRenderer::GetInstance();

    checkCudaErrors(cudaStreamSynchronize(render.m_thrdBufs[render.m_trainThrdCurIdx].curBufStream));
    checkCudaErrors(cudaStreamSynchronize(render.m_thrdBufs[render.m_trainThrdCurIdx].curBufStream2));
    MeshLearnerForwardTexPtr(render.m_thrdBufs[render.m_trainThrdCurIdx].visibleTexIdsInfo.size(),
        curTrainShTex.data_ptr<float>(),
        (uint32_t *)thrust::raw_pointer_cast(render.m_thrdBufs[render.m_trainThrdCurIdx].shTexMemLayoutDevice.data()),
        render.m_renderedPixelsCUDA.data_ptr<float>(),
        render.m_renderedPixelsMask.data_ptr<unsigned char>(),
        render.m_thrdBufs[render.m_trainThrdCurIdx].devHashFindRefFind, errCnt,
        (TexPixInfo*)thrust::raw_pointer_cast(render.m_thrdBufs[render.m_trainThrdCurIdx].visibleTexIdsInfo.data()),
        (PixLocation*)thrust::raw_pointer_cast(render.m_thrdBufs[render.m_trainThrdCurIdx].visibleTexPixLocCompact.data()),
        imgW, imgH,
        // render.g_texObjViewDirFrag2CamAndNull,
        // render.g_texObjLerpCoeffAndTexCoord,
        // render.g_texObjOriShLUandRUTexCoord,
        // render.g_texObjOriShLDandRDTexCoord,
        // render.g_texObjWorldPoseAndNull,
        // render.g_texObjEllipCoeffsAndLodLvl,
        // render.g_texObjTexScaleFactorDepthTexId,
        render.m_thrdBufs[render.m_trainThrdCurIdx].viewDirFrag2CamAndNullGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].lerpCoeffAndTexCoordGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].oriShLUandRUGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].oriShLDandRDGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].texWorldPoseAndNullGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].ellipCoeffsAndLodLvlGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].texScaleFacDepthTexIdGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].cornerAreaInfo.data_ptr<float>(),
        (TexAlignedPosWH *)(render.m_thrdBufs[render.m_trainThrdCurIdx].validShWHMapCpct.data_ptr<int32_t>()),
        (uint32_t *)(thrust::raw_pointer_cast(render.m_thrdBufs[render.m_trainThrdCurIdx].validShWHMapLayoutDevice.data())),
        (int32_t *)(render.m_thrdBufs[render.m_trainThrdCurIdx].validShNumsAll.data_ptr<int32_t>()),
        render.m_thrdBufs[render.m_trainThrdCurIdx].shPosMapCpct.data_ptr<float>(),
        (uint32_t *)thrust::raw_pointer_cast(render.m_thrdBufs[render.m_trainThrdCurIdx].posMapMemLayoutDevice.data()),
        render.m_thrdBufs[render.m_trainThrdCurIdx].botYCoeffs.data_ptr<float>(),
        render.m_thrdBufs[render.m_trainThrdCurIdx].topYCoeffs.data_ptr<float>(),
        render.m_invDistFactorEdge, render.m_invDistFactorCorner,
        shLayerNum, shOrders,
        (uint32_t*)(render.m_thrdBufs[render.m_trainThrdCurIdx].edgeNbrs.data_ptr<int32_t>()),
        (uint32_t*)(render.m_thrdBufs[render.m_trainThrdCurIdx].vertNbrs.data_ptr<int32_t>()),
        render.m_gaussianCoeffEWA,
        render.m_thrdBufs[render.m_trainThrdCurIdx].texInWorldInfo.data_ptr<float>(),
        (CurTexAlignedWH *)(render.m_thrdBufs[render.m_trainThrdCurIdx].texWHs.data_ptr<int32_t>()),
        render.m_thrdBufs[render.m_trainThrdCurIdx].meshDensities.data_ptr<float>(),
        render.m_thrdBufs[render.m_trainThrdCurIdx].meshNormals.data_ptr<float>(),
        render.m_printPixX, render.m_printPixY);


    checkCudaErrors(cudaStreamSynchronize(0));

    return {render.m_renderedPixelsCUDA};
}


tensor_list MyTrainer::backward(AutogradContext *ctx, tensor_list gradOutPuts)
{
    int shTexSize = ctx->saved_data["sh_tex_size"].toInt();
    int shLayerNum = ctx->saved_data["sh_layer_num"].toInt();
    int shOrders = ctx->saved_data["sh_orders"].toInt();
    int imgW = ctx->saved_data["img_w"].toInt();
    int imgH = ctx->saved_data["img_h"].toInt();


    auto options = torch::TensorOptions().dtype(torch::kF32).device(torch::kCUDA, 0);

    ImMeshRenderer& render = ImMeshRenderer::GetInstance();

    at::Tensor dL_dshs = torch::zeros({shTexSize}, options).requires_grad_(true).contiguous();

    checkCudaErrors(cudaStreamSynchronize(render.m_thrdBufs[render.m_trainThrdCurIdx].curBufStream));
    checkCudaErrors(cudaStreamSynchronize(render.m_thrdBufs[render.m_trainThrdCurIdx].curBufStream2));

    MeshLearnerBackwardTexPtr(render.m_thrdBufs[render.m_trainThrdCurIdx].visibleTexIdsInfo.size(),
        gradOutPuts[0].data_ptr<float>(),
        dL_dshs.data_ptr<float>(),
        (uint32_t *)thrust::raw_pointer_cast(render.m_thrdBufs[render.m_trainThrdCurIdx].shTexMemLayoutDevice.data()),
        render.m_thrdBufs[render.m_trainThrdCurIdx].devHashFindRefFind, render.m_devErrCntBackwardPtr,
        (TexPixInfo*)thrust::raw_pointer_cast(render.m_thrdBufs[render.m_trainThrdCurIdx].visibleTexIdsInfo.data()),
        (PixLocation*)thrust::raw_pointer_cast(render.m_thrdBufs[render.m_trainThrdCurIdx].visibleTexPixLocCompact.data()),
        imgW, imgH,
        // render.g_texObjViewDirFrag2CamAndNull,
        // render.g_texObjLerpCoeffAndTexCoord,
        // render.g_texObjOriShLUandRUTexCoord,
        // render.g_texObjOriShLDandRDTexCoord,
        // render.g_texObjWorldPoseAndNull,
        // render.g_texObjEllipCoeffsAndLodLvl,
        // render.g_texObjTexScaleFactorDepthTexId,
        render.m_thrdBufs[render.m_trainThrdCurIdx].viewDirFrag2CamAndNullGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].lerpCoeffAndTexCoordGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].oriShLUandRUGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].oriShLDandRDGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].texWorldPoseAndNullGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].ellipCoeffsAndLodLvlGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].texScaleFacDepthTexIdGPU,
        render.m_thrdBufs[render.m_trainThrdCurIdx].cornerAreaInfo.data_ptr<float>(),
        (TexAlignedPosWH*)(render.m_thrdBufs[render.m_trainThrdCurIdx].validShWHMapCpct.data_ptr<int32_t>()),
        (uint32_t*)(thrust::raw_pointer_cast(render.m_thrdBufs[render.m_trainThrdCurIdx].validShWHMapLayoutDevice.data())),
        (int32_t*)(render.m_thrdBufs[render.m_trainThrdCurIdx].validShNumsAll.data_ptr<int32_t>()),
        render.m_thrdBufs[render.m_trainThrdCurIdx].shPosMapCpct.data_ptr<float>(),
        (uint32_t *)thrust::raw_pointer_cast(render.m_thrdBufs[render.m_trainThrdCurIdx].posMapMemLayoutDevice.data()),
        render.m_thrdBufs[render.m_trainThrdCurIdx].botYCoeffs.data_ptr<float>(),
        render.m_thrdBufs[render.m_trainThrdCurIdx].topYCoeffs.data_ptr<float>(),
        render.m_invDistFactorEdge, render.m_invDistFactorCorner,
        shLayerNum, shOrders,
        (uint32_t*)(render.m_thrdBufs[render.m_trainThrdCurIdx].edgeNbrs.data_ptr<int32_t>()),
        (uint32_t*)(render.m_thrdBufs[render.m_trainThrdCurIdx].vertNbrs.data_ptr<int32_t>()),
        (float)render.m_highOrderSHLrMultiplier,
        render.m_gaussianCoeffEWA, render.m_thrdBufs[render.m_trainThrdCurIdx].texInWorldInfo.data_ptr<float>(),
        (CurTexAlignedWH *)(render.m_thrdBufs[render.m_trainThrdCurIdx].texWHs.data_ptr<int32_t>()),
        render.m_thrdBufs[render.m_trainThrdCurIdx].meshDensities.data_ptr<float>(),
        render.m_thrdBufs[render.m_trainThrdCurIdx].meshNormals.data_ptr<float>(),
        render.m_printPixX, render.m_printPixY);

    checkCudaErrors(cudaStreamSynchronize(0));

    return {dL_dshs, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

bool readFileConfig(const std::string& filename, FileConfig& config)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::string jsonStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    Document document;
    document.Parse(jsonStr.c_str());
    if (document.HasParseError()) {
        std::cout << "Failed to parse JSON file: " << filename << std::endl;
        return false;
    }

    // if (document.HasMember("cameraConfigPath") && document["cameraConfigPath"].IsString()) {
    //     config.cameraConfigPath = document["cameraConfigPath"].GetString();
    // }

    if (document.HasMember("cameraConfigGtPath") && document["cameraConfigGtPath"].IsString()) {
        config.cameraConfigGtPath = document["cameraConfigGtPath"].GetString();
    }

    if (document.HasMember("algoConfigPath") && document["algoConfigPath"].IsString()) {
        config.algoConfigPath = document["algoConfigPath"].GetString();
    }

    if (document.HasMember("sceneType") && document["sceneType"].IsInt()) {
        config.sceneType = (SceneType)document["sceneType"].GetInt();
    }

    if (document.HasMember("meshType") && document["meshType"].IsInt()) {
        config.meshType = (MeshType)document["meshType"].GetInt();
    }


    return true;
}



bool readCameraConfig(const std::string& filename, CameraConfig& config)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::string jsonStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    Document document;
    document.Parse(jsonStr.c_str());
    if (document.HasParseError()) {
        std::cout << "Failed to parse JSON file: " << filename << std::endl;
        return false;
    }

    if (document.HasMember("fx") && document["fx"].IsFloat()) {
        config.fx = document["fx"].GetFloat();
    }

    if (document.HasMember("fy") && document["fy"].IsFloat()) {
        config.fy = document["fy"].GetFloat();
    }

    if (document.HasMember("cx") && document["cx"].IsFloat()) {
        config.cx = document["cx"].GetFloat();
    }

    if (document.HasMember("cy") && document["cy"].IsFloat()) {
        config.cy = document["cy"].GetFloat();
    }

    if (document.HasMember("scaleFactor") && document["scaleFactor"].IsFloat()) {
        config.scaleFactor = document["scaleFactor"].GetFloat();
    }

    return true;
}

bool readAlgorithmConfig(const std::string& filename, AlgoConfig& config)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::string jsonStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    Document document;
    document.Parse(jsonStr.c_str());
    if (document.HasParseError()) {
        std::cout << "Failed to parse JSON file: " << filename << std::endl;
        return false;
    }

    if (document.HasMember("shOrder") && document["shOrder"].IsUint()) {
        config.shOrder = document["shOrder"].GetUint();
    }

    if (document.HasMember("renderW") && document["renderW"].IsUint()) {
        config.renderW = document["renderW"].GetUint();
    }

    if (document.HasMember("gtW") && document["gtW"].IsUint()) {
        config.gtW = document["gtW"].GetUint();
    }

    if (document.HasMember("gtH") && document["gtH"].IsUint()) {
        config.gtH = document["gtH"].GetUint();
    }

    if (document.HasMember("renderH") && document["renderH"].IsUint()) {
        config.renderH = document["renderH"].GetUint();
    }

    if (document.HasMember("perPoseTrainCnt") && document["perPoseTrainCnt"].IsUint()) {
        config.perPoseTrainCnt = document["perPoseTrainCnt"].GetUint();
    }

    if (document.HasMember("printPixX") && document["printPixX"].IsUint()) {
        config.printPixX = document["printPixX"].GetUint();
    }

    if (document.HasMember("printPixY") && document["printPixY"].IsUint()) {
        config.printPixY = document["printPixY"].GetUint();
    }

    if (document.HasMember("printGradTexId") && document["printGradTexId"].IsUint()) {
        config.printGradTexId = document["printGradTexId"].GetUint();
    }

    if (document.HasMember("schedulerPatience") && document["schedulerPatience"].IsUint()) {
        config.schedulerPatience = document["schedulerPatience"].GetUint();
    }

    if (document.HasMember("texResizeInterval") && document["texResizeInterval"].IsUint()) {
        config.texResizeInterval = document["texResizeInterval"].GetUint();
    }

    if (document.HasMember("testInterval") && document["testInterval"].IsUint()) {
        config.testInterval = document["testInterval"].GetUint();
    }

    if (document.HasMember("resizePatience") && document["resizePatience"].IsUint()) {
        config.resizePatience = document["resizePatience"].GetUint();
    }

    if (document.HasMember("resizeReturnCntMax") && document["resizeReturnCntMax"].IsUint()) {
        config.resizeReturnCntMax = document["resizeReturnCntMax"].GetUint();
    }

    if (document.HasMember("resizeConvergeThreshold") && document["resizeConvergeThreshold"].IsDouble()) {
        config.resizeConvergeThreshold = document["resizeConvergeThreshold"].GetDouble();
    }

    if (document.HasMember("resizeTriangleEdgeMinLen") && document["resizeTriangleEdgeMinLen"].IsDouble()) {
        config.resizeTriangleEdgeMinLen = document["resizeTriangleEdgeMinLen"].GetDouble();
    }

    if (document.HasMember("maxTexWH") && document["maxTexWH"].IsUint()) {
        config.maxTexWH = document["maxTexWH"].GetUint();
    }

    if (document.HasMember("resizeStartStep") && document["resizeStartStep"].IsUint()) {
        config.resizeStartStep = document["resizeStartStep"].GetUint();
    }

    if (document.HasMember("trainPoseCount") && document["trainPoseCount"].IsUint()) {
        config.trainPoseCount = document["trainPoseCount"].GetUint();
    }

    if (document.HasMember("saveDir") && document["saveDir"].IsString()) {
        config.saveDir = document["saveDir"].GetString();
    }

    if (document.HasMember("experimentName") && document["experimentName"].IsString()) {
        config.experimentName = document["experimentName"].GetString();
    }

    if (document.HasMember("meshPath") && document["meshPath"].IsString()) {
        config.meshPath = document["meshPath"].GetString();
    }

    if (document.HasMember("testSequencePath") && document["testSequencePath"].IsString()) {
        config.testSequencePath = document["testSequencePath"].GetString();
    }

    if (document.HasMember("evalSequencePath") && document["evalSequencePath"].IsString()) {
        config.evalSequencePath = document["evalSequencePath"].GetString();
    }

    if (document.HasMember("testGtimgPath") && document["testGtimgPath"].IsString()) {
        config.testGtimgPath = document["testGtimgPath"].GetString();
    }

    if (document.HasMember("evalImgPath") && document["evalImgPath"].IsString()) {
        config.evalImgPath = document["evalImgPath"].GetString();
    }

    if (document.HasMember("tensorboardDir") && document["tensorboardDir"].IsString()) {
        config.tensorboardDir = document["tensorboardDir"].GetString();
    }

    if (document.HasMember("gtPosePath") && document["gtPosePath"].IsString()) {
        config.gtPosePath = document["gtPosePath"].GetString();
    }

    if (document.HasMember("highOrderSHLrMultiplier") && document["highOrderSHLrMultiplier"].IsDouble()) {
        config.highOrderSHLrMultiplier = document["highOrderSHLrMultiplier"].GetDouble();
    }

    if (document.HasMember("invDistFactorEdge") && document["invDistFactorEdge"].IsDouble()) {
        config.invDistFactorEdge = document["invDistFactorEdge"].GetDouble();
    }

    if (document.HasMember("invDistFactorCorner") && document["invDistFactorCorner"].IsDouble()) {
        config.invDistFactorCorner = document["invDistFactorCorner"].GetDouble();
    }

    if (document.HasMember("shDensity") && document["shDensity"].IsDouble()) {
        config.shDensity = document["shDensity"].GetDouble();
    }

    if (document.HasMember("shDensityMax") && document["shDensityMax"].IsDouble()) {
        config.shDensityMax = document["shDensityMax"].GetDouble();
    }

    if (document.HasMember("varThresholdPSNR") && document["varThresholdPSNR"].IsDouble()) {
        config.varThresholdPSNR = document["varThresholdPSNR"].GetDouble();
    }

    if (document.HasMember("L1lossThreshold") && document["L1lossThreshold"].IsDouble()) {
        config.L1lossThreshold = document["L1lossThreshold"].GetDouble();
    }

    if (document.HasMember("resizeDepthThreshold") && document["resizeDepthThreshold"].IsDouble()) {
        config.resizeDepthThreshold = document["resizeDepthThreshold"].GetDouble();
    }

    if (document.HasMember("densityUpdateStep") && document["densityUpdateStep"].IsDouble()) {
        config.densityUpdateStep = document["densityUpdateStep"].GetDouble();
    }

    if (document.HasMember("densityUpdateStepInner") && document["densityUpdateStepInner"].IsDouble()) {
        config.densityUpdateStepInner = document["densityUpdateStepInner"].GetDouble();
    }

    if (document.HasMember("gtImageDir") && document["gtImageDir"].IsString()) {
        config.gtImageDir = document["gtImageDir"].GetString();
    }

    if (document.HasMember("gaussianCoeffEWA") && document["gaussianCoeffEWA"].IsDouble()) {
        config.gaussianCoeffEWA = document["gaussianCoeffEWA"].GetDouble();
    }

    if (document.HasMember("isRenderingMode") && document["isRenderingMode"].IsUint()) {
        config.isRenderingMode = document["isRenderingMode"].GetUint();
    }

    if (document.HasMember("savedJsonPath") && document["savedJsonPath"].IsString()) {
        config.savedJsonPath = document["savedJsonPath"].GetString();
    }

    if (document.HasMember("savedBinPath") && document["savedBinPath"].IsString()) {
        config.savedBinPath = document["savedBinPath"].GetString();
    }

    if (document.HasMember("RenderingPosePath") && document["RenderingPosePath"].IsString()) {
        config.RenderingPosePath = document["RenderingPosePath"].GetString();
    }

    if (document.HasMember("ConvergedSaveInterval") && document["ConvergedSaveInterval"].IsUint()) {
        config.ConvergedSaveInterval = document["ConvergedSaveInterval"].GetUint();
    }

    return true;
}


static unsigned long GetTimeMS()
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}



GLenum glCheckError_(const char *file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
            case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
            case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
            case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
    }
    return errorCode;
}
#define glCheckError() glCheckError_(__FILE__, __LINE__) 



string g_vertexShaderText = R"(
#version 450 core
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform vec3 camWorldPos;

layout (location = 0) in mat4 vertAttribCpct;
layout (location = 4) in vec4 meshAttribCpct;
out vec2 texCoord;
out vec3 dirFrag2Cam;
out vec3 worldPose;

out flat vec2 curTexWH;
out flat float texId;

void main()
{
    curTexWH.x = meshAttribCpct.x;
    curTexWH.y = meshAttribCpct.y;

    texId = meshAttribCpct.z;

    float texUVAll[6];
    texUVAll[0] = vertAttribCpct[0][3];
    texUVAll[1] = vertAttribCpct[1][3];
    texUVAll[2] = vertAttribCpct[2][3];
    texUVAll[3] = vertAttribCpct[3][3];
    texUVAll[4] = vertAttribCpct[3][0];
    texUVAll[5] = vertAttribCpct[3][1];

    gl_Position = proj * view * vec4(vertAttribCpct[gl_VertexID][0], vertAttribCpct[gl_VertexID][1], vertAttribCpct[gl_VertexID][2], 1.0);
    dirFrag2Cam = normalize(camWorldPos - vec3(vertAttribCpct[gl_VertexID][0], vertAttribCpct[gl_VertexID][1], vertAttribCpct[gl_VertexID][2]));
    texCoord = vec2(texUVAll[gl_VertexID * 2 + 0], texUVAll[gl_VertexID * 2 + 1]);
    worldPose = vec3(vertAttribCpct[gl_VertexID][0], vertAttribCpct[gl_VertexID][1], vertAttribCpct[gl_VertexID][2]);
};
)";


string g_fragmentShaderText = R"(
#version 450 core

precision highp float; 

float near = 0.1;
float far  = 100.0;

layout (location = 0) out vec4 g_color;
layout (location = 1) out vec4 g_lerpCoeffAndTexCoord;
layout (location = 2) out vec4 g_viewDirAndNull;
layout (location = 3) out vec4 g_oriShLUandRUCoord;
layout (location = 4) out vec4 g_oriShLDandRDCoord;
layout (location = 5) out vec4 g_texScaleFacDepthTexId;
layout (location = 6) out vec4 g_texWorldPoseAndNull;
layout (location = 7) out vec4 g_ellipCeoffsAndLodLvl;

in vec2 texCoord;
in vec3 dirFrag2Cam;
in vec3 worldPose;

in flat vec2 curTexWH;
in flat float texId;

uniform sampler2D texture1;

// interpolate at the center because SHs are in the center of each texel
void CalcCornerShCoordAndCoeffNoSampler(vec2 texCoordTmp, out vec2 coordLU, out vec2 coordRU, out vec2 coordLD, out vec2 coordRD, out vec2 coeff)
{
    float texelSizeW = 1.0 / curTexWH.x;
    float texelSizeH = 1.0 / curTexWH.y;
    float a = fract(texCoordTmp.x * curTexWH.x);
    float b = fract(texCoordTmp.y * curTexWH.y);

    if (a >= 0.5) {
        if (b >= 0.5) {
            coordLU = texCoordTmp + vec2(0.0, texelSizeH);
            coordRU = texCoordTmp + vec2(texelSizeW, texelSizeH);
            coordLD = texCoordTmp;
            coordRD = texCoordTmp + vec2(texelSizeW, 0.0);
            
            coeff.x = a - (0.5);
            coeff.y = (1.5) - b;
        } else {
            coordLU = texCoordTmp;
            coordRU = texCoordTmp + vec2(texelSizeW, 0.0);
            coordLD = texCoordTmp + vec2(0.0, -texelSizeH);
            coordRD = texCoordTmp + vec2(texelSizeW, -texelSizeH);
            
            coeff.x = a - (0.5);
            coeff.y = (0.5) - b;
        }
    } else {
        if (b >= 0.5) {
            coordLU = texCoordTmp + vec2(-texelSizeW, texelSizeH);
            coordRU = texCoordTmp + vec2(0.0, texelSizeH);
            coordLD = texCoordTmp + vec2(-texelSizeW, 0.0);
            coordRD = texCoordTmp;

            coeff.x = (0.5) + a;
            coeff.y = (1.5) - b;
        } else {
            coordLU = texCoordTmp + vec2(-texelSizeW, 0.0);
            coordRU = texCoordTmp;
            coordLD = texCoordTmp + vec2(-texelSizeW, -texelSizeH);
            coordRD = texCoordTmp + vec2(0.0, -texelSizeH);

            coeff.x = (0.5) + a;
            coeff.y = (0.5) - b;
        }
    }
}

vec3 BiLinearLerp(sampler2DArray tex, vec2 coordLU, vec2 coordRU, vec2 coordLD, vec2 coordRD, float coeffX, float coeffY, float level)
{   
    vec4 valLU = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 valRU = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 valLD = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 valRD = vec4(0.0, 0.0, 0.0, 0.0);

    valLU = texture(tex, vec3(coordLU, level), 0);
    valRU = texture(tex, vec3(coordRU, level), 0);
    valLD = texture(tex, vec3(coordLD, level), 0);
    valRD = texture(tex, vec3(coordRD, level), 0);

    vec4 interpUp = mix(valLU, valRU, coeffX);
    vec4 interpDown = mix(valLD, valRD, coeffX);

    return vec3(mix(interpUp, interpDown, coeffY).xyz);
}


float GetRealDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));    
}

vec3 filterEWA(vec2 texCord, vec2 duvx, vec2 duvy)
{
    float APrime = duvx.y * duvx.y + duvy.y * duvy.y;
    float BPrime = -2.0 * (duvx.x * duvx.y + duvy.x * duvy.y);
    float CPrime = duvx.x * duvx.x + duvy.x * duvy.x;
    float t = (duvx.x * duvy.y - duvy.x * duvx.y);
    float F = 1.0 / (t * t);

    APrime *= F;
    BPrime *= F;
    CPrime *= F;

    return vec3(APrime, BPrime, CPrime);

    // float theta = atan(BPrime / (APrime - CPrime)) * 0.5;
    // float costheta = cos(theta);
    // float sintheta = sin(theta);

    // float det = (APrime - CPrime) * (APrime - CPrime) + BPrime * BPrime;
    // float A = 0.5 * (APrime + CPrime - sqrt(det));
    // float C = 0.5 * (APrime + CPrime + sqrt(det));
    // float a = 1.0 / sqrt(A);
    // float b = 1.0 / sqrt(C);

    // float x = max(a * costheta, b * sintheta);
    // float y = max(a * sintheta, b * costheta);

    // int ss = int(floor(texCord.x * float(texW)));
    // int tt = int(floor(texCord.y * float(texH)));

    // int xx = int(floor(x * float(texW))) * 100;
    // int yy = int(floor(y * float(texH))) * 100;

    // vec3 sum = vec3(0.0, 0.0, 0.0);

    // float sumWeights = 0.0;

    // float lenOfddx = sqrt(duvx.x * duvx.x + duvx.y * duvx.y);
    // float lenOfddy = sqrt(duvy.x * duvy.x + duvy.y * duvy.y);


    // for (int i = -xx + ss; i <= xx + ss; i++) {
  	//     if (i < 0 || i >= texW) {
    //         continue;
    //     }

    //     for (int j = -yy + tt; j <= yy + tt; j++) {
    //         if (j < 0 || j >= texH) {
    //             continue;
    //         }

    //         // float s = float(i - ss) / float(texW);
    //         // float t = float(j - tt) / float(texH);

    //         vec2 offsetVec = vec2(float(i - ss) / float(texW), float(j - tt) / float(texH));
    //         float s = dot(duvx, offsetVec) / lenOfddx;
    //         float t = dot(duvy, offsetVec) / lenOfddy;


    //         float r2 = APrime * s * s + BPrime * s * t + CPrime * t * t;
    //         if (r2 < 1.0) {
    //             // float weight = 1.0;
    //             float weight = exp(-2.0 * r2) - exp(-2.0);
    //             sum += vec3(texture(texture1, vec2((float(i) / float(texW)), (float(j) / float(texH))) ).xyz) * weight;
    //             sumWeights += weight;
    //         }
    //     }
    // }

    // vec3 finalOut = sum / sumWeights;
    // return finalOut;
}

void main()
{
    float finalR, finalG, finalB;
    finalR = 0.0;
    finalG = 0.0;
    finalB = 0.0;

    vec2 coordLU, coordRU, coordLD, coordRD;
    vec2 lerpCoeff;
        
    CalcCornerShCoordAndCoeffNoSampler(texCoord, coordLU, coordRU, coordLD, coordRD, lerpCoeff);

    g_oriShLUandRUCoord = vec4(coordLU.xy, coordRU.xy);
    g_oriShLDandRDCoord = vec4(coordLD.xy, coordRD.xy);
    g_lerpCoeffAndTexCoord = vec4(lerpCoeff.xy, texCoord.xy);
    
    g_viewDirAndNull = vec4(dirFrag2Cam.xyz, 0.0);
    
    g_texScaleFacDepthTexId = vec4(0.0, 0.0, GetRealDepth(gl_FragCoord.z), texId);
    
    g_texWorldPoseAndNull = vec4(worldPose, 0.0);

    vec2 duvX = dFdx(texCoord);
    vec2 duvY = dFdy(texCoord);

    g_ellipCeoffsAndLodLvl = vec4(filterEWA(texCoord, duvX, duvY), textureQueryLod(texture1, texCoord).y);

    g_color = vec4(0.1, 0.2, 0.4, 1.0);
}
)";



static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

void MatConvert(mat4x4& outMat, std::vector<float> inMat)
{
    outMat[0][0] = inMat[0];
    outMat[0][1] = inMat[1];
    outMat[0][2] = inMat[2];
    outMat[0][3] = inMat[3];
    outMat[1][0] = inMat[4];
    outMat[1][1] = inMat[5];
    outMat[1][2] = inMat[6];
    outMat[1][3] = inMat[7];
    outMat[2][0] = inMat[8];
    outMat[2][1] = inMat[9];
    outMat[2][2] = inMat[10];
    outMat[2][3] = inMat[11];
    outMat[3][0] = inMat[12];
    outMat[3][1] = inMat[13];
    outMat[3][2] = inMat[14];
    outMat[3][3] = inMat[15];
}

void mat4x4_print(mat4x4 m)
{
    std::cout << "print mat: \n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << m[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "print mat end \n";
}


unsigned int ImMeshRenderer::savedImgCnt = 0;


// void ImMeshRenderer::UnmapIntermediateTextures()
// {
//     cudaDestroyTextureObject(g_texObjLerpCoeffAndTexCoord);
//     cudaDestroyTextureObject(g_texObjViewDirFrag2CamAndNull);
//     cudaDestroyTextureObject(g_texObjOriShLUandRUTexCoord);
//     cudaDestroyTextureObject(g_texObjOriShLDandRDTexCoord);
//     cudaDestroyTextureObject(g_texObjTexScaleFactorDepthTexId);
//     // cudaDestroyTextureObject(g_texObjTexEdgeLR);
//     cudaDestroyTextureObject(g_texObjWorldPoseAndNull);
//     // cudaDestroyTextureObject(g_texObjTexInvDistCoeffs);
//     cudaDestroyTextureObject(g_texObjEllipCoeffsAndLodLvl);
    

//     checkCudaErrors(cudaGraphicsUnmapResources(1, &g_cuResLerpCoeffAndTexCoord, 0));
//     checkCudaErrors(cudaGraphicsUnmapResources(1, &g_cuResViewDirFrag2CamAndNull, 0));
//     checkCudaErrors(cudaGraphicsUnmapResources(1, &g_cuResOriShLUandRUTexCoord, 0));
//     checkCudaErrors(cudaGraphicsUnmapResources(1, &g_cuResOriShLDandRDTexCoord, 0));
//     checkCudaErrors(cudaGraphicsUnmapResources(1, &g_cuResTexScaleFactorDepthTexId, 0));
//     // checkCudaErrors(cudaGraphicsUnmapResources(1, &g_cuResTexEdgeLR, 0));
//     checkCudaErrors(cudaGraphicsUnmapResources(1, &g_cuResWorldPoseAndNull, 0));
//     // checkCudaErrors(cudaGraphicsUnmapResources(1, &g_cuResTexInvDistCoeffs, 0));
//     checkCudaErrors(cudaGraphicsUnmapResources(1, &g_cuResEllipCoeffsAndLodLvl, 0));
    
// }

// map intermediate texture
// void ImMeshRenderer::MapIntermediateTextures()
// {
//     checkCudaErrors(cudaGraphicsMapResources(1, &g_cuResLerpCoeffAndTexCoord, 0));
//     checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&g_cuArrayLerpCoeffAndTexCoord, g_cuResLerpCoeffAndTexCoord, 0, 0));
//     memset(&g_resDescLerpCoeffAndTexCoord, 0, sizeof(g_resDescLerpCoeffAndTexCoord));
//     g_resDescLerpCoeffAndTexCoord.resType = cudaResourceTypeArray;
//     g_resDescLerpCoeffAndTexCoord.res.array.array = g_cuArrayLerpCoeffAndTexCoord;

//     memset(&g_texDescLerpCoeffAndTexCoord, 0, sizeof(g_texDescLerpCoeffAndTexCoord));
//     g_texDescLerpCoeffAndTexCoord.addressMode[0] = cudaAddressModeClamp;
//     g_texDescLerpCoeffAndTexCoord.addressMode[1] = cudaAddressModeClamp;
//     g_texDescLerpCoeffAndTexCoord.filterMode = cudaFilterModePoint;
//     g_texDescLerpCoeffAndTexCoord.readMode = cudaReadModeElementType;
//     g_texDescLerpCoeffAndTexCoord.normalizedCoords = 0;
//     cudaCreateTextureObject(&g_texObjLerpCoeffAndTexCoord, &g_resDescLerpCoeffAndTexCoord, &g_texDescLerpCoeffAndTexCoord, NULL);


//     checkCudaErrors(cudaGraphicsMapResources(1, &g_cuResViewDirFrag2CamAndNull, 0));
//     checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&g_cuArrayViewDirFrag2CamAndNull, g_cuResViewDirFrag2CamAndNull, 0, 0));
//     memset(&g_resDescViewDirFrag2CamAndNull, 0, sizeof(g_resDescViewDirFrag2CamAndNull));
//     g_resDescViewDirFrag2CamAndNull.resType = cudaResourceTypeArray;
//     g_resDescViewDirFrag2CamAndNull.res.array.array = g_cuArrayViewDirFrag2CamAndNull;

//     memset(&g_texDescViewDirFrag2CamAndNull, 0, sizeof(g_texDescViewDirFrag2CamAndNull));
//     g_texDescViewDirFrag2CamAndNull.addressMode[0] = cudaAddressModeClamp;
//     g_texDescViewDirFrag2CamAndNull.addressMode[1] = cudaAddressModeClamp;
//     g_texDescViewDirFrag2CamAndNull.filterMode = cudaFilterModePoint;
//     g_texDescViewDirFrag2CamAndNull.readMode = cudaReadModeElementType;
//     g_texDescViewDirFrag2CamAndNull.normalizedCoords = 0;
//     cudaCreateTextureObject(&g_texObjViewDirFrag2CamAndNull, &g_resDescViewDirFrag2CamAndNull, &g_texDescViewDirFrag2CamAndNull, NULL);


//     checkCudaErrors(cudaGraphicsMapResources(1, &g_cuResOriShLUandRUTexCoord, 0));
//     checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&g_cuArrayOriShLUandRUTexCoord, g_cuResOriShLUandRUTexCoord, 0, 0));
//     memset(&g_resDescOriShLUandRUTexCoord, 0, sizeof(g_resDescOriShLUandRUTexCoord));
//     g_resDescOriShLUandRUTexCoord.resType = cudaResourceTypeArray;
//     g_resDescOriShLUandRUTexCoord.res.array.array = g_cuArrayOriShLUandRUTexCoord;

//     memset(&g_texDescOriShLUandRUTexCoord, 0, sizeof(g_texDescOriShLUandRUTexCoord));
//     g_texDescOriShLUandRUTexCoord.addressMode[0] = cudaAddressModeClamp;
//     g_texDescOriShLUandRUTexCoord.addressMode[1] = cudaAddressModeClamp;
//     g_texDescOriShLUandRUTexCoord.filterMode = cudaFilterModePoint;
//     g_texDescOriShLUandRUTexCoord.readMode = cudaReadModeElementType;
//     g_texDescOriShLUandRUTexCoord.normalizedCoords = 0;
//     cudaCreateTextureObject(&g_texObjOriShLUandRUTexCoord, &g_resDescOriShLUandRUTexCoord, &g_texDescOriShLUandRUTexCoord, NULL);


//     checkCudaErrors(cudaGraphicsMapResources(1, &g_cuResOriShLDandRDTexCoord, 0));
//     checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&g_cuArrayOriShLDandRDTexCoord, g_cuResOriShLDandRDTexCoord, 0, 0));
//     memset(&g_resDescOriShLDandRDTexCoord, 0, sizeof(g_resDescOriShLDandRDTexCoord));
//     g_resDescOriShLDandRDTexCoord.resType = cudaResourceTypeArray;
//     g_resDescOriShLDandRDTexCoord.res.array.array = g_cuArrayOriShLDandRDTexCoord;

//     memset(&g_texDescOriShLDandRDTexCoord, 0, sizeof(g_texDescOriShLDandRDTexCoord));
//     g_texDescOriShLDandRDTexCoord.addressMode[0] = cudaAddressModeClamp;
//     g_texDescOriShLDandRDTexCoord.addressMode[1] = cudaAddressModeClamp;
//     g_texDescOriShLDandRDTexCoord.filterMode = cudaFilterModePoint;
//     g_texDescOriShLDandRDTexCoord.readMode = cudaReadModeElementType;
//     g_texDescOriShLDandRDTexCoord.normalizedCoords = 0;
//     cudaCreateTextureObject(&g_texObjOriShLDandRDTexCoord, &g_resDescOriShLDandRDTexCoord, &g_texDescOriShLDandRDTexCoord, NULL);


//     checkCudaErrors(cudaGraphicsMapResources(1, &g_cuResTexScaleFactorDepthTexId, 0));
//     checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&g_cuArrayTexScaleFactorDepthTexId, g_cuResTexScaleFactorDepthTexId, 0, 0));
//     memset(&g_resDescTexScaleFactorDepthTexId, 0, sizeof(g_resDescTexScaleFactorDepthTexId));
//     g_resDescTexScaleFactorDepthTexId.resType = cudaResourceTypeArray;
//     g_resDescTexScaleFactorDepthTexId.res.array.array = g_cuArrayTexScaleFactorDepthTexId;

//     memset(&g_texDescTexScaleFactorDepthTexId, 0, sizeof(g_texDescTexScaleFactorDepthTexId));
//     g_texDescTexScaleFactorDepthTexId.addressMode[0] = cudaAddressModeClamp;
//     g_texDescTexScaleFactorDepthTexId.addressMode[1] = cudaAddressModeClamp;
//     g_texDescTexScaleFactorDepthTexId.filterMode = cudaFilterModePoint;
//     g_texDescTexScaleFactorDepthTexId.readMode = cudaReadModeElementType;
//     g_texDescTexScaleFactorDepthTexId.normalizedCoords = 0;
//     cudaCreateTextureObject(&g_texObjTexScaleFactorDepthTexId, &g_resDescTexScaleFactorDepthTexId, &g_texDescTexScaleFactorDepthTexId, NULL);


//     // checkCudaErrors(cudaGraphicsMapResources(1, &g_cuResTexEdgeLR, 0));
//     // checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&g_cuArrayTexEdgeLR, g_cuResTexEdgeLR, 0, 0));
//     // memset(&g_resDescTexEdgeLR, 0, sizeof(g_resDescTexEdgeLR));
//     // g_resDescTexEdgeLR.resType = cudaResourceTypeArray;
//     // g_resDescTexEdgeLR.res.array.array = g_cuArrayTexEdgeLR;

//     // memset(&g_texDescTexEdgeLR, 0, sizeof(g_texDescTexEdgeLR));
//     // g_texDescTexEdgeLR.addressMode[0] = cudaAddressModeClamp;
//     // g_texDescTexEdgeLR.addressMode[1] = cudaAddressModeClamp;
//     // g_texDescTexEdgeLR.filterMode = cudaFilterModePoint;
//     // g_texDescTexEdgeLR.readMode = cudaReadModeElementType;
//     // g_texDescTexEdgeLR.normalizedCoords = 0;
//     // cudaCreateTextureObject(&g_texObjTexEdgeLR, &g_resDescTexEdgeLR, &g_texDescTexEdgeLR, NULL);

    
//     checkCudaErrors(cudaGraphicsMapResources(1, &g_cuResWorldPoseAndNull, 0));
//     checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&g_cuArrayWorldPoseAndNull, g_cuResWorldPoseAndNull, 0, 0));
//     memset(&g_resDescWorldPoseAndNull, 0, sizeof(g_resDescWorldPoseAndNull));
//     g_resDescWorldPoseAndNull.resType = cudaResourceTypeArray;
//     g_resDescWorldPoseAndNull.res.array.array = g_cuArrayWorldPoseAndNull;

//     memset(&g_texDescWorldPoseAndNull, 0, sizeof(g_texDescWorldPoseAndNull));
//     g_texDescWorldPoseAndNull.addressMode[0] = cudaAddressModeClamp;
//     g_texDescWorldPoseAndNull.addressMode[1] = cudaAddressModeClamp;
//     g_texDescWorldPoseAndNull.filterMode = cudaFilterModePoint;
//     g_texDescWorldPoseAndNull.readMode = cudaReadModeElementType;
//     g_texDescWorldPoseAndNull.normalizedCoords = 0;

//     cudaCreateTextureObject(&g_texObjWorldPoseAndNull, &g_resDescWorldPoseAndNull, &g_texDescWorldPoseAndNull, NULL);
    
    
    
//     // checkCudaErrors(cudaGraphicsMapResources(1, &g_cuResTexInvDistCoeffs, 0));
//     // checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&g_cuArrayTexInvDistCoeffs, g_cuResTexInvDistCoeffs, 0, 0));
//     // memset(&g_resDescTexInvDistCoeffs, 0, sizeof(g_resDescTexInvDistCoeffs));
//     // g_resDescTexInvDistCoeffs.resType = cudaResourceTypeArray;
//     // g_resDescTexInvDistCoeffs.res.array.array = g_cuArrayTexInvDistCoeffs;

//     // memset(&g_texDescTexInvDistCoeffs, 0, sizeof(g_texDescTexInvDistCoeffs));
//     // g_texDescTexInvDistCoeffs.addressMode[0] = cudaAddressModeClamp;
//     // g_texDescTexInvDistCoeffs.addressMode[1] = cudaAddressModeClamp;
//     // g_texDescTexInvDistCoeffs.filterMode = cudaFilterModePoint;
//     // g_texDescTexInvDistCoeffs.readMode = cudaReadModeElementType;
//     // g_texDescTexInvDistCoeffs.normalizedCoords = 0;
//     // cudaCreateTextureObject(&g_texObjTexInvDistCoeffs, &g_resDescTexInvDistCoeffs, &g_texDescTexInvDistCoeffs, NULL);

//     checkCudaErrors(cudaGraphicsMapResources(1, &g_cuResEllipCoeffsAndLodLvl, 0));
//     checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&g_cuArrayEllipCoeffsAndLodLvl, g_cuResEllipCoeffsAndLodLvl, 0, 0));
//     memset(&g_resDescEllipCoeffsAndLodLvl, 0, sizeof(g_resDescEllipCoeffsAndLodLvl));
//     g_resDescEllipCoeffsAndLodLvl.resType = cudaResourceTypeArray;
//     g_resDescEllipCoeffsAndLodLvl.res.array.array = g_cuArrayEllipCoeffsAndLodLvl;

//     memset(&g_texDescEllipCoeffsAndLodLvl, 0, sizeof(g_texDescEllipCoeffsAndLodLvl));
//     g_texDescEllipCoeffsAndLodLvl.addressMode[0] = cudaAddressModeClamp;
//     g_texDescEllipCoeffsAndLodLvl.addressMode[1] = cudaAddressModeClamp;
//     g_texDescEllipCoeffsAndLodLvl.filterMode = cudaFilterModePoint;
//     g_texDescEllipCoeffsAndLodLvl.readMode = cudaReadModeElementType;
//     g_texDescEllipCoeffsAndLodLvl.normalizedCoords = 0;
//     cudaCreateTextureObject(&g_texObjEllipCoeffsAndLodLvl, &g_resDescEllipCoeffsAndLodLvl, &g_texDescEllipCoeffsAndLodLvl, NULL);


// }

GLuint ImMeshRenderer::GenDummyTexture(uint32_t texW, uint32_t texH)
{
    GLuint dummyTexture = 0;

    glCheckError();
    glGenTextures(1, &dummyTexture);
    glBindTexture(GL_TEXTURE_2D, dummyTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texW, texH, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);

    return dummyTexture;
}

void ImMeshRenderer::HashValuePredifinedHandle(thrust::host_vector<HashValPreDifinedDevice>& hashValPredifinedHost, std::vector<VertNbrMemInfoTmp>& vertNbrMemInfoTmp, uint32_t& compactVertBufSize,
thrust::host_vector<TexScaleFactor>& scaleFactorHost, float* texInWorldInfoHostPtr, bool isRenderingMode)
{
    uint32_t curOffset = 0;
    uint32_t allShNumbers = 0;

    for (uint32_t i = 0; i < hashValPredifinedHost.size(); i++) {
        m_shTexHashMap[i].norm[0] = hashValPredifinedHost[i].norm[0];
        m_shTexHashMap[i].norm[1] = hashValPredifinedHost[i].norm[1];
        m_shTexHashMap[i].norm[2] = hashValPredifinedHost[i].norm[2];

        m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW = (uint16_t)hashValPredifinedHost[i].curTexW;
        m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH = (uint16_t)hashValPredifinedHost[i].curTexH;

        m_shTexHashMap[i].resizeInfo.LastTexWH.data.curTexW = (uint16_t)hashValPredifinedHost[i].curTexW;
        m_shTexHashMap[i].resizeInfo.LastTexWH.data.curTexH = (uint16_t)hashValPredifinedHost[i].curTexH;

        m_shTexHashMap[i].resizeInfo.triangleBotDistance = hashValPredifinedHost[i].triangleBotDistance;
        m_shTexHashMap[i].resizeInfo.triangleHeight = hashValPredifinedHost[i].triangleHeight;

        if (isRenderingMode == false) {
            m_shTexHashMap[i].resizeInfo.curDensity = m_initialShDensity;
        } else {
            m_shTexHashMap[i].resizeInfo.curDensity = m_readJsonDataVec[i].shDensity;
        }

        if ((m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW == 0) ||
            (m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH == 0)) {
                printf("triIdx : %d, curTexW : %d, curTexH : %d\n", i, m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW, m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH);
            }
        CurTexWH tmp((uint32_t)m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW, (uint32_t)m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH);
        if (m_texWH2dummyTex.find(tmp) == m_texWH2dummyTex.end()) {
            GLuint curTexId = GenDummyTexture(m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW, m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH);
            assert(curTexId > 0);

            m_texWH2dummyTex.insert(std::pair<CurTexWH, GLuint>(tmp, curTexId));
            m_OpenglDummyTexId.insert(curTexId);

            m_dummyOpenglTexId2TexId.insert(std::make_pair(curTexId, std::unordered_set<uint32_t>()));
            m_dummyOpenglTexId2TexId[curTexId].insert((i + 1));

            printf("gen dummy tex: WH: %d, %d \n", m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW, m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH);
        } else {
            auto find = m_texWH2dummyTex.find(tmp);
            assert(find != m_texWH2dummyTex.end());

            auto find2 = m_dummyOpenglTexId2TexId.find(find->second);
            assert(find2 != m_dummyOpenglTexId2TexId.end());
            find2->second.insert((i + 1));
        }

        m_shTexHashMap[i].neighborTexId[0] = hashValPredifinedHost[i].neighborTexId[0];
        m_shTexHashMap[i].neighborTexId[1] = hashValPredifinedHost[i].neighborTexId[1];
        m_shTexHashMap[i].neighborTexId[2] = hashValPredifinedHost[i].neighborTexId[2];

        m_shTexHashMap[i].neighborEdgeType[0] = hashValPredifinedHost[i].edgeType[0];
        m_shTexHashMap[i].neighborEdgeType[1] = hashValPredifinedHost[i].edgeType[1];
        m_shTexHashMap[i].neighborEdgeType[2] = hashValPredifinedHost[i].edgeType[2];

        m_scaleFactor[i].distPerTexelW = scaleFactorHost[i].distPerTexelW;
        m_scaleFactor[i].distPerTexelH = scaleFactorHost[i].distPerTexelH;

        vertNbrMemInfoTmp[i].offset = curOffset;

        vertNbrMemInfoTmp[i].topPtIdx = hashValPredifinedHost[i].topPtIdx;
        vertNbrMemInfoTmp[i].topVertNum = hashValPredifinedHost[i].topVertNbrNum;

        vertNbrMemInfoTmp[i].leftPtIdx = hashValPredifinedHost[i].leftPtIdx;
        vertNbrMemInfoTmp[i].leftVertNum = hashValPredifinedHost[i].leftVertNbrNum;

        vertNbrMemInfoTmp[i].rightPtIdx = hashValPredifinedHost[i].rightPtIdx;
        vertNbrMemInfoTmp[i].rightVertNum = hashValPredifinedHost[i].rightVertNbrNum;

        curOffset += (vertNbrMemInfoTmp[i].topVertNum + vertNbrMemInfoTmp[i].leftVertNum + vertNbrMemInfoTmp[i].rightVertNum);

        for (uint32_t ii = 0; ii < 6; ii++) {
            m_shTexHashMap[i].texInWorldInfo[ii] = *(texInWorldInfoHostPtr + i * 6 + ii);
        }


        allShNumbers += (m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW * m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH);
    }

    compactVertBufSize = curOffset;

    printf("dummy textures inserted num: %d\n", m_OpenglDummyTexId.size());
    printf("allShNumbers : %d\n", allShNumbers);
}

void ImMeshRenderer::MallocedDataWriteToHashMap(std::vector<VertNbrMemInfoTmp>& vertNbrMemInfoTmp, thrust::host_vector<cuco::pair<int, int>>& vertNbrGetHost)
{
    for (uint32_t i = 0; i < vertNbrMemInfoTmp.size(); i++) {
        for (uint32_t ii = 0; ii < vertNbrMemInfoTmp[i].topVertNum; ii++) {
            if (i == (vertNbrGetHost[vertNbrMemInfoTmp[i].offset + ii].second)) {
                continue;
            }
            m_shTexHashMap[i].topVertNbr.push_back(((uint32_t)(vertNbrGetHost[vertNbrMemInfoTmp[i].offset + ii].second) + 1));
        }

        for (uint32_t ii = 0; ii < vertNbrMemInfoTmp[i].leftVertNum; ii++) {
            if (i == (vertNbrGetHost[vertNbrMemInfoTmp[i].offset + vertNbrMemInfoTmp[i].topVertNum + ii].second)) {
                continue;
            }
            m_shTexHashMap[i].leftVertNbr.push_back(((uint32_t)(vertNbrGetHost[vertNbrMemInfoTmp[i].offset + vertNbrMemInfoTmp[i].topVertNum + ii].second) + 1));
        }

        for (uint32_t ii = 0; ii < vertNbrMemInfoTmp[i].rightVertNum; ii++) {
            if (i == (vertNbrGetHost[vertNbrMemInfoTmp[i].offset + vertNbrMemInfoTmp[i].topVertNum + vertNbrMemInfoTmp[i].leftVertNum + ii].second)) {
                continue;
            }
            m_shTexHashMap[i].rightVertNbr.push_back(((uint32_t)(vertNbrGetHost[vertNbrMemInfoTmp[i].offset + vertNbrMemInfoTmp[i].topVertNum + vertNbrMemInfoTmp[i].leftVertNum + ii].second) + 1));
        }
    }
}

void ImMeshRenderer::WriteCompactBuffersToGlobalHash(thrust::device_vector<float>& shPoseMapBufDev, 
    thrust::device_vector<uint32_t>& shValidCpctOffsets, std::vector<uint32_t>& shValidCpctNumbersHost,
    thrust::device_vector<TexAlignedPosWH>& validShWHMap,
    thrust::host_vector<HashValPreDifinedDevice>& hashValPredifinedHost)
{
    std::vector<float> shPoseMapBufHostStd(shPoseMapBufDev.size());
    checkCudaErrors(cudaMemcpy((char*)shPoseMapBufHostStd.data(), (char*)(thrust::raw_pointer_cast(shPoseMapBufDev.data())), shPoseMapBufDev.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<HashValPreDifinedDevice> hashValPredifinedHostStd(hashValPredifinedHost.size());

    memcpy((char*)hashValPredifinedHostStd.data(),
           (char*)(thrust::raw_pointer_cast(hashValPredifinedHost.data())),
           hashValPredifinedHost.size() * sizeof(HashValPreDifinedDevice));

    std::vector<uint32_t> shValidCpctOffsetsHost(shValidCpctOffsets.size());
    checkCudaErrors(cudaMemcpy((char*)(shValidCpctOffsetsHost.data()), (char*)(thrust::raw_pointer_cast(shValidCpctOffsets.data())), shValidCpctOffsets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::vector<TexAlignedPosWH> validShWHMapHost(validShWHMap.size());
    checkCudaErrors(cudaMemcpy((char*)(validShWHMapHost.data()), (char*)(thrust::raw_pointer_cast(validShWHMap.data())), validShWHMap.size() * sizeof(TexAlignedPosWH), cudaMemcpyDeviceToHost));

    parallel_for(blocked_range<size_t>(0, hashValPredifinedHostStd.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            uint32_t curTexValidShNum = shValidCpctNumbersHost[i];

            m_shTexHashMap[i].perTexOffset = sizeof(float) * m_shTexChannelNum * m_shLayerNum * curTexValidShNum;

            m_shTexHashMap[i].shTexValue = (float *)malloc(m_shTexHashMap[i].perTexOffset);
            memset((char *)(m_shTexHashMap[i].shTexValue), 0, m_shTexHashMap[i].perTexOffset);

            m_shTexHashMap[i].adamStateExpAvg = (float *)malloc(m_shTexHashMap[i].perTexOffset);
            memset((char *)(m_shTexHashMap[i].adamStateExpAvg), 0, m_shTexHashMap[i].perTexOffset);

            m_shTexHashMap[i].adamStateExpAvgSq = (float *)malloc(m_shTexHashMap[i].perTexOffset);
            memset((char *)(m_shTexHashMap[i].adamStateExpAvgSq), 0, m_shTexHashMap[i].perTexOffset);

            m_shTexHashMap[i].worldPoseMap = (float *)malloc(sizeof(float) * 3 * curTexValidShNum);
            memset((char *)m_shTexHashMap[i].worldPoseMap, 0, sizeof(float) * 3 * curTexValidShNum);
            memcpy((char *)m_shTexHashMap[i].worldPoseMap,
                (char *)((float*)(shPoseMapBufHostStd.data()) + 3 * shValidCpctOffsetsHost[i]),
                sizeof(float) * 3 * curTexValidShNum);

            m_shTexHashMap[i].validShWHMap = (TexAlignedPosWH*)malloc(sizeof(TexAlignedPosWH) * curTexValidShNum);
            memset((char*)(m_shTexHashMap[i].validShWHMap), 0, sizeof(TexAlignedPosWH) * curTexValidShNum);
            memcpy((char*)(m_shTexHashMap[i].validShWHMap),
                (char*)((TexAlignedPosWH*)(validShWHMapHost.data()) + shValidCpctOffsetsHost[i]),
                sizeof(TexAlignedPosWH) * curTexValidShNum);
            m_shTexHashMap[i].validShNum = curTexValidShNum;

            m_shTexHashMap[i].cornerArea[0] = hashValPredifinedHostStd[i].cornerArea[0];
            m_shTexHashMap[i].cornerArea[1] = hashValPredifinedHostStd[i].cornerArea[1];

            float curTexelH = 1.f / (float)(m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH);
            m_shTexHashMap[i].topYCoef = 1.f - 0.5f * curTexelH;
            m_shTexHashMap[i].bottomYCoef = 0.5f * curTexelH;
        }
    });
}

void ImMeshRenderer::LoadMesh(std::string path, MeshType meshType, MeshMultimaps& multimaps)
{
    PTexMesh meshReader;
    thrust::device_vector<TriangleInfoDevice> triangleArrRaw;

    meshReader.LoadMeshData(path, 3.f, meshType);
    uint32_t numFaces = meshReader.GetNumFaces(meshType);
    assert(numFaces > 0);

    // Create the multimaps and store them in unique_ptrs
    multimaps.edgeNbrMultiMapCuda = std::make_unique<cuco::static_multimap<edgeMapKeyType, edgeMapValueType, cuda::thread_scope_device, cuco::cuda_allocator<char>, edgeMapProbe>>(
        numFaces * 6,
        cuco::empty_key{emptyEdgeKeySentinel},
        cuco::empty_value{emptyEdgeValueSentinel}
    );
    
    multimaps.vertNbrMultiMapCuda = std::make_unique<cuco::static_multimap<vertMapKeyType, vertMapValueType, cuda::thread_scope_device, cuco::cuda_allocator<char>, vertMapProbe>>(
        numFaces * 7,
        cuco::empty_key{emptyVertKeySentinel},
        cuco::empty_value{emptyVertValueSentinel}
    );

    // Pass dereferenced multimaps to CalcEachTriangle
    meshReader.CalcEachTriangle(*multimaps.edgeNbrMultiMapCuda, *multimaps.vertNbrMultiMapCuda, triangleArrRaw, meshType);

    m_triangleArr = triangleArrRaw;
}

bool ImMeshRenderer::CheckCudaResultsWithJsonData(thrust::device_vector<HashValPreDifinedDevice>& hashValPredifined)
{
    thrust::host_vector<HashValPreDifinedDevice> hashValPredifinedHost = hashValPredifined;

    for (uint32_t i = 0; i < hashValPredifinedHost.size(); i++) {
        if (hashValPredifinedHost[i].curTexW != m_readJsonDataVec[i].texW ||
            hashValPredifinedHost[i].curTexH != m_readJsonDataVec[i].texH) {
                std::cerr << "Error: Mismatch in hashValPredifined and readJsonData at index " << i << 
                          " - curTexW: " << hashValPredifinedHost[i].curTexW << 
                          ", curTexH: " << hashValPredifinedHost[i].curTexH << 
                          ", readJsonData texW: " << m_readJsonDataVec[i].texW << 
                          ", texH: " << m_readJsonDataVec[i].texH << std::endl;

                hashValPredifinedHost[i].curTexW = m_readJsonDataVec[i].texW;
                hashValPredifinedHost[i].curTexH = m_readJsonDataVec[i].texH;
                // return false;
        }
    }

    hashValPredifined = hashValPredifinedHost;

    return true;
}

bool ImMeshRenderer::LoadShTexturesFromBin()
{
    std::ifstream binFile(m_algoConfig.savedBinPath, std::ios::binary);
    if (!binFile.is_open()) {
        std::cerr << "Error: Could not open binary file " << m_algoConfig.savedBinPath << std::endl;
        return false;
    }

    size_t numElements;
    binFile.read(reinterpret_cast<char*>(&numElements), sizeof(numElements));
    if (binFile.gcount() != sizeof(numElements)) {
        std::cerr << "Error: Could not read number of elements from binary file" << std::endl;
        return false;
    }

    std::cout << "Binary file: " << numElements << " elements expected" << std::endl;

    if (numElements != m_shTexHashMap.size()) {
        std::cerr << "Error: Mismatch between JSON totalEntries (" << m_shTexHashMap.size() 
                  << ") and binary numElements (" << numElements << ")" << std::endl;
        return false;
    }

    for (size_t i = 0; i < numElements; ++i) {
        uint32_t magic;
        binFile.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (binFile.gcount() != sizeof(magic) || magic != 0xDEADBEEF) {
            std::cerr << "Error: Invalid magic number for element " << i << std::endl;
            return false;
        }
        
        // Read index
        uint32_t index;
        binFile.read(reinterpret_cast<char*>(&index), sizeof(index));
        if (binFile.gcount() != sizeof(index)) {
            std::cerr << "Error: Could not read index for element " << i << std::endl;
            return false;
        }

        if (index != i) {
            std::cerr << "Error: Index mismatch for element " << i 
                      << " (expected: " << i << ", got: " << index << ")" << std::endl;
            return false;
        }
        
        // Read perTexOffset
        uint32_t perTexOffset;
        binFile.read(reinterpret_cast<char*>(&perTexOffset), sizeof(perTexOffset));
        if (binFile.gcount() != sizeof(perTexOffset)) {
            std::cerr << "Error: Could not read perTexOffset for element " << i << std::endl;
            return false;
        }
        
        // Validate index bounds
        if (index >= numElements) {
            std::cerr << "Error: Index " << index << " out of bounds (max: " << numElements - 1 << ")" << std::endl;
            return false;
        }

        if (m_shTexHashMap[i].perTexOffset != perTexOffset) {
            std::cerr << "Error: Mismatch in perTexOffset for element " << i 
                      << " (expected: " << m_shTexHashMap[i].perTexOffset 
                      << ", got: " << perTexOffset << ")" << std::endl;
            return false;
        }

        if (perTexOffset > 0) {
            // Allocate CPU memory for shTexValue (number of floats * sizeof(float))
            // Note: perTexOffset is in bytes, so we need to ensure proper alignment for float array
            size_t numFloats = perTexOffset / sizeof(float);
            if (perTexOffset % sizeof(float) != 0) {
                std::cerr << "Warning: perTexOffset " << perTexOffset << " is not aligned to float size for index " << index << std::endl;
            }
            
            // Read data directly into the allocated CPU memory
            binFile.read(reinterpret_cast<char*>(m_shTexHashMap[i].shTexValue), perTexOffset);
            if (binFile.gcount() != static_cast<std::streamsize>(perTexOffset)) {
                std::cerr << "Error: Could not read " << perTexOffset << " bytes of data for element " << i << std::endl;
                return false;
            }
        }
    }

    binFile.close();
    return true;
}

void ImMeshRenderer::InitVerts(std::string path, MeshType meshType)
{
    unsigned int cntIdx0 = 0;
    unsigned int cntIdx1 = 0;
    unsigned int cntIdx2 = 0;

    MeshMultimaps multimaps;
    LoadMesh(path, meshType, multimaps);

    unsigned long startTime = GetTimeMS();
    unsigned long endTime = GetTimeMS();

    thrust::device_vector<float> newVertexBufDevice(m_triangleArr.size() * 5 * 3);
    checkCudaErrors(cudaMemset(thrust::raw_pointer_cast(&newVertexBufDevice[0]), 0, sizeof(float) * m_triangleArr.size() * 5 * 3));

    float* texInWorldInfoPtr{nullptr};
    
    checkCudaErrors(cudaMalloc(&texInWorldInfoPtr, sizeof(float) * 6 * m_triangleArr.size()));
    checkCudaErrors(cudaMemset(texInWorldInfoPtr, 0, sizeof(float) * 6 * m_triangleArr.size()));

    thrust::device_vector<TexScaleFactor> scaleFactorDevice(m_triangleArr.size());
    thrust::device_vector<HashValPreDifinedDevice> hashValPredifined(m_triangleArr.size());
    thrust::device_vector<uint32_t> idxCntReducePoolCuda(m_triangleArr.size() * 3);
    thrust::device_vector<uint32_t> invalidKCntReducePoolCuda(m_triangleArr.size());
    thrust::device_vector<uint8_t> flipCountReducePoolCuda(m_triangleArr.size());
    thrust::device_vector<float> deltaLenReducePoolCuda(m_triangleArr.size());

    uint32_t constexpr blockSize = 64;
    uint32_t const gridSize = (m_triangleArr.size() + blockSize - 1) / blockSize;

    assert(multimaps.edgeNbrMultiMapCuda->cg_size() == multimaps.vertNbrMultiMapCuda->cg_size());
    assert(multimaps.edgeNbrMultiMapCuda->cg_size() == 1);

    auto edgeMapView = multimaps.edgeNbrMultiMapCuda->get_device_view();
    auto vertMapView = multimaps.vertNbrMultiMapCuda->get_device_view();

    startTime = GetTimeMS();

    my_atomic_ctr_type* myCountersDummy;
    checkCudaErrors(cudaMalloc(&myCountersDummy, 3 * m_triangleArr.size() * sizeof(my_atomic_ctr_type)));
    
    if (m_algoConfig.isRenderingMode == true) {
        if (m_triangleArr.size() != m_readJsonDataVec.size()) {
            std::cerr << "Error: m_triangleArr.size() != m_readJsonDataVec.size() in InitVerts!" << std::endl;
            exit(-1);
        }

        thrust::device_vector<float> shDensitiesLoaded(m_triangleArr.size());
        thrust::host_vector<float> shDensitiesLoadedHost(m_triangleArr.size());

        for (uint32_t i = 0; i < m_triangleArr.size(); i++) {
            shDensitiesLoadedHost[i] = m_readJsonDataVec[i].shDensity;
        }

        shDensitiesLoaded = shDensitiesLoadedHost;

        assign_predefined_and_calc_alloc_size_eval <1, 1, 1> <<<gridSize, blockSize>>> (
            edgeMapView,
            vertMapView,
            m_triangleArr.begin(), m_triangleArr.size(),
            scaleFactorDevice.begin(),
            hashValPredifined.begin(),
            thrust::raw_pointer_cast(newVertexBufDevice.data()),
            EdgeKeyEqual{},
            thrust::raw_pointer_cast(idxCntReducePoolCuda.data()),
            thrust::raw_pointer_cast(invalidKCntReducePoolCuda.data()),
            thrust::raw_pointer_cast(flipCountReducePoolCuda.data()),
            deltaLenReducePoolCuda.begin(),
            myCountersDummy,
            texInWorldInfoPtr,
            thrust::raw_pointer_cast(shDensitiesLoaded.data()),
            m_maxTexWH
        );

        checkCudaErrors(cudaDeviceSynchronize());

        bool ret = CheckCudaResultsWithJsonData(hashValPredifined);
        if (ret == false) {
            std::cerr << "Error: CheckCudaResultsWithJsonData failed!" << std::endl;
            exit(-1);
        }

    } else {
        assign_predefined_and_calc_alloc_size <1, 1, 1> <<<gridSize, blockSize>>> (
            edgeMapView,
            vertMapView,
            m_triangleArr.begin(), m_triangleArr.size(),
            scaleFactorDevice.begin(),
            hashValPredifined.begin(),
            thrust::raw_pointer_cast(newVertexBufDevice.data()),
            EdgeKeyEqual{},
            thrust::raw_pointer_cast(idxCntReducePoolCuda.data()),
            thrust::raw_pointer_cast(invalidKCntReducePoolCuda.data()),
            thrust::raw_pointer_cast(flipCountReducePoolCuda.data()),
            deltaLenReducePoolCuda.begin(),
            myCountersDummy,
            texInWorldInfoPtr,
            m_initialShDensity,
            m_maxTexWH
        );

        checkCudaErrors(cudaDeviceSynchronize());
    }
    

    endTime = GetTimeMS();

    PROFILE_PRINT(std::cout << "------ cuda AssignVertexBuf time : " << (double)(endTime - startTime) << " ms" << std::endl);

    thrust::host_vector<uint32_t> idxCntReducePool = idxCntReducePoolCuda;
    thrust::host_vector<uint32_t> invalidKCntReducePool = invalidKCntReducePoolCuda;
    thrust::host_vector<uint8_t> flipCountReducePool = flipCountReducePoolCuda;
    thrust::host_vector<float> deltaLenReducePool = deltaLenReducePoolCuda;

    uint32_t invalidNumK = 0;
    uint32_t flipCount = 0;
    float maxDeltaLen = -1.0f;

    checkCudaErrors(cudaDeviceSynchronize());

    m_shTexHashMap.resize(m_triangleArr.size());

    for (uint32_t ii = 0; ii < m_triangleArr.size(); ii++) {
        cntIdx0 += idxCntReducePool[ii * 3 + 0];
        cntIdx1 += idxCntReducePool[ii * 3 + 1];
        cntIdx2 += idxCntReducePool[ii * 3 + 2];

        invalidNumK += invalidKCntReducePool[ii];
        if (invalidKCntReducePool[ii] != 0) {
            printf("invalid K triarr idx: %d\n", ii);
            m_shTexHashMap[ii].resizeInfo.isConverge = true;
        }

        flipCount += flipCountReducePool[ii];

        if (deltaLenReducePool[ii] - maxDeltaLen > 1e-3f) {
            maxDeltaLen = deltaLenReducePool[ii];
        }
    }

    cout << " glm sync cntIdx0 : " << cntIdx0 << " cntIdx1: " << cntIdx1 << " cntIdx2:" << cntIdx2 << " flip time: " << flipCount << " invalidNumK : " << invalidNumK << " maxDeltaLen " << maxDeltaLen << endl;


    m_resizeInfoForCuda.resize(m_triangleArr.size());
    memset(thrust::raw_pointer_cast(m_resizeInfoForCuda.data()), 0, sizeof(ResizeInfo) * m_triangleArr.size());

    m_scaleFactor.resize(m_triangleArr.size());

    cout << "init hash finish, start to init opengl vert buffer\n";

    thrust::host_vector<HashValPreDifinedDevice> hashValPredifinedHost = hashValPredifined;
    thrust::host_vector<TexScaleFactor> scaleFactorHost = scaleFactorDevice;
    float* texInWorldInfoHostPtr = (float *)malloc(sizeof(float) * 6 * m_triangleArr.size());
    memset(texInWorldInfoHostPtr, 0, sizeof(float) * 6 * m_triangleArr.size());
    checkCudaErrors(cudaMemcpy(texInWorldInfoHostPtr, texInWorldInfoPtr, sizeof(float) * 6 * m_triangleArr.size(), cudaMemcpyDeviceToHost));


    startTime = GetTimeMS();

    std::vector<VertNbrMemInfoTmp> vertNbrMemInfoTmp(hashValPredifinedHost.size());
    uint32_t compactVertBufSize = 0;


    HashValuePredifinedHandle(hashValPredifinedHost,
        vertNbrMemInfoTmp, compactVertBufSize,
        scaleFactorHost,
        texInWorldInfoHostPtr,
        m_algoConfig.isRenderingMode);

    endTime = GetTimeMS();

    std::cout << "------ HashValuePredifinedHandle time : " << (double)(endTime - startTime) << " ms" << std::endl;

    endTime = GetTimeMS();

    std::cout << "------ HashValuePredifinedHandle cudaStreamSynchronize time : " << (double)(endTime - startTime) << " ms" << std::endl;

    checkCudaErrors(cudaDeviceSynchronize());

    startTime = GetTimeMS();

    thrust::device_vector<cuco::pair<int, int>> vertNbrGet(compactVertBufSize);

    thrust::device_vector<VertNbrMemInfoTmp> vertNbrMemInfoTmpDevice = vertNbrMemInfoTmp;

    uint32_t constexpr blockSize2 = 32;
    uint32_t const gridSize2 = (m_triangleArr.size() + blockSize2 - 1) / blockSize2;


    checkCudaErrors(cudaMemset(myCountersDummy, 0, 3 * m_triangleArr.size() * sizeof(my_atomic_ctr_type)));

    // multimaps.vertNbrMultiMapCuda->cg_size() == 1
    vert_map_nbr_write_back<blockSize2, 1, 1, 10> <<<gridSize2, blockSize2>>> (
        m_triangleArr.size(),
        vertNbrGet.begin(),
        vertMapView,
        vertNbrMemInfoTmpDevice.begin(),
        thrust::equal_to<int>{},
        myCountersDummy);

    checkCudaErrors(cudaDeviceSynchronize());

    endTime = GetTimeMS();

    std::cout << "------ vert_map_nbr_write_back time : " << (double)(endTime - startTime) << " ms" << std::endl;


    startTime = GetTimeMS();
    thrust::host_vector<cuco::pair<int, int>> vertNbrGetHost = vertNbrGet;
    MallocedDataWriteToHashMap(vertNbrMemInfoTmp, vertNbrGetHost);

    endTime = GetTimeMS();

    std::cout << "------ MallocedDataWriteToHashMap time : " << (double)(endTime - startTime) << " ms" << std::endl;

    uint32_t compactShValidMapBufSize = 0;
    thrust::device_vector<uint32_t> shValidCpctOffsets(hashValPredifinedHost.size());
    thrust::device_vector<uint32_t> shValidCpctNumbers(hashValPredifinedHost.size());
    thrust::device_vector<CurTexAlignedWH> allTriangleWHs(m_triangleArr.size());

    thrust::transform(hashValPredifined.begin(), hashValPredifined.end(), allTriangleWHs.begin(), CopyWHsFromHashValPreDifinedDeviceFunc());

    calc_valid_sh_number <<<gridSize2, blockSize2>>> (m_triangleArr.begin(),
        (CurTexAlignedWH*)(thrust::raw_pointer_cast(allTriangleWHs.data())),
        m_triangleArr.size(),
        thrust::raw_pointer_cast(shValidCpctNumbers.data()));
    
    checkCudaErrors(cudaDeviceSynchronize());
    std::vector<uint32_t> shValidCpctNumbersHost(shValidCpctNumbers.size());
    checkCudaErrors(cudaMemcpy(shValidCpctNumbersHost.data(), (char*)(thrust::raw_pointer_cast(shValidCpctNumbers.data())), shValidCpctNumbers.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint32_t validShNumSum = 0;
    for (auto it: shValidCpctNumbersHost) {
        validShNumSum += it;
    }

    thrust::exclusive_scan(shValidCpctNumbers.begin(), shValidCpctNumbers.end(), shValidCpctOffsets.begin());

    printf("valid SH num sum: %d\n", validShNumSum);

    thrust::device_vector<float> shPoseMapBufDev(validShNumSum * 3);
    thrust::device_vector<TexAlignedPosWH> validShWHMap(validShNumSum);

    checkCudaErrors(cudaMemset((char *)(thrust::raw_pointer_cast(shPoseMapBufDev.data())), 0, sizeof(float) * validShNumSum * 3));
    checkCudaErrors(cudaMemset((char *)(thrust::raw_pointer_cast(validShWHMap.data())), 0, sizeof(TexAlignedPosWH) * validShNumSum));

    assign_corerArea_posMap_validMap <<< gridSize2, blockSize2 >>> (m_triangleArr.begin(),
        hashValPredifined.begin(),
        (float *)(thrust::raw_pointer_cast(shPoseMapBufDev.data())),
        (uint32_t *)(thrust::raw_pointer_cast(shValidCpctOffsets.data())),
        (TexAlignedPosWH *)(thrust::raw_pointer_cast(validShWHMap.data())),
        m_triangleArr.size());

    checkCudaErrors(cudaDeviceSynchronize());

    endTime = GetTimeMS();

    std::cout << "------till assign_corerArea_posMap_validMap time : " << (double)(endTime - startTime) << " ms" << std::endl;

    hashValPredifinedHost = hashValPredifined;

    WriteCompactBuffersToGlobalHash(shPoseMapBufDev, shValidCpctOffsets, shValidCpctNumbersHost, validShWHMap, hashValPredifinedHost);

    if (m_algoConfig.isRenderingMode == true) {
        bool ret = LoadShTexturesFromBin();
        if (ret == false) {
            std::cerr << "Error: LoadShTexturesFromBin failed!" << std::endl;
            exit(-1);
        }
    }

    endTime = GetTimeMS();

    std::cout << "------till WriteCompactBuffersToGlobalHash time : " << (double)(endTime - startTime) << " ms" << std::endl;

    checkCudaErrors(cudaFree(myCountersDummy));
    checkCudaErrors(cudaFree(texInWorldInfoPtr));
    free(texInWorldInfoHostPtr);

    endTime = GetTimeMS();

    std::cout << "------ parallel_for AssignVertexBuf time : " << (double)(endTime - startTime) << " ms" << std::endl;

    thrust::host_vector<float> newVertexBufHost = newVertexBufDevice;
    m_vertBufData.assign(newVertexBufHost.begin(), newVertexBufHost.end());

    cout << "init opengl vert buffer end\n";

    m_texNum = newVertexBufHost.size() / 5 / 3;

    ASSERT(m_texNum == m_scaleFactor.size());
    ASSERT(m_texNum == m_triangleArr.size());
}

void ImMeshRenderer::WindowInit()
{
    glCheckError();

    GLint pack_alignment = 0;
    glGetIntegerv(GL_PACK_ALIGNMENT, &pack_alignment);
    GLint unpack_alignment = 0;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &unpack_alignment);

    cout << "opengl pack info : unpack : " << unpack_alignment << " pack : " << pack_alignment << endl;

    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    GLint maxAttach = 0;
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxAttach);

    GLint maxDrawBuf = 0;
    glGetIntegerv(GL_MAX_DRAW_BUFFERS, &maxDrawBuf);

    cout << "opengl maxAttach : << " << maxAttach << " maxDrawBuf << " << maxDrawBuf << endl;

    if (pack_alignment == 0 || unpack_alignment == 0) {
        cout << "opengl not successfully init! \n";
        exit(0);
    }
}

void ImMeshRenderer::EGLInit()
{
    // /usr/lib/x86_64-linux-gnu/libEGL.so
    const std::string lib("libEGL.so");
    m_handle = dlopen(lib.c_str(), RTLD_LAZY);

    if (nullptr == m_handle)
        m_handle = dlopen(
            "/usr/local/lib/libEGL.so",
            RTLD_LAZY); // dgx machines have this location, which is not on the lib search path

    ASSERT(m_handle, "Can't find " + lib + ", " + dlerror());

    dlerror(); // Clear any existing error

    char* error = NULL;

    // eglGetCurrentContext = (EGLContext(*)(void))dlsym(m_handle, "eglGetCurrentContext");
    // error = dlerror();
    // ASSERT(error == NULL, "Error loading eglGetCurrentContext from " + lib + ", " + error);

    eglInitialize = (EGLBoolean(*)(EGLDisplay, EGLint*, EGLint*))dlsym(m_handle, "eglInitialize");
    error = dlerror();
    ASSERT(error == NULL, "Error loading eglInitialize from " + lib + ", " + error);

    eglChooseConfig = (EGLBoolean(*)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*))dlsym(
        m_handle, "eglChooseConfig");
    error = dlerror();
    ASSERT(error == NULL, "Error loading eglChooseConfig from " + lib + ", " + error);

    eglGetProcAddress =
        (__eglMustCastToProperFunctionPointerType(*)(const char*))dlsym(m_handle, "eglGetProcAddress");
    error = dlerror();
    ASSERT(error == NULL, "Error loading eglGetProcAddress from " + lib + ", " + error);

    // eglCreatePbufferSurface =
    //     (EGLSurface(*)(EGLDisplay, EGLConfig, const EGLint*))dlsym(m_handle, "eglCreatePbufferSurface");
    // error = dlerror();
    // ASSERT(error == NULL, "Error loading eglCreatePbufferSurface from " + lib + ", " + error);

    eglBindAPI = (EGLBoolean(*)(EGLenum))dlsym(m_handle, "eglBindAPI");
    error = dlerror();
    ASSERT(error == NULL, "Error loading eglBindAPI from " + lib + ", " + error);

    eglCreateContext = (EGLContext(*)(EGLDisplay, EGLConfig, EGLContext, const EGLint*))dlsym(
        m_handle, "eglCreateContext");
    error = dlerror();
    ASSERT(error == NULL, "Error loading eglCreateContext from " + lib + ", " + error);

    eglMakeCurrent = (EGLBoolean(*)(EGLDisplay, EGLSurface, EGLSurface, EGLContext))dlsym(
        m_handle, "eglMakeCurrent");
    error = dlerror();
    ASSERT(error == NULL, "Error loading eglMakeCurrent from " + lib + ", " + error);

    EGLDeviceEXT eglDevs[32];
    EGLint numDevices;

    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");

    eglQueryDevicesEXT(32, eglDevs, &numDevices);

    ASSERT(numDevices, "Found no GPUs");

    PFNEGLQUERYDEVICEATTRIBEXTPROC eglQueryDeviceAttribEXT =
        reinterpret_cast<PFNEGLQUERYDEVICEATTRIBEXTPROC>(
            eglGetProcAddress("eglQueryDeviceAttribEXT"));

    int cudaDevice = 0;
    int eglDevId = 0;
    bool foundCudaDev = false;
    // Find the CUDA device asked for
    for (; eglDevId < numDevices; ++eglDevId) {
      EGLAttrib cudaDevNumber;

      if (eglQueryDeviceAttribEXT(eglDevs[eglDevId], EGL_CUDA_DEVICE_NV, &cudaDevNumber) ==
          EGL_FALSE)
        continue;

      if (cudaDevNumber == cudaDevice) {
        break;
      }
    }

    printf("use dev : %d, found %d\n", eglDevId, foundCudaDev);


    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    m_eglMainThrdDsp = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevs[eglDevId], 0);

    EGLint major, minor;

    auto ret = eglInitialize(m_eglMainThrdDsp, &major, &minor);
    if (ret != EGL_TRUE) {
        printf("eglInitialize fail ! err: %d\n", (uint32_t)ret);
    }

    EGLint numConfigs;
    EGLConfig eglCfg;
    eglChooseConfig(m_eglMainThrdDsp, configAttribs, &eglCfg, 1, &numConfigs);

    eglBindAPI(EGL_OPENGL_API);

    m_mainThrdCtx = eglCreateContext(m_eglMainThrdDsp, eglCfg, EGL_NO_CONTEXT, NULL);

    eglMakeCurrent(m_eglMainThrdDsp, EGL_NO_SURFACE, EGL_NO_SURFACE, m_mainThrdCtx);

    GLenum err = glewInit();
    // if (err == GLEW_ERROR_NO_GLX_DISPLAY) {
    //     std::cout << "Can't initialize EGL GLEW GLX display, may crash!" << std::endl;
    // } else if (err != GLEW_OK) {
    //     ASSERT(false, "Can't initialize EGL, glewInit failing completely.");
    // }
    printf("err = glewInit() %d\n", (uint32_t)err);
    checkCudaErrors(cudaSetDevice(0));
}

void ImMeshRenderer::ShaderInit(bool isInfer)
{
    const GLchar * vsCode = g_vertexShaderText.c_str();
    GLint compiled = 0;
   
    if (isInfer == true) {
        m_vertexShaderInfer = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(m_vertexShaderInfer, 1, &vsCode, NULL);
        glCompileShader(m_vertexShaderInfer);
        glGetShaderiv(m_vertexShaderInfer, GL_COMPILE_STATUS, &compiled);
    } else {
        m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(m_vertexShader, 1, &vsCode, NULL);
        glCompileShader(m_vertexShader);
        glGetShaderiv(m_vertexShader, GL_COMPILE_STATUS, &compiled);
    }

    if (compiled == GL_FALSE) {
        char szLog[10240] = { 0 }; //设置buf大小存储信息长度
        GLsizei logLen = 0;//返回的实际错误信息长度

        if (isInfer == true) {
            glGetShaderInfoLog(m_vertexShaderInfer, 10240, &logLen, szLog);
        } else {
            glGetShaderInfoLog(m_vertexShader, 10240, &logLen, szLog);
        }
        
        std::cout << "Compile vs shader fail error log is " <<  szLog << std::endl;
        exit(1);
    }

    const GLchar * fsCode = g_fragmentShaderText.c_str();
    compiled = 0;

    if (isInfer == true) {
        m_fragmentShaderInfer = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(m_fragmentShaderInfer, 1, &fsCode, NULL);
        glCompileShader(m_fragmentShaderInfer);
        glGetShaderiv(m_fragmentShaderInfer, GL_COMPILE_STATUS, &compiled);
    } else {
        m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);    
        glShaderSource(m_fragmentShader, 1, &fsCode, NULL);
        glCompileShader(m_fragmentShader);
        glGetShaderiv(m_fragmentShader, GL_COMPILE_STATUS, &compiled);
    }
       
    if (compiled == GL_FALSE) {
        char szLog[10240] = { 0 }; //设置buf大小存储信息长度
        GLsizei logLen = 0;//返回的实际错误信息长度

        if (isInfer == true) {
            glGetShaderInfoLog(m_fragmentShaderInfer, 10240, &logLen, szLog);
        } else {
            glGetShaderInfoLog(m_fragmentShader, 10240, &logLen, szLog);
        }
        
        std::cout << "Compile fs shader fail error log is " <<  szLog << std::endl;
        exit(1);
    }

    if (isInfer == true) {
        m_programInfer = glCreateProgram();
        glAttachShader(m_programInfer, m_vertexShaderInfer);
        glAttachShader(m_programInfer, m_fragmentShaderInfer);
        glLinkProgram(m_programInfer);
    } else {
        m_program = glCreateProgram();
        glAttachShader(m_program, m_vertexShader);
        glAttachShader(m_program, m_fragmentShader);
        glLinkProgram(m_program);
    }


    glCheckError();

}

void ImMeshRenderer::SaveImg(std::string dir, std::string name, at::Tensor img)
{
    if (!std::filesystem::exists(dir)) {
        if (std::filesystem::create_directory(dir)) {
            std::cout << "model save path create success" << std::endl;
        } else {
            std::cout << "model save path is not valid!\n";
            return;
        }
    }

    std::string fullPath = dir + name;

    torch::Tensor uchar8Tensor = img.to(torch::kU8).to(torch::kCPU);

    uchar8Tensor = uchar8Tensor.permute({1, 2, 0}); // NCHW -> NHWC

    uchar8Tensor = at::flip(uchar8Tensor, -1).contiguous();

    cv::Mat image(uchar8Tensor.sizes().data()[0], uchar8Tensor.sizes().data()[1], CV_8UC3, uchar8Tensor.data_ptr<uchar>());
    cv::imwrite(fullPath, image);
}

void ImMeshRenderer::ImgSaveCUDA(std::string dir, std::string name, bool needAddCnt) // defalut is true
{
    if (!std::filesystem::exists(dir)) {
        if (std::filesystem::create_directory(dir)) {
            std::cout << "model save path create success" << std::endl;
        } else {
            std::cout << "model save path is not valid!\n";
            return;
        }
    }

    PROFILE_PRINT(cout << "save image : " << savedImgCnt << std::endl);

    std::string file_name = "offscreen__" + std::to_string(savedImgCnt);

    if (!name.empty()) {
        file_name = file_name + name;
    }

    file_name = file_name + ".png";

    std::string fullPath = dir + file_name;

    at::Tensor cloned = m_renderedPixelsCUDA.clone().detach();

    if ((m_width != m_gtW) && (m_height != m_gtH)) {
        cloned = at::avg_pool2d(cloned, {2, 2}, {2, 2});
    }

    torch::Tensor uchar8Tensor = cloned.to(torch::kU8).to(torch::kCPU);

    uchar8Tensor = uchar8Tensor.permute({1, 2, 0}); // NCHW -> NHWC

    uchar8Tensor = at::flip(uchar8Tensor, -1).contiguous();

    cv::Mat image(uchar8Tensor.sizes().data()[0], uchar8Tensor.sizes().data()[1], CV_8UC3, uchar8Tensor.data_ptr<uchar>());
    cv::imwrite(fullPath, image);

    if (needAddCnt == true) {
        savedImgCnt++;
    }
}

void ImMeshRenderer::ImgSaveFromOPGL(std::string dir)
{
    if (!std::filesystem::exists(dir)) {
        if (std::filesystem::create_directory(dir)) {
            std::cout << "model save path create success" << std::endl;
        } else {
            std::cout << "model save path is not valid!\n";
            return;
        }
    }

    PROFILE_PRINT(cout << "save image : " << savedImgCnt << std::endl);
    float* buffer = (float*)malloc(m_width * m_height * m_channelNum * sizeof(char));
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);

    // Write image Y-flipped because OpenGL
    std::string file_name = "offscreen_OPGL__" + std::to_string(savedImgCnt) + ".png";

    std::string fullPath = dir + file_name;
    stbi_write_png(fullPath.c_str(),
                m_width, m_height, m_channelNum,
                buffer,
                m_width * m_channelNum * sizeof(char));

    free(buffer);
    savedImgCnt++;
}

void ImMeshRenderer::DrawPrepare(SceneType sceneType, bool isInfer)
{
    glCheckError();

    glViewport(0, 0, m_width, m_height);
    // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glCheckError();

    float yFovRad = atan2(m_height / 2, m_fyRender) * 2.0f;
    float xFovRad = atan2(m_width / 2, m_fxRender) * 2.0f;

    float fov = 120.0f;
    float fovRad = fov * (M_PI / 180.0f);
    if (sceneType == SCENE_TYPE_BLENDER_OUT) {
        mat4x4_perspective_wan(m_perspectiveMat, fovRad, fovRad, 0.1f, 100.0f);    
    } else {
        mat4x4_perspective_from_intrinsic(m_perspectiveMat, m_fxRender, m_fyRender, m_cxRender, m_cyRender, 0.1f, 100.0f, m_width, m_height);
    }

    if (isInfer == true) {
        m_modelLocationInfer = glGetUniformLocation(m_programInfer, "model");
        m_viewLocationInfer = glGetUniformLocation(m_programInfer, "view");
        m_projLocationInfer = glGetUniformLocation(m_programInfer, "proj");
        m_camWorldPosLocationInfer = glGetUniformLocation(m_programInfer, "camWorldPos");

    } else {
        m_modelLocation = glGetUniformLocation(m_program, "model");
        m_viewLocation = glGetUniformLocation(m_program, "view");
        m_projLocation = glGetUniformLocation(m_program, "proj");
        m_camWorldPosLocation = glGetUniformLocation(m_program, "camWorldPos");
    }

    glCheckError();
}

void ImMeshRenderer::ShTextureTensorInitCUDADummy()
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    m_shTensorAllMergeCUDA = torch::zeros({1, m_shLayerNum, 2, 2, m_shTexChannelNum}, options).fill_(0.5f);
    m_shTensorAllMergeCUDA = m_shTensorAllMergeCUDA.contiguous().set_requires_grad(true);

    cout << "init  dummy shTensor shape : " << m_shTensorAllMergeCUDA.sizes() << " elem size : " << m_shTensorAllMergeCUDA.element_size() << " ele num : " << m_shTensorAllMergeCUDA.numel() << " is contiguous:  " << m_shTensorAllMergeCUDA.is_contiguous() << endl;
}

void ImMeshRenderer::CudaInit()
{
    checkCudaErrors(cudaSetDevice(0));

    checkCudaErrors(cudaMalloc((void**)&m_devErrCntPtr, sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(m_devErrCntPtr, 0, sizeof(unsigned int)));

    checkCudaErrors(cudaMalloc((void**)&m_devErrCntBackwardPtr, sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(m_devErrCntBackwardPtr, 0, sizeof(unsigned int)));
}

void ImMeshRenderer::DoDraw(mat4x4 viewMatrix, std::vector<float> camPos, ThreadBuffer& buf)
{
    glUseProgram(m_program);

    glViewport(0, 0, m_width, m_height);

    glBindFramebuffer(GL_FRAMEBUFFER, buf.fboId);

    float clearColor[4] = { 0.f, 0.f, 0.f, 0.f };
    glClearBufferfv(GL_COLOR, 0, clearColor);
    glClearBufferfv(GL_COLOR, 1, clearColor);
    glClearBufferfv(GL_COLOR, 2, clearColor);
    glClearBufferfv(GL_COLOR, 3, clearColor);
    glClearBufferfv(GL_COLOR, 4, clearColor);

    glClearColor(0.f, 0.f, 0.f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);

    glUniform3f(m_camWorldPosLocation, camPos[0], camPos[1], camPos[2]);

    glUniformMatrix4fv(m_viewLocation, 1, GL_FALSE, (const GLfloat*) viewMatrix);
    glCheckError();

    glUniformMatrix4fv(m_projLocation, 1, GL_FALSE, (const GLfloat*) m_perspectiveMat);
    glCheckError();

    for (auto it: m_OpenglDummyTexId) {
        glBindTexture(GL_TEXTURE_2D, it);

        auto find = m_dummyTex2BufferObjs.find(it);
        assert(find != m_dummyTex2BufferObjs.end());

        glBindVertexArray(find->second.vao);

        glDrawArraysInstanced(GL_TRIANGLES, 0, 3, find->second.meshNumOfThisKind);
    }

    glBindVertexArray(0);

    glFinish();
}

void ReadGtImg(const std::string gtImageDir, std::map<std::string, at::Tensor>& gtImgMap)
{
    std::vector<std::string> filenames;
    for (const auto& entry : fs::directory_iterator(gtImageDir)) {
        if (entry.path().extension() == ".png") {
            filenames.push_back(entry.path().filename().string());
        }
    }

    for (auto it : filenames) {

        std::string fileName = gtImageDir + it;
        cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR);

        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        at::Tensor tensor = torch::from_blob(img.data, {img.size[0], img.size[1], 3}, options);

        at::Tensor tensorNew = tensor.clone().detach();
        tensorNew = at::flip(tensorNew, -1);
        tensorNew = tensorNew.permute({2, 0, 1});
        tensorNew = tensorNew.contiguous();

        at::Tensor cvtTensor = tensorNew.to(torch::kFloat32).contiguous();

        gtImgMap.insert(pair<std::string, at::Tensor>(it, cvtTensor));
    }
}

bool ReadGtPoseAndImageBlenderOut(const std::string gtPosePath, mat4x4** viewMatrix, std::vector<vector<float>>& camPosWorld)
{
    std::ifstream infile;
    infile.open(gtPosePath.data());
    // assert(infile.is_open());
    if (infile.is_open() == false) {
        return false;
    }

    mat4x4 rot180AxisZ;
    mat4x4_identity(rot180AxisZ);
    rot180AxisZ[0][0] = -1.f;
    rot180AxisZ[1][1] = -1.f;

    mat4x4 flipX;
    mat4x4_identity(flipX);
    flipX[0][0] = -1.f;

    *viewMatrix = (mat4x4*)malloc(sizeof(mat4x4) * 1000);

    mat4x4* curViewMatInvTransposeArr = *viewMatrix;

    std::string s;

    unsigned int arrCnt = 0;
    int i = 0;
    std::vector<string> tmpString;
    while (std::getline(infile, s)) {
        if (i == 4) {
            vector<float> rotTmp;
            for (int j = 0; j < tmpString.size(); j++) {
                rotTmp.push_back(std::stof(tmpString[j]));
            }

            mat4x4 translationC2w;
            mat4x4 translationW2c;
            mat4x4 rot180AlongZ;
            mat4x4 rot180AlongZFlipX;
            mat4x4 viewMatTrsp;

            MatConvert(translationC2w, rotTmp);
            mat4x4_invert(translationW2c, translationC2w);
            mat4x4_mul(rot180AlongZ, translationW2c, rot180AxisZ);
            mat4x4_mul(rot180AlongZFlipX, rot180AlongZ, flipX);
            
            mat4x4_transpose(viewMatTrsp, rot180AlongZFlipX);

            vector<float> posTmp;
            for (int j = 0; j < 3; j++) {
                posTmp.push_back(translationC2w[0][3]);
                posTmp.push_back(translationC2w[1][3]);
                posTmp.push_back(translationC2w[2][3]);
            }

            mat4x4_dup(*(curViewMatInvTransposeArr + arrCnt), viewMatTrsp);
            arrCnt++;
            camPosWorld.push_back(posTmp);

            tmpString.clear();
            i = 0;
        }
        
        string subStr;

        istringstream iss(s);
        while (iss >> subStr) {
            tmpString.push_back(subStr);
        }

        i++;
    }

    if (i == 4) {
        vector<float> rotTmp;
        for (int j = 0; j < tmpString.size(); j++) {
            rotTmp.push_back(std::stof(tmpString[j]));
        }

        mat4x4 translationC2w;
        mat4x4 translationW2c;
        mat4x4 rot180AlongZ;
        mat4x4 rot180AlongZFlipX;
        mat4x4 viewMatTrsp;

        MatConvert(translationC2w, rotTmp);
        mat4x4_invert(translationW2c, translationC2w);
        mat4x4_mul(rot180AlongZ, translationW2c, rot180AxisZ);
        mat4x4_mul(rot180AlongZFlipX, rot180AlongZ, flipX);
        
        mat4x4_transpose(viewMatTrsp, rot180AlongZFlipX);

        vector<float> posTmp;
        for (int j = 0; j < 3; j++) {
            posTmp.push_back(translationC2w[0][3]);
            posTmp.push_back(translationC2w[1][3]);
            posTmp.push_back(translationC2w[2][3]);
        }

        mat4x4_dup(*(curViewMatInvTransposeArr + arrCnt), viewMatTrsp);
        arrCnt++;
        camPosWorld.push_back(posTmp);

        tmpString.clear();
        i = 0;
    }

    infile.close();

    return true;
}

uint32_t getFileLineNum(const std::string posePath)
{
    std::ifstream infile;
    infile.open(posePath.data());
    if (infile.is_open() == false) {
        return 0;
    }

    std::string s;
    uint32_t lineNum = 0;

    while (std::getline(infile, s)) {
        lineNum++;
    }

    infile.close();

    return lineNum;
}

bool ReadGtPoseFromLIVO2_mat(const std::string posePath, mat4x4** viewMatrix, std::vector<vector<float>>& camPosWorld)
{
    uint32_t lineNum = getFileLineNum(posePath);
    if (lineNum == 0) {
        return false;
    }

    if (lineNum % 4 != 0) {
        cout << "pose file is corrupted, line num is not divisible by 4, line num : " << lineNum << endl;
        return false;
    }

    *viewMatrix = (mat4x4*)malloc(sizeof(mat4x4) * (lineNum / 4));
    mat4x4* curViewMatInvTransposeArr = *viewMatrix;

    std::ifstream infile;
    infile.open(posePath.data());
    if (infile.is_open() == false) {
        return false;
    }

    mat4x4 xyFlip;
    mat4x4_identity(xyFlip);
    xyFlip[0][0] = -1.0f;
    xyFlip[1][1] = -1.0f;

    mat4x4 LIVO2Colmap;
    mat4x4_zero(LIVO2Colmap);
    LIVO2Colmap[0][1] = -1.0f;
    LIVO2Colmap[1][2] = -1.0f;
    LIVO2Colmap[2][0] = 1.0f;
    LIVO2Colmap[3][3] = 1.0f;

    mat4x4 colmap2Opgl;
    mat4x4_identity(colmap2Opgl);
    colmap2Opgl[1][1] = -1.0f;
    colmap2Opgl[2][2] = -1.0f;


    mat4x4 flipZ;
    mat4x4_identity(flipZ);
    flipZ[2][2] = -1.f;

    mat4x4 clkWise90AlongZ;
    mat4x4_zero(clkWise90AlongZ);
    clkWise90AlongZ[0][1] = 1.0f;
    clkWise90AlongZ[1][0] = -1.0f;
    clkWise90AlongZ[2][2] = 1.0f;
    clkWise90AlongZ[3][3] = 1.0f;

    mat4x4 flipY;
    mat4x4_identity(flipY);
    flipY[1][1] = -1.f;


    std::string s;
    unsigned int arrCnt = 0;
    int i = 0;
    std::vector<string> tmpString;

    while (std::getline(infile, s)) {
        if (i == 4) {
            vector<float> rotTmp;
            for (int j = 0; j < tmpString.size(); j++) {
                rotTmp.push_back(std::stof(tmpString[j]));
            }

            mat4x4 rot;
            mat4x4 rotInv;
            mat4x4 rotInv2Opgl;
            mat4x4 rotInv2OpglFlipY;
            mat4x4 rotInv2OpglFlipYTrsp;
            mat4x4 rotInv2OpglFlipYInv;

            MatConvert(rot, rotTmp);

            mat4x4_invert(rotInv, rot);
            mat4x4_mul(rotInv2Opgl, rotInv, colmap2Opgl);
            mat4x4_mul(rotInv2OpglFlipY, rotInv2Opgl, flipY);

            mat4x4_invert(rotInv2OpglFlipYInv, rotInv2OpglFlipY);

            mat4x4_transpose(rotInv2OpglFlipYTrsp, rotInv2OpglFlipY); // row 2 col major

            mat4x4_dup(*(curViewMatInvTransposeArr + arrCnt), rotInv2OpglFlipYTrsp);

            arrCnt++;

            vector<float> posTmp;
            posTmp.push_back(rotInv2OpglFlipYInv[0][3]);
            posTmp.push_back(rotInv2OpglFlipYInv[1][3]);
            posTmp.push_back(rotInv2OpglFlipYInv[2][3]);

            camPosWorld.push_back(posTmp);

            tmpString.clear();
            i = 0;
        }

        string subStr;

        istringstream iss(s);
        while (iss >> subStr) {
            tmpString.push_back(subStr);
        }

        i++;
    }

    if (i == 4) {
        vector<float> rotTmp;
        for (int j = 0; j < tmpString.size(); j++) {
            rotTmp.push_back(std::stof(tmpString[j]));
        }

        mat4x4 rot;
        mat4x4 rotInv;
        mat4x4 rotInv2Opgl;
        mat4x4 rotInv2OpglFlipY;
        mat4x4 rotInv2OpglFlipYTrsp;
        mat4x4 rotInv2OpglFlipYInv;

        MatConvert(rot, rotTmp);
        mat4x4_invert(rotInv, rot);
        mat4x4_mul(rotInv2Opgl, rotInv, colmap2Opgl);
        mat4x4_mul(rotInv2OpglFlipY, rotInv2Opgl, flipY);

        
        mat4x4_invert(rotInv2OpglFlipYInv, rotInv2OpglFlipY);
        
        mat4x4_transpose(rotInv2OpglFlipYTrsp, rotInv2OpglFlipY); // row 2 col major

        mat4x4_dup(*(curViewMatInvTransposeArr + arrCnt), rotInv2OpglFlipYTrsp);

        arrCnt++;

        vector<float> posTmp;
        posTmp.push_back(rotInv2OpglFlipYInv[0][3]);
        posTmp.push_back(rotInv2OpglFlipYInv[1][3]);
        posTmp.push_back(rotInv2OpglFlipYInv[2][3]);

        camPosWorld.push_back(posTmp);
    }

    infile.close();

    return true;
}

bool ReadGtPoseAndImgFromLIVO2_mat(const std::string gtImageDir, const std::string gtPosePath, std::vector<at::Tensor>& gtImages, mat4x4** viewMatrix, std::vector<vector<float>>& camPosWorld)
{
    std::map<std::string, at::Tensor> gtImgMap;
    ReadGtImg(gtImageDir, gtImgMap);

    *viewMatrix = (mat4x4*)malloc(sizeof(mat4x4) * gtImgMap.size());

    mat4x4* curViewMatInvTransposeArr = *viewMatrix;

    std::ifstream infile;
    infile.open(gtPosePath.data());
    if (infile.is_open() == false) {
        return false;
    }

    mat4x4 xyFlip;
    mat4x4_identity(xyFlip);
    xyFlip[0][0] = -1.0f;
    xyFlip[1][1] = -1.0f;

    mat4x4 LIVO2Colmap;
    mat4x4_zero(LIVO2Colmap);
    LIVO2Colmap[0][1] = -1.0f;
    LIVO2Colmap[1][2] = -1.0f;
    LIVO2Colmap[2][0] = 1.0f;
    LIVO2Colmap[3][3] = 1.0f;

    mat4x4 colmap2Opgl;
    mat4x4_identity(colmap2Opgl);
    colmap2Opgl[1][1] = -1.0f;
    colmap2Opgl[2][2] = -1.0f;


    mat4x4 flipZ;
    mat4x4_identity(flipZ);
    flipZ[2][2] = -1.f;

    mat4x4 clkWise90AlongZ;
    mat4x4_zero(clkWise90AlongZ);
    clkWise90AlongZ[0][1] = 1.0f;
    clkWise90AlongZ[1][0] = -1.0f;
    clkWise90AlongZ[2][2] = 1.0f;
    clkWise90AlongZ[3][3] = 1.0f;

    mat4x4 flipY;
    mat4x4_identity(flipY);
    flipY[1][1] = -1.f;


    std::string s;
    unsigned int arrCnt = 0;
    int i = 0;
    std::vector<string> tmpString;

    while (std::getline(infile, s)) {
        if (i == 4) {
            vector<float> rotTmp;
            for (int j = 0; j < tmpString.size(); j++) {
                rotTmp.push_back(std::stof(tmpString[j]));
            }

            mat4x4 rot;
            mat4x4 rotInv;
            mat4x4 rotInv2Opgl;
            mat4x4 rotInv2OpglFlipY;
            mat4x4 rotInv2OpglFlipYTrsp;
            mat4x4 rotInv2OpglFlipYInv;

            MatConvert(rot, rotTmp);

            mat4x4_invert(rotInv, rot);
            mat4x4_mul(rotInv2Opgl, rotInv, colmap2Opgl);
            mat4x4_mul(rotInv2OpglFlipY, rotInv2Opgl, flipY);

            mat4x4_invert(rotInv2OpglFlipYInv, rotInv2OpglFlipY);

            mat4x4_transpose(rotInv2OpglFlipYTrsp, rotInv2OpglFlipY); // row 2 col major

            mat4x4_dup(*(curViewMatInvTransposeArr + arrCnt), rotInv2OpglFlipYTrsp);

            gtImages.push_back(gtImgMap[std::to_string(arrCnt) + ".png"]);

            arrCnt++;

            vector<float> posTmp;
            posTmp.push_back(rotInv2OpglFlipYInv[0][3]);
            posTmp.push_back(rotInv2OpglFlipYInv[1][3]);
            posTmp.push_back(rotInv2OpglFlipYInv[2][3]);

            camPosWorld.push_back(posTmp);

            tmpString.clear();
            i = 0;
        }

        string subStr;

        istringstream iss(s);
        while (iss >> subStr) {
            tmpString.push_back(subStr);
        }

        i++;
    }

    if (i == 4) {
        vector<float> rotTmp;
        for (int j = 0; j < tmpString.size(); j++) {
            rotTmp.push_back(std::stof(tmpString[j]));
        }

        mat4x4 rot;
        mat4x4 rotInv;
        mat4x4 rotInv2Opgl;
        mat4x4 rotInv2OpglFlipY;
        mat4x4 rotInv2OpglFlipYTrsp;
        mat4x4 rotInv2OpglFlipYInv;

        MatConvert(rot, rotTmp);
        mat4x4_invert(rotInv, rot);
        mat4x4_mul(rotInv2Opgl, rotInv, colmap2Opgl);
        mat4x4_mul(rotInv2OpglFlipY, rotInv2Opgl, flipY);

        
        mat4x4_invert(rotInv2OpglFlipYInv, rotInv2OpglFlipY);
        
        mat4x4_transpose(rotInv2OpglFlipYTrsp, rotInv2OpglFlipY); // row 2 col major

        mat4x4_dup(*(curViewMatInvTransposeArr + arrCnt), rotInv2OpglFlipYTrsp);

        gtImages.push_back(gtImgMap[std::to_string(arrCnt) + ".png"]);

        arrCnt++;

        vector<float> posTmp;
        posTmp.push_back(rotInv2OpglFlipYInv[0][3]);
        posTmp.push_back(rotInv2OpglFlipYInv[1][3]);
        posTmp.push_back(rotInv2OpglFlipYInv[2][3]);

        camPosWorld.push_back(posTmp);
    }

    infile.close();

    return true;
}

void MatConvertFromTensor(mat4x4& outMat, at::Tensor inMat)
{
    outMat[0][0] = inMat[0][0].item().toFloat();
    outMat[0][1] = inMat[0][1].item().toFloat();
    outMat[0][2] = inMat[0][2].item().toFloat();
    outMat[0][3] = inMat[0][3].item().toFloat();

    outMat[1][0] = inMat[1][0].item().toFloat();
    outMat[1][1] = inMat[1][1].item().toFloat();
    outMat[1][2] = inMat[1][2].item().toFloat();
    outMat[1][3] = inMat[1][3].item().toFloat();

    outMat[2][0] = inMat[2][0].item().toFloat();
    outMat[2][1] = inMat[2][1].item().toFloat();
    outMat[2][2] = inMat[2][2].item().toFloat();
    outMat[2][3] = inMat[2][3].item().toFloat();

    outMat[3][0] = inMat[3][0].item().toFloat();
    outMat[3][1] = inMat[3][1].item().toFloat();
    outMat[3][2] = inMat[3][2].item().toFloat();
    outMat[3][3] = inMat[3][3].item().toFloat();
}

void ReadGtImage(const std::string gtImageDir, std::vector<at::Tensor>& gtImages)
{
    std::vector<std::string> filenames;
    for (const auto& entry : fs::directory_iterator(gtImageDir)) {
        if (entry.path().extension() == ".png") {
            filenames.push_back(entry.path().filename().string());
        }
    }

    std::sort(filenames.begin(), filenames.end());

    for (auto it : filenames) {
        std::string fileName = gtImageDir + it;
        cout << fileName << endl;
        cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR);

        auto options = torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true);
        at::Tensor tensor = torch::from_blob(img.data, {img.size[0], img.size[1], 3}, options).contiguous();

        tensor = tensor.clone().detach();
        tensor = at::flip(tensor, -1);
        tensor = tensor.permute({2, 0, 1}).contiguous();


        at::Tensor cvtTensor = tensor.to(torch::kFloat32).to(torch::kCPU);

        gtImages.push_back(cvtTensor);
    }

    cout << "get gt images : " << gtImages.size() << endl;
}


double CalculatePsnr(const torch::Tensor& pred, const torch::Tensor& target) {
    torch::Tensor mse = torch::mean(torch::pow(pred - target, 2));
    double psnr = 10 * std::log10(255.0 * 255.0 / mse.item<double>());
    return psnr;
}

void ImMeshRenderer::InitRenderedResultTensor()
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    auto options2 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0);

    m_renderedPixelsCUDA = torch::zeros({3, m_height, m_width}, options).contiguous();
    m_renderedPixelsMask = torch::zeros({1, m_height, m_width}, options2).contiguous(); //.fill_(1)
}

template <typename Map, typename KeyIter, typename ValueIter>
__global__ void filtered_insert(Map map_ref,
                                KeyIter key_begin,
                                ValueIter value_begin,
                                std::size_t num_keys)
                                // int32_t* num_inserted)
{
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  std::size_t counter = 0;
  while (tid < num_keys) {
      // Map::insert returns `true` if it is the first time the given key was
      // inserted and `false` if the key already existed
      if (map_ref.insert(cuco::pair{key_begin[tid], value_begin[tid]})) {
        ++counter;  // Count number of successfully inserted keys
      }
    tid += gridDim.x * blockDim.x;
  }

  // Update global count of inserted keys
//   atomicAdd(num_inserted, counter);
}

void ImMeshRenderer::ConstructCustomHashValueAllNbrPipeLine(std::vector<uint64_t>& insertVals, std::vector<uint32_t>& curMemLayoutVec, ThreadBuffer& buf)
{
    parallel_for(blocked_range<size_t>(0, insertVals.size()), [&](const blocked_range<size_t>& range) {
        for (size_t cnt = range.begin(); cnt != range.end(); cnt++) {
            uint32_t low10bit = (curMemLayoutVec[cnt] >> 17);
            uint32_t high22bit = (cnt << 10); // texOffset
            uint32_t first32bit = low10bit | high22bit;

            uint32_t high17bit = (0x1FFFF & curMemLayoutVec[cnt]) << 15;
            uint32_t low15bit = ((uint32_t)(m_shTexHashMap[buf.curTexIdsAll[cnt] - 1].topVertNbr.size()) << 10) | ((uint32_t)(m_shTexHashMap[buf.curTexIdsAll[cnt] - 1].leftVertNbr.size()) << 5) | ((uint32_t)(m_shTexHashMap[buf.curTexIdsAll[cnt] - 1].rightVertNbr.size()));
            uint32_t second32bit = high17bit | low15bit;

            uint64_t tmp = static_cast<uint64_t>(first32bit);
            tmp = tmp << 32;
            uint64_t tmp2 = static_cast<uint64_t>(second32bit);
            insertVals[cnt] = (tmp | tmp2);
        }
    });
}

uint32_t ImMeshRenderer::CalcCValidBitNum(unsigned char validBits)
{
    if (validBits > 31) {
        cout << "invalid bits number, cannot gen value\n";
        exit(0);
    }

    uint32_t result = 1;

    for (unsigned char cnt = 0; cnt < validBits; cnt++) {
        result = ((result << 1) | 1);
    }

    return result;
}

void ImMeshRenderer::AddInvisibleTexId(uniqueTexIdsConcurrent_t& uniqueTexIds)
{
    // edge nbr is included in vert nbr
    uniqueTexIdsConcurrent_t extraTexIds;

    std::vector<uint32_t> visibleTexIdsVec;
    visibleTexIdsVec.assign(uniqueTexIds.begin(), uniqueTexIds.end());

    parallel_for(blocked_range<uint32_t>(0, uniqueTexIds.size()), AddInvisibleTexIdMultiThrd(m_shTexHashMap, extraTexIds, visibleTexIdsVec, uniqueTexIds));

    PROFILE_PRINT(std::cout << "detect extraTexId number : " << extraTexIds.size() << endl);

    if (extraTexIds.size() > 0) {
        uniqueTexIds.merge(extraTexIds);
    }
}

__global__ void reset_img_cuda(uint32_t imgW, uint32_t imgH, float* outImg, unsigned char* maskImg)
{
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int offsetPerPixelChannel = 3; // because rendered img that calc loss is RGB 3 channel
    unsigned int offsetPerPixelLine = imgW * offsetPerPixelChannel; // RGB

    if (x >= imgW || y >= imgH) {
        return;
    }

    // don't care order, reset all pixel
    *(maskImg + (imgH - y - 1) * imgW + x) = 0;
    *(outImg + (imgH - y - 1) * offsetPerPixelLine + x * offsetPerPixelChannel) = 0.f;
    *(outImg + (imgH - y - 1) * offsetPerPixelLine + x * offsetPerPixelChannel + 1) = 0.f;
    *(outImg + (imgH - y - 1) * offsetPerPixelLine + x * offsetPerPixelChannel + 2) = 0.f;
}

void ImMeshRenderer::ResetImg()
{
    dim3 block_size(16, 16, 1);
    dim3 grid_size((m_width + block_size.x - 1) / block_size.x, (m_height + block_size.y - 1) / block_size.y, 1);

    reset_img_cuda <<<grid_size, block_size>>> (m_width, m_height,
        m_renderedPixelsCUDA.data_ptr<float>(),
        m_renderedPixelsMask.data_ptr<unsigned char>());
}

void ReadDataLIVO2(AlgoConfig& algoConfig, std::vector<std::vector<float>>& camPosWorld, std::vector<at::Tensor>& gtImages, mat4x4** viewMatrix, std::vector<float>& poseConfidence)
{
    cout << " read data ReadDataLIVO2 matrix form\n";

    ReadGtPoseAndImgFromLIVO2_mat(algoConfig.gtImageDir, algoConfig.gtPosePath, gtImages, viewMatrix, camPosWorld);

    for (size_t i = 0; i < gtImages.size(); i++) {
        poseConfidence.push_back(1.f);
    }

}

bool ReadPoseFileM2Mapping(std::string path, mat4x4* curViewMatInvTransposeArr, std::vector<vector<float>>& camPosWorld)
{
    std::ifstream file(path);
    std::string line;

    if (file.is_open() == false) {
        return false;
    }

    mat4x4 rot180AxisZ;
    mat4x4_identity(rot180AxisZ);
    rot180AxisZ[0][0] = -1.f;
    rot180AxisZ[1][1] = -1.f;

    mat4x4 flipZ;
    mat4x4_identity(flipZ);
    flipZ[2][2] = -1.f;


    unsigned int arrCnt = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        vector<float> rotTmp;

        float value;
        while (iss >> value) {
            rotTmp.push_back(value);
        }

        mat4x4 curViewMat;
        mat4x4 curViewMatInv;
        mat4x4 rot180AlongZ;
        mat4x4 rot180AlongZFlipZ;
        mat4x4 curViewMatInvTranspose;

        MatConvert(curViewMat, rotTmp);
        mat4x4_invert(curViewMatInv, curViewMat);


        // mat4x4_mul(rot180AlongZ, curViewMatInv, rot180AxisZ);
        // mat4x4_mul(rot180AlongZ, curViewMat, rot180AxisZ);

        // mat4x4_mul(rot180AlongZFlipZ, rot180AlongZ, flipZ);
        mat4x4_mul(rot180AlongZFlipZ, curViewMatInv, flipZ);
        

        // mat4x4_mul(rot180AlongZ, curViewMatInv, rot180AxisZ);

        // mat4x4_transpose(curViewMatInvTranspose, rot180AlongZ);

        mat4x4_transpose(curViewMatInvTranspose, rot180AlongZFlipZ);

        vector<float> posTmp;
        for (int j = 0; j < 3; j++) {
            posTmp.push_back(curViewMat[0][3]);
            posTmp.push_back(curViewMat[1][3]);
            posTmp.push_back(curViewMat[2][3]);
        }

        mat4x4_dup(*(curViewMatInvTransposeArr + arrCnt), curViewMatInvTranspose);
        arrCnt++;
        camPosWorld.push_back(posTmp);

    }

    return true;
}

void ReadDataReplica(AlgoConfig& algoConfig, std::vector<std::vector<float>>& camPosWorld, std::vector<at::Tensor>& gtImages, mat4x4** viewMatrix, std::vector<float>& poseConfidence)
{
    cout << " read data ReadDataReplica\n";
    ReadGtImage(algoConfig.gtImageDir, gtImages);
    *viewMatrix = (mat4x4*)malloc(sizeof(mat4x4) * gtImages.size());

    ReadPoseFileM2Mapping(algoConfig.gtPosePath, *viewMatrix, camPosWorld);

    for (size_t i = 0; i < gtImages.size(); i++) {
        poseConfidence.push_back(1.f);
    }
}

// std::vector<std::vector<float>>&camPosWorldSub, std::vector<at::Tensor>& gtImagesSub, mat4x4** viewMatrixSub
void ReadDataBlenderOut(AlgoConfig& algoConfig, std::vector<std::vector<float>>& camPosWorld, std::vector<at::Tensor>& gtImages, mat4x4** viewMatrix, std::vector<float>& poseConfidence)
{
    cout << " read data ReadDataBlenderOut\n";
    ReadGtImage(algoConfig.gtImageDir, gtImages);
    ReadGtPoseAndImageBlenderOut(algoConfig.gtPosePath, viewMatrix, camPosWorld);

    for (size_t i = 0; i < gtImages.size(); i++) {
        poseConfidence.push_back(1.f);
    }
}

void ReadConfigs(std::string fileConfigPath, std::vector<std::vector<float>>& camPosWorld, std::vector<at::Tensor>& gtImages, mat4x4** viewMatrix, std::vector<float>& poseConfidence, FileConfig& fileConfig, AlgoConfig& algoConfig, CameraConfig& camGtConfig)
{
    if (readFileConfig(fileConfigPath, fileConfig) == false) {
        std::cout << "cannot read config file !" << std::endl;
    }

    if (readAlgorithmConfig(fileConfig.algoConfigPath, algoConfig) == false) {
        std::cout << "cannot read config file !" << std::endl;
    }

    // if (readCameraConfig(fileConfig.cameraConfigPath, camConfig) == false) {
    //     std::cout << "cannot read config file !" << std::endl;
    // }

    if (readCameraConfig(fileConfig.cameraConfigGtPath, camGtConfig) == false) {
        std::cout << "cannot read config file !" << std::endl;
    }

    switch (fileConfig.sceneType) {
        case SCENE_TYPE_LIVO2 :
            ReadDataLIVO2(algoConfig, camPosWorld, gtImages, viewMatrix, poseConfidence);
            break;
        
        case SCENE_TYPE_REPLICA :
            ReadDataReplica(algoConfig, camPosWorld, gtImages, viewMatrix, poseConfidence);
            break;

        case SCENE_TYPE_BLENDER_OUT :
            ReadDataBlenderOut(algoConfig, camPosWorld, gtImages, viewMatrix, poseConfidence);
            break;
        default :
            cout << "invalid scence type !!\n";
            exit(0);
    }
}

void ImMeshRenderer::InitCurBufInferTensors(ThreadBuffer& buf)
{
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
    auto optionsGPU = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

    // auto optionsI32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    auto optionsI32_GPU = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);

    uint32_t uniqueTexNum = buf.curTexIdsAll.size();

    InitThrdFixedCpyBuffer(buf, uniqueTexNum);

    buf.edgeNbrs = torch::zeros({uniqueTexNum, 6}, optionsI32_GPU).contiguous();
    buf.cornerAreaInfo = torch::zeros({uniqueTexNum, 2}, optionsGPU).contiguous();
    buf.texInWorldInfo = torch::zeros({uniqueTexNum, 6}, optionsGPU).contiguous();
    buf.texWHs = torch::zeros({uniqueTexNum}, optionsI32_GPU);
    buf.botYCoeffs = torch::zeros({uniqueTexNum}, optionsGPU);
    buf.topYCoeffs = torch::zeros({uniqueTexNum}, optionsGPU);
    buf.meshDensities = torch::zeros({uniqueTexNum}, optionsGPU);
    buf.meshNormals = torch::zeros({uniqueTexNum * 3}, optionsGPU);
    buf.validShNumsAll = torch::zeros({uniqueTexNum}, optionsI32_GPU);


    buf.validShWHMapCpctHead = buf.validShWHMapCpct.data_ptr<int32_t>();
    buf.validShNumsAllHead = buf.validShNumsAll.data_ptr<int32_t>();
    buf.shTexturesCpctHead = buf.shTexturesCpct.data_ptr<float>();
    buf.edgeNbrsHead = buf.edgeNbrs.data_ptr<int32_t>();
    buf.cornerAreaInfoHead = buf.cornerAreaInfo.data_ptr<float>();
    buf.vertNbrsHead = buf.vertNbrs.data_ptr<int32_t>();
    buf.texInWorldInfoHead = buf.texInWorldInfo.data_ptr<float>();
    buf.texWHsHead = (uint32_t *)(buf.texWHs.data_ptr<int32_t>());

    buf.shPosMapCpctHead = buf.shPosMapCpct.data_ptr<float>();
    buf.botYCoeffsHead = buf.botYCoeffs.data_ptr<float>();
    buf.topYCoeffsHead = buf.topYCoeffs.data_ptr<float>();
    buf.meshDensitiesHead = buf.meshDensities.data_ptr<float>();
    buf.meshNormalsHead = buf.meshNormals.data_ptr<float>();
}

void ImMeshRenderer::InferPreparePipeline(ThreadBuffer& buf)
{
    unsigned long startTime = GetTimeMS();
    unsigned long endTime;

    ResetImg();
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << "------till Infer ResetImg passed time : " << (double)(endTime - startTime) << " ms" << endl);
    
    buf.m_uniqueTexIds[0].clear();
    buf.m_texId2PixLocMap[0].clear();
    GenHashMaps(buf.m_uniqueTexIds[0], buf.m_texId2PixLocMap[0], buf);
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << "------till Infer GenHashMaps passed time : " << (double)(endTime - startTime) << " ms" << endl);

    ConstructCompactTex2PixBufferPipeLine(buf.m_texId2PixLocMap[0], buf.m_uniqueTexIds[0], buf);
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << "------till Infer ConstructCompactTex2PixBufferPipeLine passed time : " << (double)(endTime - startTime) << " ms" << endl);

    AddInvisibleTexId(buf.m_uniqueTexIds[0]);
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << "------till Infer AddInvisibleTexId passed time : " << (double)(endTime - startTime) << " ms" << endl);

    buf.curTexIdsAll.resize(buf.m_uniqueTexIds[0].size());
    buf.curTexIdsAll.assign(buf.m_uniqueTexIds[0].begin(), buf.m_uniqueTexIds[0].end());

    uint32_t maxValidOffset = CalcCValidBitNum(m_vertNbrAddrOffsetBit);

    ConstructCpctBuffersForCuda(buf, true);

    endTime = GetTimeMS();
    PROFILE_PRINT(cout << "------till Infer ConstructCpctBuffersForCuda passed time : " << (double)(endTime - startTime) << " ms" << endl);
    
    ConstructDeviceShHashAllNbrPipeLine(buf);
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << "------till Infer ConstructDeviceShHashAllNbrPipeLine passed time : " << (double)(endTime - startTime) << " ms" << endl);

    InitCurBufInferTensors(buf);
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << "------till Infer InitCurBufInferTensors passed time : " << (double)(endTime - startTime) << " ms" << endl);

    CopyCurInferEssentials(buf);
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << "------till Infer CopyCurInferEssentials passed time : " << (double)(endTime - startTime) << " ms" << endl);
}

bool ReadTestSequence(SceneType scene, std::string testGtPosePath, std::string testGtimgPath, mat4x4** viewMatrix, std::vector<std::vector<float>>& camPosWorld, std::vector<at::Tensor>& gtImagesInfer)
{
    std::ifstream infile;
    switch (scene)
    {
        case SCENE_TYPE_BLENDER_OUT:
            if (ReadGtPoseAndImageBlenderOut(testGtPosePath, viewMatrix, camPosWorld) == false) {
                return false;
            }

            ReadGtImage(testGtimgPath, gtImagesInfer);
            if (camPosWorld.size() <= 0) {
                return false;
            }
            break;

        case SCENE_TYPE_REPLICA:
            infile.open(testGtPosePath.data());
            if (infile.is_open() == false) {
                return false;
            }

            ReadGtImage(testGtimgPath, gtImagesInfer);
            if (gtImagesInfer.size() <= 0) {
                return false;
            }
            *viewMatrix = (mat4x4*)malloc(sizeof(mat4x4) * gtImagesInfer.size());

            if (ReadPoseFileM2Mapping(testGtPosePath, *viewMatrix, camPosWorld) == false) {
                return false;
            }
            break;
    
        case SCENE_TYPE_LIVO2:
            infile.open(testGtPosePath.data());
            if (infile.is_open() == false) {
                return false;
            }

            if (ReadGtPoseAndImgFromLIVO2_mat(testGtimgPath, testGtPosePath, gtImagesInfer, viewMatrix, camPosWorld) == false) {
                return false;
            }
            break;
    default:
        return false;
    }

    return true;
}

void ImMeshRenderer::InitInferBuffer(ThreadBuffer& buf)
{

    glCheckError();
    glGenFramebuffers(1, &buf.fboId);
    glBindFramebuffer(GL_FRAMEBUFFER, buf.fboId);

    glGenTextures(1, &buf.texRGB);
    glBindTexture(GL_TEXTURE_2D, buf.texRGB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glCheckError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    glCheckError();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf.texRGB, 0);
    glCheckError();

    glGenTextures(1, &buf.texLerpCoeffAndTexCoord);
    glBindTexture(GL_TEXTURE_2D, buf.texLerpCoeffAndTexCoord);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, buf.texLerpCoeffAndTexCoord, 0);
    glCheckError();

    glGenTextures(1, &buf.texViewDirFrag2CamAndNull);
    glBindTexture(GL_TEXTURE_2D, buf.texViewDirFrag2CamAndNull);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, buf.texViewDirFrag2CamAndNull, 0);
    glCheckError();

    glGenTextures(1, &buf.texOriShLUandRU);
    glBindTexture(GL_TEXTURE_2D, buf.texOriShLUandRU);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, buf.texOriShLUandRU, 0);
    glCheckError();

    glGenTextures(1, &buf.texOriShLDandRD);
    glBindTexture(GL_TEXTURE_2D, buf.texOriShLDandRD);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, buf.texOriShLDandRD, 0);
    glCheckError();

    glGenTextures(1, &buf.texScaleFacDepthTexId);
    glBindTexture(GL_TEXTURE_2D, buf.texScaleFacDepthTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, GL_TEXTURE_2D, buf.texScaleFacDepthTexId, 0);
    glCheckError();

    glGenTextures(1, &buf.texWorldPoseAndNull);
    glBindTexture(GL_TEXTURE_2D, buf.texWorldPoseAndNull);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, GL_TEXTURE_2D, buf.texWorldPoseAndNull, 0);
    glCheckError();

    glGenTextures(1, &buf.texEllipCoeffsAndLodLvl);
    glBindTexture(GL_TEXTURE_2D, buf.texEllipCoeffsAndLodLvl);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT7, GL_TEXTURE_2D, buf.texEllipCoeffsAndLodLvl, 0);
    glCheckError();

    glGenRenderbuffers(1, &buf.texDepthRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, buf.texDepthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, m_width, m_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buf.texDepthRBO);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! 111: " << std::endl;
        std::cout << "err code :" << glCheckFramebufferStatus(GL_FRAMEBUFFER) << endl;
    }

    GLuint attachments[8] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3,GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5, GL_COLOR_ATTACHMENT6, GL_COLOR_ATTACHMENT7};
    glDrawBuffers(8, attachments);

    glCheckError();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    buf.depthPunishmentMap = torch::zeros({m_height, m_width, 1}, options);

    buf.visibleTexPixLocCompact.resize(m_width * m_height);

    checkCudaErrors(cudaStreamCreateWithFlags(&buf.curBufStream, cudaStreamNonBlocking));
    checkCudaErrors(cudaStreamCreateWithFlags(&buf.curBufStream2, cudaStreamNonBlocking));

    checkCudaErrors(cudaMallocHost(&(buf.texScaleFacDepthTexIdCPU), m_height * m_width * 4 * sizeof(float)));
    if (buf.texScaleFacDepthTexIdCPU == nullptr) {
        printf("malloc buffer for threadBuffer's texScaleFacDepthTexIdCPU fail \n");
        exit(0);
    }
    memset(buf.texScaleFacDepthTexIdCPU, 0, m_height * m_width * 4 * sizeof(float));

    checkCudaErrors(cudaMallocHost(&(buf.viewDirFrag2CamAndNullCPU), m_height * m_width * 4 * sizeof(float)));
    if (buf.viewDirFrag2CamAndNullCPU == nullptr) {
        printf("malloc buffer for threadBuffer's viewDirFrag2CamAndNullCPU fail \n");
        exit(0);
    }
    memset(buf.viewDirFrag2CamAndNullCPU, 0, m_height * m_width * 4 * sizeof(float));

    checkCudaErrors(cudaMallocHost(&(buf.oriShLUandRUCPU), m_height * m_width * 4 * sizeof(float)));
    if (buf.oriShLUandRUCPU == nullptr) {
        printf("malloc buffer for threadBuffer's oriShLUandRUCPU fail \n");
        exit(0);
    }
    memset(buf.oriShLUandRUCPU, 0, m_height * m_width * 4 * sizeof(float));

    checkCudaErrors(cudaMallocHost(&(buf.oriShLDandRDCPU), m_height * m_width * 4 * sizeof(float)));
    if (buf.oriShLDandRDCPU == nullptr) {
        printf("malloc buffer for threadBuffer's oriShLDandRDCPU fail \n");
        exit(0);
    }
    memset(buf.oriShLDandRDCPU, 0, m_height * m_width * 4 * sizeof(float));

    checkCudaErrors(cudaMallocHost(&(buf.lerpCoeffAndTexCoordCPU), m_height * m_width * 4 * sizeof(float)));
    if (buf.lerpCoeffAndTexCoordCPU == nullptr) {
        printf("malloc buffer for threadBuffer's lerpCoeffAndTexCoordCPU fail \n");
        exit(0);
    }
    memset(buf.lerpCoeffAndTexCoordCPU, 0, m_height * m_width * 4 * sizeof(float));

    checkCudaErrors(cudaMallocHost(&(buf.texWorldPoseAndNullCPU), m_height * m_width * 4 * sizeof(float)));
    if (buf.texWorldPoseAndNullCPU == nullptr) {
        printf("malloc buffer for threadBuffer's texWorldPoseAndNullCPU fail \n");
        exit(0);
    }
    memset(buf.texWorldPoseAndNullCPU, 0, m_height * m_width * 4 * sizeof(float));

    checkCudaErrors(cudaMallocHost(&(buf.ellipCoeffsAndLodLvlCPU), m_height * m_width * 4 * sizeof(float)));
    if (buf.ellipCoeffsAndLodLvlCPU == nullptr) {
        printf("malloc buffer for threadBuffer's ellipCoeffsAndLodLvlCPU fail \n");
        exit(0);
    }
    memset(buf.ellipCoeffsAndLodLvlCPU, 0, m_height * m_width * 4 * sizeof(float));


    checkCudaErrors(cudaMalloc(&buf.texScaleFacDepthTexIdGPU, sizeof(float) * 4 * m_width * m_height));
    checkCudaErrors(cudaMemset(buf.texScaleFacDepthTexIdGPU, 0, sizeof(float) * 4 * m_width * m_height));

    checkCudaErrors(cudaMalloc(&buf.viewDirFrag2CamAndNullGPU, sizeof(float) * 4 * m_width * m_height));
    checkCudaErrors(cudaMemset(buf.viewDirFrag2CamAndNullGPU, 0, sizeof(float) * 4 * m_width * m_height));

    checkCudaErrors(cudaMalloc(&buf.oriShLUandRUGPU, sizeof(float) * 4 * m_width * m_height));
    checkCudaErrors(cudaMemset(buf.oriShLUandRUGPU, 0, sizeof(float) * 4 * m_width * m_height));

    checkCudaErrors(cudaMalloc(&buf.oriShLDandRDGPU, sizeof(float) * 4 * m_width * m_height));
    checkCudaErrors(cudaMemset(buf.oriShLDandRDGPU, 0, sizeof(float) * 4 * m_width * m_height));

    checkCudaErrors(cudaMalloc(&buf.lerpCoeffAndTexCoordGPU, sizeof(float) * 4 * m_width * m_height));
    checkCudaErrors(cudaMemset(buf.lerpCoeffAndTexCoordGPU, 0, sizeof(float) * 4 * m_width * m_height));

    checkCudaErrors(cudaMalloc(&buf.texWorldPoseAndNullGPU, sizeof(float) * 4 * m_width * m_height));
    checkCudaErrors(cudaMemset(buf.texWorldPoseAndNullGPU, 0, sizeof(float) * 4 * m_width * m_height));

    checkCudaErrors(cudaMalloc(&buf.ellipCoeffsAndLodLvlGPU, sizeof(float) * 4 * m_width * m_height));
    checkCudaErrors(cudaMemset(buf.ellipCoeffsAndLodLvlGPU, 0, sizeof(float) * 4 * m_width * m_height));

    glCheckError();
}

void ImMeshRenderer::InitThreadBuffers()
{
    checkCudaErrors(cudaSetDevice(0));

    for (uint32_t i = 0; i < NUM_BUFFERS; i++) {
        glGenFramebuffers(1, &m_thrdBufs[i].fboId);
        glBindFramebuffer(GL_FRAMEBUFFER, m_thrdBufs[i].fboId);

        glGenTextures(1, &m_thrdBufs[i].texRGB);
        glBindTexture(GL_TEXTURE_2D, m_thrdBufs[i].texRGB);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glCheckError();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        glCheckError();
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_thrdBufs[i].texRGB, 0);
        glCheckError();

        glGenTextures(1, &m_thrdBufs[i].texLerpCoeffAndTexCoord);
        glBindTexture(GL_TEXTURE_2D, m_thrdBufs[i].texLerpCoeffAndTexCoord);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, m_thrdBufs[i].texLerpCoeffAndTexCoord, 0);
        glCheckError();

        glGenTextures(1, &m_thrdBufs[i].texViewDirFrag2CamAndNull);
        glBindTexture(GL_TEXTURE_2D, m_thrdBufs[i].texViewDirFrag2CamAndNull);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, m_thrdBufs[i].texViewDirFrag2CamAndNull, 0);
        glCheckError();

        glGenTextures(1, &m_thrdBufs[i].texOriShLUandRU);
        glBindTexture(GL_TEXTURE_2D, m_thrdBufs[i].texOriShLUandRU);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, m_thrdBufs[i].texOriShLUandRU, 0);
        glCheckError();

        glGenTextures(1, &m_thrdBufs[i].texOriShLDandRD);
        glBindTexture(GL_TEXTURE_2D, m_thrdBufs[i].texOriShLDandRD);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, m_thrdBufs[i].texOriShLDandRD, 0);
        glCheckError();

        glGenTextures(1, &m_thrdBufs[i].texScaleFacDepthTexId);
        glBindTexture(GL_TEXTURE_2D, m_thrdBufs[i].texScaleFacDepthTexId);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, GL_TEXTURE_2D, m_thrdBufs[i].texScaleFacDepthTexId, 0);
        glCheckError();

        glGenTextures(1, &m_thrdBufs[i].texWorldPoseAndNull);
        glBindTexture(GL_TEXTURE_2D, m_thrdBufs[i].texWorldPoseAndNull);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, GL_TEXTURE_2D, m_thrdBufs[i].texWorldPoseAndNull, 0);
        glCheckError();

        glGenTextures(1, &m_thrdBufs[i].texEllipCoeffsAndLodLvl);
        glBindTexture(GL_TEXTURE_2D, m_thrdBufs[i].texEllipCoeffsAndLodLvl);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT7, GL_TEXTURE_2D, m_thrdBufs[i].texEllipCoeffsAndLodLvl, 0);
        glCheckError();

        glGenRenderbuffers(1, &m_thrdBufs[i].texDepthRBO);
        glBindRenderbuffer(GL_RENDERBUFFER, m_thrdBufs[i].texDepthRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, m_width, m_height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_thrdBufs[i].texDepthRBO);

        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer " << i << " is not complete! 111: " << std::endl;
            std::cout << "err code :" << glCheckFramebufferStatus(GL_FRAMEBUFFER) << endl;
        }

        GLuint attachments[8] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5, GL_COLOR_ATTACHMENT6, GL_COLOR_ATTACHMENT7};
        glDrawBuffers(8, attachments);

        glCheckError();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        m_thrdBufs[i].depthPunishmentMap = torch::zeros({m_height, m_width, 1}, options);

        m_thrdBufs[i].visibleTexPixLocCompact.resize(m_width * m_height);

        checkCudaErrors(cudaStreamCreateWithFlags(&m_thrdBufs[i].curBufStream, cudaStreamNonBlocking));
        checkCudaErrors(cudaStreamCreateWithFlags(&m_thrdBufs[i].curBufStream2, cudaStreamNonBlocking));

        checkCudaErrors(cudaMallocHost(&(m_thrdBufs[i].texScaleFacDepthTexIdCPU), m_height * m_width * 4 * sizeof(float)));
        if (m_thrdBufs[i].texScaleFacDepthTexIdCPU == nullptr) {
            printf("malloc buffer for threadBuffer's texScaleFacDepthTexIdCPU fail \n");
            exit(0);
        }
        memset(m_thrdBufs[i].texScaleFacDepthTexIdCPU, 0, m_height * m_width * 4 * sizeof(float));

        checkCudaErrors(cudaMallocHost(&(m_thrdBufs[i].viewDirFrag2CamAndNullCPU), m_height * m_width * 4 * sizeof(float)));
        if (m_thrdBufs[i].viewDirFrag2CamAndNullCPU == nullptr) {
            printf("malloc buffer for threadBuffer's viewDirFrag2CamAndNullCPU fail \n");
            exit(0);
        }
        memset(m_thrdBufs[i].viewDirFrag2CamAndNullCPU, 0, m_height * m_width * 4 * sizeof(float));

        checkCudaErrors(cudaMallocHost(&(m_thrdBufs[i].oriShLUandRUCPU), m_height * m_width * 4 * sizeof(float)));
        if (m_thrdBufs[i].oriShLUandRUCPU == nullptr) {
            printf("malloc buffer for threadBuffer's oriShLUandRUCPU fail \n");
            exit(0);
        }
        memset(m_thrdBufs[i].oriShLUandRUCPU, 0, m_height * m_width * 4 * sizeof(float));

        checkCudaErrors(cudaMallocHost(&(m_thrdBufs[i].oriShLDandRDCPU), m_height * m_width * 4 * sizeof(float)));
        if (m_thrdBufs[i].oriShLDandRDCPU == nullptr) {
            printf("malloc buffer for threadBuffer's oriShLDandRDCPU fail \n");
            exit(0);
        }
        memset(m_thrdBufs[i].oriShLDandRDCPU, 0, m_height * m_width * 4 * sizeof(float));

        checkCudaErrors(cudaMallocHost(&(m_thrdBufs[i].lerpCoeffAndTexCoordCPU), m_height * m_width * 4 * sizeof(float)));
        if (m_thrdBufs[i].lerpCoeffAndTexCoordCPU == nullptr) {
            printf("malloc buffer for threadBuffer's lerpCoeffAndTexCoordCPU fail \n");
            exit(0);
        }
        memset(m_thrdBufs[i].lerpCoeffAndTexCoordCPU, 0, m_height * m_width * 4 * sizeof(float));

        checkCudaErrors(cudaMallocHost(&(m_thrdBufs[i].texWorldPoseAndNullCPU), m_height * m_width * 4 * sizeof(float)));
        if (m_thrdBufs[i].texWorldPoseAndNullCPU == nullptr) {
            printf("malloc buffer for threadBuffer's texWorldPoseAndNullCPU fail \n");
            exit(0);
        }
        memset(m_thrdBufs[i].texWorldPoseAndNullCPU, 0, m_height * m_width * 4 * sizeof(float));

        checkCudaErrors(cudaMallocHost(&(m_thrdBufs[i].ellipCoeffsAndLodLvlCPU), m_height * m_width * 4 * sizeof(float)));
        if (m_thrdBufs[i].ellipCoeffsAndLodLvlCPU == nullptr) {
            printf("malloc buffer for threadBuffer's ellipCoeffsAndLodLvlCPU fail \n");
            exit(0);
        }
        memset(m_thrdBufs[i].ellipCoeffsAndLodLvlCPU, 0, m_height * m_width * 4 * sizeof(float));


        checkCudaErrors(cudaMalloc(&m_thrdBufs[i].texScaleFacDepthTexIdGPU, sizeof(float) * 4 * m_width * m_height));
        checkCudaErrors(cudaMemset(m_thrdBufs[i].texScaleFacDepthTexIdGPU, 0, sizeof(float) * 4 * m_width * m_height));

        checkCudaErrors(cudaMalloc(&m_thrdBufs[i].viewDirFrag2CamAndNullGPU, sizeof(float) * 4 * m_width * m_height));
        checkCudaErrors(cudaMemset(m_thrdBufs[i].viewDirFrag2CamAndNullGPU, 0, sizeof(float) * 4 * m_width * m_height));

        checkCudaErrors(cudaMalloc(&m_thrdBufs[i].oriShLUandRUGPU, sizeof(float) * 4 * m_width * m_height));
        checkCudaErrors(cudaMemset(m_thrdBufs[i].oriShLUandRUGPU, 0, sizeof(float) * 4 * m_width * m_height));

        checkCudaErrors(cudaMalloc(&m_thrdBufs[i].oriShLDandRDGPU, sizeof(float) * 4 * m_width * m_height));
        checkCudaErrors(cudaMemset(m_thrdBufs[i].oriShLDandRDGPU, 0, sizeof(float) * 4 * m_width * m_height));

        checkCudaErrors(cudaMalloc(&m_thrdBufs[i].lerpCoeffAndTexCoordGPU, sizeof(float) * 4 * m_width * m_height));
        checkCudaErrors(cudaMemset(m_thrdBufs[i].lerpCoeffAndTexCoordGPU, 0, sizeof(float) * 4 * m_width * m_height));

        checkCudaErrors(cudaMalloc(&m_thrdBufs[i].texWorldPoseAndNullGPU, sizeof(float) * 4 * m_width * m_height));
        checkCudaErrors(cudaMemset(m_thrdBufs[i].texWorldPoseAndNullGPU, 0, sizeof(float) * 4 * m_width * m_height));

        checkCudaErrors(cudaMalloc(&m_thrdBufs[i].ellipCoeffsAndLodLvlGPU, sizeof(float) * 4 * m_width * m_height));
        checkCudaErrors(cudaMemset(m_thrdBufs[i].ellipCoeffsAndLodLvlGPU, 0, sizeof(float) * 4 * m_width * m_height));

    }
}

void ImMeshRenderer::ThreadStart()
{
    std::thread renderThread(&ImMeshRenderer::RenderThreadLoop, this);
    std::thread TrainPrepareThread(&ImMeshRenderer::TrainPrepareThreadLoop, this);
    std::thread TrainThread(&ImMeshRenderer::TrainThreadLoop, this);
    
    renderThread.detach();
    TrainPrepareThread.detach();
    TrainThread.detach();
}


void ImMeshRenderer::DoRender(uint32_t bufIdx, uint32_t poseIdx)
{
    unsigned long startTime = GetTimeMS();
    unsigned long endTime;

    DoDraw(m_viewMatrix[poseIdx], m_camPosWorld[poseIdx], m_thrdBufs[bufIdx]);
    
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << g_renderThrdName << " ------till DoDraw passed time : " << (double)(endTime - startTime) << " ms" << endl);
    startTime = GetTimeMS();

    SaveTrainTextures(m_thrdBufs[bufIdx]);
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << g_renderThrdName << " ------till SaveTrainTextures passed time : " << (double)(endTime - startTime) << " ms" << endl);


    MoveTex2GPU(m_thrdBufs[bufIdx]);
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << g_renderThrdName << " ------till MoveTex2GPU passed time : " << (double)(endTime - startTime) << " ms" << endl);

    glCheckError();
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << g_renderThrdName << " ------till glReadPixels passed time : " << (double)(endTime - startTime) << " ms" << endl);
}

void ImMeshRenderer::EvalBufferInit()
{
    InitInferBuffer(m_evalBuffer.thrdBuf);

    assert(m_gtW > 0);
    assert(m_gtH > 0);

    glGenFramebuffers(1, &m_evalBuffer.gtResoFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_evalBuffer.gtResoFBO);

    glGenTextures(1, &m_evalBuffer.texGtResoRGB);
    glBindTexture(GL_TEXTURE_2D, m_evalBuffer.texGtResoRGB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_gtW, m_gtH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glCheckError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    glCheckError();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_evalBuffer.texGtResoRGB, 0);
    glCheckError();

    glGenTextures(1, &m_evalBuffer.texGtResoLerpCoeffAndTexCoord);
    glBindTexture(GL_TEXTURE_2D, m_evalBuffer.texGtResoLerpCoeffAndTexCoord);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_gtW, m_gtH, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, m_evalBuffer.texGtResoLerpCoeffAndTexCoord, 0);
    glCheckError();

    glGenTextures(1, &m_evalBuffer.texGtResoViewDirFrag2CamAndNull);
    glBindTexture(GL_TEXTURE_2D, m_evalBuffer.texGtResoViewDirFrag2CamAndNull);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_gtW, m_gtH, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, m_evalBuffer.texGtResoViewDirFrag2CamAndNull, 0);
    glCheckError();

    glGenTextures(1, &m_evalBuffer.texGtResoShLUandRU);
    glBindTexture(GL_TEXTURE_2D, m_evalBuffer.texGtResoShLUandRU);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_gtW, m_gtH, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, m_evalBuffer.texGtResoShLUandRU, 0);
    glCheckError();

    glGenTextures(1, &m_evalBuffer.texGtResoShLDandRD);
    glBindTexture(GL_TEXTURE_2D, m_evalBuffer.texGtResoShLDandRD);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_gtW, m_gtH, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, m_evalBuffer.texGtResoShLDandRD, 0);
    glCheckError();

    glGenTextures(1, &m_evalBuffer.texGtResoScaleFacDepthTexId);
    glBindTexture(GL_TEXTURE_2D, m_evalBuffer.texGtResoScaleFacDepthTexId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_gtW, m_gtH, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, GL_TEXTURE_2D, m_evalBuffer.texGtResoScaleFacDepthTexId, 0);
    glCheckError();

    glGenTextures(1, &m_evalBuffer.texGtResoWorldPoseAndNull);
    glBindTexture(GL_TEXTURE_2D, m_evalBuffer.texGtResoWorldPoseAndNull);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_gtW, m_gtH, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, GL_TEXTURE_2D, m_evalBuffer.texGtResoWorldPoseAndNull, 0);
    glCheckError();
    
    glGenTextures(1, &m_evalBuffer.texGTResoEllipCoeffsAndLodLvl);
    glBindTexture(GL_TEXTURE_2D, m_evalBuffer.texGTResoEllipCoeffsAndLodLvl);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_gtW, m_gtH, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT7, GL_TEXTURE_2D, m_evalBuffer.texGTResoEllipCoeffsAndLodLvl, 0);
    glCheckError();

    glGenRenderbuffers(1, &m_evalBuffer.gtResoDepthBuf);
    glBindRenderbuffer(GL_RENDERBUFFER, m_evalBuffer.gtResoDepthBuf);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, m_gtW, m_gtH);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_evalBuffer.gtResoDepthBuf);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete! GtResolutionTextureInit: " << std::endl;
        std::cout << "err code :" << glCheckFramebufferStatus(GL_FRAMEBUFFER) << endl;
    }

    GLuint attachments[8] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5, GL_COLOR_ATTACHMENT6, GL_COLOR_ATTACHMENT7};
    glDrawBuffers(8, attachments);
    
    glCheckError();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    m_evalBuffer.depthAndTexIdBufferGtReso = (float *)malloc(m_gtH * m_gtW * 4 * sizeof(float));
        if (m_evalBuffer.depthAndTexIdBufferGtReso == nullptr) {
            printf("malloc buffer for threadBuffer's depthAndTexIdBufferGtReso fail \n");
            exit(0);
        }
    memset(m_evalBuffer.depthAndTexIdBufferGtReso, 0, m_gtH * m_gtW * 4 * sizeof(float));
}


// ***************** if you want to use CUDA-GL interop, follow this code and adjust to your needs


// void ImMeshRenderer::RegisterTexForEval()
// {
//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResLerpCoeffAndTexCoord, m_evalBuffer.thrdBuf.texLerpCoeffAndTexCoord, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResViewDirFrag2CamAndNull, m_evalBuffer.thrdBuf.texViewDirFrag2CamAndNull, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResOriShLUandRUTexCoord, m_evalBuffer.thrdBuf.texOriShLUandRU, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResOriShLDandRDTexCoord, m_evalBuffer.thrdBuf.texOriShLDandRD, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResTexScaleFactorDepthTexId, m_evalBuffer.thrdBuf.texScaleFacDepthTexId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResWorldPoseAndNull, m_evalBuffer.thrdBuf.texWorldPoseAndNull, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResEllipCoeffsAndLodLvl, m_evalBuffer.thrdBuf.texEllipCoeffsAndLodLvl, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
// }


void ImMeshRenderer::SaveTrainTextures(ThreadBuffer& buf)
{
    unsigned long startTime = GetTimeMS();
    glBindFramebuffer(GL_FRAMEBUFFER, buf.fboId);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, buf.lerpCoeffAndTexCoordCPU);

    glReadBuffer(GL_COLOR_ATTACHMENT2);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, buf.viewDirFrag2CamAndNullCPU); // GL_FLOAT GL_RGBA32F

    glReadBuffer(GL_COLOR_ATTACHMENT3);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, buf.oriShLUandRUCPU); // GL_FLOAT GL_RGBA32F

    glReadBuffer(GL_COLOR_ATTACHMENT4);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, buf.oriShLDandRDCPU); // GL_FLOAT GL_RGBA32F

    glReadBuffer(GL_COLOR_ATTACHMENT5);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, buf.texScaleFacDepthTexIdCPU); // GL_FLOAT GL_RGBA32F

    glReadBuffer(GL_COLOR_ATTACHMENT6);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, buf.texWorldPoseAndNullCPU); // GL_FLOAT GL_RGBA32F

    glReadBuffer(GL_COLOR_ATTACHMENT7);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, buf.ellipCoeffsAndLodLvlCPU); // GL_FLOAT GL_RGBA32F
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

}

void ImMeshRenderer::MoveTex2GPU(ThreadBuffer& buf)
{
    unsigned long startTime = GetTimeMS();
    checkCudaErrors(cudaMemcpyAsync(buf.texScaleFacDepthTexIdGPU, buf.texScaleFacDepthTexIdCPU, sizeof(float) * 4 * m_width * m_height, cudaMemcpyHostToDevice, buf.curBufStream));

    checkCudaErrors(cudaMemcpyAsync(buf.viewDirFrag2CamAndNullGPU, buf.viewDirFrag2CamAndNullCPU, sizeof(float) * 4 * m_width * m_height, cudaMemcpyHostToDevice, buf.curBufStream2));

    checkCudaErrors(cudaMemcpyAsync(buf.oriShLUandRUGPU, buf.oriShLUandRUCPU, sizeof(float) * 4 * m_width * m_height, cudaMemcpyHostToDevice, buf.curBufStream));

    checkCudaErrors(cudaMemcpyAsync(buf.oriShLDandRDGPU, buf.oriShLDandRDCPU, sizeof(float) * 4 * m_width * m_height, cudaMemcpyHostToDevice, buf.curBufStream2));

    checkCudaErrors(cudaMemcpyAsync(buf.lerpCoeffAndTexCoordGPU, buf.lerpCoeffAndTexCoordCPU, sizeof(float) * 4 * m_width * m_height, cudaMemcpyHostToDevice, buf.curBufStream));

    checkCudaErrors(cudaMemcpyAsync(buf.texWorldPoseAndNullGPU, buf.texWorldPoseAndNullCPU, sizeof(float) * 4 * m_width * m_height, cudaMemcpyHostToDevice, buf.curBufStream2));

    checkCudaErrors(cudaMemcpyAsync(buf.ellipCoeffsAndLodLvlGPU, buf.ellipCoeffsAndLodLvlCPU, sizeof(float) * 4 * m_width * m_height, cudaMemcpyHostToDevice, buf.curBufStream));

    unsigned long endTime = GetTimeMS();
    PROFILE_PRINT(cout << "    ------time to MoveTex2GPU : " << (double)(endTime - startTime) << " ms" << endl);
}

void ImMeshRenderer::RenderFrameEval(uint32_t poseIdx)
{
    // RegisterTexForEval();
    DoDraw(m_viewMatrixEval[poseIdx], m_camPosWorldEval[poseIdx], m_evalBuffer.thrdBuf);
    SaveTrainTextures(m_evalBuffer.thrdBuf);
    glCheckError();
    MoveTex2GPU(m_evalBuffer.thrdBuf);

    InferPreparePipeline(m_evalBuffer.thrdBuf);
    // MapIntermediateTextures();

    auto hashMapRef = m_evalBuffer.thrdBuf.curTrainHashMap->ref(cuco::find);

    checkCudaErrors(cudaMemset(m_devErrCntPtr, 0, sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(m_devErrCntBackwardPtr, 0, sizeof(unsigned int)));

    checkCudaErrors(cudaStreamSynchronize(m_shTexturesCpctCpyStrm));
    checkCudaErrors(cudaStreamSynchronize(m_evalBuffer.thrdBuf.curBufStream));
    checkCudaErrors(cudaStreamSynchronize(m_evalBuffer.thrdBuf.curBufStream2));

    MeshLearnerForwardTexPtr(m_evalBuffer.thrdBuf.visibleTexIdsInfo.size(),
        m_evalBuffer.thrdBuf.shTexturesCpct.data_ptr<float>(), (uint32_t*)thrust::raw_pointer_cast(m_evalBuffer.thrdBuf.shTexMemLayoutDevice.data()),
        m_renderedPixelsCUDA.data_ptr<float>(),
        m_renderedPixelsMask.data_ptr<unsigned char>(),
        hashMapRef, m_devErrCntPtr,
        (TexPixInfo*)thrust::raw_pointer_cast(m_evalBuffer.thrdBuf.visibleTexIdsInfo.data()),
        (PixLocation*)thrust::raw_pointer_cast(m_evalBuffer.thrdBuf.visibleTexPixLocCompact.data()),
        m_width, m_height,
        // g_texObjViewDirFrag2CamAndNull,
        // g_texObjLerpCoeffAndTexCoord,
        // g_texObjOriShLUandRUTexCoord,
        // g_texObjOriShLDandRDTexCoord,
        // g_texObjWorldPoseAndNull,
        // g_texObjEllipCoeffsAndLodLvl,
        // g_texObjTexScaleFactorDepthTexId,
        m_evalBuffer.thrdBuf.viewDirFrag2CamAndNullGPU,
        m_evalBuffer.thrdBuf.lerpCoeffAndTexCoordGPU,
        m_evalBuffer.thrdBuf.oriShLUandRUGPU,
        m_evalBuffer.thrdBuf.oriShLDandRDGPU,
        m_evalBuffer.thrdBuf.texWorldPoseAndNullGPU,
        m_evalBuffer.thrdBuf.ellipCoeffsAndLodLvlGPU,
        m_evalBuffer.thrdBuf.texScaleFacDepthTexIdGPU,
        m_evalBuffer.thrdBuf.cornerAreaInfo.data_ptr<float>(),
        (TexAlignedPosWH*)(m_evalBuffer.thrdBuf.validShWHMapCpct.data_ptr<int32_t>()),
        (uint32_t*)(thrust::raw_pointer_cast(m_evalBuffer.thrdBuf.validShWHMapLayoutDevice.data())),
        (int32_t*)(m_evalBuffer.thrdBuf.validShNumsAll.data_ptr<int32_t>()),
        m_evalBuffer.thrdBuf.shPosMapCpct.data_ptr<float>(),
        (uint32_t *)thrust::raw_pointer_cast(m_evalBuffer.thrdBuf.posMapMemLayoutDevice.data()),
        m_evalBuffer.thrdBuf.botYCoeffs.data_ptr<float>(),
        m_evalBuffer.thrdBuf.topYCoeffs.data_ptr<float>(),
        m_invDistFactorEdge, m_invDistFactorCorner,
        m_shLayerNum, m_shOrder,
        (uint32_t*)(m_evalBuffer.thrdBuf.edgeNbrs.data_ptr<int32_t>()),
        (uint32_t*)(m_evalBuffer.thrdBuf.vertNbrs.data_ptr<int32_t>()),
        m_gaussianCoeffEWA, m_evalBuffer.thrdBuf.texInWorldInfo.data_ptr<float>(),
        (CurTexAlignedWH *)(m_evalBuffer.thrdBuf.texWHs.data_ptr<int32_t>()),
        m_evalBuffer.thrdBuf.meshDensities.data_ptr<float>(),
        m_evalBuffer.thrdBuf.meshNormals.data_ptr<float>(),
        m_printPixX, m_printPixY);
    
    // UnmapIntermediateTextures();
    // UnRegisterCudaTextures();
}

void ImMeshRenderer::DoDrawGtReso(mat4x4 viewMatrix, std::vector<float> camPos)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_evalBuffer.gtResoFBO);
    glCheckError();
    glViewport(0, 0, m_gtW, m_gtH);
    glCheckError();

    float clearColor[4] = { 0.f, 0.f, 0.f, 0.f };
    glClearBufferfv(GL_COLOR, 0, clearColor);
    glClearBufferfv(GL_COLOR, 1, clearColor);
    glClearBufferfv(GL_COLOR, 2, clearColor);
    glClearBufferfv(GL_COLOR, 3, clearColor);
    glClearBufferfv(GL_COLOR, 4, clearColor);

    glCheckError();
    glClearColor(0.f, 0.f, 0.f, 0.0f);
    glCheckError();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glCheckError();

    glUseProgram(m_program);
    glCheckError();

    glEnable(GL_DEPTH_TEST);

    glUniform3f(m_camWorldPosLocation, camPos[0], camPos[1], camPos[2]);

    glUniformMatrix4fv(m_viewLocation, 1, GL_FALSE, (const GLfloat*) viewMatrix);
    glCheckError();

    glUniformMatrix4fv(m_projLocation, 1, GL_FALSE, (const GLfloat*) m_evalBuffer.perspectiveMatGtReso);
    glCheckError();


    for (auto it: m_OpenglDummyTexId) {
        glBindTexture(GL_TEXTURE_2D, it);

        auto find = m_dummyTex2BufferObjs.find(it);
        assert(find != m_dummyTex2BufferObjs.end());

        glBindVertexArray(find->second.vao);

        glDrawArraysInstanced(GL_TRIANGLES, 0, 3, find->second.meshNumOfThisKind);
    }

    glBindVertexArray(0);

    glFinish();
}

void ImMeshRenderer::ShTextureRealloc(uint32_t meshIdx)
{
    assert((m_shTexHashMap[meshIdx].shTexValue != nullptr));
    assert((m_shTexHashMap[meshIdx].adamStateExpAvg != nullptr));
    assert((m_shTexHashMap[meshIdx].adamStateExpAvgSq != nullptr));
    assert((m_shTexHashMap[meshIdx].worldPoseMap != nullptr));
    // assert((m_shTexHashMap[meshIdx].validShWHMap != nullptr));

    // free(m_shTexHashMap[meshIdx].shTexValue);
    free(m_shTexHashMap[meshIdx].adamStateExpAvg);
    free(m_shTexHashMap[meshIdx].adamStateExpAvgSq);
    free(m_shTexHashMap[meshIdx].worldPoseMap);
    // free(m_shTexHashMap[meshIdx].validShWHMap); // still needed in tex resample stage

    m_shTexHashMap[meshIdx].perTexOffset = sizeof(float) * m_shTexChannelNum * m_shLayerNum * m_shTexHashMap[meshIdx].validShNumNew;

    // save memory, malloc when tex is resampled. otherwise memory peak occ may double because origin shTexValue still exits now

    m_shTexHashMap[meshIdx].adamStateExpAvg = (float *)malloc(m_shTexHashMap[meshIdx].perTexOffset);
    memset(m_shTexHashMap[meshIdx].adamStateExpAvg, 0, m_shTexHashMap[meshIdx].perTexOffset);

    m_shTexHashMap[meshIdx].adamStateExpAvgSq = (float *)malloc(m_shTexHashMap[meshIdx].perTexOffset);
    memset(m_shTexHashMap[meshIdx].adamStateExpAvgSq, 0, m_shTexHashMap[meshIdx].perTexOffset);

    m_shTexHashMap[meshIdx].worldPoseMap = (float *)malloc(sizeof(float) * 3 * m_shTexHashMap[meshIdx].validShNumNew);
    memset(m_shTexHashMap[meshIdx].worldPoseMap, 0, sizeof(float) * 3 * m_shTexHashMap[meshIdx].validShNumNew);

    m_shTexHashMap[meshIdx].validShWHMapNew = (TexAlignedPosWH*)malloc(sizeof(TexAlignedPosWH) * m_shTexHashMap[meshIdx].validShNumNew);
    memset(m_shTexHashMap[meshIdx].validShWHMapNew, 0, sizeof(TexAlignedPosWH) * m_shTexHashMap[meshIdx].validShNumNew);
}

void ImMeshRenderer::CalcEachMeshStatus()
{
    parallel_for(blocked_range<size_t>(0, m_shTexHashMap.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            float var = 0.0f;
            float curL1lossAvg = m_shTexHashMap[i].resizeInfo.l1LossAvg;
            for (auto it: m_shTexHashMap[i].l1Loss) {
                var += ((it - curL1lossAvg) * (it - curL1lossAvg));
            }

            var = (var / (float)(m_shTexHashMap[i].l1Loss.size()));
            m_shTexHashMap[i].resizeInfo.l1LossVar = var;
        }
    });
}

void ImMeshRenderer::UpdateMeshStatisticInfoOnce(std::vector<float>& curMeshesPsnr, std::vector<float>& curMeshesL1Loss, std::vector<float>& curMeshesDepth, std::vector<uint32_t>& uniqueTexIds)
{
    parallel_for(blocked_range<size_t>(0, uniqueTexIds.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            uint32_t curTexId = uniqueTexIds[i];
            
            // m_shTexHashMap[curTexId - 1].resizeInfo.psnrAvg = ((float)((m_shTexHashMap[curTexId - 1].resizeInfo.psnrAvg * m_shTexHashMap[curTexId - 1].psnr.size() + curMeshesPsnr[i])) / (float)((m_shTexHashMap[curTexId - 1].psnr.size() + 1)));
            // m_shTexHashMap[curTexId - 1].psnr.push_back(curMeshesPsnr[i]);

            m_shTexHashMap[curTexId - 1].resizeInfo.l1LossAvg = ((m_shTexHashMap[curTexId - 1].resizeInfo.l1LossAvg * (float)(m_shTexHashMap[curTexId - 1].l1Loss.size()) + curMeshesL1Loss[i]) / (float)(m_shTexHashMap[curTexId - 1].l1Loss.size() + 1));
            m_shTexHashMap[curTexId - 1].l1Loss.push_back(curMeshesL1Loss[i]);

            m_shTexHashMap[curTexId - 1].resizeInfo.depthAvg = ((float)(m_shTexHashMap[curTexId - 1].resizeInfo.depthAvg * m_shTexHashMap[curTexId - 1].depth.size() + curMeshesDepth[i]) / (float)(m_shTexHashMap[curTexId - 1].depth.size() + 1));
            m_shTexHashMap[curTexId - 1].depth.push_back(curMeshesDepth[i]);
        }
    });
}

void ImMeshRenderer::StactisticEachMesh()
{
    AvgPool2d avgPool2d(torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}));

    ProgressBar pb(m_camPosWorldEval.size());

    cout << "start to StactisticEachMesh " << endl;
    for (uint32_t i = 0; i < m_camPosWorldEval.size(); i++) {

        size_t freeMem, totalMem;
    
        // Returned memory in bytes
        cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);

        // printf("StactisticEachMesh free CUDA mem: %d KB, total: %d MB\n", (uint32_t)(freeMem / 1024), (uint32_t)(totalMem / 1024 / 1024));


        RenderFrameEval(i);

        memset(m_evalBuffer.depthAndTexIdBufferGtReso, 0, m_gtH * m_gtW * 4 * sizeof(float));

        DoDrawGtReso(m_viewMatrixEval[i], m_camPosWorldEval[i]);

        glBindFramebuffer(GL_FRAMEBUFFER, m_evalBuffer.gtResoFBO);
        glReadBuffer(GL_COLOR_ATTACHMENT5);
        glReadPixels(0, 0, m_gtW, m_gtH, GL_RGBA, GL_FLOAT, m_evalBuffer.depthAndTexIdBufferGtReso); // GL_FLOAT GL_RGBA32F
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glCheckError();

        unsigned int perPixOffset = 4;
        unsigned int perLineOffset = perPixOffset * m_gtW;

        std::unordered_set<uint32_t> uniqueTexIds;
        uniqueTexIds.reserve(1000000);

        std::multimap<uint32_t, PixInfo> texId2PixInfoMap;

        // depthAndTexIdBufferGtReso origin is LU
        for (size_t ii = 0; ii < m_gtH; ii++) {
            for (size_t jj = 0; jj < m_gtW; jj++) {
                if (static_cast<unsigned int>((*(m_evalBuffer.depthAndTexIdBufferGtReso + ii * perLineOffset + jj * perPixOffset + 3))) == 0) {
                    continue;
                }
                
                uint32_t texId = static_cast<unsigned int>((*(m_evalBuffer.depthAndTexIdBufferGtReso + ii * perLineOffset + jj * perPixOffset + 3)));
                float curDepth = (*(m_evalBuffer.depthAndTexIdBufferGtReso + ii * perLineOffset + jj * perPixOffset + 2));
                // PixInfo tmp((uint16_t)jj, (uint16_t)(m_gtH - 1 - ii), curDepth); // this xy, left up is origin

                PixInfo tmp((uint16_t)jj, (uint16_t)(ii), curDepth); // this xy, left up is origin; this buffer, LU is origin

                uniqueTexIds.insert(texId);

                texId2PixInfoMap.insert(std::pair<uint32_t, PixInfo>(texId, tmp));
            }
        }

        // printf("StactisticEachMesh uniqueTexIds size: %d, texId2PixInfoMap size: %d\n", uniqueTexIds.size(), texId2PixInfoMap.size());

        at::Tensor curGtImg = m_gtImagesEval[i].clone().detach().permute({1, 2, 0}).contiguous().to(torch::kCUDA, 0);

        at::Tensor renderImgDownSampled2X;
        if ((m_width != m_gtW) && (m_height != m_gtH)) {
            renderImgDownSampled2X = avgPool2d(m_renderedPixelsCUDA);
        } else {
            renderImgDownSampled2X = m_renderedPixelsCUDA;
        }

        // to HWC
        renderImgDownSampled2X = renderImgDownSampled2X.clone().detach().permute({1, 2, 0}).contiguous();

        thrust::host_vector<cuco::pair<uint32_t, PixInfo>> texId2PixInfoHost(texId2PixInfoMap.begin(), texId2PixInfoMap.end());
        thrust::device_vector<cuco::pair<uint32_t, PixInfo>> texId2PixInfoDevice = texId2PixInfoHost;

        uint32_t constexpr blockSize = 32;
        uint32_t gridSize = (uniqueTexIds.size() + blockSize - 1) / blockSize;

        thrust::device_vector<float> eachMeshPsnrDevice(uniqueTexIds.size());
        thrust::device_vector<float> eachMeshDepthDevice(uniqueTexIds.size());
        thrust::device_vector<float> eachMeshL1LossDevice(uniqueTexIds.size());

        thrust::host_vector<uint32_t> visibleTexIdsHost(uniqueTexIds.size());
        visibleTexIdsHost.assign(uniqueTexIds.begin(), uniqueTexIds.end());
        thrust::device_vector<uint32_t> visibleTexIdsDev = visibleTexIdsHost;

        checkCudaErrors(cudaMemset(thrust::raw_pointer_cast(eachMeshPsnrDevice.data()), 0, sizeof(float) * uniqueTexIds.size()));
        checkCudaErrors(cudaMemset(thrust::raw_pointer_cast(eachMeshDepthDevice.data()), 0, sizeof(float) * uniqueTexIds.size()));
        checkCudaErrors(cudaMemset(thrust::raw_pointer_cast(eachMeshL1LossDevice.data()), 0, sizeof(float) * uniqueTexIds.size()));

        statistic_each_mesh_info <<<gridSize, blockSize>>>
            (uniqueTexIds.size(), visibleTexIdsDev.begin(),
             texId2PixInfoDevice.begin(), texId2PixInfoDevice.size(),
             curGtImg.data_ptr<float>(), renderImgDownSampled2X.data_ptr<float>(),
             thrust::raw_pointer_cast(eachMeshPsnrDevice.data()),
             thrust::raw_pointer_cast(eachMeshL1LossDevice.data()),
             thrust::raw_pointer_cast(eachMeshDepthDevice.data()),
             m_gtW, m_gtH);

        checkCudaErrors(cudaDeviceSynchronize());

        thrust::host_vector<float> eachMeshPsnrHost = eachMeshPsnrDevice;
        thrust::host_vector<float> eachMeshDepthHost = eachMeshDepthDevice;
        thrust::host_vector<float> eachMeshL1LossHost = eachMeshL1LossDevice;
        
        std::vector<float> curMeshesPsnr(eachMeshPsnrHost.begin(), eachMeshPsnrHost.end());
        std::vector<float> curMeshesL1Loss(eachMeshL1LossHost.begin(), eachMeshL1LossHost.end());
        std::vector<float> curMeshesDepth(eachMeshDepthHost.begin(), eachMeshDepthHost.end());
        std::vector<uint32_t> visibleTexIdsVec(visibleTexIdsHost.begin(), visibleTexIdsHost.end());

        UpdateMeshStatisticInfoOnce(curMeshesPsnr, curMeshesL1Loss, curMeshesDepth, visibleTexIdsVec);

        m_renderedPixelsMask = torch::zeros({1, m_height, m_width}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0)).contiguous(); // .fill_(1)
        m_renderedPixelsCUDA = torch::zeros({3, m_height, m_width}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0)).contiguous();

        pb.update(i);
    }

    CalcEachMeshStatus();

    at::Tensor dummy;
    m_evalBuffer.thrdBuf.shTexturesCpct = dummy;
    m_evalBuffer.thrdBuf.edgeNbrs = dummy;
    m_evalBuffer.thrdBuf.cornerAreaInfo = dummy;
    m_evalBuffer.thrdBuf.vertNbrs = dummy;
    m_evalBuffer.thrdBuf.texInWorldInfo = dummy;
    m_evalBuffer.thrdBuf.texWHs = dummy;
    m_evalBuffer.thrdBuf.validShWHMapCpct = dummy;
    m_evalBuffer.thrdBuf.shPosMapCpct = dummy;
    m_evalBuffer.thrdBuf.botYCoeffs = dummy;
    m_evalBuffer.thrdBuf.topYCoeffs = dummy;
    m_evalBuffer.thrdBuf.validShNumsAll = dummy;
}

bool ImMeshRenderer::IsBetter(uint32_t meshIdx)
{
    if (m_shTexHashMap[meshIdx].resizeInfo.l1LossAvgLast >= m_shTexHashMap[meshIdx].resizeInfo.l1LossAvg) {
        return true;
    } else {
        return false;
    }
}

void ImMeshRenderer::print_progress_bar(int progress, int total)
{
    int bar_width = 50; // Width of the progress bar (e.g., 50 characters)
    float progress_percentage = (float)progress / total;
    int pos = bar_width * progress_percentage;

    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress_percentage * 100.0) << " %\r";
    std::cout.flush();
}

// void ImMeshRenderer::ResizeEachTexture()
// {
//     // parallel_for(blocked_range<size_t>(0, m_shTexHashMap.size()), [&](const blocked_range<size_t>& range) {
//         // for (size_t i = range.begin(); i != range.end(); i++) {
//         for (size_t i = 0; i < m_shTexHashMap.size(); i++) {
//             // if (m_shTexHashMap[i].psnrVar < m_varThresholdPSNR) {
//             //     continue;
//             // }
//             if (m_shTexHashMap[i].l1LossAvg < m_L1lossThreshold) {
//                 continue;
//             }

//             if (m_shTexHashMap[i].isConverge == true) {
//                 continue;
//             }

//             bool meshNeedResize = true;
//             TexResizeDir nextDir = TEX_OPT_MAX;
//             float densityResizeTo = 0.0f;
            
//             if (m_shTexHashMap[i].resizeCnt == 0) { // first time based on depth
//                 if (m_shTexHashMap[i].depthAvg >= m_resizeDepthThreshold) {
//                     nextDir = TEX_DOWN_RESOLUTION;
//                     densityResizeTo = (m_shTexHashMap[i].curDensity + m_densityUpdateStep);
//                 } else {
//                     nextDir = TEX_UP_RESOLUTION;
//                     densityResizeTo = (m_shTexHashMap[i].curDensity - m_densityUpdateStep);
//                 }

//                 m_shTexHashMap[i].resizeCnt++;

//                 continue;
//             }
            
//             if (IsBetter(i) == true) {
//                 if (m_shTexHashMap[i].isProbe == true) {
//                     m_shTexHashMap[i].isProbe = false;
//                     nextDir = m_shTexHashMap[i].probeDir;
//                     if (nextDir == TEX_DOWN_RESOLUTION) {
//                         densityResizeTo = (m_shTexHashMap[i].probeDensity + m_densityUpdateStep);
//                     } else if (nextDir == TEX_UP_RESOLUTION) {
//                         densityResizeTo = (m_shTexHashMap[i].probeDensity - m_densityUpdateStep);
//                     } else {
//                         printf("invalid TexResizeDir!!! : %d\n", nextDir);
//                         assert(0);
//                         exit(0);
//                     }
//                 } else {
//                     nextDir = m_shTexHashMap[i].lastResizeDir;

//                     if (nextDir == TEX_DOWN_RESOLUTION) {
//                         densityResizeTo = (m_shTexHashMap[i].curDensity + m_densityUpdateStep);
//                     } else if (nextDir == TEX_UP_RESOLUTION) {
//                         densityResizeTo = (m_shTexHashMap[i].curDensity - m_densityUpdateStep);
//                     } else {
//                         printf("invalid TexResizeDir!!! : %d\n", nextDir);
//                         assert(0);
//                         exit(0);
//                     }
//                 }


//                 m_shTexHashMap[i].badCount = 0;

//                 // if (i == needPrtIdx) {
//                 //     printf(" is Better, nextDir: %d, densityResizeTo: %f, curDensity: %f, step: %f\n", nextDir, densityResizeTo, m_shTexHashMap[i].curDensity, m_densityUpdateStep);
//                 // }
//             } else {
//                 if (m_shTexHashMap[i].isProbe == true) {
//                     nextDir = (TexResizeDir)(1 - (uint32_t)m_shTexHashMap[i].lastResizeDir);
//                     m_shTexHashMap[i].badCount = 0;
//                     m_shTexHashMap[i].returnCnt++;
//                     m_shTexHashMap[i].isProbe = false;

//                     if (m_shTexHashMap[i].returnCnt >= 3) {
//                         m_shTexHashMap[i].isConverge = true;
//                         // fallback to the backward resolution
//                     }

//                     if (nextDir == TEX_DOWN_RESOLUTION) {
//                         densityResizeTo = m_shTexHashMap[i].prevDensity + m_densityUpdateStep;
//                     } else if (nextDir == TEX_UP_RESOLUTION) {
//                         densityResizeTo = m_shTexHashMap[i].prevDensity - m_densityUpdateStep;
//                     } else {
//                         printf("invalid TexResizeDir!!! : %d\n", nextDir);
//                         assert(0);
//                         exit(0);
//                     }
//                 } else {
//                     m_shTexHashMap[i].badCount++;
//                     if (m_shTexHashMap[i].badCount > m_resizePatience) {
//                         if (m_shTexHashMap[i].isProbe == false) {
//                             m_shTexHashMap[i].isProbe = true;
//                         } else {
//                             assert(0);
//                         }
                        
//                         nextDir = m_shTexHashMap[i].lastResizeDir;
//                         if (nextDir == TEX_DOWN_RESOLUTION) {
//                             densityResizeTo = m_shTexHashMap[i].curDensity + m_densityUpdateStep;
//                         } else if (nextDir == TEX_UP_RESOLUTION) {
//                             densityResizeTo = m_shTexHashMap[i].curDensity - m_densityUpdateStep;
//                         } else {
//                             printf("invalid TexResizeDir!!! : %d\n", nextDir);
//                             assert(0);
//                             exit(0);
//                         }

//                         m_shTexHashMap[i].badCount = 0;

//                         // if (i == needPrtIdx) {
//                         //     printf(" need reverse, nextDir: %d, densityResizeTo: %f, curDensity: %f, step: %f\n", nextDir, densityResizeTo, m_shTexHashMap[i].curDensity, m_densityUpdateStep);
//                         // }
//                     } else {
//                         meshNeedResize = false;
//                     }
//                 }
//             }
            

//             if (meshNeedResize == false) {
//                 continue;
//             }

//             m_shTexHashMap[i].resizeCnt++;


//             print_progress_bar(i, m_shTexHashMap.size());
//         }
        
//     // });
// }

void ImMeshRenderer::ReassignWriteToGlobalHash(std::vector<uint32_t>& idxNeedToUpdate,
        std::vector<uint32_t>& shPosMapOffsets,
        std::vector<uint32_t>& shValidMapOffsets,
        thrust::device_vector<float>& shPoseMapBufDev,
        thrust::device_vector<TexAlignedPosWH>& validWhMapBufDev,
        thrust::device_vector<HashValPreDifinedDevice>& hashValPredifinedDev)
{
    unsigned long startTime = GetTimeMS();
    unsigned long endTime;

    std::vector<float> shPoseMapBuf(shPoseMapBufDev.size());
    std::vector<uint32_t> validShWhMapCpct(validWhMapBufDev.size());
    std::vector<HashValPreDifinedDevice> hashValPredifined(hashValPredifinedDev.size());

    // shPoseMapBuf.assign(shPoseMapBufDev.begin(), shPoseMapBufDev.end());
    checkCudaErrors(cudaMemcpy(shPoseMapBuf.data(), thrust::raw_pointer_cast(shPoseMapBufDev.data()), sizeof(float) * shPoseMapBufDev.size(), cudaMemcpyDeviceToHost));
    endTime = GetTimeMS();
    cout << "------in ReassignWriteToGlobalHash cudaMemcpy assignment1 finish passed time : " << (double)(endTime - startTime) << " ms" << endl;


    // validShWhMapCpct.assign(shValidMapBufDev.begin(), shValidMapBufDev.end());
    checkCudaErrors(cudaMemcpy(validShWhMapCpct.data(), thrust::raw_pointer_cast(validWhMapBufDev.data()), sizeof(uint32_t) * validWhMapBufDev.size(), cudaMemcpyDeviceToHost));
    endTime = GetTimeMS();
    cout << "------in ReassignWriteToGlobalHash cudaMemcpy assignment2 finish passed time : " << (double)(endTime - startTime) << " ms" << endl;


    // hashValPredifined.assign(hashValPredifinedDev.begin(), hashValPredifinedDev.end());
    checkCudaErrors(cudaMemcpy(hashValPredifined.data(), thrust::raw_pointer_cast(hashValPredifinedDev.data()), sizeof(HashValPreDifinedDevice) * hashValPredifinedDev.size(), cudaMemcpyDeviceToHost));
    endTime = GetTimeMS();
    cout << "------in ReassignWriteToGlobalHash cudaMemcpy assignment3 finish passed time : " << (double)(endTime - startTime) << " ms" << endl;

    parallel_for(blocked_range<size_t>(0, idxNeedToUpdate.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            uint32_t meshIdx = idxNeedToUpdate[i];

            memcpy((char *)m_shTexHashMap[meshIdx].worldPoseMap, (char *)(shPoseMapBuf.data() + shPosMapOffsets[i]), sizeof(float) * 3 * m_shTexHashMap[meshIdx].validShNumNew);

            memcpy((char *)m_shTexHashMap[meshIdx].validShWHMapNew, (char*)(validShWhMapCpct.data() + shValidMapOffsets[i]), sizeof(TexAlignedPosWH) * m_shTexHashMap[meshIdx].validShNumNew);

            m_shTexHashMap[meshIdx].cornerArea[0] = hashValPredifined[i].cornerArea[0];
            m_shTexHashMap[meshIdx].cornerArea[1] = hashValPredifined[i].cornerArea[1];

            float curTexelH = 1.f / (float)(m_shTexHashMap[meshIdx].resizeInfo.curTexWH.data.curTexH);
            m_shTexHashMap[meshIdx].topYCoef = 1.f - 0.5f * curTexelH;
            m_shTexHashMap[meshIdx].bottomYCoef = 0.5f * curTexelH;
        }
    });
}

void __inline__ ImMeshRenderer::FillValidSHs(float* dstStartAddr, float* curValidShData, uint32_t curTexLayerNumAll, TexAlignedPosWH* curTexValidShWH, uint32_t curValidShNum, uint32_t curTexW, uint32_t curTexH)
{
    uint32_t offsetPerLine = curTexW * SH_TEXTURE_CHANNEL_NUM;
    uint32_t offsetPerLayer = offsetPerLine * curTexH;
    uint32_t offsetPerSh = curTexLayerNumAll * SH_TEXTURE_CHANNEL_NUM;
    
    for (uint32_t i = 0; i < curValidShNum; i++) {
        uint32_t curW = (curTexValidShWH + i)->data.posW;
        uint32_t curH = (curTexValidShWH + i)->data.posH;

        float* curShDataAddr = (curValidShData + i * offsetPerSh);

        for (uint32_t curLayer = 0; curLayer < curTexLayerNumAll; curLayer++) {
            // float dataR = *(curShDataAddr + curLayer * SH_TEXTURE_CHANNEL_NUM + 0);
            // float dataG = *(curShDataAddr + curLayer * SH_TEXTURE_CHANNEL_NUM + 1);
            // float dataB = *(curShDataAddr + curLayer * SH_TEXTURE_CHANNEL_NUM + 2);
            *(dstStartAddr + curLayer * offsetPerLayer + curH * offsetPerLine + curW * SH_TEXTURE_CHANNEL_NUM + 0) = *(curShDataAddr + curLayer * SH_TEXTURE_CHANNEL_NUM + 0);
            *(dstStartAddr + curLayer * offsetPerLayer + curH * offsetPerLine + curW * SH_TEXTURE_CHANNEL_NUM + 1) = *(curShDataAddr + curLayer * SH_TEXTURE_CHANNEL_NUM + 1);
            *(dstStartAddr + curLayer * offsetPerLayer + curH * offsetPerLine + curW * SH_TEXTURE_CHANNEL_NUM + 2) = *(curShDataAddr + curLayer * SH_TEXTURE_CHANNEL_NUM + 2);
        }
    }
}

void __inline__ ImMeshRenderer::ExtractValidShs(float* dstValidShDataStartAddr, float* srcAllShs, uint32_t curTexLayerNumAll, TexAlignedPosWH* curTexValidShWH, uint32_t curTexValidShNum, uint32_t curTexW, uint32_t curTexH)
{
    uint32_t offsetPerLine = curTexW * SH_TEXTURE_CHANNEL_NUM;
    uint32_t offsetPerLayer = offsetPerLine * curTexH;
    uint32_t offsetPerSh = curTexLayerNumAll * SH_TEXTURE_CHANNEL_NUM;

    for (uint32_t i = 0; i < curTexValidShNum; i++) {
        uint32_t curW = (curTexValidShWH + i)->data.posW;
        uint32_t curH = (curTexValidShWH + i)->data.posH;

        float* dst = (dstValidShDataStartAddr + i * offsetPerSh);

        for (uint32_t curLayer = 0; curLayer < curTexLayerNumAll; curLayer++) {
            *(dst + curLayer * SH_TEXTURE_CHANNEL_NUM + 0) = *(srcAllShs + curLayer * offsetPerLayer + curH * offsetPerLine + curW * SH_TEXTURE_CHANNEL_NUM + 0);
            *(dst + curLayer * SH_TEXTURE_CHANNEL_NUM + 1) = *(srcAllShs + curLayer * offsetPerLayer + curH * offsetPerLine + curW * SH_TEXTURE_CHANNEL_NUM + 1);
            *(dst + curLayer * SH_TEXTURE_CHANNEL_NUM + 2) = *(srcAllShs + curLayer * offsetPerLayer + curH * offsetPerLine + curW * SH_TEXTURE_CHANNEL_NUM + 2);
        }
    }
}

void ImMeshRenderer::ResampleToResizedBuffer(std::vector<uint32_t>& idxNeedToUpdate, std::vector<uint32_t>& oriShTexOffsets, std::vector<uint32_t>& resizedShTexOffsets, std::vector<uint32_t> oriShTexWs, std::vector<uint32_t> oriShTexHs, std::vector<uint32_t> resizedShTexWs, std::vector<uint32_t> resizedShTexHs, std::vector<uint32_t> dividePositions)
{
    size_t freeMem, totalMem;
    
    // Returned memory in bytes
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);

    printf("before resample free mem: %d MB, total: %d MB\n", (uint32_t)(freeMem / 1024 / 1024), (uint32_t)(totalMem / 1024 / 1024));

    c10::cuda::CUDACachingAllocator::emptyCache();

    cudaMemGetInfo(&freeMem, &totalMem);

    printf("emptyCache after resample free mem: %d MB, total: %d MB\n", (uint32_t)(freeMem / 1024 / 1024), (uint32_t)(totalMem / 1024 / 1024));

    printf("dividePositions prt: \n");
    for (uint32_t i = 0; i < dividePositions.size(); i++) {
        if (i == 0) {
            printf("dividePositions idx: %d, rszBufSize %d MB, oriBufSize %d MB\n", i, (uint32_t)(4 * resizedShTexOffsets[dividePositions[i]] / 1024 /1024), (uint32_t)(4 * oriShTexOffsets[dividePositions[i]] / 1024 / 1024));
        } else {
            printf("dividePositions idx: %d, rszBufSize %d MB, oriBufSize %d MB\n", i, (uint32_t)(4 * (resizedShTexOffsets[dividePositions[i]] - resizedShTexOffsets[dividePositions[i - 1]]) / 1024 / 1024), (uint32_t)(4 * (oriShTexOffsets[dividePositions[i]] - oriShTexOffsets[dividePositions[i - 1]]) / 1024 / 1024));
        }
    }


    printMemoryUsage("ResampleToResizedBuffer111");


    unsigned long startTime = GetTimeMS();
    unsigned long endTime;

    uint32_t shTexFixedOffset = m_shTexChannelNum * m_shLayerNum;

    for (uint32_t iii = 0; iii < dividePositions.size(); iii++) {

        cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);

        printf("before loop %d free mem: %d MB, total: %d MB\n", iii, (uint32_t)(freeMem / 1024 / 1024), (uint32_t)(totalMem / 1024 / 1024));

        printMemoryUsage("dividePositions");

        uint32_t curRszShTexCpctSize = 0;
        uint32_t curOriShTexCpctSize = 0;
        uint32_t curHandleNum = 0;
        uint32_t curOriCpctStartOffset = 0;
        uint32_t curRszCpctStartOffset = 0;
        uint32_t curStartIdx = 0; // idx in std::vector<uint32_t>& idxNeedToUpdate

        if (iii == 0) {
            curRszShTexCpctSize = resizedShTexOffsets[dividePositions[iii]];
            curOriShTexCpctSize = oriShTexOffsets[dividePositions[iii]];
            curHandleNum = dividePositions[iii];
            curOriCpctStartOffset = 0;
            curRszCpctStartOffset = 0;
            curStartIdx = 0;
        } else {
            curRszShTexCpctSize = (resizedShTexOffsets[dividePositions[iii]] - resizedShTexOffsets[dividePositions[iii - 1]]);
            curOriShTexCpctSize = (oriShTexOffsets[dividePositions[iii]] - oriShTexOffsets[dividePositions[iii - 1]]);
            curHandleNum = (dividePositions[iii] - dividePositions[iii - 1]);
            curStartIdx = dividePositions[iii - 1];
            curOriCpctStartOffset = oriShTexOffsets[curStartIdx];
            curRszCpctStartOffset = resizedShTexOffsets[curStartIdx];
        }

        thrust::device_vector<float> rszSHTexCpctDev(curRszShTexCpctSize);
        thrust::host_vector<float> oriShTexCpct(curOriShTexCpctSize);
        memset(thrust::raw_pointer_cast(oriShTexCpct.data()), 0, sizeof(float) * oriShTexCpct.size());

        float* oriShTexCpctHead = thrust::raw_pointer_cast(oriShTexCpct.data());

        parallel_for(blocked_range<size_t>(0, curHandleNum), [&](const blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i != range.end(); i++) {
                FillValidSHs(oriShTexCpctHead + oriShTexOffsets[curStartIdx + i] - curOriCpctStartOffset,
                             m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].shTexValue,
                             m_shLayerNum,
                             m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].validShWHMap,
                             m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].validShNum,
                             m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].resizeInfo.LastTexWH.data.curTexW,
                             m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].resizeInfo.LastTexWH.data.curTexH);

                // memcpy((char*)(oriShTexCpctHead + oriShTexOffsets[curStartIdx + i] - curOriCpctStartOffset),
                //     m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].shTexValue,
                //     sizeof(float) * shTexFixedOffset *
                //         (uint32_t)(m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].resizeInfo.LastTexWH.data.curTexW) *
                //         (uint32_t)(m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].resizeInfo.LastTexWH.data.curTexH));
            }
        });

        endTime = GetTimeMS();
        cout << "------till oriShTexCpct prepare finish passed time : " << (double)(endTime - startTime) << " ms, cnt : " << iii << endl;

        thrust::device_vector<float> oriShTexCpctDev = oriShTexCpct;
        // delete oriShTexCpct;
        thrust::host_vector<float>().swap(oriShTexCpct);

        thrust::device_vector<uint32_t> oriShTexOffsetsDev(oriShTexOffsets.begin() + curStartIdx, oriShTexOffsets.begin() + curStartIdx + curHandleNum);
        thrust::device_vector<uint32_t> oriShTexWsDev(oriShTexWs.begin() + curStartIdx, oriShTexWs.begin() + curStartIdx + curHandleNum);
        thrust::device_vector<uint32_t> oriShTexHsDev(oriShTexHs.begin() + curStartIdx, oriShTexHs.begin() + curStartIdx + curHandleNum);

        thrust::device_vector<uint32_t> dstShTexOffsetsDev(resizedShTexOffsets.begin() + curStartIdx, resizedShTexOffsets.begin() + curStartIdx + curHandleNum);
        thrust::device_vector<uint32_t> dstShTexWsDev(resizedShTexWs.begin() + curStartIdx, resizedShTexWs.begin() + curStartIdx + curHandleNum);
        thrust::device_vector<uint32_t> dstShTexHsDev(resizedShTexHs.begin() + curStartIdx, resizedShTexHs.begin() + curStartIdx + curHandleNum);

        uint32_t constexpr blockSize = 128;
        uint32_t gridSize = (curHandleNum - 1 + blockSize) / blockSize;

        resample_each_texture <<<gridSize, blockSize>>>
            (thrust::raw_pointer_cast(oriShTexCpctDev.data()), curOriCpctStartOffset,
            thrust::raw_pointer_cast(oriShTexOffsetsDev.data()),
            thrust::raw_pointer_cast(oriShTexWsDev.data()),
            thrust::raw_pointer_cast(oriShTexHsDev.data()),
            thrust::raw_pointer_cast(rszSHTexCpctDev.data()), curRszCpctStartOffset,
            thrust::raw_pointer_cast(dstShTexOffsetsDev.data()),
            thrust::raw_pointer_cast(dstShTexWsDev.data()),
            thrust::raw_pointer_cast(dstShTexHsDev.data()),
            curHandleNum,
            m_shLayerNum);

        checkCudaErrors(cudaDeviceSynchronize());

        thrust::host_vector<float> rszSHTexCpct = rszSHTexCpctDev;
        float* rszSHTexCpctHead = thrust::raw_pointer_cast(rszSHTexCpct.data());

        parallel_for(blocked_range<size_t>(0, curHandleNum), [&](const blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i != range.end(); i++) {
                free(m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].shTexValue);
                free(m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].validShWHMap);

                m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].shTexValue = (float *)malloc(m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].perTexOffset);
                memset(m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].shTexValue, 0, m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].perTexOffset);
                m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].validShWHMap = m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].validShWHMapNew;
                m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].validShWHMapNew = nullptr;
                m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].validShNum = m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].validShNumNew;
                m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].validShNumNew = 0;

                // memcpy((char*)(m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].shTexValue),
                //     (char*)(rszSHTexCpctHead + resizedShTexOffsets[curStartIdx + i] - curRszCpctStartOffset),
                //     sizeof(float) * shTexFixedOffset *
                //         (uint32_t)(m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].resizeInfo.curTexWH.data.curTexW) *
                //         (uint32_t)(m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].resizeInfo.curTexWH.data.curTexH));

                ExtractValidShs(m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].shTexValue,
                    (rszSHTexCpctHead + resizedShTexOffsets[curStartIdx + i] - curRszCpctStartOffset),
                    m_shLayerNum,
                    m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].validShWHMap,
                    m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].validShNum,
                    m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].resizeInfo.curTexWH.data.curTexW,
                    m_shTexHashMap[idxNeedToUpdate[curStartIdx + i]].resizeInfo.curTexWH.data.curTexH);
            }
        });

        printMemoryUsage("dividePositions end");
    }



    // thrust::device_vector<float> rszSHTexCpctDev(resizedShTexCpctSize);
    
    // {
        // thrust::host_vector<float> oriShTexCpct(oriShTexCpctSize);

        // float* oriShTexCpctHead = thrust::raw_pointer_cast(oriShTexCpct.data());

        // parallel_for(blocked_range<size_t>(0, idxNeedToUpdate.size()), [&](const blocked_range<size_t>& range) {
            // for (size_t i = range.begin(); i != range.end(); i++) {
        //         memcpy((char*)(oriShTexCpctHead + oriShTexOffsets[i]),
        //             m_shTexHashMap[idxNeedToUpdate[i]].shTexValue,
        //             sizeof(float) * shTexFixedOffset *
        //                 (uint32_t)(m_shTexHashMap[idxNeedToUpdate[i]].resizeInfo.LastTexWH.data.curTexW) *
        //                 (uint32_t)(m_shTexHashMap[idxNeedToUpdate[i]].resizeInfo.LastTexWH.data.curTexH));
        //     }
        // });

        // endTime = GetTimeMS();
        // cout << "------till oriShTexCpct prepare finish passed time : " << (double)(endTime - startTime) << " ms" << endl;

        // thrust::device_vector<float> oriShTexCpctDev = oriShTexCpct;
        // thrust::device_vector<uint32_t> oriShTexOffsetsDev(oriShTexOffsets);
        // thrust::device_vector<uint32_t> oriShTexWsDev(oriShTexWs);
        // thrust::device_vector<uint32_t> oriShTexHsDev(oriShTexHs);

        // thrust::device_vector<uint32_t> dstShTexOffsetsDev(resizedShTexOffsets);
        // thrust::device_vector<uint32_t> dstShTexWsDev(resizedShTexWs);
        // thrust::device_vector<uint32_t> dstShTexHsDev(resizedShTexHs);

        // uint32_t constexpr blockSize = 128;
        // uint32_t gridSize = (idxNeedToUpdate.size() - 1 + blockSize) / blockSize;

    //     resample_each_texture <<<gridSize, blockSize>>> (thrust::raw_pointer_cast(oriShTexCpctDev.data()),
    //         thrust::raw_pointer_cast(oriShTexOffsetsDev.data()),
    //         thrust::raw_pointer_cast(oriShTexWsDev.data()),
    //         thrust::raw_pointer_cast(oriShTexHsDev.data()),
    //         thrust::raw_pointer_cast(rszSHTexCpctDev.data()),
    //         thrust::raw_pointer_cast(dstShTexOffsetsDev.data()),
    //         thrust::raw_pointer_cast(dstShTexWsDev.data()),
    //         thrust::raw_pointer_cast(dstShTexHsDev.data()),
    //         idxNeedToUpdate.size(),
    //         m_shLayerNum);
    // }

    // checkCudaErrors(cudaDeviceSynchronize());
    // endTime = GetTimeMS();
    // cout << "------till resample cuda finish passed time : " << (double)(endTime - startTime) << " ms" << endl;
    
    // thrust::host_vector<float> rszSHTexCpct = rszSHTexCpctDev;

    // float* rszSHTexCpctHead = thrust::raw_pointer_cast(rszSHTexCpct.data());

    // parallel_for(blocked_range<size_t>(0, idxNeedToUpdate.size()), [&](const blocked_range<size_t>& range) {
    //     for (size_t i = range.begin(); i != range.end(); i++) {
            // free(m_shTexHashMap[idxNeedToUpdate[i]].shTexValue);


    //         memcpy((char*)m_shTexHashMap[idxNeedToUpdate[i]].shTexValue,
    //             (char*)(rszSHTexCpctHead + resizedShTexOffsets[i]),
    //             sizeof(float) * shTexFixedOffset *
    //                 (uint32_t)(m_shTexHashMap[idxNeedToUpdate[i]].resizeInfo.curTexWH.data.curTexW) *
    //                 (uint32_t)(m_shTexHashMap[idxNeedToUpdate[i]].resizeInfo.curTexWH.data.curTexH));
    //     }
    // });

    endTime = GetTimeMS();
    cout << "------till rszSHTexCpctHead cpy to hash finish passed time : " << (double)(endTime - startTime) << " ms" << endl;
}


void ImMeshRenderer::RecalcAndFillShTextureInfo()
{
    unsigned long startTime = GetTimeMS();
    unsigned long endTime;

    uint32_t needUpdateNum = 0;
    for (uint32_t i = 0; i < m_shTexHashMap.size(); i++) {
        if (m_shTexHashMap[i].resizeInfo.needResize == true) {
            needUpdateNum++;
        }
    }

    printf("need to update mesh num: %d\n", needUpdateNum);
    if (needUpdateNum <= 0) {
        return;
    }

    printMemoryUsage("RecalcAndFillShTextureInfo111");

    std::vector<uint32_t> idxNeedToUpdate(needUpdateNum);
    std::vector<uint32_t> oriShTexOffsets(needUpdateNum + 1);
    std::vector<uint32_t> oriShTexWs(needUpdateNum);
    std::vector<uint32_t> oriShTexHs(needUpdateNum);

    std::vector<uint32_t> resizedShTexOffsets(needUpdateNum + 1);
    std::vector<uint32_t> resizedShTexWs(needUpdateNum);
    std::vector<uint32_t> resizedShTexHs(needUpdateNum);

    std::vector<uint32_t> shPosMapOffsets(needUpdateNum);
    std::vector<uint32_t> shValidMapOffsets(needUpdateNum);
    thrust::host_vector<HashValPreDifinedDevice> hashValPredifinedHost(needUpdateNum); // in this scenario, only use curTexWH and cornerAreaInfo

    printMemoryUsage("RecalcAndFillShTextureInfo222");

    uint32_t curShPosMapOffset = 0;
    uint32_t curShValidOffset = 0;
    uint32_t curOriShTexOffset = 0;
    uint32_t curRszShTexOffset = 0;

    uint32_t writePos = 0;

    uint32_t shTexFixedOffset = m_shTexChannelNum * m_shLayerNum;

    uint64_t curShTexSizeInByte = 0;
    std::vector<uint32_t> dividePositions;

    // uint64_t constexpr MAX_HANDLE_5GB_ONCE = (1024ULL * 1024ULL * 1024ULL * 5ULL);
    uint64_t constexpr MAX_HANDLE_1GB_ONCE = (1024ULL * 1024ULL * 1024ULL * 1ULL);
    for (uint32_t i = 0; i < m_shTexHashMap.size(); i++) {
        if (m_shTexHashMap[i].resizeInfo.needResize == false) {
            continue;
        }

        if (curShTexSizeInByte >= MAX_HANDLE_1GB_ONCE) {
            dividePositions.push_back(writePos);
            curShTexSizeInByte = 0;
        }

        idxNeedToUpdate[writePos] = i;
        oriShTexOffsets[writePos] = curOriShTexOffset;
        resizedShTexOffsets[writePos] = curRszShTexOffset;

        shPosMapOffsets[writePos] = curShPosMapOffset;
        shValidMapOffsets[writePos] = curShValidOffset;
        hashValPredifinedHost[writePos].curTexW = (uint32_t)(m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW);
        hashValPredifinedHost[writePos].curTexH = (uint32_t)(m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH);

        curShPosMapOffset += (m_shTexHashMap[i].validShNumNew * 3);
        curShValidOffset += (m_shTexHashMap[i].validShNumNew);

        curOriShTexOffset += shTexFixedOffset * (uint32_t)(m_shTexHashMap[i].resizeInfo.LastTexWH.data.curTexW) * (uint32_t)(m_shTexHashMap[i].resizeInfo.LastTexWH.data.curTexH);
        curRszShTexOffset += shTexFixedOffset * (uint32_t)(m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW) * (uint32_t)(m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH);

        oriShTexWs[writePos] = (uint32_t)m_shTexHashMap[i].resizeInfo.LastTexWH.data.curTexW;
        oriShTexHs[writePos] = (uint32_t)m_shTexHashMap[i].resizeInfo.LastTexWH.data.curTexH;

        resizedShTexWs[writePos] = (uint32_t)m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW;
        resizedShTexHs[writePos] = (uint32_t)m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH;

        curShTexSizeInByte += (uint64_t)(shTexFixedOffset * (uint32_t)(m_shTexHashMap[i].resizeInfo.LastTexWH.data.curTexW) * (uint32_t)(m_shTexHashMap[i].resizeInfo.LastTexWH.data.curTexH) * sizeof(float) + shTexFixedOffset * (uint32_t)(m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW) * (uint32_t)(m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH) * sizeof(float));

        writePos++;
    }

    printMemoryUsage("RecalcAndFillShTextureInfo333");

    // means max GB haven't triggerd once/ some remains not included
    if (curShTexSizeInByte != 0) {
        dividePositions.push_back(writePos);
        oriShTexOffsets[writePos] = curOriShTexOffset;
        resizedShTexOffsets[writePos] = curRszShTexOffset;
    }


    thrust::device_vector<float> shPoseMapBufDev(curShPosMapOffset);
    thrust::device_vector<TexAlignedPosWH> shValidWHMapBufDev(curShValidOffset);
    checkCudaErrors(cudaMemset((char *)(thrust::raw_pointer_cast(shPoseMapBufDev.data())), 0, sizeof(float) * curShPosMapOffset));
    checkCudaErrors(cudaMemset((char *)(thrust::raw_pointer_cast(shValidWHMapBufDev.data())), 0, sizeof(TexAlignedPosWH) * curShValidOffset));

    // idxNeedToUpdateDev.assign(idxNeedToUpdate.begin(), idxNeedToUpdate.end());
    thrust::device_vector<uint32_t> idxNeedToUpdateDev = idxNeedToUpdate;
    thrust::device_vector<uint32_t> shPosMapOffsetsDev = shPosMapOffsets;
    thrust::device_vector<uint32_t> shValidWHMapOffsetsDev = shValidMapOffsets;
    thrust::device_vector<HashValPreDifinedDevice> hashValPredifinedDev = hashValPredifinedHost;
    // delete hashValPredifinedHost;
    thrust::host_vector<HashValPreDifinedDevice>().swap(hashValPredifinedHost);

    uint32_t constexpr blockSize = 32;
    uint32_t gridSize = ((needUpdateNum + blockSize - 1) / blockSize);
    printMemoryUsage("RecalcAndFillShTextureInfo555");

    reassign_corerArea_posMap_validMap <<<gridSize, blockSize>>> ((uint32_t *)(thrust::raw_pointer_cast(idxNeedToUpdateDev.data())),
        m_triangleArr.begin(),
        hashValPredifinedDev.begin(), 
        (float *)(thrust::raw_pointer_cast(shPoseMapBufDev.data())),
        (uint32_t *)(thrust::raw_pointer_cast(shValidWHMapOffsetsDev.data())),
        (TexAlignedPosWH *)(thrust::raw_pointer_cast(shValidWHMapBufDev.data())),
        needUpdateNum);

    checkCudaErrors(cudaDeviceSynchronize());

    printMemoryUsage("before ReassignWriteToGlobalHash");

    ReassignWriteToGlobalHash(idxNeedToUpdate, shPosMapOffsets, shValidMapOffsets, shPoseMapBufDev, shValidWHMapBufDev, hashValPredifinedDev);


    endTime = GetTimeMS();
    cout << "------reassign_corerArea_posMap_validMap finish passed time : " << (double)(endTime - startTime) << " ms" << endl;

    assert(writePos == needUpdateNum);

    printMemoryUsage("ResampleToResizedBuffer");

    endTime = GetTimeMS();
    cout << "------till ResampleToResizedBuffer finish passed time : " << (double)(endTime - startTime) << " ms" << endl;

    ResampleToResizedBuffer(idxNeedToUpdate, oriShTexOffsets, resizedShTexOffsets, oriShTexWs, oriShTexHs, resizedShTexWs, resizedShTexHs, dividePositions);

    endTime = GetTimeMS();
    cout << "------till ReassignWriteToGlobalHash finish passed time : " << (double)(endTime - startTime) << " ms" << endl;
}

void ImMeshRenderer::ResizeCleanUp()
{
    parallel_for(blocked_range<size_t>(0, m_shTexHashMap.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            if (m_shTexHashMap[i].resizeInfo.needResize == true) {
                m_shTexHashMap[i].resizeInfo.needResize = false;
            }
            // m_shTexHashMap[i].needReturn = false;

            m_shTexHashMap[i].depth.clear();
            m_shTexHashMap[i].resizeInfo.depthAvg = 0.0f;

            // m_shTexHashMap[i].psnr.clear();
            // m_shTexHashMap[i].resizeInfo.psnrAvg = 0.0f;

            m_shTexHashMap[i].l1Loss.clear();
            m_shTexHashMap[i].resizeInfo.l1LossAvg = 0.0f;
            m_shTexHashMap[i].resizeInfo.l1LossVar = 0.0f;

            memset(m_shTexHashMap[i].adamStateExpAvg, 0, m_shTexHashMap[i].perTexOffset);
            memset(m_shTexHashMap[i].adamStateExpAvgSq, 0, m_shTexHashMap[i].perTexOffset);
        }
    });  
}

void ImMeshRenderer::ResizeDataPrepare(thrust::device_vector<ResizeInfo>& eachMeshResizeInfoDev)
{
    ResizeInfo* head = thrust::raw_pointer_cast(m_resizeInfoForCuda.data());

    parallel_for(blocked_range<size_t>(0, m_shTexHashMap.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            (head + i)->needResize = m_shTexHashMap[i].resizeInfo.needResize;

            (head + i)->depthAvg = m_shTexHashMap[i].resizeInfo.depthAvg;
            // (head + i)->psnrAvg = m_shTexHashMap[i].resizeInfo.psnrAvg;
            // (head + i)->psnrAvgLast = m_shTexHashMap[i].resizeInfo.psnrAvgLast;
            // (head + i)->psnrVar = m_shTexHashMap[i].resizeInfo.psnrVar;
            // (head + i)->psnrVarLast = m_shTexHashMap[i].resizeInfo.psnrVarLast;

            (head + i)->l1LossAvg = m_shTexHashMap[i].resizeInfo.l1LossAvg;
            (head + i)->l1LossAvgLast = m_shTexHashMap[i].resizeInfo.l1LossAvgLast;
            (head + i)->l1LossVar = m_shTexHashMap[i].resizeInfo.l1LossVar;
            (head + i)->l1LossVarLast = m_shTexHashMap[i].resizeInfo.l1LossVarLast;

            (head + i)->badCount = m_shTexHashMap[i].resizeInfo.badCount;

            (head + i)->lastResizeDir = m_shTexHashMap[i].resizeInfo.lastResizeDir;

            (head + i)->curDensity = m_shTexHashMap[i].resizeInfo.curDensity;
            (head + i)->prevDensity = m_shTexHashMap[i].resizeInfo.prevDensity;

            (head + i)->resizeCnt = m_shTexHashMap[i].resizeInfo.resizeCnt;
            (head + i)->returnCnt = m_shTexHashMap[i].resizeInfo.returnCnt;

            (head + i)->isConverge = m_shTexHashMap[i].resizeInfo.isConverge;
            (head + i)->isProbe = m_shTexHashMap[i].resizeInfo.isProbe;

            (head + i)->probeDensity = m_shTexHashMap[i].resizeInfo.probeDensity;
            (head + i)->probeDir = m_shTexHashMap[i].resizeInfo.probeDir;

            (head + i)->triangleBotDistance = m_shTexHashMap[i].resizeInfo.triangleBotDistance;
            (head + i)->triangleHeight = m_shTexHashMap[i].resizeInfo.triangleHeight;

            (head + i)->curTexWH.aligner = m_shTexHashMap[i].resizeInfo.curTexWH.aligner;
            (head + i)->LastTexWH.aligner = m_shTexHashMap[i].resizeInfo.LastTexWH.aligner;
        }
    });

    eachMeshResizeInfoDev = m_resizeInfoForCuda;
}

void ImMeshRenderer::ResizeResultWriteBack(thrust::device_vector<ResizeInfo>& eachMeshResizeInfoOutDev)
{
    std::vector<ResizeInfo> eachMeshResizeInfoOutStd(eachMeshResizeInfoOutDev.size());
 
    ResizeInfo* eachMeshResizeInfoOutDevHead = thrust::raw_pointer_cast(eachMeshResizeInfoOutDev.data());
    checkCudaErrors(cudaMemcpy(eachMeshResizeInfoOutStd.data(), eachMeshResizeInfoOutDevHead, sizeof(ResizeInfo) * eachMeshResizeInfoOutDev.size(), cudaMemcpyDeviceToHost));

    parallel_for(blocked_range<size_t>(0, m_shTexHashMap.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            // if (eachMeshResizeInfoOutStd[i].needResize == false) {
            //     m_shTexHashMap[i].resizeInfo.needResize = eachMeshResizeInfoOutStd[i].needResize;
            //     m_shTexHashMap[i].resizeInfo.badCount = eachMeshResizeInfoOutStd[i].badCount;
            //     continue;
            // }

            m_shTexHashMap[i].resizeInfo.needResize = eachMeshResizeInfoOutStd[i].needResize;

            // m_shTexHashMap[i].resizeInfo.psnrVarLast = eachMeshResizeInfoOutStd[i].psnrVarLast;
            m_shTexHashMap[i].resizeInfo.l1LossAvg = eachMeshResizeInfoOutStd[i].l1LossAvg;
            m_shTexHashMap[i].resizeInfo.l1LossAvgLast = eachMeshResizeInfoOutStd[i].l1LossAvgLast;
            m_shTexHashMap[i].resizeInfo.l1LossVar = eachMeshResizeInfoOutStd[i].l1LossVar;
            m_shTexHashMap[i].resizeInfo.l1LossVarLast = eachMeshResizeInfoOutStd[i].l1LossVarLast;

            m_shTexHashMap[i].resizeInfo.badCount = eachMeshResizeInfoOutStd[i].badCount;
            m_shTexHashMap[i].resizeInfo.lastResizeDir = eachMeshResizeInfoOutStd[i].lastResizeDir;

            m_shTexHashMap[i].resizeInfo.curDensity = eachMeshResizeInfoOutStd[i].curDensity;
            m_shTexHashMap[i].resizeInfo.prevDensity = eachMeshResizeInfoOutStd[i].prevDensity;

            m_shTexHashMap[i].resizeInfo.resizeCnt = eachMeshResizeInfoOutStd[i].resizeCnt;
            m_shTexHashMap[i].resizeInfo.returnCnt = eachMeshResizeInfoOutStd[i].returnCnt;

            m_shTexHashMap[i].resizeInfo.isConverge = eachMeshResizeInfoOutStd[i].isConverge;
            m_shTexHashMap[i].resizeInfo.isProbe = eachMeshResizeInfoOutStd[i].isProbe;

            m_shTexHashMap[i].resizeInfo.probeDensity = eachMeshResizeInfoOutStd[i].probeDensity;
            m_shTexHashMap[i].resizeInfo.probeDir = eachMeshResizeInfoOutStd[i].probeDir;
            

            m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW = eachMeshResizeInfoOutStd[i].curTexWH.data.curTexW;
            m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH = eachMeshResizeInfoOutStd[i].curTexWH.data.curTexH;

            m_shTexHashMap[i].resizeInfo.LastTexWH.data.curTexW = eachMeshResizeInfoOutStd[i].LastTexWH.data.curTexW;
            m_shTexHashMap[i].resizeInfo.LastTexWH.data.curTexH = eachMeshResizeInfoOutStd[i].LastTexWH.data.curTexH;
        }
    });
}

void ImMeshRenderer::AllMeshRealloc()
{
    unsigned long startTime = GetTimeMS();
    unsigned long endTime;

    parallel_for(blocked_range<size_t>(0, m_shTexHashMap.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            if (m_shTexHashMap[i].resizeInfo.needResize == false) {
                continue;
            }

            ShTextureRealloc(i);
        }
    });

    endTime = GetTimeMS();
    cout << "------AllMeshRealloc finish passed time : " << (double)(endTime - startTime) << " ms" << endl;
}

void ImMeshRenderer::ResizeEachTextureCuda()
{
    unsigned long startTime = GetTimeMS();
    unsigned long endTime;

    // thrust::host_vector<ResizeInfo> eachMeshResizeInfoHost;
    thrust::device_vector<ResizeInfo> eachMeshResizeInfoDev(m_shTexHashMap.size());
    // ResizeDataPrepare(eachMeshResizeInfoDev);

    // ResizeInfo* head = thrust::raw_pointer_cast(m_resizeInfoForCuda.data());

    std::vector<ResizeInfo> resizeInfoStd(m_shTexHashMap.size());

    parallel_for(blocked_range<size_t>(0, m_shTexHashMap.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            resizeInfoStd[i].needResize = m_shTexHashMap[i].resizeInfo.needResize;

            resizeInfoStd[i].depthAvg = m_shTexHashMap[i].resizeInfo.depthAvg;
            // resizeInfoStd[i].psnrAvg = m_shTexHashMap[i].resizeInfo.psnrAvg;
            // resizeInfoStd[i].psnrAvgLast = m_shTexHashMap[i].resizeInfo.psnrAvgLast;
            // resizeInfoStd[i].psnrVar = m_shTexHashMap[i].resizeInfo.psnrVar;
            // resizeInfoStd[i].psnrVarLast = m_shTexHashMap[i].resizeInfo.psnrVarLast;

            resizeInfoStd[i].l1LossAvg = m_shTexHashMap[i].resizeInfo.l1LossAvg;
            resizeInfoStd[i].l1LossAvgLast = m_shTexHashMap[i].resizeInfo.l1LossAvgLast;
            resizeInfoStd[i].l1LossVar = m_shTexHashMap[i].resizeInfo.l1LossVar;
            resizeInfoStd[i].l1LossVarLast = m_shTexHashMap[i].resizeInfo.l1LossVarLast;

            resizeInfoStd[i].badCount = m_shTexHashMap[i].resizeInfo.badCount;

            resizeInfoStd[i].lastResizeDir = m_shTexHashMap[i].resizeInfo.lastResizeDir;

            resizeInfoStd[i].curDensity = m_shTexHashMap[i].resizeInfo.curDensity;
            resizeInfoStd[i].prevDensity = m_shTexHashMap[i].resizeInfo.prevDensity;

            resizeInfoStd[i].resizeCnt = m_shTexHashMap[i].resizeInfo.resizeCnt;
            resizeInfoStd[i].returnCnt = m_shTexHashMap[i].resizeInfo.returnCnt;

            resizeInfoStd[i].isConverge = m_shTexHashMap[i].resizeInfo.isConverge;
            resizeInfoStd[i].isProbe = m_shTexHashMap[i].resizeInfo.isProbe;

            resizeInfoStd[i].probeDensity = m_shTexHashMap[i].resizeInfo.probeDensity;
            resizeInfoStd[i].probeDir = m_shTexHashMap[i].resizeInfo.probeDir;

            resizeInfoStd[i].triangleBotDistance = m_shTexHashMap[i].resizeInfo.triangleBotDistance;
            resizeInfoStd[i].triangleHeight = m_shTexHashMap[i].resizeInfo.triangleHeight;

            resizeInfoStd[i].curTexWH.aligner = m_shTexHashMap[i].resizeInfo.curTexWH.aligner;
            resizeInfoStd[i].LastTexWH.aligner = m_shTexHashMap[i].resizeInfo.LastTexWH.aligner;
        }
    });

    // eachMeshResizeInfoHost.assign(resizeInfoStd.begin(), resizeInfoStd.end());

    // eachMeshResizeInfoDev = eachMeshResizeInfoHost;

    ResizeInfo* eachMeshResizeInfoDevHead = thrust::raw_pointer_cast(eachMeshResizeInfoDev.data());
    checkCudaErrors(cudaMemcpy(eachMeshResizeInfoDevHead, resizeInfoStd.data(), sizeof(ResizeInfo) * resizeInfoStd.size(), cudaMemcpyHostToDevice));

    std::vector<ResizeInfo>().swap(resizeInfoStd);

    uint32_t constexpr blockSize = 32;
    uint32_t gridSize = ((m_shTexHashMap.size() + blockSize - 1) / blockSize);

    endTime = GetTimeMS();
    cout << "------before resize_each_texture_cuda passed time : " << (double)(endTime - startTime) << " ms" << endl;

    // for (uint32_t i = 0; i < m_shTexHashMap.size(); i++) {
    //     printf("idx: %d, depthAvg: %f\n", i, m_shTexHashMap[i].resizeInfo.depthAvg);
    // }

    resize_each_texture_cuda <<<gridSize, blockSize>>> (m_shTexHashMap.size(),
        (ResizeInfo *)(thrust::raw_pointer_cast(eachMeshResizeInfoDev.data())),
        m_L1lossThreshold, m_resizeDepthThreshold, m_densityUpdateStep, m_densityUpdateStepInner, m_resizePatience, m_maxTexWH, m_resizeReturnCntMax, m_resizeTriangleEdgeMinLen, m_shDensityMax);

    checkCudaErrors(cudaDeviceSynchronize());

    endTime = GetTimeMS();
    cout << "------till resize_each_texture_cuda finish passed time : " << (double)(endTime - startTime) << " ms" << endl;

    ResizeResultWriteBack(eachMeshResizeInfoDev);

    endTime = GetTimeMS();
    cout << "------till ResizeResultWriteBack finish passed time : " << (double)(endTime - startTime) << " ms" << endl;
}

void ImMeshRenderer::RecalcValidShNums()
{
    std::vector<CurTexAlignedWH> allTriangleWHs(m_triangleArr.size());
    thrust::device_vector<CurTexAlignedWH> allTriangleWHsDev(m_triangleArr.size());
    thrust::device_vector<uint32_t> shValidCpctNumbers(m_triangleArr.size());
    std::vector<uint32_t> shValidCpctNumbersHost(m_triangleArr.size());

    
    thrust::transform(m_shTexHashMap.begin(), m_shTexHashMap.end(), allTriangleWHs.begin(), CopyWHsFromGlbHashMap());
    thrust::copy(allTriangleWHs.begin(), allTriangleWHs.end(), allTriangleWHsDev.begin());

    // delete allTriangleWHs;
    std::vector<CurTexAlignedWH>().swap(allTriangleWHs);

    uint32_t constexpr blockSize = 32;
    uint32_t const gridSize = (m_triangleArr.size() + blockSize - 1) / blockSize;

    calc_valid_sh_number <<<gridSize, blockSize>>> (m_triangleArr.begin(),
        (CurTexAlignedWH*)(thrust::raw_pointer_cast(allTriangleWHsDev.data())),
        m_triangleArr.size(),
        thrust::raw_pointer_cast(shValidCpctNumbers.data()));
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::copy(shValidCpctNumbers.begin(), shValidCpctNumbers.end(), shValidCpctNumbersHost.begin());

    parallel_for(blocked_range<size_t>(0, m_shTexHashMap.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            m_shTexHashMap[i].validShNumNew = shValidCpctNumbersHost[i];
        }
    });
}

void ImMeshRenderer::ResizeShTextures()
{
    unsigned long startTime = GetTimeMS();
    unsigned long endTime;

    c10::cuda::CUDACachingAllocator::emptyCache();

    printMemoryUsage("StactisticEachMesh");

    StactisticEachMesh();
    endTime = GetTimeMS();
    cout << "------till StactisticEachMesh finish passed time : " << (double)(endTime - startTime) << " ms" << endl;


    printMemoryUsage("ResizeEachTextureCuda");
    // ResizeEachTexture();
    ResizeEachTextureCuda();
    endTime = GetTimeMS();
    cout << "------till ResizeEachTextureCuda finish passed time : " << (double)(endTime - startTime) << " ms" << endl;

    printMemoryUsage("RecalcValidShNums");
    RecalcValidShNums();
    endTime = GetTimeMS();
    cout << "------till RecalcValidShNums finish passed time : " << (double)(endTime - startTime) << " ms" << endl;

    printMemoryUsage("AllMeshRealloc");
    AllMeshRealloc();
    endTime = GetTimeMS();
    cout << "------till AllMeshRealloc finish passed time : " << (double)(endTime - startTime) << " ms" << endl;

    printMemoryUsage("RecalcAndFillShTextureInfo");
    RecalcAndFillShTextureInfo();
    endTime = GetTimeMS();
    cout << "------till RecalcAndFillShTextureInfo finish passed time : " << (double)(endTime - startTime) << " ms" << endl;

    printMemoryUsage("NewDummyTextures");
    NewDummyTextures();
    endTime = GetTimeMS();
    cout << "------till NewDummyTextures finish passed time : " << (double)(endTime - startTime) << " ms" << endl;


    printMemoryUsage("ReassignBuffeObjs");
    ReassignBuffeObjs();
    endTime = GetTimeMS();
    cout << "------till ReassignBuffeObjs finish passed time : " << (double)(endTime - startTime) << " ms" << endl;

    printMemoryUsage("ResizeCleanUp");
    ResizeCleanUp();
    endTime = GetTimeMS();
    cout << "------till ResizeCleanUp finish passed time : " << (double)(endTime - startTime) << " ms" << endl;

    printMemoryUsage("end");
}

// share VBO of renderThrd, TrainThrd only create VAOs
void ImMeshRenderer::ReassignBuffeObjsInfer()
{
    // assert(m_OpenglDummyTexId.size() == m_dummyTex2BufferObjsInfer.size());
    // free previous VAO
    for (auto it: m_OpenglDummyTexId) {
        glDeleteVertexArrays(1, &m_dummyTex2BufferObjsInfer[it].vao);
    }

    m_dummyTex2BufferObjsInfer.clear();

    for (auto it: m_OpenglDummyTexId) {
        BufferObjects curBufObj;
        curBufObj.vertVbo = m_dummyTex2BufferObjs[it].vertVbo;
        curBufObj.meshVbo = m_dummyTex2BufferObjs[it].meshVbo;
        curBufObj.meshNumOfThisKind = m_dummyTex2BufferObjs[it].meshNumOfThisKind;

        glGenVertexArrays(1, &curBufObj.vao);
        glBindVertexArray(curBufObj.vao);

        glBindBuffer(GL_ARRAY_BUFFER, curBufObj.vertVbo);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glEnableVertexAttribArray(3);

        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*) 0);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*) (4 * sizeof(float)));
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*) (8 * sizeof(float)));
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*) (12 * sizeof(float)));

        glVertexAttribDivisor(0, 1);
        glVertexAttribDivisor(1, 1);
        glVertexAttribDivisor(2, 1);
        glVertexAttribDivisor(3, 1);

        glBindBuffer(GL_ARRAY_BUFFER, curBufObj.meshVbo);
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
        glVertexAttribDivisor(4, 1);
        
        glBindVertexArray(0);

        m_dummyTex2BufferObjsInfer.insert(std::pair<GLuint, BufferObjects>(it, curBufObj));
    }
}

void ImMeshRenderer::ReassignBuffeObjs()
{
    // free previous VBO, VAO
    for (auto it: m_OpenglDummyTexId) {
        glDeleteBuffers(1, &m_dummyTex2BufferObjs[it].vertVbo);
        glDeleteBuffers(1, &m_dummyTex2BufferObjs[it].meshVbo);
        glDeleteVertexArrays(1, &m_dummyTex2BufferObjs[it].vao);
    }

    m_dummyTex2BufferObjs.clear();


    // create new VAO VBO
    uint32_t constexpr eachVertAttribStride = 16;
    uint32_t constexpr eachMeshStride = 4;

    for (auto it: m_OpenglDummyTexId) {
        std::vector<uint32_t> curTexIds(m_dummyOpenglTexId2TexId[it].begin(), m_dummyOpenglTexId2TexId[it].end());
        std::vector<float> curVertBufInstanced(eachVertAttribStride * curTexIds.size());
        std::vector<float> curMeshBufInstanced(eachMeshStride * curTexIds.size());

        float* curVertBufHead = curVertBufInstanced.data();
        float* curMeshBufHead = curMeshBufInstanced.data();

        float* srcHead = (float*)m_vertBufData.data();

        parallel_for(blocked_range<size_t>(0, curTexIds.size()), [&](const blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i != range.end(); i++) {
                // vert attributes
                float* srcCurHead = (srcHead + (curTexIds[i] - 1) * 15);
                float* dstCurVertHead = (curVertBufHead + i * eachVertAttribStride);

                *(dstCurVertHead + 0) = *(srcCurHead + 0);
                *(dstCurVertHead + 1) = *(srcCurHead + 1);
                *(dstCurVertHead + 2) = *(srcCurHead + 2);
                *(dstCurVertHead + 3) = *(srcCurHead + 3);

                *(dstCurVertHead + 4) = *(srcCurHead + 5);
                *(dstCurVertHead + 5) = *(srcCurHead + 6);
                *(dstCurVertHead + 6) = *(srcCurHead + 7);
                *(dstCurVertHead + 7) = *(srcCurHead + 4);

                *(dstCurVertHead + 8) = *(srcCurHead + 10);
                *(dstCurVertHead + 9) = *(srcCurHead + 11);
                *(dstCurVertHead + 10) = *(srcCurHead + 12);
                *(dstCurVertHead + 11) = *(srcCurHead + 8);

                *(dstCurVertHead + 12) = *(srcCurHead + 13);
                *(dstCurVertHead + 13) = *(srcCurHead + 14);
                *(dstCurVertHead + 14) = 0.0f;
                *(dstCurVertHead + 15) = *(srcCurHead + 9);

                float* dstCurMeshHead = (curMeshBufHead + i * eachMeshStride);

                *(dstCurMeshHead + 0) = (float)(m_shTexHashMap[(curTexIds[i] - 1)].resizeInfo.curTexWH.data.curTexW);
                *(dstCurMeshHead + 1) = (float)(m_shTexHashMap[(curTexIds[i] - 1)].resizeInfo.curTexWH.data.curTexH);
                *(dstCurMeshHead + 2) = (float)(curTexIds[i]);
                *(dstCurMeshHead + 3) = 99.9f;
            }
        });

        BufferObjects curBufObj;
        curBufObj.meshNumOfThisKind = curTexIds.size();
        glGenVertexArrays(1, &curBufObj.vao);
        glBindVertexArray(curBufObj.vao);

        GLuint curVertAttrVBO;
        glGenBuffers(1, &curVertAttrVBO);
        glBindBuffer(GL_ARRAY_BUFFER, curVertAttrVBO);
        glBufferData(GL_ARRAY_BUFFER, curVertBufInstanced.size() * sizeof(float), curVertBufInstanced.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glEnableVertexAttribArray(3);

        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*) 0);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*) (4 * sizeof(float)));
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*) (8 * sizeof(float)));
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*) (12 * sizeof(float)));

        glVertexAttribDivisor(0, 1);
        glVertexAttribDivisor(1, 1);
        glVertexAttribDivisor(2, 1);
        glVertexAttribDivisor(3, 1);

        curBufObj.vertVbo = curVertAttrVBO;

        GLuint curMeshAttrVBO;
        glGenBuffers(1, &curMeshAttrVBO);
        glBindBuffer(GL_ARRAY_BUFFER, curMeshAttrVBO);
        glBufferData(GL_ARRAY_BUFFER, curMeshBufInstanced.size() * sizeof(float), curMeshBufInstanced.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
        glVertexAttribDivisor(4, 1);

        curBufObj.meshVbo = curMeshAttrVBO;

        glBindVertexArray(0);

        m_dummyTex2BufferObjs.insert(std::pair<GLuint, BufferObjects>(it, curBufObj));
    }

    assert(m_dummyTex2BufferObjs.size() == m_OpenglDummyTexId.size());

}

bool ImMeshRenderer::IsConverge(uint32_t trainCnt)
{
    uint32_t noConvergeCount = 0;
    for (uint32_t i = 0; i < m_shTexHashMap.size(); i++) {
        if (m_shTexHashMap[i].resizeInfo.isConverge == false) {
            noConvergeCount++;
        }
    }

    m_recorderPtr->AddScalar("unconverged_percentage", ((float)noConvergeCount / (float)m_shTexHashMap.size()), trainCnt);

    if (((float)noConvergeCount / (float)m_shTexHashMap.size()) >= m_resizeConvergeThreshold) {
        printf("we have %d / %d  meshes not converge!, percentage: %f, converge thres: %f\n", noConvergeCount, m_shTexHashMap.size(), ((float)noConvergeCount / (float)m_shTexHashMap.size()), m_resizeConvergeThreshold);
        return false;
    } else {
        return true;
    }
}

void ImMeshRenderer::NewDummyTextures()
{
    for (uint32_t i = 0; i < m_shTexHashMap.size(); i++) {
        CurTexWH curWH((uint32_t)m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW,
                       (uint32_t)m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH);

        if (m_shTexHashMap[i].resizeInfo.needResize == false) {
            auto find = m_texWH2dummyTex.find(curWH);
            assert(find != m_texWH2dummyTex.end());
            assert(m_dummyOpenglTexId2TexId.contains(find->second));
            assert(m_OpenglDummyTexId.contains(find->second));
            continue;
        }

        CurTexWH lastWH((uint32_t)m_shTexHashMap[i].resizeInfo.LastTexWH.data.curTexW,
                        (uint32_t)m_shTexHashMap[i].resizeInfo.LastTexWH.data.curTexH);

        auto findLastWH = m_texWH2dummyTex.find(lastWH);
        assert(findLastWH != m_texWH2dummyTex.end());
        assert(m_dummyOpenglTexId2TexId.contains(findLastWH->second));
        assert(m_dummyOpenglTexId2TexId[findLastWH->second].contains((i + 1)));
        m_dummyOpenglTexId2TexId[findLastWH->second].erase((i + 1));

        if ((m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW == 0) &&
            (m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH == 0)) {
                printf("this tex : %d, have 0x0 texture !\n", i);
        }

        
        auto findCurWH = m_texWH2dummyTex.find(curWH);
        if (findCurWH == m_texWH2dummyTex.end()) {
            GLuint curTexId = GenDummyTexture(m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW, m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH);
            assert(curTexId > 0);

            m_texWH2dummyTex.insert(std::pair<CurTexWH, GLuint>(curWH, curTexId));
            m_OpenglDummyTexId.insert(curTexId);
            m_dummyOpenglTexId2TexId.insert(std::make_pair(curTexId, std::unordered_set<uint32_t>()));
            m_dummyOpenglTexId2TexId[curTexId].insert((i + 1));
            printf("NewDummyTextures gen dummy tex: WH: %d, %d \n", m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexW, m_shTexHashMap[i].resizeInfo.curTexWH.data.curTexH);
        } else {
            assert(m_dummyOpenglTexId2TexId.contains(findCurWH->second));
            m_dummyOpenglTexId2TexId[findCurWH->second].insert((i + 1));
        }
    }
}

void ImMeshRenderer::DrawPrepareGtReso(SceneType sceneType)
{
    float fov = 60.0f;
    float fovRad = fov * (M_PI / 180.0f);

    // if (sceneType == SCENE_TYPE_REPLICA || sceneType == SCENE_TYPE_BLENDER_OUT) {
    if (sceneType == SCENE_TYPE_BLENDER_OUT) {
        mat4x4_perspective_wan(m_evalBuffer.perspectiveMatGtReso, fovRad, fovRad, 0.1f, 100.0f);
    } else {
        mat4x4_perspective_from_intrinsic(m_evalBuffer.perspectiveMatGtReso, m_fxGt, m_fyGt, m_cxGt, m_cyGt, 0.1f, 100.0f, m_gtW, m_gtH);
    }
}

int32_t ImMeshRenderer::ReleaseResourse(uint32_t curThrdIdx, uint32_t bufIdxNeedRelease)
{
    uniqueTexIdsConcurrent_t().swap(m_thrdBufs[curThrdIdx].m_uniqueTexIds[bufIdxNeedRelease]);
    texId2PixLocMapConcurrent_t().swap(m_thrdBufs[curThrdIdx].m_texId2PixLocMap[bufIdxNeedRelease]);

    m_thrdBufs[curThrdIdx].m_uniqueTexIds[bufIdxNeedRelease].reserve(100000);
    // m_thrdBufs[curThrdIdx].m_texId2PixLocMap[bufIdxNeedRelease].reserve(500000);

    return 0;
}

// Save resizeInfo.curTexWH data to JSON file using multi-threading
void saveCurTexWHJson(const std::vector<HashValue>& hashMap, const std::string& filename) {    
    rapidjson::Document document;
    document.SetObject();
    rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
    
    rapidjson::Value hashMapArray(rapidjson::kArrayType);
    
    std::cout << "Saving " << hashMap.size() << " curTexWH entries to " << filename << std::endl;
    
    // Use multi-threading to prepare JSON objects
    const size_t numElements = hashMap.size();
    
    // rapidjson does not support multi-threading writing, so we will use a single thread
    const size_t numThreads = 1;

    const size_t elementsPerThread = numElements / numThreads;
    
    std::vector<std::vector<std::pair<size_t, rapidjson::Value>>> threadResults(numThreads);
    std::vector<std::thread> threads;
    
    for (size_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            size_t startIdx = t * elementsPerThread;
            size_t endIdx = (t == numThreads - 1) ? numElements : (t + 1) * elementsPerThread;
            std::cout << "Thread " << t << " processing indices " << startIdx << " to " << endIdx << std::endl;
            
            for (size_t i = startIdx; i < endIdx; i++) {
                const auto& hashValue = hashMap[i];
                
                rapidjson::Value entry(rapidjson::kObjectType);
                entry.AddMember("index", static_cast<uint64_t>(i), allocator);
                entry.AddMember("curTexW", hashValue.resizeInfo.curTexWH.data.curTexW, allocator);
                entry.AddMember("curTexH", hashValue.resizeInfo.curTexWH.data.curTexH, allocator);
                entry.AddMember("perTexOffset", hashValue.perTexOffset, allocator);
                entry.AddMember("shDensity", hashValue.resizeInfo.curDensity, allocator);
                
                threadResults[t].emplace_back(i, std::move(entry));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Combine results from all threads in order
    for (const auto& threadResult : threadResults) {
        for (const auto& entryPair : threadResult) {
            hashMapArray.PushBack(rapidjson::Value(entryPair.second, allocator), allocator);
        }
    }
    
    document.AddMember("hashMapEntries", hashMapArray, allocator);
    document.AddMember("totalEntries", static_cast<uint64_t>(hashMap.size()), allocator);
    
    // Write to file
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);
    
    file << buffer.GetString();
    file.close();
    
    std::cout << "Successfully saved curTexWH data to " << filename << std::endl;
}

void saveShTexValuesBinary(const std::vector<HashValue>& hashMap, const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    // Write number of elements
    size_t numElements = hashMap.size();
    file.write(reinterpret_cast<const char*>(&numElements), sizeof(numElements));
    
    std::cout << "Saving " << numElements << " shTexValue arrays to " << filename << std::endl;
    
    // Use multi-threading to process data, but maintain index correspondence
    const size_t numThreads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), numElements);
    const size_t elementsPerThread = numElements / numThreads;
    
    std::vector<std::vector<ElementData>> threadResults(numThreads);
    std::vector<std::thread> threads;
    
    for (size_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            size_t startIdx = t * elementsPerThread;
            size_t endIdx = (t == numThreads - 1) ? numElements : (t + 1) * elementsPerThread;
            
            for (size_t i = startIdx; i < endIdx; ++i) {
                const auto& hashValue = hashMap[i];
                
                ElementData element;
                element.magic = 0xDEADBEEF; // Magic number for identification
                element.index = i;
                element.perTexOffset = hashValue.perTexOffset;
                
                // Always save data length, even if it's 0
                if (hashValue.shTexValue && hashValue.perTexOffset > 0) {
                    const uint8_t* dataBytes = reinterpret_cast<const uint8_t*>(hashValue.shTexValue);
                    element.data.assign(dataBytes, dataBytes + hashValue.perTexOffset);
                }
                // If no data, element.data remains empty but we still save the entry
                
                threadResults[t].push_back(std::move(element));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Write all data sequentially, preserving index order
    for (size_t t = 0; t < numThreads; ++t) {
        for (const auto& element : threadResults[t]) {
            file.write(reinterpret_cast<const char*>(&element.magic), sizeof(element.magic));
            // Write index
            file.write(reinterpret_cast<const char*>(&element.index), sizeof(element.index));
            // Write data length
            file.write(reinterpret_cast<const char*>(&element.perTexOffset), sizeof(element.perTexOffset));
            // Write data (if any)
            if (element.perTexOffset > 0 && !element.data.empty()) {
                file.write(reinterpret_cast<const char*>(element.data.data()), element.data.size());
            }
        }
    }
    
    file.close();
    std::cout << "Successfully saved shTexValue data to " << filename << std::endl;
}

void saveHashMapData(uint32_t curIterNum)
{
    ImMeshRenderer& render = ImMeshRenderer::GetInstance();
    
    const auto& hashMap = render.m_shTexHashMap;
    if (hashMap.empty()) {
        std::cout << "Warning: m_shTexHashMap is empty, nothing to save" << std::endl;
        return;
    }
    
    std::cout << "Starting to save " << hashMap.size() << " elements from m_shTexHashMap..." << std::endl;
    
    // Generate filenames with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();

    std::string baseDir = render.m_algoConfig.saveDir + "/" + render.m_taskStartTime + "_" + render.m_algoConfig.experimentName + "/intermeiate_model/";
    if (!std::filesystem::exists(baseDir)) {
        std::filesystem::create_directories(baseDir);
    }

    std::string binFilename = baseDir + "ShTextures_" + std::to_string(curIterNum) + ".bin";
    std::string jsonFilename = baseDir + "ShTexturesInfo_" + std::to_string(curIterNum) + ".json";


    std::thread jsonThread([&]() {
        saveCurTexWHJson(hashMap, jsonFilename);
    });

    std::thread binThread([&]() {
        saveShTexValuesBinary(hashMap, binFilename);
    });

    binThread.join();
    jsonThread.join();
    
    std::cout << "Data saving completed!" << std::endl;
    std::cout << "Files saved:" << std::endl;
    std::cout << "  - " << binFilename << std::endl;
    std::cout << "  - " << jsonFilename << std::endl;
}

void ImMeshRenderer::RenderThreadLoop()
{
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    // m_renderThreadCtx = glfwCreateWindow(m_width, m_height, "render_thrd", nullptr, m_mainCtx);
    // if (!m_renderThreadCtx) {
    //     std::cerr << "Failed to create GLFW window for context RenderThreadLoop" << std::endl;
    //     exit(EXIT_FAILURE);
    // }

    // glfwMakeContextCurrent(m_renderThreadCtx);


    EGLDeviceEXT eglDevs[32];
    EGLint numDevices;

    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");


    eglQueryDevicesEXT(32, eglDevs, &numDevices);

    ASSERT(numDevices, "Found no GPUs");

    PFNEGLQUERYDEVICEATTRIBEXTPROC eglQueryDeviceAttribEXT =
        reinterpret_cast<PFNEGLQUERYDEVICEATTRIBEXTPROC>(
            eglGetProcAddress("eglQueryDeviceAttribEXT"));

    int cudaDevice = 0;
    int eglDevId = 0;
    bool foundCudaDev = false;
    // Find the CUDA device asked for
    for (; eglDevId < numDevices; ++eglDevId) {
      EGLAttrib cudaDevNumber;

      if (eglQueryDeviceAttribEXT(eglDevs[eglDevId], EGL_CUDA_DEVICE_NV, &cudaDevNumber) ==
          EGL_FALSE)
        continue;

      if (cudaDevNumber == cudaDevice) {
        break;
      }
    }

    printf("use dev : %d, found %d\n", eglDevId, foundCudaDev);


    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    m_eglRenderThrdDsp = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevs[eglDevId], 0);

    // m_eglRenderThrdDsp = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major, minor;

    eglInitialize(m_eglRenderThrdDsp, &major, &minor);
    EGLint numConfigs;
    EGLConfig eglCfg;
    eglChooseConfig(m_eglRenderThrdDsp, configAttribs, &eglCfg, 1, &numConfigs);
    eglBindAPI(EGL_OPENGL_API);

    m_renderThreadCtx = eglCreateContext(m_eglRenderThrdDsp, eglCfg, m_mainThrdCtx, NULL);

    eglMakeCurrent(m_eglRenderThrdDsp, EGL_NO_SURFACE, EGL_NO_SURFACE, m_renderThreadCtx);

    ReassignBuffeObjs();

    ShaderInit(false);

    DrawPrepare(m_fileConfig.sceneType, false);
    DrawPrepareGtReso(m_fileConfig.sceneType);

    InitThreadBuffers();
    EvalBufferInit();

    std::random_device rd;
    if (rd.entropy() == 0) {
        std::cerr << "Warning: Non-deterministic random number generation not available." << std::endl;
    }

    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, m_camPosWorld.size() - 1);

    // std::uniform_int_distribution<> distribSubTrain(0, m_camPosWorldSub.size() - 1);

    m_renderThrdInited.store(true);

    uint32_t trainCount = 0;
    uint32_t resizeConvergedCounter = 0;
    unsigned long lastTrainEndTime;
    unsigned long startTime;
    unsigned long endTime;

    std::future<int32_t> retReleaseResourse[NUM_BUFFERS];
    int32_t retVal = -1;

    while (m_threadRunning.load() == true) {
        {
            std::unique_lock<std::mutex> lock(m_thrdBufs[m_renderThrdCurIdx].bufMutex);
            if (m_thrdBufs[m_renderThrdCurIdx].state != READY_FOR_RENDER) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } else {
                // instantly return can cause 4 frames in buffer not to be trained, lazy, may fix later
                if (trainCount >= m_trainPoseCount) {
                    m_threadRunning.store(false);
                    lock.unlock();
                    return;
                }

                if (trainCount >= m_resizeStartStep && m_curTrainState == STATE_NOT_RESIZED) {
                    m_curTrainState = STATE_UNDER_RESIZED;
                }

                if (m_curTrainState == STATE_CONVERGE_RESIZED) {
                    if ((trainCount != 0) &&
                        (resizeConvergedCounter > 0) &&
                        (resizeConvergedCounter % m_algoConfig.ConvergedSaveInterval == 0)) {
                        if (!((m_renderThrdCurIdx == m_trainPrepThrdCurIdx) && (m_renderThrdCurIdx == m_trainThrdCurIdx))) {
                            lock.unlock();
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            continue;
                        }

                        std::cout << "saving data at train count: " << trainCount << "save interval is: " << m_algoConfig.ConvergedSaveInterval << std::endl;

                        saveHashMapData(m_thrdBufs[(m_renderThrdCurIdx + NUM_BUFFERS - 1) % NUM_BUFFERS].curTrainCount);

                        resizeConvergedCounter = 0;
                    }

                    resizeConvergedCounter++;
                }

                if (m_curTrainState == STATE_UNDER_RESIZED) {
                    if ((trainCount != 0) &&
                        (m_resizeStartStep > 0) && 
                        ((trainCount - m_resizeStartStep) % m_texResizeInterval == 0)) {
                        if (!((m_renderThrdCurIdx == m_trainPrepThrdCurIdx) && (m_renderThrdCurIdx == m_trainThrdCurIdx))) {
                            lock.unlock();
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            continue;
                        }

                        saveHashMapData(m_thrdBufs[(m_renderThrdCurIdx + NUM_BUFFERS - 1) % NUM_BUFFERS].curTrainCount);

                        ResizeShTextures();
                        m_thrdBufs[m_renderThrdCurIdx].needResetOptimizor = true;

                        m_curResizeCnt++;
                        if (m_curResizeCnt >= m_resizeMaxCnt) {
                            m_curTrainState = STATE_CONVERGE_RESIZED;
                            printf("meet max m_resizeMaxCnt, converge not resize !, cur cnt: %d, maxCnt: %d\n", m_curResizeCnt, m_resizeMaxCnt);

                        }

                        printf("cur resize cnt: %d, maxCnt: %d\n", m_curResizeCnt, m_resizeMaxCnt);
                        
                        // don't need lock, because we make sure other thrdBufs are finished before we do resize.
                        if (IsConverge(trainCount) == true) {
                            m_curTrainState = STATE_CONVERGE_RESIZED;
                        }
                    }
                }

                startTime = GetTimeMS();

                PROFILE_PRINT(cout << g_renderThrdName << " start idx: " << m_renderThrdCurIdx << " delta:  " << (double)(startTime - lastTrainEndTime) << std::endl);
                
                // static int poseIdx = 0;
                int poseIdx = 0;
                // if (m_isSubTrainMode == true) {
                //     poseIdx = distribSubTrain(gen);
                // } else {
                    poseIdx = distrib(gen);
                // }

                m_thrdBufs[m_renderThrdCurIdx].poseIdx = poseIdx;
                m_thrdBufs[m_renderThrdCurIdx].curTrainCount = trainCount;
                DoRender(m_renderThrdCurIdx, poseIdx);

                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_renderThrdName << " ------till DoRender passed time : " << (double)(endTime - startTime) << " ms" << endl);

                // m_thrdBufs[m_renderThrdCurIdx].m_uniqueTexIds.clear();
                // m_thrdBufs[m_renderThrdCurIdx].m_texId2PixLocMap.clear();

                // uniqueTexIdsConcurrent_t().swap(m_thrdBufs[m_renderThrdCurIdx].m_uniqueTexIds);
                // texId2PixLocMapConcurrent_t().swap(m_thrdBufs[m_renderThrdCurIdx].m_texId2PixLocMap);

                if (m_thrdBufs[m_renderThrdCurIdx].isFirstTime == false) {
                    retVal = retReleaseResourse[m_renderThrdCurIdx].get();
                    if (retVal != 0) {
                        printf("retReleaseResourse is invalid \n");
                        exit(0);
                    } else {
                        retVal = -1;
                    }
                } else {
                    m_thrdBufs[m_renderThrdCurIdx].isFirstTime = false;
                }
                

                retReleaseResourse[m_renderThrdCurIdx] = std::async(std::launch::async,
                    &ImMeshRenderer::ReleaseResourse,
                    &ImMeshRenderer::GetInstance(),
                    m_renderThrdCurIdx,
                    ((m_thrdBufs[m_renderThrdCurIdx].curValidIdx + 1) % 2));

                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_renderThrdName << " ------till clear passed time : " << (double)(endTime - startTime) << " ms" << endl);

                GenHashMaps(m_thrdBufs[m_renderThrdCurIdx].m_uniqueTexIds[m_thrdBufs[m_renderThrdCurIdx].curValidIdx],
                            m_thrdBufs[m_renderThrdCurIdx].m_texId2PixLocMap[m_thrdBufs[m_renderThrdCurIdx].curValidIdx],
                            m_thrdBufs[m_renderThrdCurIdx]);
                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_renderThrdName << " ------till GenHashMaps passed time : " << (double)(endTime - startTime) << " ms" << endl);

                // ConstructCompactTex2PixBufferPipeLine(m_thrdBufs[m_renderThrdCurIdx].m_texId2PixLocMap[m_thrdBufs[m_renderThrdCurIdx].curValidIdx],
                //                                       m_thrdBufs[m_renderThrdCurIdx].m_uniqueTexIds[m_thrdBufs[m_renderThrdCurIdx].curValidIdx],
                //                                       m_thrdBufs[m_renderThrdCurIdx]);

                // endTime = GetTimeMS();
                // cout << g_renderThrdName << " ------till ConstructCompactTex2PixBufferPipeLine passed time : " << (double)(endTime - startTime) << " ms" << endl;

                // m_thrdBufs[m_renderThrdCurIdx].dummyCount++;
                PROFILE_PRINT(cout << "RenderThreadLoop has rendered buffer : " << m_renderThrdCurIdx << ", dummy : " << m_thrdBufs[m_renderThrdCurIdx].dummyCount << endl);
                m_thrdBufs[m_renderThrdCurIdx].state = READY_FOR_TRAIN_PREP;
                // poseIdx = ((poseIdx + 1) % m_camPosWorld.size());
                
                lock.unlock();

                trainCount++;
                m_renderThrdCurIdx = (m_renderThrdCurIdx + 1) % NUM_BUFFERS;
                lastTrainEndTime = GetTimeMS();
            }
        }
    }

    // glfwDestroyWindow(m_renderThreadCtx);
}

void ImMeshRenderer::ConstructCpctBuffersForCuda(ThreadBuffer& buf, bool isInfer)
{
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
    auto optionsGPU = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

    // auto optionsI32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto optionsI32_GPU = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);

    uint32_t maxValidOffset = CalcCValidBitNum(m_vertNbrAddrOffsetBit);

    buf.posMapMemLayout.resize(buf.curTexIdsAll.size());
    buf.shTexMemLayout.resize(buf.curTexIdsAll.size());
    buf.validShWHMapLayout.resize(buf.curTexIdsAll.size());
    buf.vertNbrCpctMemLayout.resize(buf.curTexIdsAll.size());

    uint32_t posMapOffset = 0;
    uint32_t shTexOffset = 0;
    uint32_t validShWHMapOffset = 0;
    uint32_t vertNbrOffset = 0;

    for (uint32_t i = 0; i < buf.curTexIdsAll.size(); i++) {
        buf.posMapMemLayout[i] = posMapOffset;
        buf.shTexMemLayout[i] = shTexOffset;
        buf.validShWHMapLayout[i] = validShWHMapOffset;
        buf.vertNbrCpctMemLayout[i] = vertNbrOffset;

        posMapOffset += (m_shTexHashMap[buf.curTexIdsAll[i] - 1].validShNum * 3);
        shTexOffset += (m_shTexHashMap[buf.curTexIdsAll[i] - 1].validShNum * m_shTexChannelNum * m_shLayerNum);
        validShWHMapOffset += (m_shTexHashMap[buf.curTexIdsAll[i] - 1].validShNum);
        vertNbrOffset += (m_shTexHashMap[buf.curTexIdsAll[i] - 1].topVertNbr.size() +
            m_shTexHashMap[buf.curTexIdsAll[i] - 1].leftVertNbr.size() +
            m_shTexHashMap[buf.curTexIdsAll[i] - 1].rightVertNbr.size());
    }

    if (vertNbrOffset >= maxValidOffset) {
        std::cout << "cannot construct compact vert neighbor buffer, offset exceed!";
        exit(0);
    }

    buf.shPosMapCpct = torch::zeros({posMapOffset}, optionsGPU);
    buf.shTexturesCpct = torch::zeros({shTexOffset}, optionsGPU).set_requires_grad(true);
    
    buf.validShWHMapCpct = torch::zeros({validShWHMapOffset}, optionsI32_GPU);
    buf.vertNbrs = torch::zeros({vertNbrOffset}, optionsI32_GPU);

    buf.shPosMapCpct = buf.shPosMapCpct.contiguous();
    // buf.shTexturesCpct = buf.shTexturesCpct.contiguous();
    buf.validShWHMapCpct = buf.validShWHMapCpct.contiguous();
    buf.vertNbrs = buf.vertNbrs.contiguous();

    if (isInfer == true) {
        InitTrainPrepThrdCpyBufferDynamicInfer(posMapOffset, validShWHMapOffset, vertNbrOffset);
    } else {
        InitTrainPrepThrdCpyBufferDynamic(posMapOffset, validShWHMapOffset, vertNbrOffset);
    }
}

void ImMeshRenderer::TrainPrepareThreadLoop()
{
    while (1) {
        if (m_renderThrdInited.load() == true) {
            break;
        }

        sleep(1);
    }

    checkCudaErrors(cudaSetDevice(m_cudaDevice));
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    // // GLFWwindow* trainPrepThreadCtx = nullptr;
    // m_trainPrepThreadCtx = glfwCreateWindow(m_width, m_height, "trainPrep_thrd", nullptr, m_renderThreadCtx);
    // if (!m_trainPrepThreadCtx) {
    //     std::cerr << "Failed to create GLFW window for context TrainPrepareThreadLoop" << std::endl;
    //     exit(EXIT_FAILURE);
    // }

    // glfwMakeContextCurrent(m_trainPrepThreadCtx);
    m_eglTrainPrepThrdDsp = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major, minor;

    eglInitialize(m_eglTrainPrepThrdDsp, &major, &minor);
    EGLint numConfigs;
    EGLConfig eglCfg;
    eglChooseConfig(m_eglTrainPrepThrdDsp, configAttribs, &eglCfg, 1, &numConfigs);
    eglBindAPI(EGL_OPENGL_API);

    m_trainPrepThreadCtx = eglCreateContext(m_eglTrainPrepThrdDsp, eglCfg, m_renderThreadCtx, NULL);
    eglMakeCurrent(m_eglTrainPrepThrdDsp, EGL_NO_SURFACE, EGL_NO_SURFACE, m_trainPrepThreadCtx);

    m_trainPrepThrdInited.store(true);

    unsigned long lastTrainEndTime;

    checkCudaErrors(cudaStreamCreateWithFlags(&m_trainPrepThrdCpyStrm1, cudaStreamNonBlocking));
    checkCudaErrors(cudaStreamCreateWithFlags(&m_trainPrepThrdCpyStrm2, cudaStreamNonBlocking));

    while (m_threadRunning.load() == true) {
        {
            std::unique_lock<std::mutex> lock(m_thrdBufs[m_trainPrepThrdCurIdx].bufMutex);
            if (m_thrdBufs[m_trainPrepThrdCurIdx].state != READY_FOR_TRAIN_PREP) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            } else {
                unsigned long startTime = GetTimeMS();
                unsigned long endTime;

                // printf("TrainPrepareThreadLoop start idx: %d, delta\n", m_trainPrepThrdCurIdx, );

                PROFILE_PRINT(cout << g_trainPrepThrdName << " start idx:" << m_trainPrepThrdCurIdx << " delta:  " << (double)(startTime - lastTrainEndTime) << std::endl);


                // uniqueTexIdsConcurrent_t uniqueTexIds;
                // uniqueTexIds.reserve(100000);

                // texId2PixLocMapConcurrent_t texId2PixLocMap;
               
                // GenHashMaps(uniqueTexIds, texId2PixLocMap, m_thrdBufs[m_trainPrepThrdCurIdx]);

                // endTime = GetTimeMS();
                // cout << g_trainPrepThrdName << " ------till GenHashMaps passed time : " << (double)(endTime - startTime) << " ms" << endl;

                ConstructCompactTex2PixBufferPipeLine(m_thrdBufs[m_trainPrepThrdCurIdx].m_texId2PixLocMap[m_thrdBufs[m_trainPrepThrdCurIdx].curValidIdx],
                                                      m_thrdBufs[m_trainPrepThrdCurIdx].m_uniqueTexIds[m_thrdBufs[m_trainPrepThrdCurIdx].curValidIdx],
                                                      m_thrdBufs[m_trainPrepThrdCurIdx]);
                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainPrepThrdName << " ------till ConstructCompactTex2PixBufferPipeLine passed time : " << (double)(endTime - startTime) << " ms" << endl);

                AddInvisibleTexId(m_thrdBufs[m_trainPrepThrdCurIdx].m_uniqueTexIds[m_thrdBufs[m_trainPrepThrdCurIdx].curValidIdx]);
                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainPrepThrdName << " ------till AddInvisibleTexId passed time : " << (double)(endTime - startTime) << " ms" << endl);

                m_thrdBufs[m_trainPrepThrdCurIdx].curTexIdsAll.resize(m_thrdBufs[m_trainPrepThrdCurIdx].m_uniqueTexIds[m_thrdBufs[m_trainPrepThrdCurIdx].curValidIdx].size());
                m_thrdBufs[m_trainPrepThrdCurIdx].curTexIdsAll.assign(m_thrdBufs[m_trainPrepThrdCurIdx].m_uniqueTexIds[m_thrdBufs[m_trainPrepThrdCurIdx].curValidIdx].begin(),
                                                                      m_thrdBufs[m_trainPrepThrdCurIdx].m_uniqueTexIds[m_thrdBufs[m_trainPrepThrdCurIdx].curValidIdx].end());

                ConstructCpctBuffersForCuda(m_thrdBufs[m_trainPrepThrdCurIdx], false);
                
                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainPrepThrdName << " ------till ConstructCpctBuffersForCuda passed time : " << (double)(endTime - startTime) << " ms" << endl);

                ConstructDeviceShHashAllNbrPipeLine(m_thrdBufs[m_trainPrepThrdCurIdx]);
                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainPrepThrdName << " ------till ConstructDeviceShHashAllNbrPipeLine passed time : " << (double)(endTime - startTime) << " ms" << endl);

                InitCurBufTrainTensors(m_trainPrepThrdCurIdx);
                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainPrepThrdName << " ------till InitCurBufTrainTensors passed time : " << (double)(endTime - startTime) << " ms" << endl);

                CopyCurTrainEssentialsFixed(m_trainPrepThrdCurIdx);
                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainPrepThrdName << " ------till CopyCurTrainEssentialsFixed passed time : " << (double)(endTime - startTime) << " ms" << endl);

                // m_thrdBufs[m_trainPrepThrdCurIdx].dummyCount++;

                checkCudaErrors(cudaStreamSynchronize(m_trainPrepThrdCpyStrm1));
                checkCudaErrors(cudaStreamSynchronize(m_trainPrepThrdCpyStrm2));
                
                checkCudaErrors(cudaStreamSynchronize(0));

                PROFILE_PRINT(cout << g_trainPrepThrdName << " ------till cudaStreamSynchronize passed time : " << (double)(endTime - startTime) << " ms" << endl);
                m_thrdBufs[m_trainPrepThrdCurIdx].state = READY_FOR_TRAIN;
                PROFILE_PRINT(cout << "TrainPrepareThreadLoop end idx: " << m_trainPrepThrdCurIdx << endl);

                // printf("TrainPrepareThreadLoop has processed buffer : %d, dummy: %d\n", m_trainPrepThrdCurIdx, m_thrdBufs[m_trainPrepThrdCurIdx].dummyCount);
                lock.unlock();

                m_trainPrepThrdCurIdx = (m_trainPrepThrdCurIdx + 1) % NUM_BUFFERS;
                lastTrainEndTime = GetTimeMS();
            }
        }
    }

    // glfwDestroyWindow(m_trainPrepThreadCtx);
}

void ImMeshRenderer::InitCurBufTrainTensors(uint32_t thrdBufIdx)
{
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
    auto optionsGPU = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

    // auto optionsI32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    auto optionsI32_GPU = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);

    uint32_t uniqueTexNum = m_thrdBufs[thrdBufIdx].curTexIdsAll.size();

    InitThrdFixedCpyBuffer(m_thrdBufs[thrdBufIdx], uniqueTexNum);
    m_thrdBufs[thrdBufIdx].validShWHMapCpctHead = m_thrdBufs[thrdBufIdx].validShWHMapCpct.data_ptr<int32_t>();

    m_thrdBufs[thrdBufIdx].validShNumsAll = torch::zeros({uniqueTexNum}, optionsI32_GPU);
    m_thrdBufs[thrdBufIdx].validShNumsAllHead = m_thrdBufs[thrdBufIdx].validShNumsAll.data_ptr<int32_t>();

    m_thrdBufs[thrdBufIdx].shTexturesCpctHead = m_thrdBufs[thrdBufIdx].shTexturesCpct.data_ptr<float>();

    // m_thrdBufs[thrdBufIdx].adamExpAvg = torch::zeros_like(m_thrdBufs[thrdBufIdx].shTexturesCpct, c10::MemoryFormat::Contiguous);
    m_thrdBufs[thrdBufIdx].adamExpAvg = torch::zeros(m_thrdBufs[thrdBufIdx].shTexturesCpct.numel(), optionsGPU);
    m_thrdBufs[thrdBufIdx].adamExpAvgHead = m_thrdBufs[thrdBufIdx].adamExpAvg.data_ptr<float>();

    // m_thrdBufs[thrdBufIdx].adamExpAvgSq = torch::zeros_like(m_thrdBufs[thrdBufIdx].shTexturesCpct, c10::MemoryFormat::Contiguous);
    m_thrdBufs[thrdBufIdx].adamExpAvgSq = torch::zeros(m_thrdBufs[thrdBufIdx].shTexturesCpct.numel(), optionsGPU);
    m_thrdBufs[thrdBufIdx].adamExpAvgSqHead = m_thrdBufs[thrdBufIdx].adamExpAvgSq.data_ptr<float>();

    // printf("is pinned is contiguous : shTexturesCpct %d, %d,  adamExpAvg %d %d, adamExpAvgSq %d %d\n",
    //     m_thrdBufs[thrdBufIdx].shTexturesCpct.is_pinned(), m_thrdBufs[thrdBufIdx].shTexturesCpct.is_contiguous(),
    //     m_thrdBufs[thrdBufIdx].adamExpAvg.is_pinned(), m_thrdBufs[thrdBufIdx].adamExpAvg.is_contiguous(),
    //     m_thrdBufs[thrdBufIdx].adamExpAvgSq.is_pinned(), m_thrdBufs[thrdBufIdx].adamExpAvgSq.is_contiguous());

    // exit(0);

    // m_thrdBufs[thrdBufIdx].curTrainShPosMap = torch::zeros({uniqueTexNum, m_texH, m_texW, 3}, options);
    // m_thrdBufs[thrdBufIdx].curTrainShPosMapHead = m_thrdBufs[thrdBufIdx].curTrainShPosMap.data_ptr<float>();

    // m_thrdBufs[thrdBufIdx].curTrainShValidMap = torch::zeros({uniqueTexNum, m_texH, m_texW}, optionsI32);
    // m_thrdBufs[thrdBufIdx].curTrainShValidMapHead = m_thrdBufs[thrdBufIdx].curTrainShValidMap.data_ptr<int32_t>();

    m_thrdBufs[thrdBufIdx].edgeNbrs = torch::zeros({uniqueTexNum, 6}, optionsI32_GPU);
    m_thrdBufs[thrdBufIdx].edgeNbrsHead = m_thrdBufs[thrdBufIdx].edgeNbrs.data_ptr<int32_t>();

    m_thrdBufs[thrdBufIdx].cornerAreaInfo = torch::zeros({uniqueTexNum, 2}, optionsGPU);
    m_thrdBufs[thrdBufIdx].cornerAreaInfoHead = m_thrdBufs[thrdBufIdx].cornerAreaInfo.data_ptr<float>();

    m_thrdBufs[thrdBufIdx].texInWorldInfo = torch::zeros({uniqueTexNum, 6}, optionsGPU);
    m_thrdBufs[thrdBufIdx].texInWorldInfoHead = m_thrdBufs[thrdBufIdx].texInWorldInfo.data_ptr<float>();

    m_thrdBufs[thrdBufIdx].vertNbrsHead = m_thrdBufs[thrdBufIdx].vertNbrs.data_ptr<int32_t>();

    m_thrdBufs[thrdBufIdx].texWHs = torch::zeros({uniqueTexNum}, optionsI32_GPU);
    m_thrdBufs[thrdBufIdx].texWHsHead = (uint32_t*)(m_thrdBufs[thrdBufIdx].texWHs.data_ptr<int32_t>());

    m_thrdBufs[thrdBufIdx].botYCoeffs = torch::zeros({uniqueTexNum}, optionsGPU);
    m_thrdBufs[thrdBufIdx].botYCoeffsHead = m_thrdBufs[thrdBufIdx].botYCoeffs.data_ptr<float>();

    m_thrdBufs[thrdBufIdx].topYCoeffs = torch::zeros({uniqueTexNum}, optionsGPU);
    m_thrdBufs[thrdBufIdx].topYCoeffsHead = m_thrdBufs[thrdBufIdx].topYCoeffs.data_ptr<float>();

    m_thrdBufs[thrdBufIdx].shPosMapCpctHead = m_thrdBufs[thrdBufIdx].shPosMapCpct.data_ptr<float>();

    m_thrdBufs[thrdBufIdx].meshDensities = torch::zeros({uniqueTexNum}, optionsGPU);
    m_thrdBufs[thrdBufIdx].meshDensitiesHead = m_thrdBufs[thrdBufIdx].meshDensities.data_ptr<float>();

    m_thrdBufs[thrdBufIdx].meshNormals = torch::zeros({uniqueTexNum * 3}, optionsGPU);
    m_thrdBufs[thrdBufIdx].meshNormalsHead = m_thrdBufs[thrdBufIdx].meshNormals.data_ptr<float>();
}

void ImMeshRenderer::ConstructDeviceShHashAllNbrPipeLine(ThreadBuffer& buf)
{
    uint32_t constexpr emptyKeySentinel = 0;
    uint64_t constexpr emptyValSentinel = 0;
    uint32_t maxInTensorOffset = CalcCValidBitNum(m_inTensorOffsetBit);

    if (buf.curTexIdsAll.size() > maxInTensorOffset) {
        std::cout << "num of texId in current view is overflowed\n";
        exit(0);
    }

    assert(buf.vertNbrCpctMemLayout.size() == buf.curTexIdsAll.size());
    std::vector<uint64_t> insertValsStd(buf.curTexIdsAll.size());
    ConstructCustomHashValueAllNbrPipeLine(insertValsStd, buf.vertNbrCpctMemLayout, buf);

    thrust::device_vector<uint64_t> insertVals = insertValsStd;

    thrust::device_vector<uint32_t> insertKeys(buf.curTexIdsAll);

    auto constexpr load_factor = 0.5f;
    std::size_t const capacity = std::ceil(buf.curTexIdsAll.size() / load_factor);

    buf.curTrainHashMap = nullptr;
    buf.curTrainHashMap = std::make_unique<cuco::static_map<uint32_t, uint64_t>> (capacity,
                cuco::empty_key{emptyKeySentinel},
                cuco::empty_value{emptyValSentinel},
                thrust::equal_to<uint32_t>{},
                cuco::linear_probing<1, cuco::default_hash_function<uint32_t>>{});
    
    auto insert_ref = buf.curTrainHashMap->ref(cuco::insert);
    // thrust::device_vector<int32_t> numInserted(1);

    int constexpr blockSize = 256;
    int const gridSize = (buf.curTexIdsAll.size() + blockSize - 1) / blockSize;
    filtered_insert <<<gridSize, blockSize>>>(insert_ref,
                                              insertKeys.begin(),
                                              insertVals.begin(),
                                              buf.curTexIdsAll.size());
                                            //   numInserted.data().get());

    // std::cout << "Number of keys inserted: " << numInserted[0] << std::endl;
    PROFILE_PRINT(std::cout << "Number of elements in hash map: " << buf.curTrainHashMap->size() << std::endl);
}

void ImMeshRenderer::ConstructCompactTex2PixBufferPipeLine(texId2PixLocMapConcurrent_t& texId2PixLocMap, uniqueTexIdsConcurrent_t& uniqueTexIds, ThreadBuffer& buf)
{
    unsigned long startTime = GetTimeMS();
    unsigned long endTime;
    // thrust::host_vector<TexPixInfo> visibleTexIdsInfoHost(uniqueTexIds.size());
    std::vector<TexPixInfo> visibleTexIdsInfoStd(uniqueTexIds.size());
    std::vector<uint32_t> visibleTexIdsVec;
    visibleTexIdsVec.assign(uniqueTexIds.begin(), uniqueTexIds.end());

    uint32_t idx = 0;
    uint32_t curOffset = 0;

    std::vector<PixLocation> visibleTexPixLocCompactStd(m_width * m_height);

    std::mutex offsetUpdateLock;

    parallel_for(blocked_range<uint32_t>(0, uniqueTexIds.size()), ConstructCompactTex2PixBufferMultiThrd(offsetUpdateLock, visibleTexIdsInfoStd, visibleTexPixLocCompactStd, visibleTexIdsVec, texId2PixLocMap, idx, curOffset));


    // for (auto it: uniqueTexIds) {
    //     visibleTexIdsInfoHost[idx].texId = it;

    //     // auto findNum = texId2PixLocMap.count(it);

    //     typename texId2PixLocMapConcurrent_t::const_accessor access;
    //     // assert(texId2PixLocMap.find(access, it) == true);
    //     auto ret = texId2PixLocMap.find(access, it);
    //     assert(ret == true);
    //     uint32_t findNum = access->second.size();

    //     for (uint32_t ii = 0; ii < findNum; ii++) {
    //         visibleTexPixLocCompactStd[curOffset + ii].x = access->second[ii].x;
    //         visibleTexPixLocCompactStd[curOffset + ii].y = access->second[ii].y;
    //     }

    //     access.release();

    //     if (findNum <= 0) {
    //         printf("cannot find texId : %d-------------\n", it);
    //         exit(0);
    //     }

    //     visibleTexIdsInfoHost[idx].offset = curOffset;
    //     visibleTexIdsInfoHost[idx].pixNum = findNum;

    //     curOffset += findNum;
    //     idx++;
    // }

    assert(idx == uniqueTexIds.size());
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << "fill visibleTexIdsInfoHost, count all iterm time: : " << (double)(endTime - startTime) << " ms" << endl);

    buf.visibleTexIdsInfo = visibleTexIdsInfoStd;
    buf.visibleTexPixLocCompact = visibleTexPixLocCompactStd;
    // // thrust::host_vector<cuco::pair<uint32_t, PixLocation>> rawMapData(texId2PixLocMap.begin(), texId2PixLocMap.end());
    // thrust::host_vector<cuco::pair<uint32_t, PixLocation>> rawMapData(curOffset);
    // uint32_t writePos = 0;
    // for (auto it = texId2PixLocMap.begin(); it != texId2PixLocMap.end(); it++) {
    //     for (uint32_t count = 0; count < it->second.size(); count++) {
    //         rawMapData[writePos] = cuco::pair<uint32_t, PixLocation>(it->first, it->second[count]);
    //         writePos++;
    //     }
    // }

    // assert(writePos == curOffset);


    // thrust::device_vector<cuco::pair<uint32_t, PixLocation>> rawMapDataDevice = rawMapData;

    // uint32_t constexpr blockSize2 = 32;
    // auto gridSizePerTexId = (visibleTexIdsInfoHost.size() + blockSize2 - 1) / blockSize2;

    // fill_compact_buffer_per_tex_id <<<gridSizePerTexId, blockSize2>>> (buf.visibleTexIdsInfo.begin(), buf.visibleTexIdsInfo.size(), buf.visibleTexPixLocCompact.begin(), rawMapDataDevice.begin(), rawMapData.size());

    endTime = GetTimeMS();
    PROFILE_PRINT(cout << "till fill_compact_buffer_per_tex_id time: : " << (double)(endTime - startTime) << " ms" << endl);

}


struct GenHashMapMultiThrd {
    texId2PixLocMapConcurrent_t& m_texId2PixLocMap;
    uniqueTexIdsConcurrent_t& m_uniqueTexIds;
    float* m_depthAndTexIdBuffer{nullptr};
    uint32_t m_imgW;
    uint32_t m_imgH;

    GenHashMapMultiThrd(texId2PixLocMapConcurrent_t& texId2PixLocMap_,
        uniqueTexIdsConcurrent_t& uniqueTexIds_,
        float* depthAndTexIdBuffer_,
        uint32_t imgW_, 
        uint32_t imgH_): m_texId2PixLocMap(texId2PixLocMap_), m_uniqueTexIds(uniqueTexIds_), m_depthAndTexIdBuffer(depthAndTexIdBuffer_), m_imgW(imgW_), m_imgH(imgH_) {}

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
        uint32_t perLineOffset = 4 * m_imgW;
        for (uint32_t i = range.begin(); i != range.end(); i++) {
            if (static_cast<unsigned int>((*(m_depthAndTexIdBuffer + i * 4 + 3))) == 0) {
                continue;
            }

            uint32_t texId = static_cast<unsigned int>((*(m_depthAndTexIdBuffer + i * 4 + 3)));

            PixLocation tmp((i % m_imgW), (m_imgH - 1 - (i / m_imgW)));
            m_uniqueTexIds.insert(texId);

            MapInsertPushBack(m_texId2PixLocMap, texId, tmp);            
        }
    }
};

void ImMeshRenderer::GenHashMaps(uniqueTexIdsConcurrent_t& uniqueTexIds, texId2PixLocMapConcurrent_t& texId2PixLocMap, ThreadBuffer& buf)
{
    unsigned int perPixOffset = 4;
    unsigned int perLineOffset = perPixOffset * m_width;

    parallel_for(blocked_range<uint32_t>(0, m_height * m_width), GenHashMapMultiThrd(texId2PixLocMap, uniqueTexIds, buf.texScaleFacDepthTexIdCPU, m_width, m_height));

    PROFILE_PRINT(cout << "uniqueTexIds size: " << uniqueTexIds.size() << ", texId2PixLocMap size: " << texId2PixLocMap.size() << endl);
}


void ImMeshRenderer::ClearHashMapData()
{
    std::cout << "Clearing existing hashmap data..." << std::endl;
    
    for (auto& hashValue : m_shTexHashMap) {
        // Free CPU memory for shTexValue
        if (hashValue.shTexValue != nullptr) {
            delete[] hashValue.shTexValue;
            hashValue.shTexValue = nullptr;
        }
        
        // Note: We don't free other pointers here as they might be managed elsewhere
        // If needed, add cleanup for other allocated memory
    }
    
    std::cout << "Hashmap data cleared." << std::endl;
}

void ImMeshRenderer::EnableProfilingPrints(bool enable)
{
    g_enableProfilingPrints = enable;
    std::cout << "Profiling prints " << (enable ? "ENABLED" : "DISABLED") << std::endl;
}

bool ImMeshRenderer::IsProfilingEnabled()
{
    return g_enableProfilingPrints;
}

bool ImMeshRenderer::LoadHashMapData(const std::string& binFilename, const std::string& jsonFilename)
{
    std::cout << "Loading data from:" << std::endl;
    std::cout << "  - JSON file: " << jsonFilename << std::endl;
    std::cout << "  - Binary file: " << binFilename << std::endl;
    
    // Step 1: Load JSON data to get texture WH information and validate structure
    rapidjson::Document jsonDoc;
    std::ifstream jsonFile(jsonFilename);
    if (!jsonFile.is_open()) {
        std::cerr << "Error: Could not open JSON file " << jsonFilename << std::endl;
        return false;
    }
    
    std::string jsonContent((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());
    jsonFile.close();
    
    if (jsonDoc.Parse(jsonContent.c_str()).HasParseError()) {
        std::cerr << "Error: Failed to parse JSON file" << std::endl;
        return false;
    }
    
    if (!jsonDoc.HasMember("hashMapEntries") || !jsonDoc.HasMember("totalEntries")) {
        std::cerr << "Error: JSON file missing required fields" << std::endl;
        return false;
    }
    
    const auto& hashMapEntries = jsonDoc["hashMapEntries"];
    uint64_t totalEntries = jsonDoc["totalEntries"].GetUint64();
    
    std::cout << "JSON loaded: " << hashMapEntries.Size() << " entries, total expected: " << totalEntries << std::endl;
    
    // Create a map for quick JSON data lookup by index
    std::unordered_map<uint32_t, const rapidjson::Value*> jsonDataMap;
    for (const auto& entry : hashMapEntries.GetArray()) {
        if (!entry.HasMember("index") || !entry.HasMember("perTexOffset") || 
            !entry.HasMember("curTexW") || !entry.HasMember("curTexH")) {
            std::cerr << "Error: JSON entry missing required fields" << std::endl;
            return false;
        }
        uint32_t index = entry["index"].GetUint();
        jsonDataMap[index] = &entry;
    }
    
    // Step 2: Load binary data
    std::ifstream binFile(binFilename, std::ios::binary);
    if (!binFile.is_open()) {
        std::cerr << "Error: Could not open binary file " << binFilename << std::endl;
        return false;
    }
    
    // Read number of elements
    size_t numElements;
    binFile.read(reinterpret_cast<char*>(&numElements), sizeof(numElements));
    if (binFile.gcount() != sizeof(numElements)) {
        std::cerr << "Error: Could not read number of elements from binary file" << std::endl;
        return false;
    }
    
    std::cout << "Binary file: " << numElements << " elements expected" << std::endl;
    
    if (numElements != totalEntries) {
        std::cerr << "Error: Mismatch between JSON totalEntries (" << totalEntries 
                  << ") and binary numElements (" << numElements << ")" << std::endl;
        return false;
    }
    
    // Clear and resize the hashmap
    // ClearHashMapData();  // Clean up any existing data
    // m_shTexHashMap.clear();
    // m_shTexHashMap.resize(numElements);
    
    // // Initialize all entries
    // for (size_t i = 0; i < numElements; ++i) {
    //     HashValue& hashValue = m_shTexHashMap[i];
        
    //     // Initialize pointers to nullptr
    //     hashValue.shTexValue = nullptr;
    //     hashValue.adamStateExpAvg = nullptr;
    //     hashValue.adamStateExpAvgSq = nullptr;
    //     hashValue.worldPoseMap = nullptr;
    //     hashValue.validShWHMap = nullptr;
    //     hashValue.validShWHMapNew = nullptr;
        
    //     // Initialize other fields with defaults
    //     hashValue.perTexOffset = 0;
    //     hashValue.bottomYCoef = 0.0f;
    //     hashValue.topYCoef = 0.0f;
    //     hashValue.validShNum = 0;
    //     hashValue.validShNumNew = 0;
        
    //     // Initialize arrays
    //     memset(hashValue.neighborTexId, 0, sizeof(hashValue.neighborTexId));
    //     for (int j = 0; j < 3; ++j) {
    //         hashValue.neighborEdgeType[j] = EDGE_TYPE_CUDA_MAX;
    //     }
    //     memset(hashValue.cornerArea, 0, sizeof(hashValue.cornerArea));
    //     memset(hashValue.texInWorldInfo, 0, sizeof(hashValue.texInWorldInfo));
    //     memset(hashValue.norm, 0, sizeof(hashValue.norm));
        
    //     // Clear vectors
    //     hashValue.topVertNbr.clear();
    //     hashValue.leftVertNbr.clear();
    //     hashValue.rightVertNbr.clear();
    //     hashValue.depth.clear();
    //     hashValue.psnr.clear();
    //     hashValue.l1Loss.clear();
    // }
    
    m_readJsonDataVec.resize(numElements);
    // Step 3: Read binary data and populate hashmap directly
    for (size_t i = 0; i < numElements; ++i) {
        // Read magic number
        uint32_t magic;
        binFile.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (binFile.gcount() != sizeof(magic) || magic != 0xDEADBEEF) {
            std::cerr << "Error: Invalid magic number for element " << i << std::endl;
            return false;
        }
        
        // Read index
        uint32_t index;
        binFile.read(reinterpret_cast<char*>(&index), sizeof(index));
        if (binFile.gcount() != sizeof(index)) {
            std::cerr << "Error: Could not read index for element " << i << std::endl;
            return false;
        }

        if (index != i) {
            std::cerr << "Error: Index mismatch for element " << i 
                      << " (expected: " << i << ", got: " << index << ")" << std::endl;
            return false;
        }
        
        // Read perTexOffset
        uint32_t perTexOffset;
        binFile.read(reinterpret_cast<char*>(&perTexOffset), sizeof(perTexOffset));
        if (binFile.gcount() != sizeof(perTexOffset)) {
            std::cerr << "Error: Could not read perTexOffset for element " << i << std::endl;
            return false;
        }
        
        // Validate index bounds
        if (index >= numElements) {
            std::cerr << "Error: Index " << index << " out of bounds (max: " << numElements - 1 << ")" << std::endl;
            return false;
        }
        
        // Validate against JSON data
        auto jsonIt = jsonDataMap.find(index);
        if (jsonIt == jsonDataMap.end()) {
            std::cerr << "Error: Index " << index << " found in binary but not in JSON" << std::endl;
            return false;
        }
        
        const rapidjson::Value* jsonEntry = jsonIt->second;
        uint32_t jsonPerTexOffset = (*jsonEntry)["perTexOffset"].GetUint();
        if (perTexOffset != jsonPerTexOffset) {
            std::cerr << "Error: perTexOffset mismatch for index " << index 
                      << " (binary: " << perTexOffset << ", JSON: " << jsonPerTexOffset << ")" << std::endl;
            return false;
        }
        
        // Set basic fields in hashmap
        // HashValue& hashValue = m_shTexHashMap[index];
        // hashValue.perTexOffset = perTexOffset;
        
        // // Set texture WH from JSON
        // hashValue.resizeInfo.curTexWH.data.curTexW = (*jsonEntry)["curTexW"].GetUint();
        // hashValue.resizeInfo.curTexWH.data.curTexH = (*jsonEntry)["curTexH"].GetUint();
        // hashValue.resizeInfo.curDensity = (*jsonEntry)["shDensity"].GetFloat();

        m_readJsonDataVec[index].index = index;
        m_readJsonDataVec[index].texW = (*jsonEntry)["curTexW"].GetUint();
        m_readJsonDataVec[index].texH = (*jsonEntry)["curTexH"].GetUint();
        m_readJsonDataVec[index].perTexOffset = perTexOffset;
        m_readJsonDataVec[index].shDensity = (*jsonEntry)["shDensity"].GetFloat();

        // Directly allocate CPU memory and read data if any
        if (perTexOffset > 0) {
            // Allocate CPU memory for shTexValue (number of floats * sizeof(float))
            // Note: perTexOffset is in bytes, so we need to ensure proper alignment for float array
            size_t numFloats = perTexOffset / sizeof(float);
            if (perTexOffset % sizeof(float) != 0) {
                std::cerr << "Warning: perTexOffset " << perTexOffset << " is not aligned to float size for index " << index << std::endl;
            }
            

            float* tmp = new float[numFloats];
            
            // Read data directly into the allocated CPU memory
            binFile.read(reinterpret_cast<char*>(tmp), perTexOffset);
            if (binFile.gcount() != static_cast<std::streamsize>(perTexOffset)) {
                std::cerr << "Error: Could not read " << perTexOffset << " bytes of data for element " << i << std::endl;
                delete[] tmp;
                tmp = nullptr;
                return false;
            }

            delete[] tmp;
            tmp = nullptr;
        }
        
        // std::cout << "Loaded index " << index << ": curTexW = " 
        //           << hashValue.resizeInfo.curTexWH.data.curTexW 
        //           << ", curTexH = " << hashValue.resizeInfo.curTexWH.data.curTexH 
        //           << ", perTexOffset = " << perTexOffset 
        //           << ", curDensity = " << hashValue.resizeInfo.curDensity 
        //           << ", shTexValue = " << (hashValue.shTexValue ? "allocated" : "null") << std::endl;
    }
    
    binFile.close();
    
    std::cout << "Successfully loaded " << numElements << " entries into m_shTexHashMap (CPU memory)" << std::endl;
    std::cout << "Data loading completed!" << std::endl;
    
    return true;
}

void ImMeshRenderer::Rendering(std::string outDir)
{
    glCheckError();
    ReassignBuffeObjs();

    glCheckError();
    ReassignBuffeObjsInfer();

    glCheckError();
    ShaderInit(true);

    glCheckError();
    DrawPrepare(m_fileConfig.sceneType, true);

    glCheckError();
    InitInferBuffer(m_inferBuffer);

    AvgPool2d avgPool2d(torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}));

    ProgressBar progressBar(m_camPosWorldRenderingMode.size());

    std::cout << "-------------start rendering , pose number : " << m_camPosWorldRenderingMode.size() << std::endl;
    for (size_t i = 0; i < m_camPosWorldRenderingMode.size(); i++) {
        DoDrawInfer(*(m_viewMatrixRenderingMode + i), m_camPosWorldRenderingMode[i]);
        SaveTrainTextures(m_inferBuffer);
        MoveTex2GPU(m_inferBuffer);
        InferPreparePipeline(m_inferBuffer);

        auto hashMapRef = m_inferBuffer.curTrainHashMap->ref(cuco::find);

        checkCudaErrors(cudaMemset(m_devErrCntPtr, 0, sizeof(unsigned int)));
        checkCudaErrors(cudaMemset(m_devErrCntBackwardPtr, 0, sizeof(unsigned int)));

        checkCudaErrors(cudaStreamSynchronize(m_shTexturesCpctCpyStrm));
        checkCudaErrors(cudaStreamSynchronize(m_inferBuffer.curBufStream));
        checkCudaErrors(cudaStreamSynchronize(m_inferBuffer.curBufStream2));

        MeshLearnerForwardTexPtr(m_inferBuffer.visibleTexIdsInfo.size(),
            m_inferBuffer.shTexturesCpct.data_ptr<float>(), (uint32_t*)thrust::raw_pointer_cast(m_inferBuffer.shTexMemLayoutDevice.data()),
            m_renderedPixelsCUDA.data_ptr<float>(),
            m_renderedPixelsMask.data_ptr<unsigned char>(),
            hashMapRef, m_devErrCntPtr,
            (TexPixInfo*)thrust::raw_pointer_cast(m_inferBuffer.visibleTexIdsInfo.data()),
            (PixLocation*)thrust::raw_pointer_cast(m_inferBuffer.visibleTexPixLocCompact.data()),
            m_width, m_height,
            // g_texObjViewDirFrag2CamAndNull,
            // g_texObjLerpCoeffAndTexCoord,
            // g_texObjOriShLUandRUTexCoord,
            // g_texObjOriShLDandRDTexCoord,
            // g_texObjWorldPoseAndNull,
            // g_texObjEllipCoeffsAndLodLvl,
            // g_texObjTexScaleFactorDepthTexId,
            m_inferBuffer.viewDirFrag2CamAndNullGPU,
            m_inferBuffer.lerpCoeffAndTexCoordGPU,
            m_inferBuffer.oriShLUandRUGPU,
            m_inferBuffer.oriShLDandRDGPU,
            m_inferBuffer.texWorldPoseAndNullGPU,
            m_inferBuffer.ellipCoeffsAndLodLvlGPU,
            m_inferBuffer.texScaleFacDepthTexIdGPU,
            m_inferBuffer.cornerAreaInfo.data_ptr<float>(),
            (TexAlignedPosWH*)(m_inferBuffer.validShWHMapCpct.data_ptr<int32_t>()),
            (uint32_t*)(thrust::raw_pointer_cast(m_inferBuffer.validShWHMapLayoutDevice.data())),
            (int32_t*)(m_inferBuffer.validShNumsAll.data_ptr<int32_t>()),
            m_inferBuffer.shPosMapCpct.data_ptr<float>(),
            (uint32_t *)thrust::raw_pointer_cast(m_inferBuffer.posMapMemLayoutDevice.data()),
            m_inferBuffer.botYCoeffs.data_ptr<float>(),
            m_inferBuffer.topYCoeffs.data_ptr<float>(),
            m_invDistFactorEdge, m_invDistFactorCorner,
            m_shLayerNum, m_shOrder,
            (uint32_t*)(m_inferBuffer.edgeNbrs.data_ptr<int32_t>()),
            (uint32_t*)(m_inferBuffer.vertNbrs.data_ptr<int32_t>()),
            m_gaussianCoeffEWA, m_inferBuffer.texInWorldInfo.data_ptr<float>(),
            (CurTexAlignedWH *)(m_inferBuffer.texWHs.data_ptr<int32_t>()),
            m_inferBuffer.meshDensities.data_ptr<float>(),
            m_inferBuffer.meshNormals.data_ptr<float>(),
            m_printPixX, m_printPixY);
        
        checkCudaErrors(cudaDeviceSynchronize());

        at::Tensor renderImgDownSampled2X;

        if ((m_width != m_gtW) && (m_height != m_gtH)) {
            renderImgDownSampled2X = avgPool2d(m_renderedPixelsCUDA);
        } else {
            renderImgDownSampled2X = m_renderedPixelsCUDA;
        }

        SaveImg(outDir, std::to_string(i) + ".png", renderImgDownSampled2X);

        m_renderedPixelsMask = torch::zeros({1, m_height, m_width}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0)).contiguous(); //.fill_(1)
        m_renderedPixelsCUDA = torch::zeros({3, m_height, m_width}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0)).contiguous();

        progressBar.update(i);
    }
}

bool ImMeshRenderer::ReadValidationSequence(SceneType scene, std::string posePath, mat4x4** viewMatrix, std::vector<std::vector<float>>& camPosWorld)
{
    std::ifstream infile;

    switch (scene) {
        case SCENE_TYPE_BLENDER_OUT:
            if (ReadGtPoseAndImageBlenderOut(posePath, viewMatrix, camPosWorld) == false) {
                return false;
            }

            break;

        case SCENE_TYPE_REPLICA:
            std::cout << "not supported yet!" << std::endl;
            exit(0);

        case SCENE_TYPE_LIVO2:
            infile.open(posePath.data());
            if (infile.is_open() == false) {
                return false;
            }

            if (ReadGtPoseFromLIVO2_mat(posePath, viewMatrix, camPosWorld) == false) {
                return false;
            }

            break;
        
        default:
            return false;
    }

    return true;
}


void ImMeshRenderer::TrainThreadLoop()
{    
    while (1) {
        if (m_trainPrepThrdInited.load() == true) {
            break;
        }

        sleep(1);
    }

    checkCudaErrors(cudaSetDevice(m_cudaDevice));

    EGLDeviceEXT eglDevs[32];
    EGLint numDevices;

    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");

    eglQueryDevicesEXT(32, eglDevs, &numDevices);

    ASSERT(numDevices, "Found no GPUs");

    PFNEGLQUERYDEVICEATTRIBEXTPROC eglQueryDeviceAttribEXT =
        reinterpret_cast<PFNEGLQUERYDEVICEATTRIBEXTPROC>(
            eglGetProcAddress("eglQueryDeviceAttribEXT"));

    int cudaDevice = 0;
    int eglDevId = 0;
    bool foundCudaDev = false;
    // Find the CUDA device asked for
    for (; eglDevId < numDevices; ++eglDevId) {
      EGLAttrib cudaDevNumber;

      if (eglQueryDeviceAttribEXT(eglDevs[eglDevId], EGL_CUDA_DEVICE_NV, &cudaDevNumber) ==
          EGL_FALSE)
        continue;

      if (cudaDevNumber == cudaDevice) {
        break;
      }
    }

    printf("use dev : %d, found %d\n", eglDevId, foundCudaDev);


    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    m_eglTrainThrdDsp = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevs[eglDevId], 0);

    // m_eglTrainThrdDsp = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major, minor;

    eglInitialize(m_eglTrainThrdDsp, &major, &minor);
    EGLint numConfigs;
    EGLConfig eglCfg;

    eglChooseConfig(m_eglTrainThrdDsp, configAttribs, &eglCfg, 1, &numConfigs);
    eglBindAPI(EGL_OPENGL_API);

    m_trainThreadCtx = eglCreateContext(m_eglTrainThrdDsp, eglCfg, m_mainThrdCtx, NULL);
    eglMakeCurrent(m_eglTrainThrdDsp, EGL_NO_SURFACE, EGL_NO_SURFACE, m_trainThreadCtx);

    ReassignBuffeObjsInfer();

    glCheckError();
    ShaderInit(true);
    glCheckError();
    DrawPrepare(m_fileConfig.sceneType, true);

    glCheckError();
    InitInferBuffer(m_inferBuffer);

    // std::string currentTime = GetCurrentTimeAsString();
    vector<at::Tensor> shTensorVec;
    shTensorVec.push_back(m_shTensorAllMergeCUDA);

    auto adamOpt = torch::optim::AdamOptions(0.0025);
    // torch::optim::Adam optimizer(shTensorVec, adamOpt); // follow 3dGS
    std::shared_ptr<torch::optim::Adam> optimizerPtr = make_shared<torch::optim::Adam> (shTensorVec, adamOpt);

    vector<float> schedulerLrMin;
    schedulerLrMin.push_back(1e-7);

    std::unique_ptr<torch::optim::ReduceLROnPlateauScheduler> schedulerPtr = make_unique<torch::optim::ReduceLROnPlateauScheduler> ((*optimizerPtr), 
        torch::optim::ReduceLROnPlateauScheduler::SchedulerMode::min, 0.9, m_schedulerPatience, 1e-6,
        torch::optim::ReduceLROnPlateauScheduler::ThresholdMode::rel, 0, schedulerLrMin, 1e-8, false);

    // torch::optim::ReduceLROnPlateauScheduler scheduler(optimizer, 
    //     torch::optim::ReduceLROnPlateauScheduler::SchedulerMode::min, 0.9, m_schedulerPatience, 1e-6,
    //     torch::optim::ReduceLROnPlateauScheduler::ThresholdMode::rel, 0, schedulerLrMin, 1e-8, false);

    torch::nn::SmoothL1Loss smoothL1(torch::nn::SmoothL1LossOptions().reduction(torch::kNone).beta(10.0));
    AvgPool2d avgPool2d(torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}));

    unsigned int errCntForward, errCntBackward;

    unsigned long startTime;
    unsigned long endTime;

    unsigned long curTrainStartTime;
    unsigned long lastTrainEndTime = GetTimeMS();

    // bool saveFlag = false;

    checkCudaErrors(cudaStreamCreateWithFlags(&m_shTexturesCpctCpyStrm, cudaStreamNonBlocking));
    checkCudaErrors(cudaStreamCreateWithFlags(&m_adamExpAvgCpyStrm, cudaStreamNonBlocking));
    checkCudaErrors(cudaStreamCreateWithFlags(&m_adamExpAvgSqCpyStrm, cudaStreamNonBlocking));

    ProgressBar progressBar(m_trainPoseCount);

    while (m_threadRunning.load() == true) {
        {
            std::unique_lock<std::mutex> lock(m_thrdBufs[m_trainThrdCurIdx].bufMutex);
            if (m_thrdBufs[m_trainThrdCurIdx].state != READY_FOR_TRAIN) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            } else {
                if (m_thrdBufs[m_trainThrdCurIdx].needResetOptimizor == true) {
                    schedulerPtr = nullptr;
                    optimizerPtr = make_shared<torch::optim::Adam> (shTensorVec, adamOpt);
                    schedulerPtr = make_unique<torch::optim::ReduceLROnPlateauScheduler> ((*optimizerPtr), 
                        torch::optim::ReduceLROnPlateauScheduler::SchedulerMode::min, 0.9, m_schedulerPatience, 1e-6,
                        torch::optim::ReduceLROnPlateauScheduler::ThresholdMode::rel, 0, schedulerLrMin, 1e-8, false);

                    // saveFlag = true;
                    ReassignBuffeObjsInfer();
                }

                PROFILE_PRINT(cout << "TrainThreadLoop start, idx: " << m_trainThrdCurIdx << endl);
                m_trainThrdCurTrainCount = m_thrdBufs[m_trainThrdCurIdx].curTrainCount;

                curTrainStartTime = GetTimeMS();
                startTime = curTrainStartTime;

                // if (m_isSubTrainMode == true) {
                //     std::cout << "start sub train pose : " << m_thrdBufs[m_trainThrdCurIdx].poseIdx << " time delta from now to prev train:  " << (double)(curTrainStartTime - lastTrainEndTime) << std::endl;
                // } else {
                    PROFILE_PRINT(std::cout << " start pose : " << m_thrdBufs[m_trainThrdCurIdx].poseIdx << " time delta from now to prev train:  " << (double)(curTrainStartTime - lastTrainEndTime) << std::endl);
                // }
                
                ResetImg();

                InitTrainThrdCpyBuffer(m_trainThrdCurIdx);
                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainThrdName << " till InitTrainThrdCpyBuffer time: : " << (double)(endTime - startTime) << " ms" << endl);

                CopyCurTrainEssentials(m_trainThrdCurIdx);
                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainThrdName << " till CopyCurTrainEssentials time: : " << (double)(endTime - startTime) << " ms" << endl);

                PrepOptimizer(m_trainThrdCurIdx, optimizerPtr);
                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainThrdName << " till PrepOptimizer time: : " << (double)(endTime - startTime) << " ms" << endl);

                // RegisterCudaTextures(m_trainThrdCurIdx);
                // endTime = GetTimeMS();
                // cout << "till RegisterCudaTextures time: : " << (double)(endTime - startTime) << " ms" << endl;

                // MapIntermediateTextures();
                // endTime = GetTimeMS();
                // cout << "till MapIntermediateTextures time: : " << (double)(endTime - startTime) << " ms" << endl;

                m_thrdBufs[m_trainThrdCurIdx].devHashFindRefFind = m_thrdBufs[m_trainThrdCurIdx].curTrainHashMap->ref(cuco::find);
                // checkCudaErrors(cudaMemset(m_devErrCntPtr, 0, sizeof(unsigned int)));
                // checkCudaErrors(cudaMemset(m_devErrCntBackwardPtr, 0, sizeof(unsigned int)));

                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainThrdName << " before train prep time: : " << (double)(endTime - startTime) << " ms" << endl);

                for (unsigned int j = 0; j < m_algoConfig.perPoseTrainCnt; j++) {
                    optimizerPtr->zero_grad();

                    at::Tensor outColor = MyTrainer::apply(m_thrdBufs[m_trainThrdCurIdx].shTexturesCpct, m_devErrCntPtr, m_shLayerNum, m_shOrder,  m_width, m_height);

                    at::Tensor outColorDownSampled2X;
                    if ((m_width != m_gtW) && (m_height != m_gtH)) {
                        outColorDownSampled2X = avgPool2d(outColor);
                    } else {
                        outColorDownSampled2X = outColor;
                    }

                    at::Tensor loss1st;
                    // if (m_isSubTrainMode == true) {
                    //     // if (saveFlag == true) {

                    //     //     Infer(m_algoConfig.saveDir + "/" + m_taskStartTime + "_" + m_algoConfig.experimentName + "/", "after_resize_gt_saved", m_shLayerNum, 0);

                    //     // }

                    //     loss1st = smoothL1(outColorDownSampled2X,
                    //         m_gtImagesSub[m_thrdBufs[m_trainThrdCurIdx].poseIdx].to(torch::kCUDA, 0));
                    // } else {
                        loss1st = smoothL1(outColorDownSampled2X,
                            m_gtImages[m_thrdBufs[m_trainThrdCurIdx].poseIdx].to(torch::kCUDA, 0));
                    // }
                    

                    // loss1st = m_poseConfidenceVec[m_thrdBufs[m_trainThrdCurIdx].poseIdx] * loss1st;
                    
                    // loss1st = loss1st * m_thrdBufs[m_trainThrdCurIdx].depthPunishmentMap;
                    loss1st = loss1st.sum();
                    loss1st.backward();
                    optimizerPtr->step();
                    // scheduler.step(loss1st.item().toFloat());
                    schedulerPtr->step(loss1st.mean().item().toFloat());

                    at::Tensor newMask = torch::zeros({1, m_gtH, m_gtW}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0)).contiguous();
                    if ((m_width != m_gtW) && (m_height != m_gtH)) {
                        const dim3 blockSize(32, 32, 1);
                        const dim3 gridSize(((m_gtW - 1 + 32) / 32), ((m_gtH - 1 + 32) / 32), 1);
                        
                        custom_maxpool_2x2 <<<gridSize, blockSize>>> (m_width, m_height, m_gtW, m_gtH,
                            m_renderedPixelsMask.data_ptr<unsigned char>(),
                            newMask.data_ptr<unsigned char>());
                        
                        checkCudaErrors(cudaStreamSynchronize(0));
                    } else if ((m_width == m_gtW) && (m_height == m_gtH)) {
                        newMask = m_renderedPixelsMask;
                    } else {
                        printf("m_width and m_gtW are invalid\n");
                        exit(0);
                    }


                    endTime = GetTimeMS();
                    PROFILE_PRINT(cout << g_trainThrdName << " till max_pool2d finished : : " << (double)(endTime - startTime) << " ms" << endl);

                    // m_renderedPixelsMask = m_renderedPixelsMask.contiguous().view({-1, 1}).squeeze();
                    // at::Tensor validIdx = torch::nonzero(m_renderedPixelsMask);

                    newMask = newMask.contiguous().view({-1, 1}).squeeze();
                    at::Tensor validIdx = torch::nonzero(newMask);

                    outColorDownSampled2X = outColorDownSampled2X.clone().detach().permute({1, 2, 0}).contiguous().view({-1, 3});

                    at::Tensor gtImageTmp;
                    // if (m_isSubTrainMode == true) {
                    //     gtImageTmp = m_gtImagesSub[m_thrdBufs[m_trainThrdCurIdx].poseIdx].clone().detach().permute({1, 2, 0}).contiguous().view({-1, 3}).to(torch::kCUDA, 0);
                    // } else {
                        gtImageTmp = m_gtImages[m_thrdBufs[m_trainThrdCurIdx].poseIdx].clone().detach().permute({1, 2, 0}).contiguous().view({-1, 3}).to(torch::kCUDA, 0);
                    // }
                    

                    
                    outColorDownSampled2X = outColorDownSampled2X.index_select(0, validIdx.squeeze());
                    gtImageTmp = gtImageTmp.index_select(0, validIdx.squeeze());
                    double psnr = CalculatePsnr(outColorDownSampled2X, gtImageTmp);

                    
                    m_recorderPtr->AddScalar("psnr", psnr, m_trainThrdCurTrainCount * m_algoConfig.perPoseTrainCnt + j);
                    m_recorderPtr->AddScalar("loss", loss1st.mean().item().toFloat(), m_trainThrdCurTrainCount * m_algoConfig.perPoseTrainCnt + j);
                    m_recorderPtr->AddScalar("lr", optimizerPtr->param_groups()[0].options().get_lr(), m_trainThrdCurTrainCount * m_algoConfig.perPoseTrainCnt + j);

                    // if (j == 0) {
                    //     string str;
                    //     if (m_isSubTrainMode == true) {
                    //         str = "pose_sub_" + std::to_string(m_thrdBufs[m_trainThrdCurIdx].poseIdx) + "_start";
                    //     } else {
                    //         str = "pose_" + std::to_string(m_thrdBufs[m_trainThrdCurIdx].poseIdx) + "_start";
                    //     }
                        
                    // }

                    // if (j == (m_algoConfig.perPoseTrainCnt - 1)) {
                    //     string str;
                    //     if (m_isSubTrainMode == true) {
                    //         str = "pose_sub_" + std::to_string(m_thrdBufs[m_trainThrdCurIdx].poseIdx) + "_end";
                    //     } else {
                    //         str = "pose_" + std::to_string(m_thrdBufs[m_trainThrdCurIdx].poseIdx) + "_end";
                    //     }
                    // }

                    m_renderedPixelsMask = torch::zeros({1, m_height, m_width}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0)).contiguous(); // .fill_(1)

                    m_renderedPixelsCUDA = torch::zeros({3, m_height, m_width}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0)).contiguous();
                }

                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainThrdName << " ------till time to train one pose, per : " << m_algoConfig.perPoseTrainCnt << " time: " << (double)(endTime - startTime) << " ms" << endl);

                // checkCudaErrors(cudaMemcpy(&errCntForward, m_devErrCntPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost));
                // checkCudaErrors(cudaMemcpy(&errCntBackward, m_devErrCntBackwardPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost));

                string str2;
                // if (m_isSubTrainMode == true) {
                //     cout << " hash err Count in sub train pose << " << m_thrdBufs[m_trainThrdCurIdx].poseIdx <<  " forward : " << errCntForward << " backward err : " << errCntBackward << endl;
                //     str2 = "pose_sub_" + std::to_string(m_thrdBufs[m_trainThrdCurIdx].poseIdx) + "_end";
                // } else {
                    PROFILE_PRINT(cout << g_trainThrdName << " hash err Count in pose << " << m_thrdBufs[m_trainThrdCurIdx].poseIdx <<  " forward : " << errCntForward << " backward err : " << errCntBackward << endl);
                    str2 = "pose_" + std::to_string(m_thrdBufs[m_trainThrdCurIdx].poseIdx) + "_end";
                // }

                // UnmapIntermediateTextures();
                // UnRegisterCudaTextures();
                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainThrdName << " till UnmapIntermediateTextures time: : " << (double)(endTime - startTime) << " ms" << endl);

                TrainedWriteBackPipeline(m_trainThrdCurIdx);
                endTime = GetTimeMS();
                PROFILE_PRINT(cout << g_trainThrdName << " till TrainedWriteBackPipeline time: : " << (double)(endTime - startTime) << " ms" << endl);

                // m_thrdBufs[m_trainThrdCurIdx].dummyCount++;
                

                PROFILE_PRINT(cout  << "TrainThreadLoop has processed buffer : " << m_trainThrdCurIdx << ", dummy : " << m_thrdBufs[m_trainThrdCurIdx].dummyCount << endl);

                if (m_trainThrdCurTrainCount % m_testInterval == 0 && m_trainThrdCurTrainCount != 0) {
                    if (m_needInfer == true) {
                        Infer(m_algoConfig.saveDir + "/" + m_taskStartTime + "_" + m_algoConfig.experimentName + "/", str2, m_shLayerNum, (m_trainThrdCurTrainCount + 1) * m_algoConfig.perPoseTrainCnt);
                    }
                }

                PROFILE_PRINT(cout << "TrainThreadLoop end, idx: " << m_trainThrdCurIdx << endl);
                m_thrdBufs[m_trainThrdCurIdx].curValidIdx = ((m_thrdBufs[m_trainThrdCurIdx].curValidIdx + 1) % 2);
                
                m_thrdBufs[m_trainThrdCurIdx].state = READY_FOR_RENDER;

                progressBar.update(m_trainThrdCurTrainCount);
                lock.unlock();
                m_trainThrdCurIdx = (m_trainThrdCurIdx + 1) % NUM_BUFFERS;
                lastTrainEndTime = GetTimeMS();
            }
        }
    }

    // glfwDestroyWindow(m_trainThreadCtx);
}

void ImMeshRenderer::Infer(std::string dir, std::string prefix, unsigned int curTrainLayerNum, unsigned int cnt)
{
    double sum = 0.0;
    glCheckError();
    // RegisterCudaTexturesInfer();
    glCheckError();

    AvgPool2d avgPool2d(torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}));

    ProgressBar progressBar(m_camPosWorldInfer.size());

    cout << "-------------start infer , pose number : " << m_camPosWorldInfer.size() << std::endl;

    for (size_t i = 0; i < m_camPosWorldInfer.size(); i++) {
        DoDrawInfer(*(m_viewMatrixInfer + i), m_camPosWorldInfer[i]);
        SaveTrainTextures(m_inferBuffer);
        MoveTex2GPU(m_inferBuffer);
        InferPreparePipeline(m_inferBuffer);
        // MapIntermediateTextures();

        auto hashMapRef = m_inferBuffer.curTrainHashMap->ref(cuco::find);

        checkCudaErrors(cudaMemset(m_devErrCntPtr, 0, sizeof(unsigned int)));
        checkCudaErrors(cudaMemset(m_devErrCntBackwardPtr, 0, sizeof(unsigned int)));

        checkCudaErrors(cudaStreamSynchronize(m_shTexturesCpctCpyStrm));
        checkCudaErrors(cudaStreamSynchronize(m_inferBuffer.curBufStream));
        checkCudaErrors(cudaStreamSynchronize(m_inferBuffer.curBufStream2));

        MeshLearnerForwardTexPtr(m_inferBuffer.visibleTexIdsInfo.size(),
            m_inferBuffer.shTexturesCpct.data_ptr<float>(), (uint32_t*)thrust::raw_pointer_cast(m_inferBuffer.shTexMemLayoutDevice.data()),
            m_renderedPixelsCUDA.data_ptr<float>(),
            m_renderedPixelsMask.data_ptr<unsigned char>(),
            hashMapRef, m_devErrCntPtr,
            (TexPixInfo*)thrust::raw_pointer_cast(m_inferBuffer.visibleTexIdsInfo.data()),
            (PixLocation*)thrust::raw_pointer_cast(m_inferBuffer.visibleTexPixLocCompact.data()),
            m_width, m_height,
            // g_texObjViewDirFrag2CamAndNull,
            // g_texObjLerpCoeffAndTexCoord,
            // g_texObjOriShLUandRUTexCoord,
            // g_texObjOriShLDandRDTexCoord,
            // g_texObjWorldPoseAndNull,
            // g_texObjEllipCoeffsAndLodLvl,
            // g_texObjTexScaleFactorDepthTexId,
            m_inferBuffer.viewDirFrag2CamAndNullGPU,
            m_inferBuffer.lerpCoeffAndTexCoordGPU,
            m_inferBuffer.oriShLUandRUGPU,
            m_inferBuffer.oriShLDandRDGPU,
            m_inferBuffer.texWorldPoseAndNullGPU,
            m_inferBuffer.ellipCoeffsAndLodLvlGPU,
            m_inferBuffer.texScaleFacDepthTexIdGPU,
            m_inferBuffer.cornerAreaInfo.data_ptr<float>(),
            (TexAlignedPosWH*)(m_inferBuffer.validShWHMapCpct.data_ptr<int32_t>()),
            (uint32_t*)(thrust::raw_pointer_cast(m_inferBuffer.validShWHMapLayoutDevice.data())),
            (int32_t*)(m_inferBuffer.validShNumsAll.data_ptr<int32_t>()),
            m_inferBuffer.shPosMapCpct.data_ptr<float>(),
            (uint32_t *)thrust::raw_pointer_cast(m_inferBuffer.posMapMemLayoutDevice.data()),
            m_inferBuffer.botYCoeffs.data_ptr<float>(),
            m_inferBuffer.topYCoeffs.data_ptr<float>(),
            m_invDistFactorEdge, m_invDistFactorCorner,
            curTrainLayerNum, m_shOrder,
            (uint32_t*)(m_inferBuffer.edgeNbrs.data_ptr<int32_t>()),
            (uint32_t*)(m_inferBuffer.vertNbrs.data_ptr<int32_t>()),
            m_gaussianCoeffEWA, m_inferBuffer.texInWorldInfo.data_ptr<float>(),
            (CurTexAlignedWH *)(m_inferBuffer.texWHs.data_ptr<int32_t>()),
            m_inferBuffer.meshDensities.data_ptr<float>(),
            m_inferBuffer.meshNormals.data_ptr<float>(),
            m_printPixX, m_printPixY);

        checkCudaErrors(cudaDeviceSynchronize());

        at::Tensor renderImgDownSampled2X;

        if ((m_width != m_gtW) && (m_height != m_gtH)) {
            renderImgDownSampled2X = avgPool2d(m_renderedPixelsCUDA);
        } else {
            renderImgDownSampled2X = m_renderedPixelsCUDA;
        }

        at::Tensor newMask = torch::zeros({1, m_gtH, m_gtW}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0)).contiguous();
        if ((m_width != m_gtW) && (m_height != m_gtH)) {
            // m_renderedPixelsMask = m_renderedPixelsMask.to(torch::kFloat32);
            // m_renderedPixelsMask = at::max_pool2d(m_renderedPixelsMask, {2, 2}, {2, 2});
            // m_renderedPixelsMask = m_renderedPixelsMask.to(torch::kUInt8);

            const dim3 blockSize(32, 32, 1);
            const dim3 gridSize(((m_gtW - 1 + 32) / 32), ((m_gtH - 1 + 32) / 32), 1);
            custom_maxpool_2x2 <<<gridSize, blockSize>>> (m_width, m_height, m_gtW, m_gtH,
                            m_renderedPixelsMask.data_ptr<unsigned char>(),
                            newMask.data_ptr<unsigned char>());
                        
            checkCudaErrors(cudaStreamSynchronize(0));
        } else if ((m_width == m_gtW) && (m_height == m_gtH)) {
            newMask = m_renderedPixelsMask;
        } else {
            printf("infer m_width and m_gtW are invalid\n");
            exit(0);
        }
        
        newMask = newMask.contiguous().view({-1, 1}).squeeze();
        at::Tensor validIdx = torch::nonzero(newMask);


        // m_renderedPixelsMask = m_renderedPixelsMask.contiguous().view({-1, 1}).squeeze();
        // at::Tensor validIdx = torch::nonzero(m_renderedPixelsMask);

        renderImgDownSampled2X = renderImgDownSampled2X.clone().detach().permute({1, 2, 0}).contiguous().view({-1, 3});
        at::Tensor gtImageTmp = m_gtImagesInfer[i].permute({1, 2, 0}).contiguous().view({-1, 3}).to(torch::kCUDA, 0);

        renderImgDownSampled2X = renderImgDownSampled2X.index_select(0, validIdx.squeeze());
        gtImageTmp = gtImageTmp.index_select(0, validIdx.squeeze());
        double psnr = CalculatePsnr(renderImgDownSampled2X, gtImageTmp);


        sum += psnr;
        static unsigned int testCnt = 0;
        m_recorderPtr->AddScalar("test_psnr", psnr, testCnt);
        testCnt++;

        string str = "_testPose_" + std::to_string(i);

        if (i == (m_camPosWorldInfer.size() - 1)) {
            ImgSaveCUDA(dir, prefix + str);
        } else {
            ImgSaveCUDA(dir, prefix + str, false);
        }

        // UnmapIntermediateTextures();
        
        m_renderedPixelsMask = torch::zeros({1, m_height, m_width}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0)).contiguous(); //.fill_(1)
        m_renderedPixelsCUDA = torch::zeros({3, m_height, m_width}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0)).contiguous();

        progressBar.update(i);
    }

    m_recorderPtr->AddScalar("test_avg_psnr", (sum / m_camPosWorldInfer.size()), cnt);

    at::Tensor dummy;
    m_inferBuffer.shTexturesCpct = dummy;
    m_inferBuffer.edgeNbrs = dummy;
    m_inferBuffer.cornerAreaInfo = dummy;
    m_inferBuffer.vertNbrs = dummy;
    m_inferBuffer.texInWorldInfo = dummy;
    m_inferBuffer.texWHs = dummy;
    m_inferBuffer.validShWHMapCpct = dummy;
    m_inferBuffer.shPosMapCpct = dummy;
    m_inferBuffer.botYCoeffs = dummy;
    m_inferBuffer.topYCoeffs = dummy;
    m_inferBuffer.validShNumsAll = dummy;

    // UnRegisterCudaTextures();
}

void ImMeshRenderer::DoDrawInfer(mat4x4 viewMatrix, std::vector<float> camPos)
{
    glCheckError();
    glUseProgram(m_programInfer);
    glCheckError();

    glViewport(0, 0, m_width, m_height);

    glBindFramebuffer(GL_FRAMEBUFFER, m_inferBuffer.fboId);
    glCheckError();

    // glDisable(GL_CULL_FACE);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glCheckError();

    float clearColor[4] = { 0.f, 0.f, 0.f, 0.f };
    glClearBufferfv(GL_COLOR, 0, clearColor);
    glClearBufferfv(GL_COLOR, 1, clearColor);
    glClearBufferfv(GL_COLOR, 2, clearColor);
    glClearBufferfv(GL_COLOR, 3, clearColor);
    glClearBufferfv(GL_COLOR, 4, clearColor);
    glCheckError();

    glClearColor(0.f, 0.f, 0.f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glCheckError();

    glEnable(GL_DEPTH_TEST);
    glCheckError();

    glUniform3f(m_camWorldPosLocationInfer, camPos[0], camPos[1], camPos[2]);
    glCheckError();

    glUniformMatrix4fv(m_viewLocationInfer, 1, GL_FALSE, (const GLfloat*) viewMatrix);
    glCheckError();

    glUniformMatrix4fv(m_projLocationInfer, 1, GL_FALSE, (const GLfloat*) m_perspectiveMat);
    glCheckError();

    
    // glDrawArrays(GL_TRIANGLES, 0, 3); GL_LINES
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    // glBindVertexArray(m_trainThreadVAO);

    for (auto it: m_OpenglDummyTexId) {
        glBindTexture(GL_TEXTURE_2D, it);

        auto find = m_dummyTex2BufferObjsInfer.find(it);
        assert(find != m_dummyTex2BufferObjsInfer.end());

        glBindVertexArray(find->second.vao);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 3, find->second.meshNumOfThisKind);
    }

    glBindVertexArray(0);

    glFinish();
}

void ImMeshRenderer::InitTrainPrepThrdCpyBufferDynamic(uint32_t posMapBufSize, uint32_t validShWHMapSize, uint32_t vertNbrsSize)
{
    unsigned long startTime = GetTimeMS();

    if (m_shPosMapCpctCpyBuf == nullptr) {
        checkCudaErrors(cudaMallocHost(&m_shPosMapCpctCpyBuf, posMapBufSize * sizeof(float)));
        m_shPosMapCpctCpyBufSize = posMapBufSize;
    } else {
        if (posMapBufSize > m_shPosMapCpctCpyBufSize) {
            checkCudaErrors(cudaFreeHost(m_shPosMapCpctCpyBuf));
            m_shPosMapCpctCpyBuf = nullptr;

            checkCudaErrors(cudaMallocHost(&m_shPosMapCpctCpyBuf, posMapBufSize * sizeof(float)));
            m_shPosMapCpctCpyBufSize = posMapBufSize;
        }
    }

    if (m_validShWHMapCpctCpyBuf == nullptr) {
        checkCudaErrors(cudaMallocHost(&m_validShWHMapCpctCpyBuf, validShWHMapSize * sizeof(int32_t)));
        m_validShWHMapCpctCpyBufSize = validShWHMapSize;
    } else {
        if (validShWHMapSize > m_validShWHMapCpctCpyBufSize) {
            checkCudaErrors(cudaFreeHost(m_validShWHMapCpctCpyBuf));
            m_validShWHMapCpctCpyBuf = nullptr;

            checkCudaErrors(cudaMallocHost(&m_validShWHMapCpctCpyBuf, validShWHMapSize * sizeof(int32_t)));
            m_validShWHMapCpctCpyBufSize = validShWHMapSize;
        }
    }

    if (m_vertNbrsCpyBuf == nullptr) {
        checkCudaErrors(cudaMallocHost(&m_vertNbrsCpyBuf, vertNbrsSize * sizeof(int32_t)));
        m_vertNbrsCpyBufSize = vertNbrsSize;
    } else {
        if (vertNbrsSize > m_vertNbrsCpyBufSize) {
            checkCudaErrors(cudaFreeHost(m_vertNbrsCpyBuf));
            m_vertNbrsCpyBuf = nullptr;

            checkCudaErrors(cudaMallocHost(&m_vertNbrsCpyBuf, vertNbrsSize * sizeof(int32_t)));
            m_vertNbrsCpyBufSize = vertNbrsSize;
        }
    }
}

void ImMeshRenderer::InitTrainPrepThrdCpyBufferDynamicInfer(uint32_t posMapBufSize, uint32_t validShWHMapSize, uint32_t vertNbrsSize)
{
    unsigned long startTime = GetTimeMS();

    if (m_shPosMapCpctCpyBufInfer == nullptr) {
        checkCudaErrors(cudaMallocHost(&m_shPosMapCpctCpyBufInfer, posMapBufSize * sizeof(float)));
        m_shPosMapCpctCpyBufSizeInfer = posMapBufSize;
    } else {
        if (posMapBufSize > m_shPosMapCpctCpyBufSizeInfer) {
            checkCudaErrors(cudaFreeHost(m_shPosMapCpctCpyBufInfer));
            m_shPosMapCpctCpyBufInfer = nullptr;

            checkCudaErrors(cudaMallocHost(&m_shPosMapCpctCpyBufInfer, posMapBufSize * sizeof(float)));
            m_shPosMapCpctCpyBufSizeInfer = posMapBufSize;
        }
    }

    if (m_validShWHMapCpctCpyBufInfer == nullptr) {
        checkCudaErrors(cudaMallocHost(&m_validShWHMapCpctCpyBufInfer, validShWHMapSize * sizeof(int32_t)));
        m_validShWHMapCpctCpyBufSizeInfer = validShWHMapSize;
    } else {
        if (validShWHMapSize > m_validShWHMapCpctCpyBufSizeInfer) {
            checkCudaErrors(cudaFreeHost(m_validShWHMapCpctCpyBufInfer));
            m_validShWHMapCpctCpyBufInfer = nullptr;

            checkCudaErrors(cudaMallocHost(&m_validShWHMapCpctCpyBufInfer, validShWHMapSize * sizeof(int32_t)));
            m_validShWHMapCpctCpyBufSizeInfer = validShWHMapSize;
        }
    }

    if (m_vertNbrsCpyBufInfer == nullptr) {
        checkCudaErrors(cudaMallocHost(&m_vertNbrsCpyBufInfer, vertNbrsSize * sizeof(int32_t)));
        m_vertNbrsCpyBufSizeInfer = vertNbrsSize;
    } else {
        if (vertNbrsSize > m_vertNbrsCpyBufSizeInfer) {
            checkCudaErrors(cudaFreeHost(m_vertNbrsCpyBufInfer));
            m_vertNbrsCpyBufInfer = nullptr;

            checkCudaErrors(cudaMallocHost(&m_vertNbrsCpyBufInfer, vertNbrsSize * sizeof(int32_t)));
            m_vertNbrsCpyBufSizeInfer = vertNbrsSize;
        }
    }
}

void ImMeshRenderer::InitThrdFixedCpyBuffer(ThreadBuffer& buf, uint32_t elementNum)
{
    unsigned long startTime = GetTimeMS();

    if ((buf.m_validShNumsAllCpyBuf == nullptr) &&
        (buf.m_edgeNbrsCpyBuf == nullptr) &&
	    (buf.m_texWHsCpyBuf == nullptr) &&
        (buf.m_cornerAreaInfoCpyBuf == nullptr) &&
        (buf.m_texInWorldInfoCpyBuf == nullptr) &&
        (buf.m_botYCoeffsCpyBuf == nullptr) &&
        (buf.m_topYCoeffsCpyBuf == nullptr) &&
        (buf.m_meshDensitiesCpyBuf == nullptr) &&
        (buf.m_meshNormalsCpyBuf == nullptr)) {
        checkCudaErrors(cudaMallocHost(&(buf.m_validShNumsAllCpyBuf), elementNum * sizeof(int32_t)));
        checkCudaErrors(cudaMallocHost(&(buf.m_edgeNbrsCpyBuf), elementNum * 6 * sizeof(int32_t)));
        checkCudaErrors(cudaMallocHost(&(buf.m_texWHsCpyBuf), elementNum * sizeof(int32_t)));

        checkCudaErrors(cudaMallocHost(&(buf.m_cornerAreaInfoCpyBuf), elementNum * sizeof(float) * 2));
        checkCudaErrors(cudaMallocHost(&(buf.m_texInWorldInfoCpyBuf), elementNum * sizeof(float) * 6));
        checkCudaErrors(cudaMallocHost(&(buf.m_botYCoeffsCpyBuf), elementNum * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&(buf.m_topYCoeffsCpyBuf), elementNum * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&(buf.m_meshDensitiesCpyBuf), elementNum * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&(buf.m_meshNormalsCpyBuf), elementNum * sizeof(float) * 3));

        buf.m_cpyBufCurElemNum = elementNum;
        return;
    }

    if (elementNum > buf.m_cpyBufCurElemNum) {
        checkCudaErrors(cudaFreeHost(buf.m_validShNumsAllCpyBuf));
        checkCudaErrors(cudaFreeHost(buf.m_edgeNbrsCpyBuf));
        checkCudaErrors(cudaFreeHost(buf.m_texWHsCpyBuf));

        checkCudaErrors(cudaFreeHost(buf.m_cornerAreaInfoCpyBuf));
        checkCudaErrors(cudaFreeHost(buf.m_texInWorldInfoCpyBuf));
        checkCudaErrors(cudaFreeHost(buf.m_botYCoeffsCpyBuf));
        checkCudaErrors(cudaFreeHost(buf.m_topYCoeffsCpyBuf));
        checkCudaErrors(cudaFreeHost(buf.m_meshDensitiesCpyBuf));
        checkCudaErrors(cudaFreeHost(buf.m_meshNormalsCpyBuf));

        buf.m_validShNumsAllCpyBuf = nullptr;
        buf.m_edgeNbrsCpyBuf = nullptr;
        buf.m_texWHsCpyBuf = nullptr;
        buf.m_cornerAreaInfoCpyBuf = nullptr;
        buf.m_texInWorldInfoCpyBuf = nullptr;
        buf.m_botYCoeffsCpyBuf = nullptr;
        buf.m_topYCoeffsCpyBuf = nullptr;
        buf.m_meshDensitiesCpyBuf = nullptr;
        buf.m_meshNormalsCpyBuf = nullptr;

        checkCudaErrors(cudaMallocHost(&(buf.m_validShNumsAllCpyBuf), elementNum * sizeof(int32_t)));
        checkCudaErrors(cudaMallocHost(&(buf.m_edgeNbrsCpyBuf), elementNum * 6 * sizeof(int32_t)));
        checkCudaErrors(cudaMallocHost(&(buf.m_texWHsCpyBuf), elementNum * sizeof(int32_t)));

        checkCudaErrors(cudaMallocHost(&(buf.m_cornerAreaInfoCpyBuf), elementNum * sizeof(float) * 2));
        checkCudaErrors(cudaMallocHost(&(buf.m_texInWorldInfoCpyBuf), elementNum * sizeof(float) * 6));
        checkCudaErrors(cudaMallocHost(&(buf.m_botYCoeffsCpyBuf), elementNum * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&(buf.m_topYCoeffsCpyBuf), elementNum * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&(buf.m_meshDensitiesCpyBuf), elementNum * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&(buf.m_meshNormalsCpyBuf), elementNum * sizeof(float) * 3));

        buf.m_cpyBufCurElemNum = elementNum;
    }
}

void ImMeshRenderer::InitTrainThrdCpyBuffer(uint32_t thrdBufIdx)
{
    unsigned long startTime = GetTimeMS();

    if (m_shTexturesCpctCpyBuf == nullptr) {
        checkCudaErrors(cudaMallocHost(&m_shTexturesCpctCpyBuf, m_thrdBufs[thrdBufIdx].shTexturesCpct.numel() * sizeof(float)));
        m_shTexturesCpctCpyBufBytes = (m_thrdBufs[thrdBufIdx].shTexturesCpct.numel());
    } else {
        if ((m_thrdBufs[thrdBufIdx].shTexturesCpct.numel()) > m_shTexturesCpctCpyBufBytes) {
            checkCudaErrors(cudaFreeHost(m_shTexturesCpctCpyBuf));
            m_shTexturesCpctCpyBuf = nullptr;

            checkCudaErrors(cudaMallocHost(&m_shTexturesCpctCpyBuf, m_thrdBufs[thrdBufIdx].shTexturesCpct.numel() * sizeof(float)));
            m_shTexturesCpctCpyBufBytes = (m_thrdBufs[thrdBufIdx].shTexturesCpct.numel());
        }
    }

    if (m_adamExpAvgCpyBuf == nullptr) {
        checkCudaErrors(cudaMallocHost(&m_adamExpAvgCpyBuf, m_thrdBufs[thrdBufIdx].adamExpAvg.numel() * sizeof(float)));
        m_adamExpAvgCpyBufBytes = m_thrdBufs[thrdBufIdx].adamExpAvg.numel();
    } else {
        if ((m_thrdBufs[thrdBufIdx].adamExpAvg.numel()) > m_adamExpAvgCpyBufBytes) {
            checkCudaErrors(cudaFreeHost(m_adamExpAvgCpyBuf));
            m_adamExpAvgCpyBuf = nullptr;

            checkCudaErrors(cudaMallocHost(&m_adamExpAvgCpyBuf, m_thrdBufs[thrdBufIdx].adamExpAvg.numel() * sizeof(float)));
            m_adamExpAvgCpyBufBytes = m_thrdBufs[thrdBufIdx].adamExpAvg.numel();
        }
    }

    if (m_adamExpAvgSqCpyBuf == nullptr) {
        checkCudaErrors(cudaMallocHost(&m_adamExpAvgSqCpyBuf, m_thrdBufs[thrdBufIdx].adamExpAvgSq.numel() * sizeof(float)));
        m_adamExpAvgSqCpyBufBytes = m_thrdBufs[thrdBufIdx].adamExpAvgSq.numel();
    } else {
        if ((m_thrdBufs[thrdBufIdx].adamExpAvgSq.numel()) > m_adamExpAvgSqCpyBufBytes) {
            checkCudaErrors(cudaFreeHost(m_adamExpAvgSqCpyBuf));
            m_adamExpAvgSqCpyBuf = nullptr;

            checkCudaErrors(cudaMallocHost(&m_adamExpAvgSqCpyBuf, m_thrdBufs[thrdBufIdx].adamExpAvgSq.numel() * sizeof(float)));
            m_adamExpAvgSqCpyBufBytes = m_thrdBufs[thrdBufIdx].adamExpAvgSq.numel();
        }
    }
}

void ImMeshRenderer::TrainedWriteBackPipeline(uint32_t thrdBufIdx)
{
    unsigned long startTime = GetTimeMS();
    unsigned long endTime;

    checkCudaErrors(cudaMemcpyAsync(m_shTexturesCpctCpyBuf, m_thrdBufs[thrdBufIdx].shTexturesCpct.data_ptr<float>(),
        m_thrdBufs[thrdBufIdx].shTexturesCpct.numel() * sizeof(float), cudaMemcpyDeviceToHost, m_shTexturesCpctCpyStrm));

    checkCudaErrors(cudaMemcpyAsync(m_adamExpAvgCpyBuf, m_thrdBufs[thrdBufIdx].adamExpAvg.data_ptr<float>(),
        m_thrdBufs[thrdBufIdx].adamExpAvg.numel() * sizeof(float), cudaMemcpyDeviceToHost, m_adamExpAvgCpyStrm));

    checkCudaErrors(cudaMemcpyAsync(m_adamExpAvgSqCpyBuf, m_thrdBufs[thrdBufIdx].adamExpAvgSq.data_ptr<float>(),
        m_thrdBufs[thrdBufIdx].adamExpAvgSq.numel() * sizeof(float), cudaMemcpyDeviceToHost, m_adamExpAvgSqCpyStrm));


    checkCudaErrors(cudaStreamSynchronize(m_shTexturesCpctCpyStrm));

    parallel_for(blocked_range<size_t>(0, m_thrdBufs[thrdBufIdx].curTexIdsAll.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            memcpy((char*)(m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].shTexValue),
                   (char *)(m_shTexturesCpctCpyBuf + m_thrdBufs[thrdBufIdx].shTexMemLayout[i]),
                   m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].perTexOffset);
        }
    });
    
    checkCudaErrors(cudaStreamSynchronize(m_adamExpAvgCpyStrm));
    parallel_for(blocked_range<size_t>(0, m_thrdBufs[thrdBufIdx].curTexIdsAll.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            memcpy((char*)(m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].adamStateExpAvg),
                   (char *)(m_adamExpAvgCpyBuf + m_thrdBufs[thrdBufIdx].shTexMemLayout[i]),
                   m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].perTexOffset);
        }
    });

    checkCudaErrors(cudaStreamSynchronize(m_adamExpAvgSqCpyStrm));
    parallel_for(blocked_range<size_t>(0, m_thrdBufs[thrdBufIdx].curTexIdsAll.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            memcpy((char*)(m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].adamStateExpAvgSq),
                   (char *)(m_adamExpAvgSqCpyBuf + m_thrdBufs[thrdBufIdx].shTexMemLayout[i]),
                   m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].perTexOffset);
        }
    });

    // endTime = GetTimeMS();
    // cout << g_trainThrdName << " ------to CPU time : " << (double)(endTime - startTime) << " ms" << endl;

    // float* shTexturesCpctHead = m_thrdBufs[thrdBufIdx].shTexturesCpct.data_ptr<float>();
    // float* shTexturesCpctHead = m_shTexturesCpctCpyBuf;
    // float* adamExpAvgHead = m_adamExpAvgCpyBuf;
    // float* adamExpAvgHead = m_thrdBufs[thrdBufIdx].adamExpAvg.data_ptr<float>();
    // float* adamExpAvgSqHead = m_adamExpAvgSqCpyBuf;
    // float* adamExpAvgSqHead = m_thrdBufs[thrdBufIdx].adamExpAvgSq.data_ptr<float>();

    // parallel_for(blocked_range<size_t>(0, m_thrdBufs[thrdBufIdx].curTexIdsAll.size()), [&](const blocked_range<size_t>& range) {
    //     for (size_t i = range.begin(); i != range.end(); i++) {
    //         memcpy((char*)(m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].shTexValue),
    //                (char *)(shTexturesCpctHead + m_thrdBufs[thrdBufIdx].shTexMemLayout[i]),
    //                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].perTexOffset);

    //         memcpy((char*)(m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].adamStateExpAvg),
    //                (char *)(adamExpAvgHead + m_thrdBufs[thrdBufIdx].shTexMemLayout[i]),
    //                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].perTexOffset);

    //         memcpy((char*)(m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].adamStateExpAvgSq),
    //                (char *)(adamExpAvgSqHead + m_thrdBufs[thrdBufIdx].shTexMemLayout[i]),
    //                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].perTexOffset);
    //     }
    // });

    endTime = GetTimeMS();

    PROFILE_PRINT(cout << g_trainThrdName << __func__ << " time : " << (double)(endTime - startTime) << " ms" << endl);
    at::Tensor dummy;
    m_thrdBufs[thrdBufIdx].shTexturesCpct = dummy;
    m_thrdBufs[thrdBufIdx].adamExpAvg = dummy;
    m_thrdBufs[thrdBufIdx].adamExpAvgSq = dummy;
    m_thrdBufs[thrdBufIdx].edgeNbrs = dummy;
    m_thrdBufs[thrdBufIdx].cornerAreaInfo = dummy;
    m_thrdBufs[thrdBufIdx].vertNbrs = dummy;
    m_thrdBufs[thrdBufIdx].texInWorldInfo = dummy;
    m_thrdBufs[thrdBufIdx].texWHs = dummy;
    m_thrdBufs[thrdBufIdx].shPosMapCpct = dummy;
    m_thrdBufs[thrdBufIdx].validShWHMapCpct = dummy;
    m_thrdBufs[thrdBufIdx].botYCoeffs = dummy;
    m_thrdBufs[thrdBufIdx].topYCoeffs = dummy;
    m_thrdBufs[thrdBufIdx].validShNumsAll = dummy;
}

// void ImMeshRenderer::UnRegisterCudaTextures()
// {
//     // checkCudaErrors(cudaGraphicsUnregisterResource(m_cuResRenderPixel));
//     checkCudaErrors(cudaGraphicsUnregisterResource(g_cuResLerpCoeffAndTexCoord));
//     checkCudaErrors(cudaGraphicsUnregisterResource(g_cuResViewDirFrag2CamAndNull));
//     checkCudaErrors(cudaGraphicsUnregisterResource(g_cuResOriShLUandRUTexCoord));
//     checkCudaErrors(cudaGraphicsUnregisterResource(g_cuResOriShLDandRDTexCoord));
//     checkCudaErrors(cudaGraphicsUnregisterResource(g_cuResTexScaleFactorDepthTexId));
//     checkCudaErrors(cudaGraphicsUnregisterResource(g_cuResWorldPoseAndNull));
//     checkCudaErrors(cudaGraphicsUnregisterResource(g_cuResEllipCoeffsAndLodLvl));
// }

// void ImMeshRenderer::RegisterCudaTextures(uint32_t thrdBufIdx)
// {
//     // checkCudaErrors(cudaGraphicsGLRegisterImage(&m_cuResRenderPixel, m_thrdBufs[thrdBufIdx].texRGB, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResLerpCoeffAndTexCoord, m_thrdBufs[thrdBufIdx].texLerpCoeffAndTexCoord, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResViewDirFrag2CamAndNull, m_thrdBufs[thrdBufIdx].texViewDirFrag2CamAndNull, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResOriShLUandRUTexCoord, m_thrdBufs[thrdBufIdx].texOriShLUandRU, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResOriShLDandRDTexCoord, m_thrdBufs[thrdBufIdx].texOriShLDandRD, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResTexScaleFactorDepthTexId, m_thrdBufs[thrdBufIdx].texScaleFacDepthTexId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     // checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResTexEdgeLR, m_texEdgeLR, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResWorldPoseAndNull, m_thrdBufs[thrdBufIdx].texWorldPoseAndNull, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     // checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResTexInvDistCoeffs, m_texInvDistCoeffs, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResEllipCoeffsAndLodLvl, m_thrdBufs[thrdBufIdx].texEllipCoeffsAndLodLvl, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
// }


// void ImMeshRenderer::RegisterCudaTexturesInfer()
// {
//     // checkCudaErrors(cudaGraphicsUnregisterResource(m_cuResRenderPixel));
//     // checkCudaErrors(cudaGraphicsGLRegisterImage(&m_cuResRenderPixel, m_inferBuffer.texRGB, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResLerpCoeffAndTexCoord, m_inferBuffer.texLerpCoeffAndTexCoord, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResViewDirFrag2CamAndNull, m_inferBuffer.texViewDirFrag2CamAndNull, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResOriShLUandRUTexCoord, m_inferBuffer.texOriShLUandRU, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResOriShLDandRDTexCoord, m_inferBuffer.texOriShLDandRD, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResTexScaleFactorDepthTexId, m_inferBuffer.texScaleFacDepthTexId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     // checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResTexEdgeLR, m_texEdgeLR, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResWorldPoseAndNull, m_inferBuffer.texWorldPoseAndNull, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     // checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResTexInvDistCoeffs, m_texInvDistCoeffs, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

//     checkCudaErrors(cudaGraphicsGLRegisterImage(&g_cuResEllipCoeffsAndLodLvl, m_inferBuffer.texEllipCoeffsAndLodLvl, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
// }


void ImMeshRenderer::PrepOptimizer(uint32_t thrdBufIdx, std::shared_ptr<torch::optim::Adam> optimizer)
{
    std::vector<torch::optim::OptimizerParamGroup>& CurParamGrp = optimizer->param_groups();
    assert(CurParamGrp.size() == 1);
    assert(CurParamGrp[0].params().size() == 1);

    static bool isFirstTime = true;
    int64_t stepOld = 0;
    if (isFirstTime == true) {
        isFirstTime = false;
        stepOld = 0;
    } else if (m_thrdBufs[thrdBufIdx].needResetOptimizor == true) {
        stepOld = 0;
        m_thrdBufs[thrdBufIdx].needResetOptimizor = false;
    } else {
        auto paramStateOld = optimizer->state().find(CurParamGrp[0].params()[0].unsafeGetTensorImpl());
        if (paramStateOld == optimizer->state().end()) {
            cout << "cannot find cur adam state \n";
            exit(1);
        }

        stepOld = (static_cast<torch::optim::AdamParamState&>(*optimizer->state()[CurParamGrp[0].params()[0].unsafeGetTensorImpl()])).step();
    }

    optimizer->state().erase(CurParamGrp[0].params()[0].unsafeGetTensorImpl());
    CurParamGrp[0].params()[0] = m_thrdBufs[thrdBufIdx].shTexturesCpct;

    auto state = std::make_unique<torch::optim::AdamParamState>();

    state->step(stepOld);
    state->exp_avg(m_thrdBufs[thrdBufIdx].adamExpAvg);
    state->exp_avg_sq(m_thrdBufs[thrdBufIdx].adamExpAvgSq);
    optimizer->state()[m_thrdBufs[thrdBufIdx].shTexturesCpct.unsafeGetTensorImpl()] = std::move(state);
}

void ImMeshRenderer::CopyCurInferEssentials(ThreadBuffer& buf)
{
    unsigned long startTime = GetTimeMS();
    unsigned long endTime;

    if (m_shTexturesCpctCpyBuf == nullptr) {
        checkCudaErrors(cudaMallocHost(&m_shTexturesCpctCpyBuf, buf.shTexturesCpct.numel() * sizeof(float)));
        m_shTexturesCpctCpyBufBytes = buf.shTexturesCpct.numel();
    } else {
        if ((buf.shTexturesCpct.numel() > m_shTexturesCpctCpyBufBytes)) {
            checkCudaErrors(cudaFreeHost(m_shTexturesCpctCpyBuf));
            m_shTexturesCpctCpyBuf = nullptr;

            checkCudaErrors(cudaMallocHost(&m_shTexturesCpctCpyBuf, buf.shTexturesCpct.numel() * sizeof(float)));
            m_shTexturesCpctCpyBufBytes = buf.shTexturesCpct.numel();
        }
    }

    parallel_for(blocked_range<size_t>(0, buf.curTexIdsAll.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            uint32_t curTexValidShNum = m_shTexHashMap[buf.curTexIdsAll[i] - 1].validShNum;

            buf.m_validShNumsAllCpyBuf[i] = curTexValidShNum;

            memcpy((char *)(m_shTexturesCpctCpyBuf + buf.shTexMemLayout[i]),
                   (char *)(m_shTexHashMap[buf.curTexIdsAll[i] - 1].shTexValue),
                   m_shTexHashMap[buf.curTexIdsAll[i] - 1].perTexOffset);

            *(buf.m_meshNormalsCpyBuf + i * 3 + 0) = m_shTexHashMap[buf.curTexIdsAll[i] - 1].norm[0];
            *(buf.m_meshNormalsCpyBuf + i * 3 + 1) = m_shTexHashMap[buf.curTexIdsAll[i] - 1].norm[1];
            *(buf.m_meshNormalsCpyBuf + i * 3 + 2) = m_shTexHashMap[buf.curTexIdsAll[i] - 1].norm[2];
            
            buf.m_botYCoeffsCpyBuf[i] = m_shTexHashMap[buf.curTexIdsAll[i] - 1].bottomYCoef;
            buf.m_topYCoeffsCpyBuf[i] = m_shTexHashMap[buf.curTexIdsAll[i] - 1].topYCoef;
            buf.m_meshDensitiesCpyBuf[i] = m_shTexHashMap[buf.curTexIdsAll[i] - 1].resizeInfo.curDensity;

            memcpy((char *)(m_shPosMapCpctCpyBufInfer + buf.posMapMemLayout[i]),
                   (char *)(m_shTexHashMap[buf.curTexIdsAll[i] - 1].worldPoseMap),
                   sizeof(float) * 3 * curTexValidShNum);
            
            memcpy((char*)(m_validShWHMapCpctCpyBufInfer + buf.validShWHMapLayout[i]),
                   (char*)(m_shTexHashMap[buf.curTexIdsAll[i] - 1].validShWHMap),
                    sizeof(TexAlignedPosWH) * curTexValidShNum);

            buf.m_edgeNbrsCpyBuf[6 * i + 0] = m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborTexId[0];
            buf.m_edgeNbrsCpyBuf[6 * i + 1] = m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborEdgeType[0];
            buf.m_edgeNbrsCpyBuf[6 * i + 2] = m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborTexId[1];
            buf.m_edgeNbrsCpyBuf[6 * i + 3] = m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborEdgeType[1];
            buf.m_edgeNbrsCpyBuf[6 * i + 4] = m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborTexId[2];
            buf.m_edgeNbrsCpyBuf[6 * i + 5] = m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborEdgeType[2];

            
            memcpy((char *)(buf.m_texInWorldInfoCpyBuf + i * 6),
                   (char *)(m_shTexHashMap[buf.curTexIdsAll[i] - 1].texInWorldInfo),
                   sizeof(float) * 6);

            buf.m_texWHsCpyBuf[i] = m_shTexHashMap[buf.curTexIdsAll[i] - 1].resizeInfo.curTexWH.aligner;

            buf.m_cornerAreaInfoCpyBuf[i * 2 + 0] = m_shTexHashMap[buf.curTexIdsAll[i] - 1].cornerArea[0];
            buf.m_cornerAreaInfoCpyBuf[i * 2 + 1] = m_shTexHashMap[buf.curTexIdsAll[i] - 1].cornerArea[1];

            memcpy((char *)(m_vertNbrsCpyBufInfer + buf.vertNbrCpctMemLayout[i]),
                   (char *)(m_shTexHashMap[buf.curTexIdsAll[i] - 1].topVertNbr.data()),
                   m_shTexHashMap[buf.curTexIdsAll[i] - 1].topVertNbr.size() * sizeof(uint32_t));

            memcpy((char *)(m_vertNbrsCpyBufInfer + buf.vertNbrCpctMemLayout[i] + m_shTexHashMap[buf.curTexIdsAll[i] - 1].topVertNbr.size()),
                   (char *)(m_shTexHashMap[buf.curTexIdsAll[i] - 1].leftVertNbr.data()),
                   m_shTexHashMap[buf.curTexIdsAll[i] - 1].leftVertNbr.size() * sizeof(uint32_t));

            memcpy((char *)(m_vertNbrsCpyBufInfer + buf.vertNbrCpctMemLayout[i] + m_shTexHashMap[buf.curTexIdsAll[i] - 1].topVertNbr.size() + m_shTexHashMap[buf.curTexIdsAll[i] - 1].leftVertNbr.size()),
                   (char *)(m_shTexHashMap[buf.curTexIdsAll[i] - 1].rightVertNbr.data()),
                   m_shTexHashMap[buf.curTexIdsAll[i] - 1].rightVertNbr.size() * sizeof(uint32_t));
            



            // memcpy((char *)(buf.curTrainShTexHead + i * m_offsetPerTexture), (char*)m_shTexHashMap[buf.curTexIdsAll[i] - 1].shTexValue.data_ptr<float>(), sizeof(float) * m_shTexChannelNum * m_texW * m_texH * m_shLayerNum);

            // memcpy((char *)(buf.curTrainShPosMapHead + i * offsetShPosMap), (char*)m_shTexHashMap[buf.curTexIdsAll[i] - 1].worldPoseMap, sizeof(float) * offsetShPosMap);

            // // memcpy((char *)(curInferTexNbrInfoHead + i * 3), (char*)m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborTexId, sizeof(uint32_t) * 3);

            // // *((uint32_t *)(curInferTexNbrInfoHead + i * 6 + 0)) = validEdgeNbr[i * 3 + 0].texId;
            // // *((uint32_t *)(curInferTexNbrInfoHead + i * 6 + 1)) = validEdgeNbr[i * 3 + 0].edgeType;
            // *((uint32_t *)(buf.edgeNbrsHead + i * 6 + 0)) = m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborTexId[0];
            // *((uint32_t *)(buf.edgeNbrsHead + i * 6 + 1)) = m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborEdgeType[0];

            // // *((uint32_t *)(curInferTexNbrInfoHead + i * 6 + 2)) = validEdgeNbr[i * 3 + 1].texId;
            // // *((uint32_t *)(curInferTexNbrInfoHead + i * 6 + 3)) = validEdgeNbr[i * 3 + 1].edgeType;
            // *((uint32_t *)(buf.edgeNbrsHead + i * 6 + 2)) = m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborTexId[1];
            // *((uint32_t *)(buf.edgeNbrsHead + i * 6 + 3)) = m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborEdgeType[1];

            // // *((uint32_t *)(curInferTexNbrInfoHead + i * 6 + 4)) = validEdgeNbr[i * 3 + 2].texId;
            // // *((uint32_t *)(curInferTexNbrInfoHead + i * 6 + 5)) = validEdgeNbr[i * 3 + 2].edgeType;
            // *((uint32_t *)(buf.edgeNbrsHead + i * 6 + 4)) = m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborTexId[2];
            // *((uint32_t *)(buf.edgeNbrsHead + i * 6 + 5)) = m_shTexHashMap[buf.curTexIdsAll[i] - 1].neighborEdgeType[2];

            // memcpy((char *)(buf.texInWorldInfoHead + i * 6), (char *)m_shTexHashMap[buf.curTexIdsAll[i] - 1].texInWorldInfo, sizeof(float) * 6);

            // memcpy((char *)(buf.cornerAreaInfoHead + i * 2), (char*)m_shTexHashMap[buf.curTexIdsAll[i] - 1].cornerArea, sizeof(float) * 2);

            // // memcpy((char *)(curInferVertNbrBufHead + curMemLayoutVec[i].offset), (char *)(curMemLayoutVec[i].topVertNbrValid.data()), curMemLayoutVec[i].topVertNbrValid.size() * sizeof(uint32_t));

            // // memcpy((char *)(curInferVertNbrBufHead + curMemLayoutVec[i].offset + curMemLayoutVec[i].topVertNbrValid.size()), (char *)(curMemLayoutVec[i].leftVertNbrValid.data()), curMemLayoutVec[i].leftVertNbrValid.size() * sizeof(uint32_t));

            // // memcpy((char *)(curInferVertNbrBufHead + curMemLayoutVec[i].offset + curMemLayoutVec[i].topVertNbrValid.size() + curMemLayoutVec[i].leftVertNbrValid.size()), (char *)(curMemLayoutVec[i].rightVertNbrValid.data()), curMemLayoutVec[i].rightVertNbrValid.size() * sizeof(uint32_t));

            // memcpy((char *)(buf.vertNbrsHead + buf.curMemLayoutVec[i]), (char *)(m_shTexHashMap[buf.curTexIdsAll[i] - 1].topVertNbr.data()), m_shTexHashMap[buf.curTexIdsAll[i] - 1].topVertNbr.size() * sizeof(uint32_t));

            // memcpy((char *)(buf.vertNbrsHead + buf.curMemLayoutVec[i] + m_shTexHashMap[buf.curTexIdsAll[i] - 1].topVertNbr.size()), (char *)(m_shTexHashMap[buf.curTexIdsAll[i] - 1].leftVertNbr.data()), m_shTexHashMap[buf.curTexIdsAll[i] - 1].leftVertNbr.size() * sizeof(uint32_t));

            // memcpy((char *)(buf.vertNbrsHead + buf.curMemLayoutVec[i] + m_shTexHashMap[buf.curTexIdsAll[i] - 1].topVertNbr.size() + m_shTexHashMap[buf.curTexIdsAll[i] - 1].leftVertNbr.size()), (char *)(m_shTexHashMap[buf.curTexIdsAll[i] - 1].rightVertNbr.data()), m_shTexHashMap[buf.curTexIdsAll[i] - 1].rightVertNbr.size() * sizeof(uint32_t));
        }
    });

    // printf("corner info : \n");
    // for (uint32_t ii = 0; ii < m_curInferCornerAreaInfo.sizes().data()[0]; ii++) {
    //     for (uint32_t jj = 0; jj < m_curInferCornerAreaInfo.sizes().data()[1]; jj++) {
    //         printf("%f ", m_curInferCornerAreaInfo[ii][jj].item().toFloat());
    //     }
    //     printf("\n");
    // }
    // buf.shTexturesCpct = buf.shTexturesCpct.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);

    checkCudaErrors(cudaMemcpyAsync(buf.shTexturesCpctHead, m_shTexturesCpctCpyBuf, buf.shTexturesCpct.numel() * sizeof(float), cudaMemcpyDeviceToHost, m_shTexturesCpctCpyStrm));

    // buf.validShNumsAll = buf.validShNumsAll.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);
    // buf.validShWHMapCpct = buf.validShWHMapCpct.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);
    
    // buf.edgeNbrs = buf.edgeNbrs.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);
    // buf.cornerAreaInfo = buf.cornerAreaInfo.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);
    // buf.vertNbrs = buf.vertNbrs.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);
    // buf.texInWorldInfo = buf.texInWorldInfo.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);
    // buf.texWHs = buf.texWHs.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);
    // buf.shPosMapCpct = buf.shPosMapCpct.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);
    // buf.botYCoeffs = buf.botYCoeffs.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);
    // buf.topYCoeffs = buf.topYCoeffs.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);
    // buf.meshDensities = buf.meshDensities.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);
    // buf.meshNormals = buf.meshNormals.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);

    checkCudaErrors(cudaMemcpyAsync(buf.validShNumsAllHead, buf.m_validShNumsAllCpyBuf, buf.validShNumsAll.numel() * sizeof(int32_t), cudaMemcpyHostToDevice, buf.curBufStream));
    checkCudaErrors(cudaMemcpyAsync(buf.validShWHMapCpctHead, m_validShWHMapCpctCpyBufInfer, buf.validShWHMapCpct.numel() * sizeof(int32_t), cudaMemcpyHostToDevice, buf.curBufStream2));
    checkCudaErrors(cudaMemcpyAsync(buf.shPosMapCpctHead, m_shPosMapCpctCpyBufInfer, buf.shPosMapCpct.numel() * sizeof(float), cudaMemcpyHostToDevice, buf.curBufStream));
 
    checkCudaErrors(cudaMemcpyAsync(buf.edgeNbrsHead, buf.m_edgeNbrsCpyBuf, buf.edgeNbrs.numel() * sizeof(int32_t), cudaMemcpyHostToDevice, buf.curBufStream2));
    checkCudaErrors(cudaMemcpyAsync(buf.vertNbrsHead, m_vertNbrsCpyBufInfer, buf.vertNbrs.numel() * sizeof(int32_t), cudaMemcpyHostToDevice, buf.curBufStream));
    checkCudaErrors(cudaMemcpyAsync(buf.texWHsHead, buf.m_texWHsCpyBuf, buf.texWHs.numel() * sizeof(int32_t), cudaMemcpyHostToDevice, buf.curBufStream2));
    

    checkCudaErrors(cudaMemcpyAsync(buf.cornerAreaInfoHead, buf.m_cornerAreaInfoCpyBuf, buf.cornerAreaInfo.numel() * sizeof(float), cudaMemcpyHostToDevice, buf.curBufStream));
    checkCudaErrors(cudaMemcpyAsync(buf.texInWorldInfoHead, buf.m_texInWorldInfoCpyBuf, buf.texInWorldInfo.numel() * sizeof(float), cudaMemcpyHostToDevice, buf.curBufStream2));

    checkCudaErrors(cudaMemcpyAsync(buf.botYCoeffsHead, buf.m_botYCoeffsCpyBuf, buf.botYCoeffs.numel() * sizeof(float), cudaMemcpyHostToDevice, buf.curBufStream));
    checkCudaErrors(cudaMemcpyAsync(buf.topYCoeffsHead, buf.m_topYCoeffsCpyBuf, buf.topYCoeffs.numel() * sizeof(float), cudaMemcpyHostToDevice, buf.curBufStream2));

    checkCudaErrors(cudaMemcpyAsync(buf.meshDensitiesHead, buf.m_meshDensitiesCpyBuf, buf.meshDensities.numel() * sizeof(float), cudaMemcpyHostToDevice, buf.curBufStream));
    checkCudaErrors(cudaMemcpyAsync(buf.meshNormalsHead, buf.m_meshNormalsCpyBuf, buf.meshNormals.numel() * sizeof(float), cudaMemcpyHostToDevice, buf.curBufStream2));
 
    buf.posMapMemLayoutDevice = buf.posMapMemLayout;
    buf.validShWHMapLayoutDevice = buf.validShWHMapLayout;
    buf.shTexMemLayoutDevice = buf.shTexMemLayout;
}

void ImMeshRenderer::CopyCurTrainEssentialsFixed(uint32_t thrdBufIdx)
{
    unsigned long startTime = GetTimeMS();
    unsigned long endTime;

    parallel_for(blocked_range<size_t>(0, m_thrdBufs[thrdBufIdx].curTexIdsAll.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            uint32_t curTexValidShNum = m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].validShNum;

            m_thrdBufs[thrdBufIdx].m_validShNumsAllCpyBuf[i] = curTexValidShNum;

            *(m_thrdBufs[thrdBufIdx].m_meshNormalsCpyBuf + 3 * i + 0) = m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].norm[0];
            *(m_thrdBufs[thrdBufIdx].m_meshNormalsCpyBuf + 3 * i + 1) = m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].norm[1];
            *(m_thrdBufs[thrdBufIdx].m_meshNormalsCpyBuf + 3 * i + 2) = m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].norm[2];

            m_thrdBufs[thrdBufIdx].m_meshDensitiesCpyBuf[i] = m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].resizeInfo.curDensity;

            m_thrdBufs[thrdBufIdx].m_botYCoeffsCpyBuf[i] =
                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].bottomYCoef;
            m_thrdBufs[thrdBufIdx].m_topYCoeffsCpyBuf[i] =
                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].topYCoef;

            memcpy((char *)(m_shPosMapCpctCpyBuf + m_thrdBufs[thrdBufIdx].posMapMemLayout[i]),
                   (char *)m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].worldPoseMap,
                   sizeof(float) * 3 * curTexValidShNum);

            memcpy((char *)(m_validShWHMapCpctCpyBuf + m_thrdBufs[thrdBufIdx].validShWHMapLayout[i]),
                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].validShWHMap,
                sizeof(TexAlignedPosWH) * m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].validShNum);

            *((uint32_t *)(m_thrdBufs[thrdBufIdx].m_edgeNbrsCpyBuf + i * 6 + 0)) =
                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].neighborTexId[0];
            *((uint32_t *)(m_thrdBufs[thrdBufIdx].m_edgeNbrsCpyBuf + i * 6 + 1)) =
                (uint32_t)m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].neighborEdgeType[0];
            *((uint32_t *)(m_thrdBufs[thrdBufIdx].m_edgeNbrsCpyBuf + i * 6 + 2)) =
                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].neighborTexId[1];
            *((uint32_t *)(m_thrdBufs[thrdBufIdx].m_edgeNbrsCpyBuf + i * 6 + 3)) =
                (uint32_t)m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].neighborEdgeType[1];
            *((uint32_t *)(m_thrdBufs[thrdBufIdx].m_edgeNbrsCpyBuf + i * 6 + 4)) =
                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].neighborTexId[2];
            *((uint32_t *)(m_thrdBufs[thrdBufIdx].m_edgeNbrsCpyBuf + i * 6 + 5)) =
                (uint32_t)m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].neighborEdgeType[2];


            memcpy((char *)(m_thrdBufs[thrdBufIdx].m_texInWorldInfoCpyBuf + i * 6),
                   (char *)m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].texInWorldInfo,
                   sizeof(float) * 6);
            
            *((uint32_t *)((m_thrdBufs[thrdBufIdx].m_texWHsCpyBuf + i))) = m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].resizeInfo.curTexWH.aligner;

            m_thrdBufs[thrdBufIdx].m_cornerAreaInfoCpyBuf[i * 2 + 0] = m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].cornerArea[0];
            m_thrdBufs[thrdBufIdx].m_cornerAreaInfoCpyBuf[i * 2 + 1] = m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].cornerArea[1];



            memcpy((char *)(m_vertNbrsCpyBuf + m_thrdBufs[thrdBufIdx].vertNbrCpctMemLayout[i]),
                   (char *)(m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].topVertNbr.data()),
                   m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].topVertNbr.size() * sizeof(uint32_t));

            memcpy((char *)(m_vertNbrsCpyBuf + m_thrdBufs[thrdBufIdx].vertNbrCpctMemLayout[i] + m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].topVertNbr.size()),
                   (char *)(m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].leftVertNbr.data()),
                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].leftVertNbr.size() * sizeof(uint32_t));

            memcpy((char *)(m_vertNbrsCpyBuf + m_thrdBufs[thrdBufIdx].vertNbrCpctMemLayout[i] + m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].topVertNbr.size() + m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].leftVertNbr.size()),
                   (char *)(m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].rightVertNbr.data()),
                   m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].rightVertNbr.size() * sizeof(uint32_t));
        }
    });

    endTime = GetTimeMS();
    PROFILE_PRINT(cout << g_trainPrepThrdName << " ------till CopyCurTrainEssentialsFixed parallel_for passed time : " << (double)(endTime - startTime) << " ms" << endl);


    // m_thrdBufs[thrdBufIdx].validShNumsAll = m_thrdBufs[thrdBufIdx].validShNumsAll.to(torch::kCUDA, 0).contiguous();
    // m_thrdBufs[thrdBufIdx].shPosMapCpct = m_thrdBufs[thrdBufIdx].shPosMapCpct.to(torch::kCUDA, 0).contiguous();
    // m_thrdBufs[thrdBufIdx].validShWHMapCpct = m_thrdBufs[thrdBufIdx].validShWHMapCpct.to(torch::kCUDA, 0).contiguous();
    // m_thrdBufs[thrdBufIdx].edgeNbrs = m_thrdBufs[thrdBufIdx].edgeNbrs.to(torch::kCUDA, 0).contiguous();
    // m_thrdBufs[thrdBufIdx].vertNbrs = m_thrdBufs[thrdBufIdx].vertNbrs.to(torch::kCUDA, 0).contiguous();
    // m_thrdBufs[thrdBufIdx].cornerAreaInfo = m_thrdBufs[thrdBufIdx].cornerAreaInfo.to(torch::kCUDA, 0).contiguous();
    // m_thrdBufs[thrdBufIdx].texInWorldInfo = m_thrdBufs[thrdBufIdx].texInWorldInfo.to(torch::kCUDA, 0).contiguous();
    // m_thrdBufs[thrdBufIdx].texWHs = m_thrdBufs[thrdBufIdx].texWHs.to(torch::kCUDA, 0).contiguous();
    // m_thrdBufs[thrdBufIdx].botYCoeffs = m_thrdBufs[thrdBufIdx].botYCoeffs.to(torch::kCUDA, 0).contiguous();
    // m_thrdBufs[thrdBufIdx].topYCoeffs = m_thrdBufs[thrdBufIdx].topYCoeffs.to(torch::kCUDA, 0).contiguous();
    // m_thrdBufs[thrdBufIdx].meshDensities = m_thrdBufs[thrdBufIdx].meshDensities.to(torch::kCUDA, 0).contiguous();
    // m_thrdBufs[thrdBufIdx].meshNormals = m_thrdBufs[thrdBufIdx].meshNormals.to(torch::kCUDA, 0).contiguous();

    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].shPosMapCpctHead, m_shPosMapCpctCpyBuf, m_thrdBufs[thrdBufIdx].shPosMapCpct.numel() * sizeof(float), cudaMemcpyHostToDevice, m_trainPrepThrdCpyStrm1));

    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].validShWHMapCpctHead, m_validShWHMapCpctCpyBuf, m_thrdBufs[thrdBufIdx].validShWHMapCpct.numel() * sizeof(int32_t),  cudaMemcpyHostToDevice, m_trainPrepThrdCpyStrm2));

    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].vertNbrsHead, m_vertNbrsCpyBuf, m_thrdBufs[thrdBufIdx].vertNbrs.numel() * sizeof(int32_t), cudaMemcpyHostToDevice, m_trainPrepThrdCpyStrm1));

    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].validShNumsAllHead, m_thrdBufs[thrdBufIdx].m_validShNumsAllCpyBuf, m_thrdBufs[thrdBufIdx].validShNumsAll.numel() * sizeof(int32_t), cudaMemcpyHostToDevice, m_thrdBufs[thrdBufIdx].curBufStream));

    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].edgeNbrsHead, m_thrdBufs[thrdBufIdx].m_edgeNbrsCpyBuf, m_thrdBufs[thrdBufIdx].edgeNbrs.numel() * sizeof(int32_t), cudaMemcpyHostToDevice, m_thrdBufs[thrdBufIdx].curBufStream2));
    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].texWHsHead, m_thrdBufs[thrdBufIdx].m_texWHsCpyBuf, m_thrdBufs[thrdBufIdx].texWHs.numel() * sizeof(int32_t), cudaMemcpyHostToDevice, m_thrdBufs[thrdBufIdx].curBufStream));

    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].cornerAreaInfoHead, m_thrdBufs[thrdBufIdx].m_cornerAreaInfoCpyBuf, m_thrdBufs[thrdBufIdx].cornerAreaInfo.numel() * sizeof(float), cudaMemcpyHostToDevice, m_thrdBufs[thrdBufIdx].curBufStream2));
    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].texInWorldInfoHead, m_thrdBufs[thrdBufIdx].m_texInWorldInfoCpyBuf, m_thrdBufs[thrdBufIdx].texInWorldInfo.numel() * sizeof(float), cudaMemcpyHostToDevice, m_thrdBufs[thrdBufIdx].curBufStream));

    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].botYCoeffsHead, m_thrdBufs[thrdBufIdx].m_botYCoeffsCpyBuf, m_thrdBufs[thrdBufIdx].botYCoeffs.numel() * sizeof(float), cudaMemcpyHostToDevice, m_thrdBufs[thrdBufIdx].curBufStream2));
    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].topYCoeffsHead, m_thrdBufs[thrdBufIdx].m_topYCoeffsCpyBuf, m_thrdBufs[thrdBufIdx].topYCoeffs.numel() * sizeof(float), cudaMemcpyHostToDevice, m_thrdBufs[thrdBufIdx].curBufStream));

    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].meshDensitiesHead, m_thrdBufs[thrdBufIdx].m_meshDensitiesCpyBuf, m_thrdBufs[thrdBufIdx].meshDensities.numel() * sizeof(float), cudaMemcpyHostToDevice, m_thrdBufs[thrdBufIdx].curBufStream2));
    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].meshNormalsHead, m_thrdBufs[thrdBufIdx].m_meshNormalsCpyBuf, m_thrdBufs[thrdBufIdx].meshNormals.numel() * sizeof(float), cudaMemcpyHostToDevice, m_thrdBufs[thrdBufIdx].curBufStream));

    m_thrdBufs[thrdBufIdx].posMapMemLayoutDevice = m_thrdBufs[thrdBufIdx].posMapMemLayout;
    m_thrdBufs[thrdBufIdx].validShWHMapLayoutDevice = m_thrdBufs[thrdBufIdx].validShWHMapLayout;
    m_thrdBufs[thrdBufIdx].shTexMemLayoutDevice = m_thrdBufs[thrdBufIdx].shTexMemLayout;

    endTime = GetTimeMS();
    PROFILE_PRINT(cout << g_trainPrepThrdName << " ------till CopyCurTrainEssentialsFixed to GPU passed time : " << (double)(endTime - startTime) << " ms" << endl);
}

void ImMeshRenderer::CopyCurTrainEssentials(uint32_t thrdBufIdx)
{
    unsigned long startTime = GetTimeMS();
    unsigned long endTime;

    // parallel_for(blocked_range<size_t>(0, m_thrdBufs[thrdBufIdx].curTexIdsAll.size()), [&](const blocked_range<size_t>& range) {
    //     for (size_t i = range.begin(); i != range.end(); i++) {
    //         memcpy((char *)(m_thrdBufs[thrdBufIdx].shTexturesCpctHead + m_thrdBufs[thrdBufIdx].shTexMemLayout[i]),
    //                (char*)m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].shTexValue,
    //                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].perTexOffset);

    //         memcpy((char *)(m_thrdBufs[thrdBufIdx].adamExpAvgHead + m_thrdBufs[thrdBufIdx].shTexMemLayout[i]),
    //                (char*)m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].adamStateExpAvg,
    //                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].perTexOffset);

    //         memcpy((char *)(m_thrdBufs[thrdBufIdx].adamExpAvgSqHead + m_thrdBufs[thrdBufIdx].shTexMemLayout[i]),
    //                (char*)m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].adamStateExpAvgSq,
    //                m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].perTexOffset);
    //     }
    // });

    parallel_for(blocked_range<size_t>(0, m_thrdBufs[thrdBufIdx].curTexIdsAll.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            memcpy((char *)(m_shTexturesCpctCpyBuf + m_thrdBufs[thrdBufIdx].shTexMemLayout[i]),
                   (char*)m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].shTexValue,
                   m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].perTexOffset);
        }
    });
    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].shTexturesCpctHead, m_shTexturesCpctCpyBuf, m_thrdBufs[thrdBufIdx].shTexturesCpct.numel() * sizeof(float), cudaMemcpyHostToDevice, m_shTexturesCpctCpyStrm));

    endTime = GetTimeMS();
    PROFILE_PRINT(cout << g_trainThrdName << " ------copy m_shTexturesCpctCpyBuf time : " << (double)(endTime - startTime) << " ms" << endl);

    parallel_for(blocked_range<size_t>(0, m_thrdBufs[thrdBufIdx].curTexIdsAll.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            memcpy((char *)(m_adamExpAvgCpyBuf + m_thrdBufs[thrdBufIdx].shTexMemLayout[i]),
                   (char*)m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].adamStateExpAvg,
                   m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].perTexOffset);
        }
    });
    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].adamExpAvgHead, m_adamExpAvgCpyBuf, m_thrdBufs[thrdBufIdx].adamExpAvg.numel() * sizeof(float), cudaMemcpyHostToDevice, m_adamExpAvgCpyStrm));
    endTime = GetTimeMS();
    PROFILE_PRINT(cout << g_trainThrdName << " ------copy m_adamExpAvgCpyBuf time : " << (double)(endTime - startTime) << " ms" << endl);

    parallel_for(blocked_range<size_t>(0, m_thrdBufs[thrdBufIdx].curTexIdsAll.size()), [&](const blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            memcpy((char *)(m_adamExpAvgSqCpyBuf + m_thrdBufs[thrdBufIdx].shTexMemLayout[i]),
                   (char*)m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].adamStateExpAvgSq,
                   m_shTexHashMap[m_thrdBufs[thrdBufIdx].curTexIdsAll[i] - 1].perTexOffset);
        }
    });
    checkCudaErrors(cudaMemcpyAsync(m_thrdBufs[thrdBufIdx].adamExpAvgSqHead, m_adamExpAvgSqCpyBuf, m_thrdBufs[thrdBufIdx].adamExpAvgSq.numel() * sizeof(float), cudaMemcpyHostToDevice, m_adamExpAvgSqCpyStrm));

    endTime = GetTimeMS();
    PROFILE_PRINT(cout << g_trainThrdName << " ------call m_adamExpAvgSqCpyBuf time : " << (double)(endTime - startTime) << " ms" << endl);


    checkCudaErrors(cudaStreamSynchronize(m_shTexturesCpctCpyStrm));
    checkCudaErrors(cudaStreamSynchronize(m_adamExpAvgCpyStrm));
    checkCudaErrors(cudaStreamSynchronize(m_adamExpAvgSqCpyStrm));

    // m_thrdBufs[thrdBufIdx].shTexturesCpct = m_thrdBufs[thrdBufIdx].shTexturesCpct.to(torch::kCUDA, 0).contiguous().set_requires_grad(true);
    // m_thrdBufs[thrdBufIdx].adamExpAvg = m_thrdBufs[thrdBufIdx].adamExpAvg.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);
    // m_thrdBufs[thrdBufIdx].adamExpAvgSq = m_thrdBufs[thrdBufIdx].adamExpAvgSq.to(torch::kCUDA, 0).contiguous().set_requires_grad(false);

    endTime = GetTimeMS();
    PROFILE_PRINT(cout << g_trainThrdName << " ------till move to GPU time : " << (double)(endTime - startTime) << " ms" << endl);
}

at::Tensor GetWindow(uint32_t winSize, float sigma=1.5f, uint32_t channel=3)
{
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU, 0);
    assert(winSize > 0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

    auto coords = torch::arange(winSize, options);
    coords = coords - (winSize / 2);

    // g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    // torch::full({1}, sigma, std::nullopt, options);
    // torch::full({1}, 2.0f, std::nullopt, options);
    auto g = torch::exp((-torch::pow(coords, torch::full({1}, 2.0f, std::nullopt, options))) / (2.0f * torch::pow(torch::full({1}, sigma, std::nullopt, options), torch::full({1}, 2.0f, std::nullopt, options))));
    g = g / g.sum();

    g = g.reshape({1, 1, 1, -1}).repeat({channel, 1, 1, 1});

    return g;
}

at::Tensor GaussianFilter(at::Tensor x, at::Tensor window1D, uint32_t channel=3)
{
    // at::Tensor dummy;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

    torch::Tensor bias = torch::zeros({3}, options);

    // at::IntArrayRef stride = {1};
    // at::IntArrayRef stride = {1};
    // at::IntArrayRef padding = {1, 1};
    // at::IntArrayRef dilation = {1, 1};

    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> padding = {0, 0};
    std::vector<int64_t> dilation = {1, 1};

    // auto out = at::conv2d(x, window1D, bias, stride, padding, dilation, channel);
    auto out = at::conv2d(x, window1D, torch::Tensor(), stride, padding, dilation, channel);
    out = at::conv2d(out, window1D.transpose(2, 3), torch::Tensor(), stride, padding, dilation, channel);
    return out;
}

at::Tensor SSIM(const at::Tensor pred, const at::Tensor gt, uint32_t winSize, float sigma=1.5f, uint32_t channel=3, float dataRange=1.f)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

    at::Tensor window = GetWindow(winSize);

    // cout << "gausiian kernel : " << window << endl;

    const float K1 = 0.01f;
    const float K2 = 0.03f;

    auto C1 = pow((K1 * dataRange), 2.0f);
    auto C2 = pow((K2 * dataRange), 2.0f);

    auto mu1 = GaussianFilter(pred, window);
    auto mu2 = GaussianFilter(gt, window);
    auto sigma1_sq = GaussianFilter(pred * pred, window);
    auto sigma2_sq = GaussianFilter(gt * gt, window);
    auto sigma12 = GaussianFilter(pred * gt, window);

    // auto gt_CPU = gt.to(torch::kCPU);
    // auto gt_sq_sum_CPU = (gt_CPU * gt_CPU).sum();
    // printf("gt_sq_sum CPU : %f\n", gt_sq_sum_CPU.item().toFloat());

    // auto gt_sq_sum = (gt * gt).sum();
    // printf("gt_sq_sum : %f\n", gt_sq_sum.item().toFloat());

    // printf("sigma2_sq : %f\n", sigma2_sq.sum().item().toFloat());

    // torch::full({1}, 2.0f, std::nullopt, options)
    // auto mu1_sq = mu1.pow(torch::full({1}, 2.0f, std::nullopt, options));
    auto mu1_sq = mu1.pow(2.0f);
    // auto mu2_sq = mu2.pow(torch::full({1}, 2.0f, std::nullopt, options));
    auto mu2_sq = at::pow(mu2, 2.0f);
    auto mu1_mu2 = mu1 * mu2;

    // auto mu2_CPU = mu2.to(torch::kCPU);
    // auto mu2_sq_CPU = at::pow(mu2_CPU, 2.0f);

    // printf("mu2_sq_CPU sum : %f\n", mu2_sq_CPU.sum().item().toFloat());
    // printf("mu2_sq sum : %f\n", mu2_sq.sum().item().toFloat());


    sigma1_sq = sigma1_sq - mu1_sq;
    sigma2_sq = sigma2_sq - mu2_sq;
    sigma12 = sigma12 - mu1_mu2;

    auto cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2);
    cs_map = at::relu(cs_map);

    auto ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map;

    auto ssim_val = ssim_map.mean({1, 2, 3});  // reduce along CHW
    // cs = cs_map.mean({1, 2, 3});

    // printf("mu1 : %f, mu2: %f, sigma1_sq: %f, sigma2_sq: %f, sigma12 %f\n",
    // mu1.sum().item().toFloat(),
    // mu2.sum().item().toFloat(),
    // sigma1_sq.sum().item().toFloat(),
    // sigma2_sq.sum().item().toFloat(),
    // sigma12.sum().item().toFloat());

    return ssim_val;
}

void CpyCfgFileToOut(std::string dir, string srcPath)
{
    if (!std::filesystem::exists(dir)) {
        if (std::filesystem::create_directory(dir)) {
            std::cout << "model save path create success" << std::endl;
        } else {
            std::cout << "model save path is not valid!\n";
            return;
        }
    }

    try {
        std::filesystem::copy(srcPath, dir, std::filesystem::copy_options::overwrite_existing);
    }

    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "copy file err: " << e.what() << '\n';
        exit(0);
    }

}

// Signal handler for Ctrl+C
void signalHandler(int signal) {
    static bool isCtrlCPressed = false;
    if (isCtrlCPressed) {
        exit(0);
    }

    ImMeshRenderer& render = ImMeshRenderer::GetInstance();

    if (render.m_algoConfig.isRenderingMode) {
        std::cout << "Validation mode is enabled. Ctrl+C will not save data." << std::endl;
        return;
    }

    isCtrlCPressed = true;

    std::cout << "\n---------- Ctrl+C detected! Saving data..." << std::endl;

    render.m_threadRunning.store(false);

    sleep(1);

    saveHashMapData(render.m_trainThrdCurTrainCount);
}


int main(void)
{
    size_t perFrameMaxStackSize = 0;
    checkCudaErrors(cudaDeviceGetLimit(&perFrameMaxStackSize, cudaLimitStackSize));

    std::cout << "loss sum perFrameMaxStackSize : " << perFrameMaxStackSize << std::endl; 

    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 2048));

    checkCudaErrors(cudaDeviceGetLimit(&perFrameMaxStackSize, cudaLimitStackSize));

    std::cout << "update get perFrameMaxStackSize : " << perFrameMaxStackSize << std::endl; 

    ImMeshRenderer& render = ImMeshRenderer::GetInstance();

    // Set up signal handler for Ctrl+C to save data
    std::signal(SIGINT, signalHandler);
    std::cout << "Signal handler set up. Press Ctrl+C to save m_shTexHashMap data." << std::endl;

    render.m_taskStartTime = GetCurrentTimeAsString();
    // FileConfig fileConfig;
    // AlgoConfig algoConfig;
    CameraConfig camGtConfig;

    ReadConfigs("config/configFile.json", render.m_camPosWorld, render.m_gtImages, &render.m_viewMatrix, render.m_poseConfidenceVec, render.m_fileConfig, render.m_algoConfig, camGtConfig);

    CpyCfgFileToOut(render.m_algoConfig.saveDir + "/" + render.m_taskStartTime + "_" + render.m_algoConfig.experimentName + "/", render.m_fileConfig.algoConfigPath);

    cout << "read alg config highOrderSHLrMultiplier " << render.m_algoConfig.highOrderSHLrMultiplier << " gaussianCoeffEWA " << render.m_algoConfig.gaussianCoeffEWA << "shDensity " << render.m_algoConfig.shDensity << " shDensityMax :" << render.m_algoConfig.shDensityMax << endl;

    std::string logDir = render.m_algoConfig.tensorboardDir + render.m_taskStartTime + "_" + render.m_algoConfig.experimentName + "/";
    // nlptk::Recorder recorder(logDir);
    render.m_recorderPtr = std::make_unique<nlptk::Recorder>(logDir);

    if (!render.m_recorderPtr->Ready()) {
        cout << "Failed to initialize tensorboard recorder. Exit!";
        return -1;
    }

    cout << "traj pose size : " << render.m_camPosWorld.size() << endl;

    
    render.SetConfig(camGtConfig.fx * camGtConfig.scaleFactor,
        camGtConfig.fy * camGtConfig.scaleFactor,
        camGtConfig.cx * camGtConfig.scaleFactor,
        camGtConfig.cy * camGtConfig.scaleFactor,
        render.m_algoConfig.shOrder, (float)render.m_algoConfig.highOrderSHLrMultiplier, render.m_algoConfig.printPixX, render.m_algoConfig.printPixY, render.m_algoConfig.printGradTexId, render.m_algoConfig.schedulerPatience, (float)render.m_algoConfig.invDistFactorEdge, (float)render.m_algoConfig.invDistFactorCorner, (float)render.m_algoConfig.gaussianCoeffEWA, (float)render.m_algoConfig.shDensity, render.m_algoConfig.gtW, render.m_algoConfig.gtH, render.m_algoConfig.texResizeInterval, camGtConfig, (float)render.m_algoConfig.varThresholdPSNR, (float)render.m_algoConfig.resizeDepthThreshold, (float)render.m_algoConfig.densityUpdateStep, (float)render.m_algoConfig.densityUpdateStepInner, render.m_algoConfig.renderW, render.m_algoConfig.renderH, render.m_algoConfig.testInterval, (float)render.m_algoConfig.L1lossThreshold, render.m_algoConfig.resizePatience, render.m_algoConfig.trainPoseCount, render.m_algoConfig.maxTexWH, render.m_algoConfig.resizeStartStep, render.m_algoConfig.resizeReturnCntMax, (float)render.m_algoConfig.resizeConvergeThreshold, (float)render.m_algoConfig.resizeTriangleEdgeMinLen, (float)render.m_algoConfig.shDensityMax);

    if (render.m_algoConfig.isRenderingMode == true) {
        bool retVal = render.ReadValidationSequence(render.m_fileConfig.sceneType, render.m_algoConfig.RenderingPosePath, &render.m_viewMatrixRenderingMode, render.m_camPosWorldRenderingMode);
        if (retVal == false) {
            cout << "read validation sequence failed, exit!" << endl;
            exit(0);
        }


    } else {
        render.m_needInfer = ReadTestSequence(render.m_fileConfig.sceneType, render.m_algoConfig.testSequencePath, render.m_algoConfig.testGtimgPath, &render.m_viewMatrixInfer, render.m_camPosWorldInfer, render.m_gtImagesInfer);

        if (render.m_algoConfig.evalSequencePath == render.m_algoConfig.gtPosePath and
            render.m_algoConfig.evalImgPath == render.m_algoConfig.gtImageDir) {
            printf("------use train image as eval---------- \n");
            render.m_viewMatrixEval = render.m_viewMatrix;
            render.m_camPosWorldEval = render.m_camPosWorld;
            render.m_gtImagesEval = render.m_gtImages;
        } else {
            ReadTestSequence(render.m_fileConfig.sceneType, render.m_algoConfig.evalSequencePath, render.m_algoConfig.evalImgPath, &render.m_viewMatrixEval, render.m_camPosWorldEval, render.m_gtImagesEval);
        }
        

        if (render.m_needInfer == true) {
            cout << "will start infer task, pose num: " << render.m_camPosWorldInfer.size() << endl;
        }
    }

    render.EGLInit();

    render.WindowInit();

    if (render.m_algoConfig.isRenderingMode) {
        render.LoadHashMapData(render.m_algoConfig.savedBinPath, render.m_algoConfig.savedJsonPath);
    }

    render.InitVerts(render.m_algoConfig.meshPath, render.m_fileConfig.meshType);


    render.CudaInit();

    // render.DrawPrepare(render.m_fileConfig.sceneType);

    render.ShTextureTensorInitCUDADummy();

    render.InitRenderedResultTensor();

    // render.MapIntermediateTextures();

    // render.InitThreadBuffers();

    if (render.m_algoConfig.isRenderingMode) {
        render.Rendering(render.m_algoConfig.saveDir + "/" + render.m_taskStartTime + "_rendering_mode/");
    } else {
        render.ThreadStart();

        while (1)
        {
            printf("Mesh Learner Running ...\n");
            sleep(2);

            ImMeshRenderer& render = ImMeshRenderer::GetInstance();

            // if (render.m_threadRunning == false) {
            if (render.m_threadRunning.load() == false) {
                break;
            }


            size_t freeMem, totalMem;
        
            // Returned memory in bytes
            cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);

            printf("free mem: %d MB, total: %d MB\n", (uint32_t)(freeMem / 1024 / 1024), (uint32_t)(totalMem / 1024 / 1024));
        }
    }

    google::protobuf::ShutdownProtobufLibrary();
    
    return 0;
}


