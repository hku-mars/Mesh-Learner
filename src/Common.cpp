#include "include/Common.hpp"

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