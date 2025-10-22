// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include "PTexLib.h"

#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/picojson.h>
#include <Eigen/Geometry>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <experimental/filesystem>
#include <fstream>


#include "Assert.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <experimental/filesystem>
#include <fstream>
#include <set>
#include "omp.h"

#include <iomanip>

#include <glm/glm.hpp>

namespace cg = cooperative_groups;



// void PTexMesh::PLYParse(const std::string& filename)
// {
//   std::vector<std::string> comments;
//   std::vector<std::string> objInfo;

//   std::string lastElement;
//   std::string lastProperty;

//   enum Properties { POSITION = 0, NORMAL, COLOR, NUM_PROPERTIES };

//   size_t numVertices = 0;

//   size_t positionDimensions = 0;
//   size_t normalDimensions = 0;
//   size_t colorDimensions = 0;

//   std::vector<Properties> vertexLayout;

//   size_t numFaces = 0;

//   std::ifstream file(filename, std::ios::binary);

//   // Header parsing
//   {
//     std::string line;

//     while (std::getline(file, line)) {
//       std::istringstream ls(line);
//       std::string token;
//       ls >> token;

//       if (token == "ply" || token == "PLY" || token == "") {
//         // Skip preamble line
//         continue;
//       } else if (token == "comment") {
//         // Just store these incase
//         comments.push_back(line.erase(0, 8));
//       } else if (token == "format") {
//         // We can only parse binary data, so check that's what it is
//         std::string s;
//         ls >> s;
//         ASSERT(
//             s == "binary_little_endian",
//             "Can only parse binary files... why are you using ASCII anyway?");
//       } else if (token == "element") {
//         std::string name;
//         size_t size;
//         ls >> name >> size;

//         if (name == "vertex") {
//           // Pull out the number of vertices
//           numVertices = size;
//         } else if (name == "face") {
//           // Pull out number of faces
//           numFaces = size;
//         } else {
//           ASSERT(false, "Can't parse element (%)", name);
//         }

//         // Keep track of what element we parsed last to associate the properties that follow
//         lastElement = name;
//       } else if (token == "property") {
//         std::string type, name;
//         ls >> type;

//         // Special parsing for list properties (e.g. faces)
//         bool isList = false;

//         if (type == "list") {
//           isList = true;

//           std::string countType;
//           ls >> countType >> type;

//           ASSERT(
//               countType == "uchar" || countType == "uint8",
//               "Don't understand count type (%)",
//               countType);

//           ASSERT(type == "int" || type == "uint", "Don't understand index type (%)", type);

//           ASSERT(
//               lastElement == "face",
//               "Only expecting list after face element, not after (%)",
//               lastElement);
//         }

//         ASSERT(
//             type == "float" || type == "int" || type == "uint" || type == "uchar" || type == "uint8",
//             "Don't understand type (%)",
//             type);

//         ls >> name;

//         // Collecting vertex property information
//         if (lastElement == "vertex") {
//           ASSERT(type != "int", "Don't support 32-bit integer properties");

//           // Position information
//           if (name == "x") {
//             positionDimensions = 1;
//             vertexLayout.push_back(Properties::POSITION);
//             ASSERT(type == "float", "Don't support 8-bit integer positions");
//           } else if (name == "y") {
//             ASSERT(lastProperty == "x", "Properties should follow x, y, z, (w) order");
//             positionDimensions = 2;
//           } else if (name == "z") {
//             ASSERT(lastProperty == "y", "Properties should follow x, y, z, (w) order");
//             positionDimensions = 3;
//           } else if (name == "w") {
//             ASSERT(lastProperty == "z", "Properties should follow x, y, z, (w) order");
//             positionDimensions = 4;
//           }

//           // Normal information
//           if (name == "nx") {
//             normalDimensions = 1;
//             vertexLayout.push_back(Properties::NORMAL);
//             ASSERT(type == "float", "Don't support 8-bit integer normals");
//           } else if (name == "ny") {
//             ASSERT(lastProperty == "nx", "Properties should follow nx, ny, nz order");
//             normalDimensions = 2;
//           } else if (name == "nz") {
//             ASSERT(lastProperty == "ny", "Properties should follow nx, ny, nz order");
//             normalDimensions = 3;
//           }

//           // Color information
//           if (name == "red") {
//             colorDimensions = 1;
//             vertexLayout.push_back(Properties::COLOR);
//             ASSERT(type == "uchar" || type == "uint8", "Don't support non-8-bit integer colors");
//           } else if (name == "green") {
//             ASSERT(
//                 lastProperty == "red", "Properties should follow red, green, blue, (alpha) order");
//             colorDimensions = 2;
//           } else if (name == "blue") {
//             ASSERT(
//                 lastProperty == "green",
//                 "Properties should follow red, green, blue, (alpha) order");
//             colorDimensions = 3;
//           } else if (name == "alpha") {
//             ASSERT(
//                 lastProperty == "blue", "Properties should follow red, green, blue, (alpha) order");
//             colorDimensions = 4;
//           }
//         } else if (lastElement == "face") {
//           ASSERT(isList, "No idea what to do with properties following faces");
//         } else {
//           ASSERT(false, "No idea what to do with properties before elements");
//         }

//         lastProperty = name;
//       } else if (token == "obj_info") {
//         // Just store these incase
//         objInfo.push_back(line.erase(0, 9));
//       } else if (token == "end_header") {
//         // Done reading!
//         break;
//       } else {
//         // Something unrecognised
//         ASSERT(false);
//       }
//     }

//     // Check things make sense.
//     ASSERT(numVertices > 0);
//     ASSERT(positionDimensions > 0);
//   }

//   m_originalMesh.vbo.Reinitialise(numVertices, 1);
//   m_originalMesh.vbo.Fill(Eigen::Vector4f(0, 0, 0, 1));

//   if (normalDimensions) {
//     m_originalMesh.nbo.Reinitialise(numVertices, 1);
//     m_originalMesh.nbo.Fill(Eigen::Vector3f(0, 0, 0));
//   }

//   if (colorDimensions) {
//     m_originalMesh.cbo.Reinitialise(numVertices, 1);
//     m_originalMesh.cbo.Fill(Eigen::Matrix<unsigned char, 4, 1>(0, 0, 0, 255));
//   }

//   // Can only be FLOAT32 or UINT8
//   const size_t positionBytes = positionDimensions * sizeof(float); // floats
//   const size_t normalBytes = normalDimensions * sizeof(float); // floats
//   const size_t colorBytes = colorDimensions * sizeof(uint8_t); // bytes

//   const size_t vertexPacketSizeBytes = positionBytes + normalBytes + colorBytes;

//   size_t positionOffsetBytes = 0;
//   size_t normalOffsetBytes = 0;
//   size_t colorOffsetBytes = 0;

//   size_t offsetSoFarBytes = 0;

//   for (size_t i = 0; i < vertexLayout.size(); i++) {
//     if (vertexLayout[i] == Properties::POSITION) {
//       positionOffsetBytes = offsetSoFarBytes;
//       offsetSoFarBytes += positionBytes;
//     } else if (vertexLayout[i] == Properties::NORMAL) {
//       normalOffsetBytes = offsetSoFarBytes;
//       offsetSoFarBytes += normalBytes;
//     } else if (vertexLayout[i] == Properties::COLOR) {
//       colorOffsetBytes = offsetSoFarBytes;
//       offsetSoFarBytes += colorBytes;
//     } else {
//       ASSERT(false);
//     }
//   }

//   // Close after parsing header and re-open memory mapped
//   const size_t postHeader = file.tellg();

//   file.close();

//   const size_t fileSize = std::experimental::filesystem::v1::file_size(filename);

//   int fd = open(filename.c_str(), O_RDONLY, 0);
//   void* mmappedData = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);

//   // Parse each vertex packet and unpack
//   char* bytes = &(((char*)mmappedData)[postHeader]);

//   for (size_t i = 0; i < numVertices; i++) {
//     char* nextBytes = &bytes[vertexPacketSizeBytes * i];

//     memcpy(m_originalMesh.vbo[i].data(), &nextBytes[positionOffsetBytes], positionBytes);

//     if (normalDimensions)
//       memcpy(m_originalMesh.nbo[i].data(), &nextBytes[normalOffsetBytes], normalBytes);

//     if (colorDimensions)
//       memcpy(m_originalMesh.cbo[i].data(), &nextBytes[colorOffsetBytes], colorBytes);
//   }

//   const size_t bytesSoFar = postHeader + vertexPacketSizeBytes * numVertices;

//   bytes = &(((char*)mmappedData)[postHeader + vertexPacketSizeBytes * numVertices]);

//   if (numFaces > 0) {
//     // Read first face to get number of indices;
//     const uint8_t faceDimensions = *bytes;

//     ASSERT(faceDimensions == 3 || faceDimensions == 4);

//     // scanNet++ is 3
//     std::cout << "get vertex number of each mesh " << faceDimensions << std::endl;

//     // exit(0);

//     const size_t countBytes = 1;
//     const size_t faceBytes = faceDimensions * sizeof(uint32_t); // uint32_t
//     const size_t facePacketSizeBytes = countBytes + faceBytes;

//     const size_t predictedFaces = (fileSize - bytesSoFar) / facePacketSizeBytes;

//     // Not sure what to do here
//     //    if(predictedFaces < numFaces)
//     //    {
//     //        std::cout << "Skipping " << numFaces - predictedFaces << " missing faces" <<
//     //        std::endl;
//     //    }
//     //    else if(numFaces < predictedFaces)
//     //    {
//     //        std::cout << "Ignoring " << predictedFaces - numFaces << " extra faces" << std::endl;
//     //    }

//     numFaces = std::min(numFaces, predictedFaces);

//     m_originalMesh.ibo.Reinitialise(numFaces * faceDimensions, 1);

//     for (size_t i = 0; i < numFaces; i++) {
//       char* nextBytes = &bytes[facePacketSizeBytes * i];

//       memcpy(&m_originalMesh.ibo[i * faceDimensions], &nextBytes[countBytes], faceBytes);
//     }

//     m_originalMesh.polygonStride = faceDimensions;
//   } else {
//     m_originalMesh.polygonStride = 0;
//   }

//   munmap(mmappedData, fileSize);

//   close(fd);
// }

void PTexMesh::PLYParseJianHengLiu(const std::string& filename)
{
  std::vector<std::string> comments;
  std::vector<std::string> objInfo;

  std::string lastElement;
  std::string lastProperty;

  enum Properties { POSITION = 0, NORMAL, COLOR, NUM_PROPERTIES };

  size_t numVertices = 0;

  size_t positionDimensions = 0;
  size_t normalDimensions = 0;
  size_t colorDimensions = 0;

  std::vector<Properties> vertexLayout;

  size_t numFaces = 0;

  std::ifstream file(filename, std::ios::binary);

  // Header parsing
  {
    std::string line;

    while (std::getline(file, line)) {
      std::istringstream ls(line);
      std::string token;
      ls >> token;

      if (token == "ply" || token == "PLY" || token == "") {
        // Skip preamble line
        continue;
      } else if (token == "comment") {
        // Just store these incase
        comments.push_back(line.erase(0, 8));
      } else if (token == "format") {
        // We can only parse binary data, so check that's what it is
        std::string s;
        ls >> s;
        ASSERT(
            s == "binary_little_endian",
            "Can only parse binary files... why are you using ASCII anyway?");
      } else if (token == "element") {
        std::string name;
        size_t size;
        ls >> name >> size;

        if (name == "vertex") {
          // Pull out the number of vertices
          numVertices = size;
        } else if (name == "face") {
          // Pull out number of faces
          numFaces = size;
        } else {
          ASSERT(false, "Can't parse element (%)", name);
        }

        // Keep track of what element we parsed last to associate the properties that follow
        lastElement = name;
      } else if (token == "property") {
        std::string type, name;
        ls >> type;

        // Special parsing for list properties (e.g. faces)
        bool isList = false;

        if (type == "list") {
          isList = true;

          std::string countType;
          ls >> countType >> type;

          ASSERT(
              countType == "int",
              "Don't understand count type (%)",
              countType);

          ASSERT(type == "int", "Don't understand index type (%)", type);

          ASSERT(
              lastElement == "face",
              "Only expecting list after face element, not after (%)",
              lastElement);
        }

        ASSERT(
            type == "float" || type == "int" || type == "uint" || type == "uchar" || type == "uint8",
            "Don't understand type (%)",
            type);

        ls >> name;

        // Collecting vertex property information
        if (lastElement == "vertex") {
          ASSERT(type != "int", "Don't support 32-bit integer properties");

          // Position information
          if (name == "x") {
            positionDimensions = 1;
            vertexLayout.push_back(Properties::POSITION);
            ASSERT(type == "float", "Don't support 8-bit integer positions");
          } else if (name == "y") {
            ASSERT(lastProperty == "x", "Properties should follow x, y, z, (w) order");
            positionDimensions = 2;
          } else if (name == "z") {
            ASSERT(lastProperty == "y", "Properties should follow x, y, z, (w) order");
            positionDimensions = 3;
          } else if (name == "w") {
            ASSERT(lastProperty == "z", "Properties should follow x, y, z, (w) order");
            positionDimensions = 4;
          }

          // Normal information
          if (name == "nx") {
            normalDimensions = 1;
            vertexLayout.push_back(Properties::NORMAL);
            ASSERT(type == "float", "Don't support 8-bit integer normals");
          } else if (name == "ny") {
            ASSERT(lastProperty == "nx", "Properties should follow nx, ny, nz order");
            normalDimensions = 2;
          } else if (name == "nz") {
            ASSERT(lastProperty == "ny", "Properties should follow nx, ny, nz order");
            normalDimensions = 3;
          }

          // Color information
          if (name == "red") {
            colorDimensions = 1;
            vertexLayout.push_back(Properties::COLOR);
            ASSERT(type == "uchar", "Don't support non-8-bit integer colors");
          } else if (name == "green") {
            ASSERT(
                lastProperty == "red", "Properties should follow red, green, blue, (alpha) order");
            colorDimensions = 2;
          } else if (name == "blue") {
            ASSERT(
                lastProperty == "green",
                "Properties should follow red, green, blue, (alpha) order");
            colorDimensions = 3;
          } else if (name == "alpha") {
            ASSERT(
                lastProperty == "blue", "Properties should follow red, green, blue, (alpha) order");
            colorDimensions = 4;
          }
        } else if (lastElement == "face") {
          ASSERT(isList, "No idea what to do with properties following faces");
        } else {
          ASSERT(false, "No idea what to do with properties before elements");
        }

        lastProperty = name;
      } else if (token == "obj_info") {
        // Just store these incase
        objInfo.push_back(line.erase(0, 9));
      } else if (token == "end_header") {
        // Done reading!
        break;
      } else {
        // Something unrecognised
        ASSERT(false);
      }
    }

    // Check things make sense.
    ASSERT(numVertices > 0);
    ASSERT(positionDimensions == 3);
    ASSERT(colorDimensions == 3);
  }

  // m_originalMesh.vbo.Reinitialise(numVertices, 1);
  m_originalMesh.m_vboVec.resize(numVertices * 3);
  memset(m_originalMesh.m_vboVec.data(), 0, sizeof(float) * numVertices * 3);
  // m_originalMesh.vbo.Fill(Eigen::Vector4f(0, 0, 0, 1));

  if (normalDimensions) {
    // m_originalMesh.nbo.Reinitialise(numVertices, 1);
    // m_originalMesh.nbo.Fill(Eigen::Vector3f(0, 0, 0));
  }

  if (colorDimensions) {
    // m_originalMesh.cbo.Reinitialise(numVertices, 1);
    m_originalMesh.m_cboVec.resize(numVertices * 3);
    memset(m_originalMesh.m_cboVec.data(), 0, sizeof(unsigned char) * numVertices * 3);
    // m_originalMesh.cbo.Fill(Eigen::Matrix<unsigned char, 4, 1>(0, 0, 0, 255));
  }

  // Can only be FLOAT32 or UINT8
  const size_t positionBytes = positionDimensions * sizeof(float); // floats
  const size_t normalBytes = normalDimensions * sizeof(float); // floats
  const size_t colorBytes = colorDimensions * sizeof(uint8_t); // bytes

  const size_t vertexPacketSizeBytes = positionBytes + normalBytes + colorBytes;

  size_t positionOffsetBytes = 0;
  size_t normalOffsetBytes = 0;
  size_t colorOffsetBytes = 0;

  size_t offsetSoFarBytes = 0;

  for (size_t i = 0; i < vertexLayout.size(); i++) {
    if (vertexLayout[i] == Properties::POSITION) {
      positionOffsetBytes = offsetSoFarBytes;
      offsetSoFarBytes += positionBytes;
    } else if (vertexLayout[i] == Properties::NORMAL) {
      normalOffsetBytes = offsetSoFarBytes;
      offsetSoFarBytes += normalBytes;
    } else if (vertexLayout[i] == Properties::COLOR) {
      colorOffsetBytes = offsetSoFarBytes;
      offsetSoFarBytes += colorBytes;
    } else {
      ASSERT(false);
    }
  }

  // Close after parsing header and re-open memory mapped
  const size_t postHeader = file.tellg();

  file.close();

  const size_t fileSize = std::experimental::filesystem::v1::file_size(filename);

  int fd = open(filename.c_str(), O_RDONLY, 0);
  void* mmappedData = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);

  // Parse each vertex packet and unpack
  char* bytes = &(((char*)mmappedData)[postHeader]);

  for (size_t i = 0; i < numVertices; i++) {
    char* nextBytes = &bytes[vertexPacketSizeBytes * i];

    // memcpy(m_originalMesh.vbo[i].data(), &nextBytes[positionOffsetBytes], positionBytes);
    memcpy((char *)(((float *)m_originalMesh.m_vboVec.data()) + i * positionDimensions), &nextBytes[positionOffsetBytes], positionBytes);
    

    if (normalDimensions)
      // memcpy(m_originalMesh.nbo[i].data(), &nextBytes[normalOffsetBytes], normalBytes);

    if (colorDimensions)
      // memcpy(m_originalMesh.cbo[i].data(), &nextBytes[colorOffsetBytes], colorBytes);
      memcpy((((unsigned char *)m_originalMesh.m_cboVec.data()) + i * 3), &nextBytes[colorOffsetBytes], colorBytes);
  }

  const size_t bytesSoFar = postHeader + vertexPacketSizeBytes * numVertices;

  bytes = &(((char*)mmappedData)[postHeader + vertexPacketSizeBytes * numVertices]);

  if (numFaces > 0) {
    // Read first face to get number of indices;
    const int32_t faceDimensions = *((int32_t *)bytes);

    // ASSERT(faceDimensions == 3 || faceDimensions == 4);
    ASSERT(faceDimensions == 3);

    // scanNet++ is 3
    std::cout << "get vertex number of each mesh " << faceDimensions << std::endl;

    // exit(0);

    const size_t countBytes = 4;
    const size_t faceBytes = faceDimensions * sizeof(int32_t); // uint32_t
    const size_t facePacketSizeBytes = countBytes + faceBytes;

    const size_t predictedFaces = (fileSize - bytesSoFar) / facePacketSizeBytes;

    // Not sure what to do here
    //    if(predictedFaces < numFaces)
    //    {
    //        std::cout << "Skipping " << numFaces - predictedFaces << " missing faces" <<
    //        std::endl;
    //    }
    //    else if(numFaces < predictedFaces)
    //    {
    //        std::cout << "Ignoring " << predictedFaces - numFaces << " extra faces" << std::endl;
    //    }

    std::cout << "actual predictedFaces " << predictedFaces << " infile numFaces: " << numFaces << std::endl;
    numFaces = std::min(numFaces, predictedFaces);

    // m_originalMesh.iboInt32.Reinitialise(numFaces * faceDimensions, 1);
    m_originalMesh.m_iboVec.resize(numFaces * faceDimensions);
    // m_originalMesh.m_faceNormal.Reinitialise(numFaces, 1);
    m_originalMesh.m_faceNormalVec.resize(numFaces * 3);

    for (size_t i = 0; i < numFaces; i++) {
      char* nextBytes = &bytes[facePacketSizeBytes * i];

      memcpy(&m_originalMesh.m_iboVec[i * faceDimensions], &nextBytes[countBytes], faceBytes);

      glm::dvec3 pt0(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 0],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 1],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 2]);

      glm::dvec3 pt1(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 0],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 1],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 2]);

      glm::dvec3 pt2(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 0],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 1],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 2]);
      
      // Eigen::Vector3f pt1 = m_originalMesh.vbo[m_originalMesh.m_iboVec[i * faceDimensions + 1]].head(3);
      
      // Eigen::Vector3f pt2 = m_originalMesh.vbo[m_originalMesh.m_iboVec[i * faceDimensions + 2]].head(3);

      glm::dvec3 vec01 = glm::normalize(pt1 - pt0);
      glm::dvec3 vec12 = glm::normalize(pt2 - pt1);

      glm::dvec3 crossedDouble1 = glm::cross(vec01, vec12);
      double realLen1 = GetAccurateLength(crossedDouble1);

      glm::dvec3 normal1(crossedDouble1.x / realLen1, crossedDouble1.y / realLen1, crossedDouble1.z / realLen1);

      // m_originalMesh.m_faceNormal[i] = (vec01.cross(vec12)).normalized();
      m_originalMesh.m_faceNormalVec[i * 3 + 0] = normal1.x;
      m_originalMesh.m_faceNormalVec[i * 3 + 1] = normal1.y;
      m_originalMesh.m_faceNormalVec[i * 3 + 2] = normal1.z;
    }

    m_originalMesh.polygonStride = faceDimensions;
  } else {
    m_originalMesh.polygonStride = 0;
  }

  munmap(mmappedData, fileSize);

  close(fd);
}

void PTexMesh::PLYParseBlenderOut(const std::string& filename)
{
    size_t numVertices = 0;
    size_t numFaces = 0;
    size_t numEdges = 0;

    size_t positionDimensions = 0;
    size_t normalDimensions = 0;

    enum Properties { POSITION = 0, NORMAL, COLOR, NUM_PROPERTIES };
    std::vector<Properties> vertexLayout;

    std::string lastElement;
    std::string lastProperty;

    std::ifstream file(filename, std::ios::binary);

    std::string line;

    while (std::getline(file, line)) {
        std::istringstream ls(line);
        std::string token;
        ls >> token;

        if (token == "ply" || token == "PLY" || token == "") {
            // Skip preamble line
            continue;
        } else if (token == "format") {
            std::string s;
            ls >> s;
            ASSERT(
                s == "binary_little_endian",
                "Can only parse binary files... why are you using ASCII anyway?");
        } else if (token == "element") {
            std::string name;
            size_t size;
            ls >> name >> size;

            if (name == "vertex") {
                numVertices = size;
            } else if (name == "face") {
              // Pull out number of faces
              numFaces = size;
            } else if (name == "edge") {
              numEdges = size;
            } else {
              ASSERT(false, "Can't parse element (%)", name);
            }

            // Keep track of what element we parsed last to associate the properties that follow
            lastElement = name;
        } else if (token == "property") {
            std::string type, name;
            ls >> type;

            bool isList = false;
            if (type == "list") {
              isList = true;

              std::string countType;
              ls >> countType >> type;

              ASSERT(
                  countType == "uchar" || countType == "uint8",
                  "Don't understand count type (%)",
                  countType);

              ASSERT(type == "int" || type == "uint", "Don't understand index type (%)", type);

              ASSERT(
                  lastElement == "face",
                  "Only expecting list after face element, not after (%)",
                  lastElement);
            }

            ASSERT(
            type == "float" || type == "int" || type == "uint" || type == "uchar" || type == "uint8",
            "Don't understand type (%)",
            type);

            ls >> name;

            if (lastElement == "vertex") {
                ASSERT(type != "int", "Don't support 32-bit integer properties");

                if (name == "x") {
                  positionDimensions = 1;
                  vertexLayout.push_back(Properties::POSITION);
                  ASSERT(type == "float", "Don't support 8-bit integer positions");
                } else if (name == "y") {
                  ASSERT(lastProperty == "x", "Properties should follow x, y, z, (w) order");
                  positionDimensions = 2;
                } else if (name == "z") {
                  ASSERT(lastProperty == "y", "Properties should follow x, y, z, (w) order");
                  positionDimensions = 3;
                } else if (name == "w") {
                  ASSERT(lastProperty == "z", "Properties should follow x, y, z, (w) order");
                  positionDimensions = 4;
                }

                if (name == "nx") {
                  normalDimensions = 1;
                  vertexLayout.push_back(Properties::NORMAL);
                  ASSERT(type == "float", "Don't support 8-bit integer normals");
                } else if (name == "ny") {
                  ASSERT(lastProperty == "nx", "Properties should follow nx, ny, nz order");
                  normalDimensions = 2;
                } else if (name == "nz") {
                  ASSERT(lastProperty == "ny", "Properties should follow nx, ny, nz order");
                  normalDimensions = 3;
                }

            } else if (lastElement == "face") {
              ASSERT(isList, "No idea what to do with properties following faces");
            } else if (lastElement == "edge") {
              continue;
            } else {
              ASSERT(false, "No idea what to do with properties before elements");
            }

            lastProperty = name;
        } else if (token == "comment") {
          continue;
        } else if (token == "end_header") {
          break;
        } else {
          // Something unrecognised
          ASSERT(false);
        }
    }

    ASSERT(numVertices > 0);
    ASSERT(positionDimensions == 3);
    ASSERT(normalDimensions == 0);

    // m_originalMesh.vbo.Reinitialise(numVertices, 1);
    // m_originalMesh.vbo.Fill(Eigen::Vector4f(0, 0, 0, 1));

    m_originalMesh.m_vboVec.resize(numVertices * 3);
    memset(m_originalMesh.m_vboVec.data(), 0, sizeof(float) * numVertices * 3);

    // m_originalMesh.m_ptIdx2FaceInfoHashMap.resize(numVertices);

    // for (size_t i = 0; i < numVertices; i++) {
    //   m_originalMesh.m_ptIdx2FaceInfoHashMap[i].faceIdx.resize(0);
    //   m_originalMesh.m_ptIdx2FaceInfoHashMap[i].areaSum = 0.f;
    // }

    if (normalDimensions) {
      // m_originalMesh.nbo.Reinitialise(numVertices, 1);
      // m_originalMesh.nbo.Fill(Eigen::Vector3f(0, 0, 0));
    }

    const size_t positionBytes = positionDimensions * sizeof(float); // floats
    const size_t normalBytes = normalDimensions * sizeof(float); // floats

    const size_t vertexPacketSizeBytes = positionBytes + normalBytes;
  
    size_t positionOffsetBytes = 0;
    size_t normalOffsetBytes = 0;

    size_t offsetSoFarBytes = 0;

    for (size_t i = 0; i < vertexLayout.size(); i++) {
    if (vertexLayout[i] == Properties::POSITION) {
      positionOffsetBytes = offsetSoFarBytes;
      offsetSoFarBytes += positionBytes;
    } else if (vertexLayout[i] == Properties::NORMAL) {
      normalOffsetBytes = offsetSoFarBytes;
      offsetSoFarBytes += normalBytes;
    } else if (vertexLayout[i] == Properties::COLOR) {
    } else {
      ASSERT(false);
    }
  }

  const size_t postHeader = file.tellg();
  file.close();

  const size_t fileSize = std::experimental::filesystem::v1::file_size(filename);
  int fd = open(filename.c_str(), O_RDONLY, 0);
  void* mmappedData = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);

  // Parse each vertex packet and unpack
  char* bytes = &(((char*)mmappedData)[postHeader]);

  for (size_t i = 0; i < numVertices; i++) {
    char* nextBytes = &bytes[vertexPacketSizeBytes * i];

    // memcpy(m_originalMesh.vbo[i].data(), &nextBytes[positionOffsetBytes], positionBytes);
    memcpy((char *)(((float *)m_originalMesh.m_vboVec.data()) + i * positionDimensions), &nextBytes[positionOffsetBytes], positionBytes);

    if (normalDimensions) {
      // memcpy(m_originalMesh.nbo[i].data(), &nextBytes[normalOffsetBytes], normalBytes);
    }

  }

  // std::ofstream outFile("campus_outputMeshPts.txt");
  // for (int i = 0; i < numVertices; i++) {
  //   outFile << m_originalMesh.vbo[i](0) << " " << m_originalMesh.vbo[i](1) <<  " " << m_originalMesh.vbo[i](2) << std::endl;
  // }

  // outFile.close();
  // exit(0);


  const size_t bytesSoFar = postHeader + vertexPacketSizeBytes * numVertices;

  bytes = &(((char*)mmappedData)[postHeader + vertexPacketSizeBytes * numVertices]);

  if (numFaces > 0) {
    // Read first face to get number of indices;
    const uint8_t faceDimensions = *bytes;
    ASSERT(faceDimensions == 3);

    const size_t countBytes = 1;
    const size_t faceBytes = faceDimensions * sizeof(uint32_t); // uint32_t
    const size_t facePacketSizeBytes = countBytes + faceBytes;

    const size_t predictedFaces = (fileSize - bytesSoFar) / facePacketSizeBytes;

    numFaces = std::min(numFaces, predictedFaces);

    // m_originalMesh.ibo.Reinitialise(numFaces * faceDimensions, 1);
    m_originalMesh.m_iboVec.resize(numFaces * faceDimensions);

    // m_originalMesh.m_faceNormal.Reinitialise(numFaces, 1);
    m_originalMesh.m_faceNormalVec.resize(numFaces * 3);
    // m_originalMesh.m_faceArea.Reinitialise(numFaces, 1);

    for (size_t i = 0; i < numFaces; i++) {
      char* nextBytes = &bytes[facePacketSizeBytes * i];

      // memcpy(&m_originalMesh.ibo[i * faceDimensions], &nextBytes[countBytes], faceBytes);
      m_originalMesh.m_iboVec[i * faceDimensions + 0] = (int32_t)(*(((uint32_t *)(nextBytes + 1) + 0)));
      m_originalMesh.m_iboVec[i * faceDimensions + 1] = (int32_t)(*(((uint32_t *)(nextBytes + 1) + 1)));
      m_originalMesh.m_iboVec[i * faceDimensions + 2] = (int32_t)(*(((uint32_t *)(nextBytes + 1) + 2)));


      // Eigen::Vector3f pt0 = m_originalMesh.vbo[m_originalMesh.ibo[i * faceDimensions]].head(3);
      // Eigen::Vector3f pt1 = m_originalMesh.vbo[m_originalMesh.ibo[i * faceDimensions + 1]].head(3);
      // Eigen::Vector3f pt2 = m_originalMesh.vbo[m_originalMesh.ibo[i * faceDimensions + 2]].head(3);

      glm::dvec3 pt0(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 0],
                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 1],
                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 2]);

      glm::dvec3 pt1(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 0],
                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 1],
                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 2]);

      glm::dvec3 pt2(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 0],
                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 1],
                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 2]);

      glm::dvec3 vec01 = glm::normalize(pt1 - pt0);
      glm::dvec3 vec12 = glm::normalize(pt2 - pt1);

      glm::dvec3 crossedDouble1 = glm::cross(vec01, vec12);
      double realLen1 = GetAccurateLength(crossedDouble1);

      glm::dvec3 normal1(crossedDouble1.x / realLen1, crossedDouble1.y / realLen1, crossedDouble1.z / realLen1);

      m_originalMesh.m_faceNormalVec[i * 3 + 0] = normal1.x;
      m_originalMesh.m_faceNormalVec[i * 3 + 1] = normal1.y;
      m_originalMesh.m_faceNormalVec[i * 3 + 2] = normal1.z;

      // std::cout << " face " << i << " normal:" << m_originalMesh.m_faceNormal[i] << std::endl;
      // m_originalMesh.m_faceArea[i] = 0.5 * (vec01.cross(vec12)).norm();

      // m_originalMesh.m_ptIdx2FaceInfoHashMap[m_originalMesh.ibo[i * faceDimensions]].faceIdx.push_back(i);
      // m_originalMesh.m_ptIdx2FaceInfoHashMap[m_originalMesh.ibo[i * faceDimensions]].areaSum += m_originalMesh.m_faceArea[i];

      // m_originalMesh.m_ptIdx2FaceInfoHashMap[m_originalMesh.ibo[i * faceDimensions + 1]].faceIdx.push_back(i);
      // m_originalMesh.m_ptIdx2FaceInfoHashMap[m_originalMesh.ibo[i * faceDimensions + 1]].areaSum += m_originalMesh.m_faceArea[i];

      // m_originalMesh.m_ptIdx2FaceInfoHashMap[m_originalMesh.ibo[i * faceDimensions + 2]].faceIdx.push_back(i);
      // m_originalMesh.m_ptIdx2FaceInfoHashMap[m_originalMesh.ibo[i * faceDimensions + 2]].areaSum += m_originalMesh.m_faceArea[i];
    }

    m_originalMesh.polygonStride = faceDimensions;
  } else {
    m_originalMesh.polygonStride = 0;
  }

  munmap(mmappedData, fileSize);

  close(fd);

  // if (IsFaceNormalValid() == false) {
  //   std::cout << "invalid face normal! \n";
  //   exit(0);
  // }

  // for (size_t cnt = 0; cnt < numVertices; cnt++) {
  //   std::cout << std::setprecision(10) << " vert:  " << m_originalMesh.vbo[cnt](0) << " " << m_originalMesh.vbo[cnt](1) << " " << m_originalMesh.vbo[cnt](2) << " norm: " << m_originalMesh.nbo[cnt](0) << " " << m_originalMesh.nbo[cnt](1) << " " << m_originalMesh.nbo[cnt](2) << std::endl;
  // }

  // exit(0);
}

bool PTexMesh::IsFaceNormalValid()
{
  // for (size_t i = 0; i < (m_originalMesh.nbo.size()); i++) {
  //   Eigen::Vector3f normTmp(0.f, 0.f, 0.f);
  //   for (auto it: m_originalMesh.m_ptIdx2FaceInfoHashMap[i].faceIdx) {
  //     normTmp += m_originalMesh.m_faceNormal[it] * (m_originalMesh.m_faceArea[it] / m_originalMesh.m_ptIdx2FaceInfoHashMap[i].areaSum);
  //   }
  //   if (m_originalMesh.nbo[i].isApprox(normTmp, 1e-3) == false) {
  //     return false;
  //   }
  // }

  // return true;
}

void PTexMesh::PLYParseImMesh(const std::string& filename)
{
    size_t numVertices = 0;
    size_t numFaces = 0;

    size_t positionDimensions = 0;

    enum Properties { POSITION = 0, NORMAL, COLOR, NUM_PROPERTIES };
    std::vector<Properties> vertexLayout;

    std::string lastElement;
    std::string lastProperty;

    std::ifstream file(filename, std::ios::binary);

    std::string line;

    while (std::getline(file, line)) {
        std::istringstream ls(line);
        std::string token;
        ls >> token;

        if (token == "ply" || token == "PLY" || token == "") {
            // Skip preamble line
            continue;
        } else if (token == "format") {
            std::string s;
            ls >> s;
            ASSERT(
                s == "binary_little_endian",
                "Can only parse binary files... why are you using ASCII anyway?");
        } else if (token == "element") {
            std::string name;
            size_t size;
            ls >> name >> size;

            if (name == "vertex") {
                numVertices = size;
            } else if (name == "face") {
              // Pull out number of faces
              numFaces = size;
            } else {
              ASSERT(false, "Can't parse element (%)", name);
            }

            // Keep track of what element we parsed last to associate the properties that follow
            lastElement = name;
        } else if (token == "property") {
            std::string type, name;
            ls >> type;

            bool isList = false;
            if (type == "list") {
              isList = true;

              std::string countType;
              ls >> countType >> type;

              ASSERT(
                  countType == "uchar" || countType == "uint8",
                  "Don't understand count type (%)",
                  countType);

              ASSERT(type == "int" || type == "uint", "Don't understand index type (%)", type);

              ASSERT(
                  lastElement == "face",
                  "Only expecting list after face element, not after (%)",
                  lastElement);
            }

            ASSERT(
            type == "float" || type == "int" || type == "uint" || type == "uchar" || type == "uint8",
            "Don't understand type (%)",
            type);

            ls >> name;

            if (lastElement == "vertex") {
                ASSERT(type != "int", "Don't support 32-bit integer properties");

                if (name == "x") {
                  positionDimensions = 1;
                  vertexLayout.push_back(Properties::POSITION);
                  ASSERT(type == "float", "Don't support 8-bit integer positions");
                } else if (name == "y") {
                  ASSERT(lastProperty == "x", "Properties should follow x, y, z, (w) order");
                  positionDimensions = 2;
                } else if (name == "z") {
                  ASSERT(lastProperty == "y", "Properties should follow x, y, z, (w) order");
                  positionDimensions = 3;
                } else if (name == "w") {
                  ASSERT(lastProperty == "z", "Properties should follow x, y, z, (w) order");
                  positionDimensions = 4;
                }

            } else if (lastElement == "face") {
              ASSERT(isList, "No idea what to do with properties following faces");
            } else {
              ASSERT(false, "No idea what to do with properties before elements");
            }

            lastProperty = name;
        } else if (token == "comment") {
          continue;
        } else if (token == "end_header") {
          break;
        } else {
          // Something unrecognised
          ASSERT(false);
        }
    }

    ASSERT(numVertices > 0);
    ASSERT(positionDimensions > 0);

    // m_originalMesh.vbo.Reinitialise(numVertices, 1);
    // m_originalMesh.vbo.Fill(Eigen::Vector4f(0, 0, 0, 1));

    m_originalMesh.m_vboVec.resize(numVertices * 3);
    memset(m_originalMesh.m_vboVec.data(), 0, sizeof(float) * numVertices * 3);

    const size_t positionBytes = positionDimensions * sizeof(float); // floats

    const size_t vertexPacketSizeBytes = positionBytes;
    size_t positionOffsetBytes = 0;
    size_t offsetSoFarBytes = 0;

    for (size_t i = 0; i < vertexLayout.size(); i++) {
    if (vertexLayout[i] == Properties::POSITION) {
      positionOffsetBytes = offsetSoFarBytes;
      offsetSoFarBytes += positionBytes;
    } else if (vertexLayout[i] == Properties::NORMAL) {
    } else if (vertexLayout[i] == Properties::COLOR) {
    } else {
      ASSERT(false);
    }
  }

  const size_t postHeader = file.tellg();
  file.close();

  const size_t fileSize = std::experimental::filesystem::v1::file_size(filename);
  int fd = open(filename.c_str(), O_RDONLY, 0);
  void* mmappedData = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);

  // Parse each vertex packet and unpack
  char* bytes = &(((char*)mmappedData)[postHeader]);

  for (size_t i = 0; i < numVertices; i++) {
    char* nextBytes = &bytes[vertexPacketSizeBytes * i];

    // memcpy(m_originalMesh.vbo[i].data(), &nextBytes[positionOffsetBytes], positionBytes);
    memcpy((char *)(((float *)m_originalMesh.m_vboVec.data()) + i * positionDimensions), &nextBytes[positionOffsetBytes], positionBytes);
  }

  // std::ofstream outFile("campus_outputMeshPts.txt");
  // for (int i = 0; i < numVertices; i++) {
  //   outFile << m_originalMesh.vbo[i](0) << " " << m_originalMesh.vbo[i](1) <<  " " << m_originalMesh.vbo[i](2) << std::endl;
  // }

  // outFile.close();
  // exit(0);


  const size_t bytesSoFar = postHeader + vertexPacketSizeBytes * numVertices;

  bytes = &(((char*)mmappedData)[postHeader + vertexPacketSizeBytes * numVertices]);

  if (numFaces > 0) {
    // Read first face to get number of indices;
    const uint8_t faceDimensions = *bytes;
    ASSERT(faceDimensions == 3);

    const size_t countBytes = 1;
    const size_t faceBytes = faceDimensions * sizeof(int32_t); // uint32_t
    const size_t facePacketSizeBytes = countBytes + faceBytes;

    const size_t predictedFaces = (fileSize - bytesSoFar) / facePacketSizeBytes;

    numFaces = std::min(numFaces, predictedFaces);

    // m_originalMesh.ibo.Reinitialise(numFaces * faceDimensions, 1);
    m_originalMesh.m_iboVec.resize(numFaces * faceDimensions);


    // m_originalMesh.m_faceNormal.Reinitialise(numFaces, 1);
    m_originalMesh.m_faceNormalVec.resize(numFaces * 3);

    for (size_t i = 0; i < numFaces; i++) {
      char* nextBytes = &bytes[facePacketSizeBytes * i];

      // memcpy(&m_originalMesh.ibo[i * faceDimensions], &nextBytes[countBytes], faceBytes);
      m_originalMesh.m_iboVec[i * faceDimensions + 0] = *((int32_t *)(nextBytes + 1));
      m_originalMesh.m_iboVec[i * faceDimensions + 1] = *(((int32_t *)(nextBytes + 1)) + 1);
      m_originalMesh.m_iboVec[i * faceDimensions + 2] = *(((int32_t *)(nextBytes + 1)) + 2);

      Eigen::Vector3f pt0(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 0],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 1],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 2]);
      // Eigen::Vector3f pt0 = m_originalMesh.vbo[m_originalMesh.ibo[i * faceDimensions + 0]].head(3);

      Eigen::Vector3f pt1(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 0],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 1],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 2]);
      // Eigen::Vector3f pt1 = m_originalMesh.vbo[m_originalMesh.ibo[i * faceDimensions + 1]].head(3);


      Eigen::Vector3f pt2(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 0],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 1],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 2]);
      // Eigen::Vector3f pt2 = m_originalMesh.vbo[m_originalMesh.ibo[i * faceDimensions + 2]].head(3);

      Eigen::Vector3f vec01 = pt1 - pt0;
      Eigen::Vector3f vec12 = pt2 - pt1;

      m_originalMesh.m_faceNormalVec[i * 3 + 0] = ((vec01.cross(vec12)).normalized())(0);
      m_originalMesh.m_faceNormalVec[i * 3 + 1] = ((vec01.cross(vec12)).normalized())(1);
      m_originalMesh.m_faceNormalVec[i * 3 + 2] = ((vec01.cross(vec12)).normalized())(2);
      // m_originalMesh.m_faceNormal[i] = (vec01.cross(vec12)).normalized();

    }

    m_originalMesh.polygonStride = faceDimensions;
  } else {
    m_originalMesh.polygonStride = 0;
  }

  munmap(mmappedData, fileSize);

  close(fd);
}

// void PTexMesh::PLYParseDouble(const std::string& filename)
// {
//   std::vector<std::string> comments;
//   std::vector<std::string> objInfo;

//   std::string lastElement;
//   std::string lastProperty;

//   enum Properties { POSITION = 0, NORMAL, COLOR, NUM_PROPERTIES };

//   size_t numVertices = 0;

//   size_t positionDimensions = 0;
//   size_t normalDimensions = 0;
//   size_t colorDimensions = 0;

//   std::vector<Properties> vertexLayout;

//   size_t numFaces = 0;

//   std::ifstream file(filename, std::ios::binary);

//   // Header parsing
//   {
//     std::string line;

//     while (std::getline(file, line)) {
//       std::istringstream ls(line);
//       std::string token;
//       ls >> token;

//       if (token == "ply" || token == "PLY" || token == "") {
//         // Skip preamble line
//         continue;
//       } else if (token == "comment") {
//         // Just store these incase
//         comments.push_back(line.erase(0, 8));
//       } else if (token == "format") {
//         // We can only parse binary data, so check that's what it is
//         std::string s;
//         ls >> s;
//         ASSERT(
//             s == "binary_little_endian",
//             "Can only parse binary files... why are you using ASCII anyway?");
//       } else if (token == "element") {
//         std::string name;
//         size_t size;
//         ls >> name >> size;

//         if (name == "vertex") {
//           // Pull out the number of vertices
//           numVertices = size;
//         } else if (name == "face") {
//           // Pull out number of faces
//           numFaces = size;
//         } else {
//           ASSERT(false, "Can't parse element (%)", name);
//         }

//         // Keep track of what element we parsed last to associate the properties that follow
//         lastElement = name;
//       } else if (token == "property") {
//         std::string type, name;
//         ls >> type;

//         // Special parsing for list properties (e.g. faces)
//         bool isList = false;

//         if (type == "list") {
//           isList = true;

//           std::string countType;
//           ls >> countType >> type;

//           ASSERT(
//               countType == "uchar",
//               "Don't understand count type (%)",
//               countType);

//           ASSERT(type == "uint", "Don't understand index type (%)", type);

//           ASSERT(
//               lastElement == "face",
//               "Only expecting list after face element, not after (%)",
//               lastElement);
//         }

//         ASSERT(
//             type == "double" || type == "uint" || type == "uchar",
//             "Don't understand type (%)",
//             type);

//         ls >> name;

//         // Collecting vertex property information
//         if (lastElement == "vertex") {
//           ASSERT(type != "int", "Don't support 32-bit integer properties");

//           // Position information
//           if (name == "x") {
//             positionDimensions = 1;
//             vertexLayout.push_back(Properties::POSITION);
//             ASSERT(type == "double", "this func only support double type position");
//           } else if (name == "y") {
//             ASSERT(lastProperty == "x", "Properties should follow x, y, z, (w) order");
//             positionDimensions = 2;
//           } else if (name == "z") {
//             ASSERT(lastProperty == "y", "Properties should follow x, y, z, (w) order");
//             positionDimensions = 3;
//           } else if (name == "w") {
//             ASSERT(lastProperty == "z", "Properties should follow x, y, z, (w) order");
//             positionDimensions = 4;
//           }

//           // Normal information
//           if (name == "nx") {
//             normalDimensions = 1;
//             vertexLayout.push_back(Properties::NORMAL);
//             ASSERT(type == "double", "this func only support double type normal");
//           } else if (name == "ny") {
//             ASSERT(lastProperty == "nx", "Properties should follow nx, ny, nz order");
//             normalDimensions = 2;
//           } else if (name == "nz") {
//             ASSERT(lastProperty == "ny", "Properties should follow nx, ny, nz order");
//             normalDimensions = 3;
//           }

//           // Color information
//           if (name == "red") {
//             colorDimensions = 1;
//             vertexLayout.push_back(Properties::COLOR);
//             ASSERT(type == "uchar" || type == "uint8", "Don't support non-8-bit integer colors");
//           } else if (name == "green") {
//             ASSERT(
//                 lastProperty == "red", "Properties should follow red, green, blue, (alpha) order");
//             colorDimensions = 2;
//           } else if (name == "blue") {
//             ASSERT(
//                 lastProperty == "green",
//                 "Properties should follow red, green, blue, (alpha) order");
//             colorDimensions = 3;
//           } else if (name == "alpha") {
//             ASSERT(
//                 lastProperty == "blue", "Properties should follow red, green, blue, (alpha) order");
//             colorDimensions = 4;
//           }
//         } else if (lastElement == "face") {
//           ASSERT(isList, "No idea what to do with properties following faces");
//         } else {
//           ASSERT(false, "No idea what to do with properties before elements");
//         }

//         lastProperty = name;
//       } else if (token == "obj_info") {
//         // Just store these incase
//         objInfo.push_back(line.erase(0, 9));
//       } else if (token == "end_header") {
//         // Done reading!
//         break;
//       } else {
//         // Something unrecognised
//         ASSERT(false);
//       }
//     }

//     // Check things make sense.
//     ASSERT(numVertices > 0);
//     ASSERT(positionDimensions > 0);
//   }

//   m_originalMesh.vboDouble.Reinitialise(numVertices, 1);
//   m_originalMesh.vboDouble.Fill(Eigen::Vector4d(0, 0, 0, 1));

//   if (normalDimensions) {
//     m_originalMesh.nboDouble.Reinitialise(numVertices, 1);
//     m_originalMesh.nboDouble.Fill(Eigen::Vector4d(0, 0, 0, 1));
//   }

//   if (colorDimensions) {
//     m_originalMesh.cbo.Reinitialise(numVertices, 1);
//     m_originalMesh.cbo.Fill(Eigen::Matrix<unsigned char, 4, 1>(0, 0, 0, 255));
//   }

//   // Can only be FLOAT32 or UINT8
//   const size_t positionBytes = positionDimensions * sizeof(double); // floats
//   const size_t normalBytes = normalDimensions * sizeof(double); // floats
//   const size_t colorBytes = colorDimensions * sizeof(uint8_t); // bytes

//   const size_t vertexPacketSizeBytes = positionBytes + normalBytes + colorBytes;

//   size_t positionOffsetBytes = 0;
//   size_t normalOffsetBytes = 0;
//   size_t colorOffsetBytes = 0;

//   size_t offsetSoFarBytes = 0;

//   for (size_t i = 0; i < vertexLayout.size(); i++) {
//     if (vertexLayout[i] == Properties::POSITION) {
//       positionOffsetBytes = offsetSoFarBytes;
//       offsetSoFarBytes += positionBytes;
//     } else if (vertexLayout[i] == Properties::NORMAL) {
//       normalOffsetBytes = offsetSoFarBytes;
//       offsetSoFarBytes += normalBytes;
//     } else if (vertexLayout[i] == Properties::COLOR) {
//       colorOffsetBytes = offsetSoFarBytes;
//       offsetSoFarBytes += colorBytes;
//     } else {
//       ASSERT(false);
//     }
//   }

//   // Close after parsing header and re-open memory mapped
//   const size_t postHeader = file.tellg();

//   file.close();

//   const size_t fileSize = std::experimental::filesystem::v1::file_size(filename);

//   int fd = open(filename.c_str(), O_RDONLY, 0);
//   void* mmappedData = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);

//   // Parse each vertex packet and unpack
//   char* bytes = &(((char*)mmappedData)[postHeader]);

//   for (size_t i = 0; i < numVertices; i++) {
//     char* nextBytes = &bytes[vertexPacketSizeBytes * i];

//     memcpy(m_originalMesh.vboDouble[i].data(), &nextBytes[positionOffsetBytes], positionBytes);

//     if (normalDimensions)
//       memcpy(m_originalMesh.nboDouble[i].data(), &nextBytes[normalOffsetBytes], normalBytes);

//     if (colorDimensions)
//       memcpy(m_originalMesh.cbo[i].data(), &nextBytes[colorOffsetBytes], colorBytes);
//   }

//   const size_t bytesSoFar = postHeader + vertexPacketSizeBytes * numVertices;

//   bytes = &(((char*)mmappedData)[postHeader + vertexPacketSizeBytes * numVertices]);

//   if (numFaces > 0) {
//     // Read first face to get number of indices;
//     const unsigned char faceDimensions = *bytes;

//     ASSERT(faceDimensions == 3);

//     const size_t countBytes = 1;
//     const size_t faceBytes = faceDimensions * sizeof(uint32_t); // uint32_t
//     const size_t facePacketSizeBytes = countBytes + faceBytes;

//     const size_t predictedFaces = (fileSize - bytesSoFar) / facePacketSizeBytes;

//     // Not sure what to do here
//     //    if(predictedFaces < numFaces)
//     //    {
//     //        std::cout << "Skipping " << numFaces - predictedFaces << " missing faces" <<
//     //        std::endl;
//     //    }
//     //    else if(numFaces < predictedFaces)
//     //    {
//     //        std::cout << "Ignoring " << predictedFaces - numFaces << " extra faces" << std::endl;
//     //    }

//     numFaces = std::min(numFaces, predictedFaces);

//     // uint32_t
//     m_originalMesh.ibo.Reinitialise(numFaces * faceDimensions, 1);

//     for (size_t i = 0; i < numFaces; i++) {
//       char* nextBytes = &bytes[facePacketSizeBytes * i];

//       memcpy(&m_originalMesh.ibo[i * faceDimensions], &nextBytes[countBytes], faceBytes);
//     }

//     m_originalMesh.polygonStride = faceDimensions;
//   } else {
//     m_originalMesh.polygonStride = 0;
//   }

//   munmap(mmappedData, fileSize);

//   close(fd);
// }

double PTexMesh::GetAccurateLength(const glm::dvec3 v)
{
	// return sqrtl(v.x * v.x + v.y * v.y + v.z * v.z);
    return hypot(hypot((double)v.x, (double)v.y), (double)v.z);
}

void PTexMesh::PLYParseReplica(const std::string& filename)
{
  std::vector<std::string> comments;
  std::vector<std::string> objInfo;

  std::string lastElement;
  std::string lastProperty;

  enum Properties { POSITION = 0, NORMAL, COLOR, NUM_PROPERTIES };

  size_t numVertices = 0;

  size_t positionDimensions = 0;
  size_t normalDimensions = 0;
  size_t colorDimensions = 0;

  std::vector<Properties> vertexLayout;

  size_t numFaces = 0;

  std::ifstream file(filename, std::ios::binary);

  // Header parsing
  {
    std::string line;

    while (std::getline(file, line)) {
      std::istringstream ls(line);
      std::string token;
      ls >> token;

      if (token == "ply" || token == "PLY" || token == "") {
        // Skip preamble line
        continue;
      } else if (token == "comment") {
        // Just store these incase
        comments.push_back(line.erase(0, 8));
      } else if (token == "format") {
        // We can only parse binary data, so check that's what it is
        std::string s;
        ls >> s;
        ASSERT(
            s == "binary_little_endian",
            "Can only parse binary files... why are you using ASCII anyway?");
      } else if (token == "element") {
        std::string name;
        size_t size;
        ls >> name >> size;

        if (name == "vertex") {
          // Pull out the number of vertices
          numVertices = size;
        } else if (name == "face") {
          // Pull out number of faces
          numFaces = size;
        } else {
          ASSERT(false, "Can't parse element (%)", name);
        }

        // Keep track of what element we parsed last to associate the properties that follow
        lastElement = name;
      } else if (token == "property") {
        std::string type, name;
        ls >> type;

        // Special parsing for list properties (e.g. faces)
        bool isList = false;

        if (type == "list") {
          isList = true;

          std::string countType;
          ls >> countType >> type;

          ASSERT(
              countType == "uchar" || countType == "uint8",
              "Don't understand count type (%)",
              countType);

          ASSERT(type == "int" || type == "uint", "Don't understand index type (%)", type);

          ASSERT(
              lastElement == "face",
              "Only expecting list after face element, not after (%)",
              lastElement);
        }

        ASSERT(
            type == "float" || type == "int" || type == "uint" || type == "uchar" || type == "uint8",
            "Don't understand type (%)",
            type);

        ls >> name;

        // Collecting vertex property information
        if (lastElement == "vertex") {
          ASSERT(type != "int", "Don't support 32-bit integer properties");

          // Position information
          if (name == "x") {
            positionDimensions = 1;
            vertexLayout.push_back(Properties::POSITION);
            ASSERT(type == "float", "Don't support 8-bit integer positions");
          } else if (name == "y") {
            ASSERT(lastProperty == "x", "Properties should follow x, y, z, (w) order");
            positionDimensions = 2;
          } else if (name == "z") {
            ASSERT(lastProperty == "y", "Properties should follow x, y, z, (w) order");
            positionDimensions = 3;
          } else if (name == "w") {
            ASSERT(lastProperty == "z", "Properties should follow x, y, z, (w) order");
            positionDimensions = 4;
          }

          // Normal information
          if (name == "nx") {
            normalDimensions = 1;
            vertexLayout.push_back(Properties::NORMAL);
            ASSERT(type == "float", "Don't support 8-bit integer normals");
          } else if (name == "ny") {
            ASSERT(lastProperty == "nx", "Properties should follow nx, ny, nz order");
            normalDimensions = 2;
          } else if (name == "nz") {
            ASSERT(lastProperty == "ny", "Properties should follow nx, ny, nz order");
            normalDimensions = 3;
          }

          // Color information
          if (name == "red") {
            colorDimensions = 1;
            vertexLayout.push_back(Properties::COLOR);
            ASSERT(type == "uchar" || type == "uint8", "Don't support non-8-bit integer colors");
          } else if (name == "green") {
            ASSERT(
                lastProperty == "red", "Properties should follow red, green, blue, (alpha) order");
            colorDimensions = 2;
          } else if (name == "blue") {
            ASSERT(
                lastProperty == "green",
                "Properties should follow red, green, blue, (alpha) order");
            colorDimensions = 3;
          } else if (name == "alpha") {
            ASSERT(
                lastProperty == "blue", "Properties should follow red, green, blue, (alpha) order");
            colorDimensions = 4;
          }
        } else if (lastElement == "face") {
          ASSERT(isList, "No idea what to do with properties following faces");
        } else {
          ASSERT(false, "No idea what to do with properties before elements");
        }

        lastProperty = name;
      } else if (token == "obj_info") {
        // Just store these incase
        objInfo.push_back(line.erase(0, 9));
      } else if (token == "end_header") {
        // Done reading!
        break;
      } else {
        // Something unrecognised
        ASSERT(false);
      }
    }

    // Check things make sense.
    ASSERT(numVertices > 0);
    ASSERT(positionDimensions == 3);
  }

  // m_originalMesh.vbo.Reinitialise(numVertices, 1);
  // m_originalMesh.vbo.Fill(Eigen::Vector4f(0, 0, 0, 1));

  m_originalMesh.m_vboVec.resize(numVertices * 3);
  memset(m_originalMesh.m_vboVec.data(), 0, sizeof(float) * numVertices * 3);

  if (normalDimensions) {
    // m_originalMesh.nbo.Reinitialise(numVertices, 1);
    // m_originalMesh.nbo.Fill(Eigen::Vector3f(0, 0, 0));
  }

  if (colorDimensions) {
    // m_originalMesh.cbo.Reinitialise(numVertices, 1);
    // m_originalMesh.cbo.Fill(Eigen::Matrix<unsigned char, 4, 1>(0, 0, 0, 255));
  }

  // Can only be FLOAT32 or UINT8
  const size_t positionBytes = positionDimensions * sizeof(float); // floats
  const size_t normalBytes = normalDimensions * sizeof(float); // floats
  const size_t colorBytes = colorDimensions * sizeof(uint8_t); // bytes

  const size_t vertexPacketSizeBytes = positionBytes + normalBytes + colorBytes;

  size_t positionOffsetBytes = 0;
  size_t normalOffsetBytes = 0;
  size_t colorOffsetBytes = 0;

  size_t offsetSoFarBytes = 0;

  for (size_t i = 0; i < vertexLayout.size(); i++) {
    if (vertexLayout[i] == Properties::POSITION) {
      positionOffsetBytes = offsetSoFarBytes;
      offsetSoFarBytes += positionBytes;
    } else if (vertexLayout[i] == Properties::NORMAL) {
      normalOffsetBytes = offsetSoFarBytes;
      offsetSoFarBytes += normalBytes;
    } else if (vertexLayout[i] == Properties::COLOR) {
      colorOffsetBytes = offsetSoFarBytes;
      offsetSoFarBytes += colorBytes;
    } else {
      ASSERT(false);
    }
  }

  // Close after parsing header and re-open memory mapped
  const size_t postHeader = file.tellg();

  file.close();

  const size_t fileSize = std::experimental::filesystem::v1::file_size(filename);

  int fd = open(filename.c_str(), O_RDONLY, 0);
  void* mmappedData = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);

  // Parse each vertex packet and unpack
  char* bytes = &(((char*)mmappedData)[postHeader]);

  for (size_t i = 0; i < numVertices; i++) {
    char* nextBytes = &bytes[vertexPacketSizeBytes * i];

    // memcpy(m_originalMesh.vbo[i].data(), &nextBytes[positionOffsetBytes], positionBytes);
    memcpy((char *)(((float *)m_originalMesh.m_vboVec.data()) + i * positionDimensions), &nextBytes[positionOffsetBytes], positionBytes);

    // if (normalDimensions)
    //   memcpy(m_originalMesh.nbo[i].data(), &nextBytes[normalOffsetBytes], normalBytes);

    // if (colorDimensions)
    //   memcpy(m_originalMesh.cbo[i].data(), &nextBytes[colorOffsetBytes], colorBytes);
  }

  const size_t bytesSoFar = postHeader + vertexPacketSizeBytes * numVertices;

  bytes = &(((char*)mmappedData)[postHeader + vertexPacketSizeBytes * numVertices]);

  if (numFaces > 0) {
    // Read first face to get number of indices;
    const uint8_t faceDimensions = *bytes;

    ASSERT(faceDimensions == 4);

    // scanNet++ is 3
    std::cout << "get vertex number of each mesh " << faceDimensions << std::endl;

    // exit(0);

    const size_t countBytes = 1;
    const size_t faceBytes = faceDimensions * sizeof(int32_t); // uint32_t
    const size_t facePacketSizeBytes = countBytes + faceBytes;

    const size_t predictedFaces = (fileSize - bytesSoFar) / facePacketSizeBytes;

    // Not sure what to do here
    //    if(predictedFaces < numFaces)
    //    {
    //        std::cout << "Skipping " << numFaces - predictedFaces << " missing faces" <<
    //        std::endl;
    //    }
    //    else if(numFaces < predictedFaces)
    //    {
    //        std::cout << "Ignoring " << predictedFaces - numFaces << " extra faces" << std::endl;
    //    }

    numFaces = std::min(numFaces, predictedFaces);

    // m_originalMesh.ibo.Reinitialise(numFaces * faceDimensions, 1);
    m_originalMesh.m_iboVec.resize(numFaces * faceDimensions);


    // m_originalMesh.m_faceNormal.Reinitialise(numFaces * 2, 1);
    m_originalMesh.m_faceNormalVec.resize(numFaces * 2 * 3);

    for (size_t i = 0; i < numFaces; i++) {
      char* nextBytes = &bytes[facePacketSizeBytes * i];

      // memcpy(&m_originalMesh.ibo[i * faceDimensions], &nextBytes[countBytes], faceBytes);
      m_originalMesh.m_iboVec[i * faceDimensions + 0] = *((int32_t *)(nextBytes + 1));
      m_originalMesh.m_iboVec[i * faceDimensions + 1] = *(((int32_t *)(nextBytes + 1)) + 1);
      m_originalMesh.m_iboVec[i * faceDimensions + 2] = *(((int32_t *)(nextBytes + 1)) + 2);
      m_originalMesh.m_iboVec[i * faceDimensions + 3] = *(((int32_t *)(nextBytes + 1)) + 3);

      // Eigen::Vector3f pt0 = m_originalMesh.vbo[m_originalMesh.ibo[i * faceDimensions]].head(3);
      // Eigen::Vector3f pt0(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 0],
      //                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 1],
      //                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 2]);
      
      glm::dvec3 pt0(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 0],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 1],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 0] * positionDimensions + 2]);

      // Eigen::Vector3f pt1 = m_originalMesh.vbo[m_originalMesh.ibo[i * faceDimensions + 1]].head(3);
      // Eigen::Vector3f pt1(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 0],
      //                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 1],
      //                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 2]);

      glm::dvec3 pt1(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 0],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 1],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 1] * positionDimensions + 2]);

      // Eigen::Vector3f pt2 = m_originalMesh.vbo[m_originalMesh.ibo[i * faceDimensions + 2]].head(3);
      // Eigen::Vector3f pt2(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 0],
      //                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 1],
      //                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 2]);

      glm::dvec3 pt2(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 0],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 1],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 2] * positionDimensions + 2]);

      // Eigen::Vector3f pt3 = m_originalMesh.vbo[m_originalMesh.ibo[i * faceDimensions + 3]].head(3);
      // Eigen::Vector3f pt3(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 3] * positionDimensions + 0],
      //                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 3] * positionDimensions + 1],
      //                     m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 3] * positionDimensions + 2]);

      glm::dvec3 pt3(m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 3] * positionDimensions + 0],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 3] * positionDimensions + 1],
                          m_originalMesh.m_vboVec[m_originalMesh.m_iboVec[i * faceDimensions + 3] * positionDimensions + 2]);

      // Eigen::Vector3f vec01 = (pt1 - pt0).normalized();
      // Eigen::Vector3f vec12 = (pt2 - pt1).normalized();

      // Eigen::Vector3f vec02 = (pt2 - pt0).normalized();
      // Eigen::Vector3f vec23 = (pt3 - pt2).normalized();

      // glm::vec3 vec01 = pt1 - pt0;
      // glm::vec3 vec12 = pt2 - pt1;

      // glm::vec3 vec02 = pt2 - pt0;
      // glm::vec3 vec23 = pt3 - pt2;

      glm::dvec3 vec01 = glm::normalize(pt1 - pt0);
      glm::dvec3 vec12 = glm::normalize(pt2 - pt1);

      glm::dvec3 vec02 = glm::normalize(pt2 - pt0);
      glm::dvec3 vec23 = glm::normalize(pt3 - pt2);

      glm::dvec3 crossedDouble1 = glm::cross(vec01, vec12);
      glm::dvec3 crossedDouble2 = glm::cross(vec02, vec23);

      double realLen1 = GetAccurateLength(crossedDouble1);
      double realLen2 = GetAccurateLength(crossedDouble2);

      glm::dvec3 normal1(crossedDouble1.x / realLen1, crossedDouble1.y / realLen1, crossedDouble1.z / realLen1);
      glm::dvec3 normal2(crossedDouble2.x / realLen2, crossedDouble2.y / realLen2, crossedDouble2.z / realLen2);

      // m_originalMesh.m_faceNormalVec[i * 2 * 3 + 0] = ((vec01.cross(vec12)).normalized())(0);
      // m_originalMesh.m_faceNormalVec[i * 2 * 3 + 1] = ((vec01.cross(vec12)).normalized())(1);
      // m_originalMesh.m_faceNormalVec[i * 2 * 3 + 2] = ((vec01.cross(vec12)).normalized())(2);

      // glm::vec3 normal1 = glm::normalize(glm::cross(vec01, vec12));
      m_originalMesh.m_faceNormalVec[i * 2 * 3 + 0] = normal1.x;
      m_originalMesh.m_faceNormalVec[i * 2 * 3 + 1] = normal1.y;
      m_originalMesh.m_faceNormalVec[i * 2 * 3 + 2] = normal1.z;


      // m_originalMesh.m_faceNormalVec[i * 2 * 3 + 3] = ((vec02.cross(vec23)).normalized())(0);
      // m_originalMesh.m_faceNormalVec[i * 2 * 3 + 4] = ((vec02.cross(vec23)).normalized())(1);
      // m_originalMesh.m_faceNormalVec[i * 2 * 3 + 5] = ((vec02.cross(vec23)).normalized())(2);

      // glm::vec3 normal2 = glm::normalize(glm::cross(vec02, vec23));
      m_originalMesh.m_faceNormalVec[i * 2 * 3 + 3] = normal2.x;
      m_originalMesh.m_faceNormalVec[i * 2 * 3 + 4] = normal2.y;
      m_originalMesh.m_faceNormalVec[i * 2 * 3 + 5] = normal2.z;
      

    }

    m_originalMesh.polygonStride = faceDimensions;
  } else {
    m_originalMesh.polygonStride = 0;
  }

  munmap(mmappedData, fileSize);

  close(fd);
}

// void PTexMesh::CalcEachTriangleImMeshDouble(std::vector<TriangleInfo>& triangleArr)
// {
//     size_t numFaces = m_originalMesh.ibo.size() / 3;
    

//     for (size_t cnt = 0; cnt < numFaces; cnt++){
//       TriangleInfo oneTriangle1;

//       // oneTriangle1.vert.push_back(m_originalMesh.vboDouble[m_originalMesh.ibo[cnt * 3 + 0]].head(3).cast<float>());
//       // oneTriangle1.vert.push_back(m_originalMesh.vboDouble[m_originalMesh.ibo[cnt * 3 + 1]].head(3).cast<float>());
//       // oneTriangle1.vert.push_back(m_originalMesh.vboDouble[m_originalMesh.ibo[cnt * 3 + 2]].head(3).cast<float>());

//       oneTriangle1.norm = Eigen::Vector3f::Zero();
      
//       triangleArr.push_back(oneTriangle1);
//     }

//     // for (size_t i = 0; i < m_splitMeshData.size(); i++) {
//     //   size_t numFaces = m_splitMeshData[i].ibo.size() / 4;
//     //   for ()

//     // }

//     std::cout << "num of mesh in ibo : " << numFaces << std::endl;
//     std::cout << "num of triangles out : " << triangleArr.size() << std::endl;
// }

void PTexMesh::CalcEachTriangleImMesh(std::vector<TriangleInfoDevice>& triangleArr)
{
    size_t numFaces = m_originalMesh.m_iboVec.size() / 3;

    triangleArr.resize(numFaces);

    std::vector<uint32_t> reduceErrCnt(numFaces);

    // parallel_for(blocked_range<uint32_t>(0, triangleArr.size()), ApplyCalcEachTriangleImMesh(m_edgeMapConcurrent, m_vertNbrMapConcurrent, m_originalMesh, triangleArr, reduceErrCnt));


    // CalcEachTriangleCuda(triangleArr, numFaces, MESH_TYPE_IM_MESH);

    // for (size_t cnt = 0; cnt < numFaces; cnt++){
    //   TriangleInfo oneTriangle1;

    //   // oneTriangle1.vert.push_back(m_originalMesh.vbo[m_originalMesh.ibo[cnt * 3 + 0]].head(3));
    //   // oneTriangle1.vert.push_back(m_originalMesh.vbo[m_originalMesh.ibo[cnt * 3 + 1]].head(3));
    //   // oneTriangle1.vert.push_back(m_originalMesh.vbo[m_originalMesh.ibo[cnt * 3 + 2]].head(3));

    //   oneTriangle1.norm = Eigen::Vector3f::Zero();
      
    //   triangleArr.push_back(oneTriangle1);
    // }

    uint32_t errCntAll = 0;
    for (auto it: reduceErrCnt) {
      errCntAll += it;
    }

    std::cout << "inserting triangles errCntAll: " << errCntAll << std::endl;
    std::cout << "num of mesh in ibo : " << numFaces << std::endl;
    std::cout << "num of triangles out : " << triangleArr.size() << std::endl;
}

// template <uint32_t tile_size, typename VertKeyIter, typename VertMapViewType>
// __global__ void validate_vert_map_kernel(VertKeyIter iboDeviceFirst, uint32_t keyNum, VertMapViewType vertMap, uint32_t* maxVertNbrNum, uint32_t* cannotFindNum)
// {
//   auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
//   auto tid = threadIdx.x + blockIdx.x * blockDim.x;
//   std::size_t thread_num_matches = 0;

//   if (tid < keyNum) {
//     auto key = *(iboDeviceFirst + tid);

//     thread_num_matches = vertMap.count(tile, key);
//     atomicMax(maxVertNbrNum, thread_num_matches);

//     if (thread_num_matches == 0) {
//       atomicAdd(cannotFindNum, 1);
//     }

//     // no loop stride
//     // tid += gridDim.x * blockDim.x;
//   }

// }


// template <uint32_t tile_size, typename EdgeKeyIter, typename EdgeMapViewType, typename KeyEqual>
// __global__ void validate_edge_map_kernel(EdgeKeyIter first, uint32_t keyNum, EdgeMapViewType edgeMap, KeyEqual equalFunc, uint32_t* zeroNbrCnt, uint32_t* oneNbrCnt, uint32_t* twoNbrCnt, uint32_t* moreThanTwoNbrCnt, uint32_t* maxEdgeNbr)
// {
//   auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
//   auto tid = threadIdx.x + blockIdx.x * blockDim.x;
//   std::size_t thread_num_matches = 0;

//   if (tid < keyNum) {
//     auto key = *(first + tid);

//     thread_num_matches = edgeMap.count(tile, key, equalFunc);

//     if (thread_num_matches == 0) {
//       atomicAdd(zeroNbrCnt, 1);
//       // printf("edge cannot be find: %d, %d\n", static_cast<EdgeKeyCuda>(*(first + threadIdx.x + blockIdx.x * blockDim.x)).m_e0, static_cast<EdgeKeyCuda>(*(first + threadIdx.x + blockIdx.x * blockDim.x)).m_e1);
//     }

//     if (thread_num_matches == 1) {
//       atomicAdd(oneNbrCnt, 1);
//     }

//     if (thread_num_matches == 2) {
//       atomicAdd(twoNbrCnt, 1);
//     }

//     if (thread_num_matches > 2) {
//       atomicAdd(moreThanTwoNbrCnt, 1);
//     }

//     atomicMax(maxEdgeNbr, thread_num_matches);

//     // tid += gridDim.x * blockDim.x;
//   }

 
// }


// template<uint32_t tile_size, typename EdgeKeyIter, typename EdgeMapViewType, typename KeyEqual>
// void PTexMesh::ValidateEdgeTriangleRelationCuda(EdgeKeyIter first, uint32_t keyNum, EdgeMapViewType& edgeMapView, KeyEqual equalFunc)
// {
//     thrust::device_vector<uint32_t> numFinded(1);
//     auto constexpr block_size = 128;
//     auto constexpr stride     = 1;
//     auto const grid_size = (keyNum + stride * block_size - 1) / (stride * block_size);

//     validate_edge_map_kernel<tile_size><<<grid_size, block_size>>>(first, keyNum, edgeMapView, equalFunc, numFinded.data().get());

// }

void PTexMesh::ValidateEdgeTriangleRelation()
{
  uint32_t invalidEntry = 0;
  uint32_t invalidEntryConcurrent = 0;
  for (auto kv : m_edgeMap) {
    if (kv.second.size() > 2) { // one edge have 2 triangle max
      std::cout << "invalid edge-triangle relation in m_edgeMap, edge v1: " << kv.first.m_v1 << " v2: " << kv.first.m_v2 << " trignale num : " << kv.second.size() << std::endl;

      std::cout << " tri idx : \n";
      for (auto it : kv.second) {
        std::cout << it << " ";
      }
      std::cout << std::endl;
      // m_edgeMap.erase(kv.first);
      invalidEntry++;
    }
  }

  for (auto it: m_edgeMapConcurrent) {
    if (it.second.size() > 2) {
      std::cout << "invalid edge-triangle relation in m_edgeMapConcurrent, edge v1: " << it.first.m_v1 << " v2: " << it.first.m_v2 << " trignale num : " << it.second.size() << std::endl;

      std::cout << " tri idx : \n";
      for (auto it : it.second) {
        std::cout << it << " ";
      }
      std::cout << std::endl;
      // m_edgeMapConcurrent.erase(it.first);
      invalidEntryConcurrent++;
    }
  }

  std::cout << "invalidEntry : " << invalidEntry << " invalidEntryConcurrent :  " << invalidEntryConcurrent << std::endl;
}

// template <typename EdgeMap, typename VertMap, typename EdgeKeySet>
// void PTexMesh::PrintHashMapAndTriangleArrCuda(std::vector<TriangleInfoDevice>& triangleArr, EdgeMap& edgeNbrMultiMapDevice, VertMap& vertNbrMultiMapDevice, EdgeKeySet& edgeKeysSet)
// {
//   // printf(" printf all triangleArr \n");
//   // for (auto it : triangleArr) {
//   //   printf("vert (%f, %f, %f), (%f, %f, %f), (%f, %f, %f), norm (%f, %f, %f), vertIdx (%u, %u, %u)\n", it.vert[0](0), it.vert[0](1), it.vert[0](2),
//   //           it.vert[1](0), it.vert[1](1), it.vert[1](2),
//   //           it.vert[2](0), it.vert[2](1), it.vert[2](2),
//   //           it.norm(0), it.norm(1), it.norm(2),
//   //           it.vertIdx[0], it.vertIdx[1], it.vertIdx[2]);
//   // }

//   // printf("m_edgeMapConcurrent size : %u\n", m_edgeMapConcurrent.size());
//   // printf("m_vertNbrMapConcurrent size : %u\n", m_vertNbrMapConcurrent.size());

//   // printf("m_edgeMap size : %u\n", m_edgeMap.size());
//   // printf("m_vertNeighborMap size : %u\n", m_vertNeighborMap.size());

//   // for (auto it = m_edgeMapConcurrent.begin(); it != m_edgeMapConcurrent.end(); it++) {
//   //   printf("edge %u %u info ", it->first.m_v1, it->first.m_v2);
//   //   for (auto it2: it->second) {
//   //     printf(" %d ", it2);
//   //   }

//   //   printf("\n");
//   // }

//   // for (auto it : m_vertNbrMapConcurrent) {
//   //   printf("vertId  %u  Nbr texId ", it.first);
//   //   for (auto it2: it.second) {
//   //     printf(" %u ", it2);
//   //   }

//   //   printf("\n");
//   // }

//   // for (auto it = m_edgeMap.begin(); it != m_edgeMap.end(); it++) {
//   //   printf("edge %u %u info ", it->first.m_v1, it->first.m_v2);
//   //   for (auto it2: it->second) {
//   //     printf(" %d ", it2);
//   //   }

//   //   printf("\n");
//   // }

//   // for (auto it : m_vertNeighborMap) {
//   //   printf("vertId  %u  Nbr texId ", it.first);
//   //   for (auto it2: it.second) {
//   //     printf(" %u ", it2);
//   //   }

//   //   printf("\n");
//   // }

//   // // uint32_t sumAll = 0;
//   // // for (auto it: reduceCnt) {
//   // //   sumAll += it;
//   // // }

//   // // printf("insert concurrent map err cnt : %u\n", sumAll);
// }



void PTexMesh::PrintHashMapAndTriangleArr(std::vector<TriangleInfo>& triangleArr)
{
  printf(" printf all triangleArr \n");
  for (auto it : triangleArr) {
    printf("vert (%f, %f, %f), (%f, %f, %f), (%f, %f, %f), norm (%f, %f, %f), vertIdx (%u, %u, %u)\n", it.vert[0](0), it.vert[0](1), it.vert[0](2),
            it.vert[1](0), it.vert[1](1), it.vert[1](2),
            it.vert[2](0), it.vert[2](1), it.vert[2](2),
            it.norm(0), it.norm(1), it.norm(2),
            it.vertIdx[0], it.vertIdx[1], it.vertIdx[2]);
  }

  printf("m_edgeMapConcurrent size : %u\n", m_edgeMapConcurrent.size());
  printf("m_vertNbrMapConcurrent size : %u\n", m_vertNbrMapConcurrent.size());

  printf("m_edgeMap size : %u\n", m_edgeMap.size());
  printf("m_vertNeighborMap size : %u\n", m_vertNeighborMap.size());

  for (auto it = m_edgeMapConcurrent.begin(); it != m_edgeMapConcurrent.end(); it++) {
    printf("edge %u %u info ", it->first.m_v1, it->first.m_v2);
    for (auto it2: it->second) {
      printf(" %d ", it2);
    }

    printf("\n");
  }

  for (auto it : m_vertNbrMapConcurrent) {
    printf("vertId  %u  Nbr texId ", it.first);
    for (auto it2: it.second) {
      printf(" %u ", it2);
    }

    printf("\n");
  }

  for (auto it = m_edgeMap.begin(); it != m_edgeMap.end(); it++) {
    printf("edge %u %u info ", it->first.m_v1, it->first.m_v2);
    for (auto it2: it->second) {
      printf(" %d ", it2);
    }

    printf("\n");
  }

  for (auto it : m_vertNeighborMap) {
    printf("vertId  %u  Nbr texId ", it.first);
    for (auto it2: it.second) {
      printf(" %u ", it2);
    }

    printf("\n");
  }

  // uint32_t sumAll = 0;
  // for (auto it: reduceCnt) {
  //   sumAll += it;
  // }

  // printf("insert concurrent map err cnt : %u\n", sumAll);
}

// template <typename Int32It, typename floatIt>
// __device__ void update_triangle_info(TriangleInfoDevice* triangleInfoDeviceInput,
//   Int32It iboDeviceFirst,
//   floatIt vboDeviceFirst,
//   floatIt faceNormalDeviceFirst,
//   uint32_t tid)
// {
//     TriangleInfoDevice tmp1;
//     TriangleInfoDevice tmp2;

//     tmp1.vert[0] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 0)) * 3 + 0);
//     tmp1.vert[1] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 0)) * 3 + 1);
//     tmp1.vert[2] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 0)) * 3 + 2);

//     tmp1.vert[3] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 1)) * 3 + 0);
//     tmp1.vert[4] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 1)) * 3 + 1);
//     tmp1.vert[5] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 1)) * 3 + 2);

//     tmp1.vert[6] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 2)) * 3 + 0);
//     tmp1.vert[7] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 2)) * 3 + 1);
//     tmp1.vert[8] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 2)) * 3 + 2);



//     // tmp1.vert[0] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 0];
//     // tmp1.vert[1] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 1];
//     // tmp1.vert[2] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 2];

//     // tmp1.vert[3] = vboDevice[iboDevice[tid * 4 + 1] * 3 + 0];
//     // tmp1.vert[4] = vboDevice[iboDevice[tid * 4 + 1] * 3 + 1];
//     // tmp1.vert[5] = vboDevice[iboDevice[tid * 4 + 1] * 3 + 2];

//     // tmp1.vert[6] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 0];
//     // tmp1.vert[7] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 1];
//     // tmp1.vert[8] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 2];

//     tmp1.norm[0] = *(faceNormalDeviceFirst + tid * 2 * 3 + 0);
//     tmp1.norm[1] = *(faceNormalDeviceFirst + tid * 2 * 3 + 1);
//     tmp1.norm[2] = *(faceNormalDeviceFirst + tid * 2 * 3 + 2);

//     // tmp1.norm[0] = faceNormalDevice[tid * 2 * 3 + 0];
//     // tmp1.norm[1] = faceNormalDevice[tid * 2 * 3 + 1];
//     // tmp1.norm[2] = faceNormalDevice[tid * 2 * 3 + 2];

//     tmp1.vertIdx[0] = (uint32_t)(*(iboDeviceFirst + tid * 4 + 0));
//     tmp1.vertIdx[1] = (uint32_t)(*(iboDeviceFirst + tid * 4 + 1));
//     tmp1.vertIdx[2] = (uint32_t)(*(iboDeviceFirst + tid * 4 + 2));

//     // tmp1.vertIdx[0] = iboDevice[tid * 4 + 0];
//     // tmp1.vertIdx[1] = iboDevice[tid * 4 + 1];
//     // tmp1.vertIdx[2] = iboDevice[tid * 4 + 2];


//     tmp2.vert[0] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 0)) * 3 + 0);
//     tmp2.vert[1] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 0)) * 3 + 1);
//     tmp2.vert[2] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 0)) * 3 + 2);

//     // tmp2.vert[0] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 0];
//     // tmp2.vert[1] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 1];
//     // tmp2.vert[2] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 2];

//     tmp2.vert[3] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 2)) * 3 + 0);
//     tmp2.vert[4] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 2)) * 3 + 1);
//     tmp2.vert[5] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 2)) * 3 + 2);

//     // tmp2.vert[3] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 0];
//     // tmp2.vert[4] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 1];
//     // tmp2.vert[5] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 2];

//     tmp2.vert[6] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 3)) * 3 + 0);
//     tmp2.vert[7] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 3)) * 3 + 1);
//     tmp2.vert[8] = *(vboDeviceFirst + (*(iboDeviceFirst + tid * 4 + 3)) * 3 + 2);

//     // tmp2.vert[6] = vboDevice[iboDevice[tid * 4 + 3] * 3 + 0];
//     // tmp2.vert[7] = vboDevice[iboDevice[tid * 4 + 3] * 3 + 1];
//     // tmp2.vert[8] = vboDevice[iboDevice[tid * 4 + 3] * 3 + 2];


//     tmp2.norm[0] = *(faceNormalDeviceFirst + tid * 2 * 3 + 3);
//     tmp2.norm[1] = *(faceNormalDeviceFirst + tid * 2 * 3 + 4);
//     tmp2.norm[2] = *(faceNormalDeviceFirst + tid * 2 * 3 + 5);

//     // tmp2.norm[0] = faceNormalDevice[tid * 2 * 3 + 3];
//     // tmp2.norm[1] = faceNormalDevice[tid * 2 * 3 + 4];
//     // tmp2.norm[2] = faceNormalDevice[tid * 2 * 3 + 5];

//     tmp2.vertIdx[0] = (uint32_t)(*(iboDeviceFirst + tid * 4 + 0));
//     tmp2.vertIdx[1] = (uint32_t)(*(iboDeviceFirst + tid * 4 + 2));
//     tmp2.vertIdx[2] = (uint32_t)(*(iboDeviceFirst + tid * 4 + 3));

//     // tmp2.vertIdx[0] = iboDevice[tid * 4 + 0];
//     // tmp2.vertIdx[1] = iboDevice[tid * 4 + 2];
//     // tmp2.vertIdx[2] = iboDevice[tid * 4 + 3];


//     triangleInfoDeviceInput[tid * 2 + 0] = tmp1;
//     triangleInfoDeviceInput[tid * 2 + 1] = tmp2;
// }


// template <typename vertMap, typename tileType, typename InputIt>
// __device__ void update_vert_map(vertMap& vertMapInsertView, tileType& tile, InputIt& iboDeviceFirst, uint32_t tid)
// {
//   vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 4 + 0)), (int32_t)(tid * 2 + 0)});
//   vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 4 + 0)), (int32_t)(tid * 2 + 1)});

//   vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 4 + 1)), (int32_t)(tid * 2 + 0)});

//   vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 4 + 2)), (int32_t)(tid * 2 + 0)});
//   vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 4 + 2)), (int32_t)(tid * 2 + 1)});

//   vertMapInsertView.insert(tile, cuco::pair{(int32_t)(*(iboDeviceFirst + tid * 4 + 3)), (int32_t)(tid * 2 + 1)});

//   // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 0]), (int32_t)(tid * 2 + 0)});
//   // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 0]), (int32_t)(tid * 2 + 1)});

//   // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 1]), (int32_t)(tid * 2 + 0)});

//   // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 2]), (int32_t)(tid * 2 + 0)});
//   // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 2]), (int32_t)(tid * 2 + 1)});

//   // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 3]), (int32_t)(tid * 2 + 1)});
// }


// template <typename edgeMap, typename tileType, typename SetRef, typename InputIt>
// __device__ void update_edge_map(InputIt& iboVecFirst, tileType& tile, edgeMap& edgeMapInsertView, SetRef& edgeKeysSet, uint32_t tid)
// {
//   auto edge01 = EdgeKeyCuda((*(iboVecFirst + tid * 4 + 0)), (*(iboVecFirst + tid * 4 + 1)));
//   // auto edge01 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 0]), (uint32_t)(iboDevice[tid * 4 + 1]));
//   edgeMapInsertView.insert(tile, cuco::pair{edge01, (int32_t)(tid * 2 + 0)});

//   auto edge12 = EdgeKeyCuda((*(iboVecFirst + tid * 4 + 1)), (*(iboVecFirst + tid * 4 + 2)));
//   // auto edge12 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 1]), (uint32_t)(iboDevice[tid * 4 + 2]));
//   edgeMapInsertView.insert(tile, cuco::pair{edge12, (int32_t)(tid * 2 + 0)});

//   auto edge02 = EdgeKeyCuda((*(iboVecFirst + tid * 4 + 0)), (*(iboVecFirst + tid * 4 + 2)));
//   // auto edge02 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 0]), (uint32_t)(iboDevice[tid * 4 + 2]));
//   edgeMapInsertView.insert(tile, cuco::pair{edge02, (int32_t)(tid * 2 + 0)});

//   //triangle 2
//   edgeMapInsertView.insert(tile, cuco::pair{edge02, (int32_t)(tid * 2 + 1)});

//   auto edge23 = EdgeKeyCuda((*(iboVecFirst + tid * 4 + 2)), (*(iboVecFirst + tid * 4 + 3)));
//   // auto edge23 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 2]), (uint32_t)(iboDevice[tid * 4 + 3]));
//   edgeMapInsertView.insert(tile, cuco::pair{edge23, (int32_t)(tid * 2 + 1)});

//   auto edge03 = EdgeKeyCuda((*(iboVecFirst + tid * 4 + 0)), (*(iboVecFirst + tid * 4 + 3)));
//   // auto edge03 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 0]), (uint32_t)(iboDevice[tid * 4 + 3]));
//   edgeMapInsertView.insert(tile, cuco::pair{edge03, (int32_t)(tid * 2 + 1)});

//   edgeKeysSet.insert(tile, edge01);
//   edgeKeysSet.insert(tile, edge12);
//   edgeKeysSet.insert(tile, edge02);
//   edgeKeysSet.insert(tile, edge23);
//   edgeKeysSet.insert(tile, edge03);
// }


// template <typename Int32It, typename floatIt>
// __global__ void insert_triangle_info_quad(Int32It iboDeviceFirst,
//   TriangleInfoDevice* triangleInfoDeviceInput,
//   floatIt faceNormalDeviceFirst,
//   floatIt vboDeviceFirst,
//   uint32_t numFaces)
// {
//   auto tid = threadIdx.x + blockIdx.x * blockDim.x;

//   while (tid < numFaces) {
//     update_triangle_info(triangleInfoDeviceInput, iboDeviceFirst, vboDeviceFirst, faceNormalDeviceFirst, tid);
//     tid += gridDim.x * blockDim.x;
//   }
// }

// template <uint32_t tile_size, typename edgeMap, typename SetRef, typename InputIt>
// __global__ void insert_edge_map_quad(edgeMap edgeMapInsertView,
// SetRef edgeKeysSet,
// InputIt iboVecFirst,
// uint32_t numFaces)
// {
//   auto tid = threadIdx.x + blockIdx.x * blockDim.x;
//   // std::size_t counter = 0;
//   auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());

//   while (tid < numFaces) {
//     update_edge_map(iboVecFirst, tile, edgeMapInsertView, edgeKeysSet, tid);
//     tid += gridDim.x * blockDim.x;
//   }

// }


// template <uint32_t tile_size, typename vertMap, typename InputIt>
// __global__ void insert_vert_map_quad(vertMap vertMapInsertView, InputIt iboDeviceFirst, uint32_t numFaces)
// {
//   auto tid = threadIdx.x + blockIdx.x * blockDim.x;
//   // std::size_t counter = 0;
//   auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());

//   while (tid < numFaces) {
//     update_vert_map(vertMapInsertView, tile, iboDeviceFirst, tid);
//     tid += gridDim.x * blockDim.x;
//   }
// }


template <uint32_t tile_size, typename edgeMap, typename vertMap, typename int32It, typename floatIt, typename SetRef>
__global__ void custom_insert_quad(edgeMap edgeMapInsertView,
vertMap vertMapInsertView,
SetRef edgeKeysSet,
int32It iboDevice,
TriangleInfoDevice* triangleInfoDeviceInput,
floatIt faceNormalDevice,
floatIt vboDevice,
int* num_inserted, uint32_t numFaces)
{
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  // std::size_t counter = 0;
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());

  while (tid < numFaces) {
    // TriangleInfoDevice tmp1;
    // TriangleInfoDevice tmp2;

    // tmp1.vert[0] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 0];
    // tmp1.vert[1] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 1];
    // tmp1.vert[2] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 2];

    // tmp1.vert[3] = vboDevice[iboDevice[tid * 4 + 1] * 3 + 0];
    // tmp1.vert[4] = vboDevice[iboDevice[tid * 4 + 1] * 3 + 1];
    // tmp1.vert[5] = vboDevice[iboDevice[tid * 4 + 1] * 3 + 2];

    // tmp1.vert[6] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 0];
    // tmp1.vert[7] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 1];
    // tmp1.vert[8] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 2];

    // tmp1.norm[0] = faceNormalDevice[tid * 2 * 3 + 0];
    // tmp1.norm[1] = faceNormalDevice[tid * 2 * 3 + 1];
    // tmp1.norm[2] = faceNormalDevice[tid * 2 * 3 + 2];

    // tmp1.vertIdx[0] = iboDevice[tid * 4 + 0];
    // tmp1.vertIdx[1] = iboDevice[tid * 4 + 1];
    // tmp1.vertIdx[2] = iboDevice[tid * 4 + 2];


    // tmp2.vert[0] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 0];
    // tmp2.vert[1] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 1];
    // tmp2.vert[2] = vboDevice[iboDevice[tid * 4 + 0] * 3 + 2];

    // tmp2.vert[3] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 0];
    // tmp2.vert[4] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 1];
    // tmp2.vert[5] = vboDevice[iboDevice[tid * 4 + 2] * 3 + 2];

    // tmp2.vert[6] = vboDevice[iboDevice[tid * 4 + 3] * 3 + 0];
    // tmp2.vert[7] = vboDevice[iboDevice[tid * 4 + 3] * 3 + 1];
    // tmp2.vert[8] = vboDevice[iboDevice[tid * 4 + 3] * 3 + 2];

    // tmp2.norm[0] = faceNormalDevice[tid * 2 * 3 + 3];
    // tmp2.norm[1] = faceNormalDevice[tid * 2 * 3 + 4];
    // tmp2.norm[2] = faceNormalDevice[tid * 2 * 3 + 5];

    // tmp2.vertIdx[0] = iboDevice[tid * 4 + 0];
    // tmp2.vertIdx[1] = iboDevice[tid * 4 + 2];
    // tmp2.vertIdx[2] = iboDevice[tid * 4 + 3];


    // triangleInfoDeviceInput[tid * 2 + 0] = tmp1;
    // triangleInfoDeviceInput[tid * 2 + 1] = tmp2;

    update_triangle_info(triangleInfoDeviceInput, iboDevice, vboDevice, faceNormalDevice, tid);


    // triangle1
    // auto edge01 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 0]), (uint32_t)(iboDevice[tid * 4 + 1]));
    // edgeMapInsertView.insert(tile, cuco::pair{edge01, (int32_t)(tid * 2 + 0)});

    // auto edge12 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 1]), (uint32_t)(iboDevice[tid * 4 + 2]));
    // edgeMapInsertView.insert(tile, cuco::pair{edge12, (int32_t)(tid * 2 + 0)});

    // auto edge02 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 0]), (uint32_t)(iboDevice[tid * 4 + 2]));
    // edgeMapInsertView.insert(tile, cuco::pair{edge02, (int32_t)(tid * 2 + 0)});

    // //triangle 2
    // edgeMapInsertView.insert(tile, cuco::pair{edge02, (int32_t)(tid * 2 + 1)});

    // auto edge23 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 2]), (uint32_t)(iboDevice[tid * 4 + 3]));
    // edgeMapInsertView.insert(tile, cuco::pair{edge23, (int32_t)(tid * 2 + 1)});

    // auto edge03 = EdgeKeyCuda((uint32_t)(iboDevice[tid * 4 + 0]), (uint32_t)(iboDevice[tid * 4 + 3]));
    // edgeMapInsertView.insert(tile, cuco::pair{edge03, (int32_t)(tid * 2 + 1)});

    update_edge_map(iboDevice, tile, edgeMapInsertView, edgeKeysSet, tid);


    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 0]), (int32_t)(tid * 2 + 0)});
    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 0]), (int32_t)(tid * 2 + 1)});

    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 1]), (int32_t)(tid * 2 + 0)});

    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 2]), (int32_t)(tid * 2 + 0)});
    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 2]), (int32_t)(tid * 2 + 1)});

    // vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 4 + 3]), (int32_t)(tid * 2 + 1)});
    update_vert_map(vertMapInsertView, tile, iboDevice, tid);

    
    // edgeKeysSet.insert(tile, edge01);
    // edgeKeysSet.insert(tile, edge12);
    // edgeKeysSet.insert(tile, edge02);
    // edgeKeysSet.insert(tile, edge23);
    // edgeKeysSet.insert(tile, edge03);

    // testSet.insert(tile, (int)tid);

    tid += gridDim.x * blockDim.x;
    }
}


// template <uint32_t tile_size, typename edgeMap, typename vertMap, typename SetRef>
// __global__ void custom_insert(edgeMap edgeMapInsertView,
// vertMap vertMapInsertView,
// SetRef edgeKeysSet,
// thrust::device_vector<int32_t> iboDevice,
// TriangleInfoDevice* triangleInfoDeviceInput,
// thrust::device_vector<float> faceNormalDevice,
// thrust::device_vector<float> vboDevice,
// int* num_inserted, uint32_t numFaces)
// {
//   auto tid = threadIdx.x + blockIdx.x * blockDim.x;
//   // std::size_t counter = 0;

//   while (tid < numFaces) {
//     auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
//     TriangleInfoDevice tmp;

//     tmp.vert[0] = vboDevice[iboDevice[tid * 3 + 0] * 3 + 0];
//     tmp.vert[1] = vboDevice[iboDevice[tid * 3 + 0] * 3 + 1];
//     tmp.vert[2] = vboDevice[iboDevice[tid * 3 + 0] * 3 + 2];

//     tmp.vert[3] = vboDevice[iboDevice[tid * 3 + 1] * 3 + 0];
//     tmp.vert[4] = vboDevice[iboDevice[tid * 3 + 1] * 3 + 1];
//     tmp.vert[5] = vboDevice[iboDevice[tid * 3 + 1] * 3 + 2];

//     tmp.vert[6] = vboDevice[iboDevice[tid * 3 + 2] * 3 + 0];
//     tmp.vert[7] = vboDevice[iboDevice[tid * 3 + 2] * 3 + 1];
//     tmp.vert[8] = vboDevice[iboDevice[tid * 3 + 2] * 3 + 2];

//     tmp.norm[0] = faceNormalDevice[tid * 3 + 0];
//     tmp.norm[1] = faceNormalDevice[tid * 3 + 1];
//     tmp.norm[2] = faceNormalDevice[tid * 3 + 2];

//     tmp.vertIdx[0] = iboDevice[tid * 3 + 0];
//     tmp.vertIdx[1] = iboDevice[tid * 3 + 1];
//     tmp.vertIdx[2] = iboDevice[tid * 3 + 2];

//     triangleInfoDeviceInput[tid] = tmp;


//     // edge 01
//     edgeMapInsertView.insert(tile, cuco::pair{EdgeKeyCuda((uint32_t)(iboDevice[tid * 3 + 0]), (uint32_t)(iboDevice[tid * 3 + 1])), (int32_t)tid});

//     // edge 12
//     edgeMapInsertView.insert(tile, cuco::pair{EdgeKeyCuda((uint32_t)(iboDevice[tid * 3 + 1]), (uint32_t)(iboDevice[tid * 3 + 2])), (int32_t)tid});

//     // edge 02
//     edgeMapInsertView.insert(tile, cuco::pair{EdgeKeyCuda((uint32_t)(iboDevice[tid * 3 + 0]), (uint32_t)(iboDevice[tid * 3 + 2])), (int32_t)tid});

//     vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 3 + 0]), (int32_t)tid});
//     vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 3 + 1]), (int32_t)tid});
//     vertMapInsertView.insert(tile, cuco::pair{(int32_t)(iboDevice[tid * 3 + 2]), (int32_t)tid});

//     tid += gridDim.x * blockDim.x;
//   }

//   // std::size_t counter = 1;
//   //   atomicAdd(num_inserted, counter);
// }

template <typename SetRef, typename edgeMap>
__global__ void wan_cooperative_insert(SetRef set, edgeMap mapView, std::size_t n)
{
  namespace cg = cooperative_groups;

  constexpr auto cg_size = SetRef::cg_size;

  auto tile = cg::tiled_partition<cg_size>(cg::this_thread_block());

  int64_t const loop_stride = gridDim.x * blockDim.x / cg_size;
  int64_t idx               = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;

  while (idx < n) {
    // set.insert(tile, *(keys + idx));
    mapView.insert(tile, cuco::pair{(int)idx, (int)(idx + 122)});
    set.insert(tile, (int)idx);
    idx += loop_stride;
  }
}

void PTexMesh::GetCudaLastErr()
{
  cudaError_t err;

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
}

// void PTexMesh::CalcEachTriangleCuda(std::vector<TriangleInfoDevice>& triangleArr, uint32_t numFaces, MeshType meshType)
// {
//   using edgeMapKeyType   = EdgeKeyCuda;
//   using edgeMapValueType = int;

//   using vertMapKeyType   = int;
//   using vertMapValueType = int;

//   unsigned long startTime = {0};
//   unsigned long endTime = {0};

//   edgeMapKeyType  emptyEdgeKeySentinel{-1, -1};
//   edgeMapValueType constexpr emptyEdgeValueSentinel = -1;

//   vertMapKeyType emptyVertKeySentinel = -1;
//   vertMapValueType constexpr emptyVertValueSentinel = -1;

//   std::cout << "triangle faces: " << numFaces << std::endl;

//   thrust::device_vector<int32_t> iboDevice;
//   thrust::device_vector<TriangleInfoDevice> triangleInfoDevice;
//   thrust::device_vector<float> faceNormalDevice;
//   thrust::device_vector<float> vboDevice;

//   // triangleInfoDevice.resize(numFaces);
//   faceNormalDevice = m_originalMesh.m_faceNormalVec;
//   iboDevice = m_originalMesh.m_iboVec;
//   vboDevice = m_originalMesh.m_vboVec;

//   using edgeMapProbe = cuco::legacy::linear_probing<1, EdgeKeyCudaHasher>;
//   using vertMapProbe = cuco::legacy::linear_probing<1, cuco::default_hash_function<vertMapKeyType>>;
//   using edgeSetProbe = cuco::linear_probing<1, EdgeKeyCudaHasher>;

//   cuco::static_multimap<edgeMapKeyType, edgeMapValueType, cuda::thread_scope_device, cuco::cuda_allocator<char>, edgeMapProbe> edgeNbrMultiMapDevice{numFaces * 6,
//                                       cuco::empty_key{emptyEdgeKeySentinel},
//                                       cuco::empty_value{emptyEdgeValueSentinel}};

//   cuco::static_multimap<vertMapKeyType, vertMapValueType, cuda::thread_scope_device, cuco::cuda_allocator<char>, vertMapProbe> vertNbrMultiMapDevice{numFaces * 6,
//                                       cuco::empty_key{emptyVertKeySentinel},
//                                       cuco::empty_value{emptyVertValueSentinel}};

//   // cuco::static_set<edgeMapKeyType> edgeKeysSet(numFaces * 6, cuco::empty_key{emptyEdgeKeySentinel});
//   auto edgeKeysSet = cuco::static_set{cuco::extent<std::size_t>{numFaces * 5},
//                                cuco::empty_key{emptyEdgeKeySentinel},
//                                EdgeKeyEqual{},
//                                edgeSetProbe{}};


//   // auto edgeKeysSetTest = cuco::static_set{cuco::extent<std::size_t>{numFaces * 6},
//   //                                      cuco::empty_key{emptyVertKeySentinel},
//   //                                      thrust::equal_to<int>{},
//   //                                      cuco::double_hashing<1, cuco::default_hash_function<int>>{}};

  
//   // wan_cooperative_insert<<<128, 128>>>(edgeKeysSetTest.ref(cuco::insert), vertMapInsertView, numFaces);
//   // cudaStreamSynchronize(0);

//   // thrust::device_vector<int> allKeysTest(1'000'0000);
//   // auto const allKeysTestEnd = edgeKeysSetTest.retrieve_all(allKeysTest.begin());
//   // auto const numTest             = std::distance(allKeysTest.begin(), allKeysTestEnd);

//   // std::cout << "----- allKeysTest in set number: " << numTest << std::endl;



//   assert(edgeNbrMultiMapDevice.cg_size() == vertNbrMultiMapDevice.cg_size());

//   thrust::device_vector<int> num_inserted(1);
//   auto constexpr block_size = 128;
//   auto constexpr stride     = 1;
//   auto const grid_size = (edgeNbrMultiMapDevice.cg_size() * numFaces + stride * block_size - 1) / (stride * block_size);

//   auto edgeMapInsertView = edgeNbrMultiMapDevice.get_device_mutable_view();
//   auto vertMapInsertView = vertNbrMultiMapDevice.get_device_mutable_view();



//     switch (meshType)
//   {
//   case MESH_TYPE_M2_MAPPING:
//   case MESH_TYPE_IM_MESH:
//     std::cout << "into MESH_TYPE_M2_MAPPING MESH_TYPE_IM_MESH kernel \n";
//     // triangleInfoDevice.resize(numFaces);
//     // std::cout << "triangleInfoDevice size : " << triangleInfoDevice.size() << std::endl;
//     // custom_insert<edgeNbrMultiMapDevice.cg_size()><<<grid_size, block_size>>>(edgeMapInsertView, vertMapInsertView, edgeKeysSet.ref(cuco::insert), iboDevice, (TriangleInfoDevice *)thrust::raw_pointer_cast(triangleInfoDevice.data()), faceNormalDevice, vboDevice, num_inserted.data().get(), numFaces);
//     break;

//   case MESH_TYPE_REPLICA:
//     std::cout << "into MESH_TYPE_REPLICA kernel \n";
//     triangleInfoDevice.resize(numFaces * 2);
//     std::cout << "triangleInfoDevice size : " << triangleInfoDevice.size() << std::endl;

//     printf("into MESH_TYPE_REPLICA kernel  iboDevice size: %d, faceNormalDevice: %d, vboDevice: %d, numFaces : %d\n", iboDevice.size(), faceNormalDevice.size(), vboDevice.size(), numFaces);

//     printf("grid_size, block_size : %d, %d\n", grid_size, block_size);

//     startTime = GetTimeMS();
//     insert_triangle_info_quad<<<grid_size, block_size>>> (iboDevice.begin(),
//         (TriangleInfoDevice *)thrust::raw_pointer_cast(triangleInfoDevice.data()),
//         faceNormalDevice.begin(),
//         vboDevice.begin(),
//         numFaces);

//     GetCudaLastErr();
//     checkCudaErrors(cudaDeviceSynchronize());

//     endTime = GetTimeMS();
//     std::cout << "insert_triangle_info_quad kernel run time : " << (double)(endTime - startTime) << "ms" << std::endl;

//     startTime = GetTimeMS();
//     insert_edge_map_quad<edgeNbrMultiMapDevice.cg_size()> <<<grid_size, block_size>>> (edgeMapInsertView,
//         edgeKeysSet.ref(cuco::insert),
//         iboDevice.begin(),
//         numFaces);

//     GetCudaLastErr();
//     checkCudaErrors(cudaDeviceSynchronize());

//     endTime = GetTimeMS();
//     std::cout << "insert_edge_map_quad kernel run time : " << (double)(endTime - startTime) << "ms" << std::endl;

//     startTime = GetTimeMS();
//     insert_vert_map_quad<edgeNbrMultiMapDevice.cg_size()> <<<grid_size, block_size>>> (vertMapInsertView, iboDevice.begin(), numFaces);

//     GetCudaLastErr();
//     checkCudaErrors(cudaDeviceSynchronize());

//     endTime = GetTimeMS();
//     std::cout << "insert_vert_map_quad kernel run time : " << (double)(endTime - startTime) << "ms" << std::endl;

//     // custom_insert_quad<edgeNbrMultiMapDevice.cg_size()> <<<1024, 1024>>>(
//     //   edgeMapInsertView,
//     //   vertMapInsertView,
//     //   edgeKeysSet.ref(cuco::insert),
//     //   iboDevice.begin(),
//     //   (TriangleInfoDevice *)thrust::raw_pointer_cast(triangleInfoDevice.data()),
//     //   faceNormalDevice.begin(),
//     //   vboDevice.begin(),
//     //   num_inserted.data().get(),
//     //   numFaces);

//     //   GetCudaLastErr();
//     //   checkCudaErrors(cudaDeviceSynchronize());
//     break;
  
//   default:
//     break;
//   }


//   startTime = GetTimeMS();
//   thrust::copy(triangleInfoDevice.begin(), triangleInfoDevice.end(), triangleArr.begin());
//   endTime = GetTimeMS();

//   std::cout << "triangleInfoDevice copy time : " << (double)(endTime - startTime) << "ms" << std::endl;

//   // edgeMapKeyType
//   thrust::device_vector<edgeMapKeyType> allEdgeKeys(numFaces * 5);
//   auto const allEdgeKeysEnd = edgeKeysSet.retrieve_all(allEdgeKeys.begin());
//   auto const keyNum             = std::distance(allEdgeKeys.begin(), allEdgeKeysEnd);

//   std::cout << "----- allEdgeKeys in set number: " << keyNum << std::endl;

//   auto edgeMapReadOnlyView = edgeNbrMultiMapDevice.get_device_view();

//   thrust::device_vector<uint32_t> zeroEdgeNbrCnt(1);
//   thrust::device_vector<uint32_t> oneEdgeNbrCnt(1);
//   thrust::device_vector<uint32_t> twoEdgeNbrCnt(1);
//   thrust::device_vector<uint32_t> moreThanTwoEdgeNbrCnt(1);
//   thrust::device_vector<uint32_t> maxEdgeNbr(1);

//   auto constexpr block_size_2 = 128;
//   auto constexpr stride_2     = 1;
//   auto const grid_size_2 = (keyNum + stride_2 * block_size_2 - 1) / (stride_2 * block_size_2);

//   printf("block_size_2 %d, grid_size_2 : %d\n", block_size_2, grid_size_2);

//   validate_edge_map_kernel<edgeNbrMultiMapDevice.cg_size()><<<grid_size_2, block_size_2>>>(allEdgeKeys.begin(), keyNum, edgeMapReadOnlyView, EdgeKeyEqual{}, zeroEdgeNbrCnt.data().get(), oneEdgeNbrCnt.data().get(), twoEdgeNbrCnt.data().get(), moreThanTwoEdgeNbrCnt.data().get(), maxEdgeNbr.data().get());

//   std::cout << "key type: zero nbr: " << zeroEdgeNbrCnt[0] << ", one nbr: " << oneEdgeNbrCnt[0] <<  ", two nbr: " <<  twoEdgeNbrCnt[0] << ", morn than 2 nbr: "<< moreThanTwoEdgeNbrCnt[0] << " max br cnt: " << maxEdgeNbr[0] << std::endl;


//   std::unordered_set<int32_t> uniqueIdxs(m_originalMesh.m_iboVec.begin(), m_originalMesh.m_iboVec.end());
//   std::vector<int32_t> uniqueKeys;
//   uniqueKeys.assign(uniqueIdxs.begin(), uniqueIdxs.end());

//   printf("ibo all size: %d, unique idx size : %d\n", m_originalMesh.m_iboVec.size(), uniqueIdxs.size());

//   auto constexpr block_size_3 = 128;
//   auto constexpr stride_3     = 1;
//   auto const grid_size_3 = (uniqueIdxs.size() + stride_3 * block_size_3 - 1) / (stride_3 * block_size_3);
//   thrust::device_vector<int32_t> validateVertKeys;
//   validateVertKeys = uniqueKeys;
//   thrust::device_vector<uint32_t> maxVertNbrNum(1);
//   thrust::device_vector<uint32_t> cannotFindNum(1);

//   auto vertMapReadOnlyView = vertNbrMultiMapDevice.get_device_view();
//   validate_vert_map_kernel<vertNbrMultiMapDevice.cg_size()><<<grid_size_3, block_size_3>>>(validateVertKeys.begin(), uniqueIdxs.size(), vertMapReadOnlyView, maxVertNbrNum.data().get(), cannotFindNum.data().get());

//   printf("block_size_3 %d, grid_size_3 : %d\n", block_size_3, grid_size_3);
//   std::cout << "max vert nbr num :  " << maxVertNbrNum[0] << " cannot find num:  " <<  cannotFindNum[0] << std::endl;

// }

void PTexMesh::PrintProgressBar(int progress, int total)
{
    int barWidth = 50; // 进度条的宽度
    float ratio = static_cast<float>(progress) / total;
    int pos = barWidth * ratio;

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "="; // 已完成部分
        else if (i == pos) std::cout << ">"; // 当前部分
        else std::cout << " "; // 未完成部分
    }
    std::cout << "] " << static_cast<int>(ratio * 100) << " %\r"; // \r 返回到行首
    std::cout.flush(); // 刷新输出
}

unsigned long PTexMesh::GetTimeMS()
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}


// std::vector<MeshData> PTexMesh::SplitMesh(const MeshData& mesh, const float splitSize)
// {
//   std::vector<uint32_t> verts;
//   verts.resize(mesh.vbo.size());

//   auto Part1By2 = [](uint64_t x) {
//     x &= 0x1fffff; // mask off lower 21 bits
//     x = (x | (x << 32)) & 0x1f00000000ffff;
//     x = (x | (x << 16)) & 0x1f0000ff0000ff;
//     x = (x | (x << 8)) & 0x100f00f00f00f00f;
//     x = (x | (x << 4)) & 0x10c30c30c30c30c3;
//     x = (x | (x << 2)) & 0x1249249249249249;
//     return x;
//   };

//   auto EncodeMorton3 = [&Part1By2](const Eigen::Vector3i& v) {
//     return (Part1By2(v(2)) << 2) + (Part1By2(v(1)) << 1) + Part1By2(v(0));
//   };

//   Eigen::AlignedBox3f boundingBox;

//   for (size_t i = 0; i < mesh.vbo.Area(); i++) {
//     boundingBox.extend(mesh.vbo[i].head<3>());
//   }

// // calculate vertex grid position and code
// #pragma omp parallel for
//   for (size_t i = 0; i < mesh.vbo.size(); i++) {
//     const Eigen::Vector3f p = mesh.vbo[i].head<3>();
//     Eigen::Vector3f pi = (p - boundingBox.min()) / splitSize;
//     verts[i] = EncodeMorton3(pi.cast<int>());
//   }

//   // data structure for sorting faces
//   struct SortFace {
//     uint32_t index[4];
//     uint32_t code;
//     size_t originalFace;
//   };

//   // fill per-face data structures (including codes)
//   size_t numFaces = mesh.ibo.size() / 4;
//   std::vector<SortFace> faces;
//   faces.resize(numFaces);

// #pragma omp parallel for
//   for (size_t i = 0; i < numFaces; i++) {
//     faces[i].originalFace = i;
//     faces[i].code = std::numeric_limits<uint32_t>::max();
//     for (int j = 0; j < 4; j++) {
//       faces[i].index[j] = mesh.ibo[i * 4 + j];

//       // face code is minimum of referenced vertices codes
//       faces[i].code = std::min(faces[i].code, verts[faces[i].index[j]]);
//     }
//   }

//   // sort faces by code
//   std::sort(faces.begin(), faces.end(), [](const SortFace& f1, const SortFace& f2) -> bool {
//     return (f1.code < f2.code);
//   });

//   // find face chunk start indices
//   std::vector<uint32_t> chunkStart;
//   chunkStart.push_back(0);
//   uint32_t prevCode = faces[0].code;
//   for (size_t i = 1; i < faces.size(); i++) {
//     if (faces[i].code != prevCode) {
//       chunkStart.push_back(i);
//       prevCode = faces[i].code;
//     }
//   }

//   chunkStart.push_back(faces.size());
//   size_t numChunks = chunkStart.size() - 1;

//   size_t maxFaces = 0;
//   for (size_t i = 0; i < numChunks; i++) {
//     uint32_t chunkSize = chunkStart[i + 1] - chunkStart[i];
//     if (chunkSize > maxFaces)
//       maxFaces = chunkSize;
//   }

//   // create new mesh for each chunk of faces
//   std::vector<MeshData> subMeshes;

//   for (size_t i = 0; i < numChunks; i++) {
//     subMeshes.emplace_back(4);
//   }

// #pragma omp parallel for
//   for (size_t i = 0; i < numChunks; i++) {
//     uint32_t chunkSize = chunkStart[i + 1] - chunkStart[i];

//     std::vector<uint32_t> refdVerts;
//     std::unordered_map<uint32_t, uint32_t> refdVertsMap;
//     subMeshes[i].ibo.Reinitialise(chunkSize * 4, 1);

//     for (size_t j = 0; j < chunkSize; j++) {
//       size_t faceIdx = chunkStart[i] + j;
//       for (int k = 0; k < 4; k++) {
//         uint32_t vertIndex = faces[faceIdx].index[k];
//         uint32_t newIndex = 0;

//         auto it = refdVertsMap.find(vertIndex);

//         if (it == refdVertsMap.end()) {
//           // vertex not found, add
//           newIndex = refdVerts.size();
//           refdVerts.push_back(vertIndex);
//           refdVertsMap[vertIndex] = newIndex;
//         } else {
//           // found, use existing index
//           newIndex = it->second;
//         }
//         subMeshes[i].ibo[j * 4 + k] = newIndex;
//       }
//     }

//     // add referenced vertices to submesh
//     subMeshes[i].vbo.Reinitialise(refdVerts.size(), 1);
//     subMeshes[i].nbo.Reinitialise(refdVerts.size(), 1);
//     for (size_t j = 0; j < refdVerts.size(); j++) {
//       uint32_t index = refdVerts[j];
//       subMeshes[i].vbo[j] = mesh.vbo[index];
//       subMeshes[i].nbo[j] = mesh.nbo[index];
//     }
//   }

//   return subMeshes;
// }

void PTexMesh::LoadMeshDataJianHengLiu(const std::string& meshFile, float splitSize)
{
  // Load the meshes
    ASSERT(pangolin::FileExists(meshFile));
    PLYParseJianHengLiu(meshFile);

    ASSERT(m_originalMesh.polygonStride == 3, "Must be a triangle mesh!");
    std::cout << " LoadMeshDataJianHengLiu done" << std::endl;
}

void PTexMesh::LoadMeshDataBlenderOut(const std::string& meshFile, float splitSize)
{
  // Load the meshes
    ASSERT(pangolin::FileExists(meshFile));
    PLYParseBlenderOut(meshFile);

    ASSERT(m_originalMesh.polygonStride == 3, "Must be a quad mesh!");
    std::cout << " LoadMeshDataBlenderOut done" << std::endl;
}

void PTexMesh::LoadMeshDataReplica(const std::string& meshFile, float splitSize)
{
  // Load the meshes
    ASSERT(pangolin::FileExists(meshFile));
    PLYParseReplica(meshFile);

    ASSERT(m_originalMesh.polygonStride == 4, "Must be a quad mesh!");

    // if (splitSize > 0.0f) {
    // std::cout << "Splitting mesh... ";
    // std::cout.flush();
    // m_splitMeshData = SplitMesh(m_originalMesh, splitSize);
    // } else {
    //   m_splitMeshData.emplace_back(std::move(m_originalMesh));
    // }

    std::cout << " LoadMeshDataReplica done" << std::endl;
    
}

uint32_t PTexMesh::GetNumFaces(MeshType meshType)
{
  switch (meshType) {
        case MESH_TYPE_IM_MESH:
            return (m_originalMesh.m_iboVec.size() / 3);
        
        case MESH_TYPE_REPLICA:
            return (m_originalMesh.m_iboVec.size() / 4);

        case MESH_TYPE_BLENDER_OUT:
            return (m_originalMesh.m_iboVec.size() / 3);
        case MESH_TYPE_M2_MAPPING:
            return (m_originalMesh.m_iboVec.size() / 3);

        default:
            return 0;
    }

    return 0;
}

void PTexMesh::LoadMeshData(const std::string& meshFile, float splitSize, MeshType meshType)
{
  switch (meshType) {
        case MESH_TYPE_IM_MESH:
            LoadMeshDataImMesh(meshFile, 3.f);
            break;
        
        case MESH_TYPE_REPLICA:
            LoadMeshDataReplica(meshFile, 3.f);
            break;

        case MESH_TYPE_BLENDER_OUT:
            LoadMeshDataBlenderOut(meshFile, 3.f);
            break;

        case MESH_TYPE_M2_MAPPING:
            LoadMeshDataJianHengLiu(meshFile, 3.f);
            break;

        default:
            std::cout << "invalid mesh type !!\n";
            exit(0);
    }
}

void PTexMesh::LoadMeshDataImMesh(const std::string& meshFile, float splitSize) {
  // Load the meshes
    ASSERT(pangolin::FileExists(meshFile));
    PLYParseImMesh(meshFile);

    ASSERT(m_originalMesh.polygonStride == 3, "Must be a triangle mesh!");

    // if (splitSize > 0.0f) {
    // std::cout << "Splitting mesh... ";
    // std::cout.flush();
    // m_splitMeshData = SplitMesh(m_originalMesh, splitSize);
    // } else {
    //   m_splitMeshData.emplace_back(std::move(m_originalMesh));
    // }

    std::cout << " LoadMeshDataImMesh done" << std::endl;
    
}

__global__ void my_initialize_atomic(my_atomic_ctr_type* atms, uint32_t numOfAtms)
{
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < numOfAtms) {
    atms[tid].store(0, cuda::memory_order_relaxed);
  }
}

