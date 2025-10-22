// Copyright (c) 2023 Mininglamp Tech. Inc. (Liang Zhao)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils/image.h"

#include <algorithm>

#include "glog/logging.h"

#include "stb_image.h"          // NOLINT(build/include_subdir)
#include "stb_image_write.h"    // NOLINT(build/include_subdir)

using std::string;
using std::vector;

namespace nlptk {

Image::Image() : width_(0), height_(0), colorspace_(0), data_("") {
}

Image::Image(const vector<u_char>& d, uint32_t w, uint32_t h, uint32_t c)
    : Image(d.data(), w, h, c) {
  if (d.size() != w * h * c) {
    width_ = height_ = colorspace_ = 0;
    data_.clear();
  }
}

Image::Image(const string& d, uint32_t w, uint32_t h, uint32_t c)
    : Image(reinterpret_cast<const u_char*>(d.data()), w, h, c) {
  if (d.size() != w * h * c) {
    LOG(ERROR) << "Invalid piexl data!";
    width_ = height_ = colorspace_ = 0;
    data_.clear();
  }
}

Image::Image(const u_char* d, uint32_t w, uint32_t h, uint32_t c)
    : width_(w), height_(h), colorspace_(c), data_("") {
  data_.reserve(w * h * c);
  data_.append(reinterpret_cast<const char*>(d), w * h * c);
}

Image* Image::Load(const string& file_path) {
  int w, h, c;
  stbi_uc* pixels = stbi_load(file_path.c_str(), &w, &h, &c, STBI_default);
  if (nullptr == pixels) {
    LOG(ERROR) << "Failed to load image " << file_path;
    return nullptr;
  }

  auto img = new Image(pixels, w, h, c);
  stbi_image_free(pixels);
  return img;
}

Image* Image::LoadFromMem(const string& buf) {
  return LoadFromMem(reinterpret_cast<const u_char*>(buf.data()), buf.size());
}

Image* Image::LoadFromMem(u_char const* buf, size_t len) {
  int w, h, c;
  stbi_uc* pixels = stbi_load_from_memory(buf, len, &w, &h, &c, STBI_default);
  if (nullptr == pixels) {
    LOG(ERROR) << "Failed to load image from memory";
    return nullptr;
  }

  auto img = new Image(pixels, w, h, c);
  stbi_image_free(pixels);
  return img;
}

Image::~Image() {
}

const string& Image::Data() const {
  return data_;
}

uint32_t Image::Width() const {
  return width_;
}

uint32_t Image::Height() const {
  return height_;
}

uint32_t Image::Channel() const {
  return colorspace_;
}

int Image::Write(const string& file_path, Type type) const {
  switch (type) {
    case Type::kBMP: {
      return stbi_write_bmp(file_path.c_str(), width_, height_, colorspace_,
                            data_.data());
    }

    // case Type::kHDR: {
    //   // float data
    //   return stbi_write_hdr(file_path.c_str(), width_, height_, colorspace_,
    //                         data_.data());
    // }

    case Type::kJPG: {
      return stbi_write_jpg(file_path.c_str(), width_, height_, colorspace_,
                            data_.data(), 95);
    }

    case Type::kPNG: {
      return stbi_write_png(file_path.c_str(), width_, height_, colorspace_,
                            data_.data(), 0);
    }

    case Type::kTGA: {
      return stbi_write_tga(file_path.c_str(), width_, height_, colorspace_,
                            data_.data());
    }

    default: {
      LOG(ERROR) << "Unsupport image type";
      return -1;
    }
  }

  return -1;
}

void bufcpy(void* context, void* data, int size) {
  string& buf = *static_cast<string*>(context);
  const char* start = static_cast<const char*>(data);
  buf.append(start, size);
}

int Image::Write(const char* data, uint32_t w,
                 uint32_t h, uint32_t c, string* buf,
                 Type type) {
  switch (type) {
    case Type::kBMP: {
      return stbi_write_bmp_to_func(bufcpy, buf, w, h, c, data);
    }

    // case Type::kHDR: {
    //   // float data
    //   return stbi_write_hdr_to_func(bufcpy, buf, w, h, c, data);
    // }

    case Type::kJPG: {
      return stbi_write_jpg_to_func(bufcpy, buf, w, h, c, data, 95);
    }

    case Type::kPNG: {
      return stbi_write_png_to_func(bufcpy, buf, w, h, c, data, 0);
    }

    case Type::kTGA: {
      return stbi_write_tga_to_func(bufcpy, buf, w, h, c, data);
    }

    default: {
      LOG(ERROR) << "Unsupport image type";
      return -1;
    }
  }

  return -1;
}

int Image::Write(string* buf, Type type) const {
  return Write(data_.data(), width_, height_, colorspace_, buf, type);
}

}  // namespace nlptk
