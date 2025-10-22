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

#ifndef UTILS_IMAGE_H_
#define UTILS_IMAGE_H_

#include <string>
#include <vector>

namespace nlptk {

class Image {
 public:
  enum class Type : uint8_t {
    kBMP,
    kHDR,   // not supported right now!
    kJPG,
    kPNG,
    kTGA,
  };

  static Image* Load(const std::string& file_path);

  static Image* LoadFromMem(const std::string& buf);

  static Image* LoadFromMem(u_char const *buf, size_t length);

  static int Write(const char* data, uint32_t width, uint32_t height,
                   uint32_t channel, std::string* buf, Type type = Type::kPNG);

 public:
  Image(const std::vector<u_char>& data, uint32_t width, uint32_t height,
        uint32_t channel);

  Image(const std::string& data, uint32_t width, uint32_t height,
        uint32_t channel);

  virtual ~Image();

  const std::string& Data() const;

  uint32_t Width() const;

  uint32_t Height() const;

  uint32_t Channel() const;

  int Write(const std::string& file_path, Type type = Type::kPNG) const;

  int Write(std::string* buf, Type type = Type::kPNG) const;

 private:
  Image();

  Image(const u_char* data, uint32_t width, uint32_t height, uint32_t channe);

  uint32_t      width_;
  uint32_t      height_;
  uint32_t      colorspace_;
  std::string   data_;
};

}  // namespace nlptk

#endif  // UTILS_IMAGE_H_
