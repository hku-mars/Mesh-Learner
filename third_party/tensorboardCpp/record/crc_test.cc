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

#include "record/crc.h"

#include <string>

#include "gtest/gtest.h"

namespace nlptk {

using std::string;

TEST(CRC, CRC32c) {
  string content = "";
  auto checksum = CRC32c(content.c_str(), content.size());
  EXPECT_EQ(0x00000000, checksum);

  content = "121aba";
  checksum = CRC32c(content.c_str(), content.size());
  EXPECT_EQ(0x4D5AD9C8, checksum);

  content = "qwassfaf";
  checksum = CRC32c(content.c_str(), content.size());
  EXPECT_EQ(0xCB14B9B2, checksum);
}

TEST(CRC, MaskedCRC32c) {
  string content = "";
  auto checksum = MaskedCRC32c(content.c_str(), content.size());
  EXPECT_EQ(0xA282EAD8, checksum);

  content = "121aba";
  checksum = MaskedCRC32c(content.c_str(), content.size());
  EXPECT_EQ(0x5613858D, checksum);

  content = "qwassfaf";
  checksum = MaskedCRC32c(content.c_str(), content.size());
  EXPECT_EQ(0x15E88101, checksum);
}

}  // namespace nlptk
