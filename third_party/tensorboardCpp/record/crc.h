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

#ifndef RECORD_CRC_H_
#define RECORD_CRC_H_

#include <cstddef>
#include <cstdint>

namespace nlptk {

// Cycle Redundance Check, CRC

uint32_t CRC32c(const char* buf, size_t len);

uint32_t MaskedCRC32c(const char* buf, size_t len);

}  // namespace nlptk

#endif  // RECORD_CRC_H_
