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

#ifndef RECORD_UTILS_H_
#define RECORD_UTILS_H_

#include <string>

namespace nlptk {

class Env {
 public:
  static const std::string HostName;
};

class StringUtil {
 public:
  static bool EmptyOrSpaces(const std::string& str);

  static std::string Format(const std::string& format, ...);

  static int AppendFormat(std::string* dst, const std::string& format, ...);
};

double Timestamp();

int MakeDirs(const std::string& dir, bool parents = true);

bool IsExisted(const std::string& path);

bool IsDirectory(const std::string& path);

std::string GetDirectoryPath(const std::string& path, bool absolute = true);

std::string AbsolutePath(const std::string& path);

std::string JoinPath(const std::string& path, const std::string& sub_path);

}  // namespace nlptk

#endif  // RECORD_UTILS_H_

