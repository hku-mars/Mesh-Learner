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

#include "record/utils.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <chrono>           // NOLINT(build/c++11)
#include <cstdarg>
#include <vector>

#include "glog/logging.h"

namespace nlptk {

using std::chrono::microseconds;
using std::chrono::system_clock;
using std::string;
using std::vector;

#define MAXHOSTNAMELEN 256

namespace internal {

string GetHostName() {
  char hostname[MAXHOSTNAMELEN];
  if (gethostname(hostname, MAXHOSTNAMELEN) == -1) {
    LOG(ERROR) << "Failed to get hostname due to " << strerror(errno);
    return "";
  }

  hostname[MAXHOSTNAMELEN - 1] = '\0';
  return hostname;
}

int AppendFormat(string* dst, const string& format, std::va_list ap) {
  typename string::value_type stack_buf[1024];
  va_list ap_copy;
  va_copy(ap_copy, ap);
  int ret =
      std::vsnprintf(stack_buf, 1024, format.c_str(), ap_copy);  // NOLINT
  va_end(ap_copy);

  if (ret >= 0 && ret < static_cast<int>(1024)) {
    dst->append(stack_buf, ret);
    return ret;
  }

  int len = 1024;
  while (true) {
    if (ret < 0) {
      if (errno != 0 && errno != EOVERFLOW) {
        return ret;
      }

      len *= 2;
    } else {
      len = ret + 1;
    }

    if (len > 32 * 1024 * 1024) {
      LOG(ERROR) << "Unable to format the requested string due to size.";
      return -1;
    }

    vector<string::value_type> buf(len);
    va_copy(ap_copy, ap);
    ret = std::vsnprintf(buf.data(), len, format.c_str(), ap_copy);
    va_end(ap_copy);

    if ((ret >= 0) && (ret < len)) {
      dst->append(buf.data(), ret);
      return ret;
    }
  }

  return 0;
}

}  // namespace internal

const string Env::HostName = internal::GetHostName();  // NOLINT(runtime/string)

static const char kSpaces[] = " \t\r\n";
bool StringUtil::EmptyOrSpaces(const string& str) {
  return str.empty() || str.find_first_not_of(kSpaces) == string::npos;
}

string StringUtil::Format(const string& format, ...) {
  std::va_list ap;
  va_start(ap, format);
  string text;
  auto code = internal::AppendFormat(&text, format, ap);
  va_end(ap);

  if (code < 0) {
    text.clear();
  }

  return text;
}

int StringUtil::AppendFormat(string* dst, const string& format, ...) {
  assert(dst);

  std::va_list ap;
  va_start(ap, format);
  int ret = internal::AppendFormat(dst, format, ap);
  va_end(ap);

  return ret;
}

double Timestamp() {
  auto tp = std::chrono::time_point_cast<microseconds>(system_clock::now());
  return tp.time_since_epoch().count() / 1000000.0;
}

int MakeDirs(const string& dir, bool parents) {
  if (IsExisted(dir)) {
    if (IsDirectory(dir)) {
      return 0;
    } else {
      LOG(ERROR) << "Failed create directory '" << dir << "': File exists";
      return -1;
    }
  }

  auto parent_path = GetDirectoryPath(dir);
  if (!IsExisted(parent_path)) {
    if (!parents) {
      LOG(ERROR) << "Failed create directory '" << dir
                 << "': No such file or directory";
      return -1;
    }

    if (MakeDirs(parent_path) < 0) {
      return -1;
    }
  }

  if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) {
    LOG(ERROR) << "Failed to create tensorboard log dir '" << dir
               << "', due to: " << strerror(errno);
    return -1;
  }

  return 0;
}

bool IsExisted(const string& path) {
  return access(path.c_str(), F_OK) != -1;
}

bool IsDirectory(const string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0) {
    return S_ISDIR(st.st_mode);
  }

  return false;
}

string GetDirectoryPath(const string& path, bool absolute) {
  auto root = path;
  if (absolute) {
    root = AbsolutePath(path);
  }

  auto pos = root.find_last_of('/');
  if (pos != path.npos) {
    return path.substr(0, pos);
  }

  return absolute ? "/" : ".";
}

string AbsolutePath(const string& path) {
  char full_path[PATH_MAX];
  if (realpath(path.c_str(), full_path) != nullptr) {
    return full_path;
  }

  return path;
}

string JoinPath(const string& path, const string& sub_path) {
  if (!path.empty() && path.back() == '/') {
    return path + sub_path;
  } else {
    return path + "/" + sub_path;
  }
}

}  // namespace nlptk
