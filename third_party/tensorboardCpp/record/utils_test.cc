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

#include <cstring>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

namespace nlptk {

using std::string;

#define MAXHOSTNAMELEN 256

string _GetHostName() {
  char hostname[MAXHOSTNAMELEN];
  if (gethostname(hostname, MAXHOSTNAMELEN) == -1) {
    LOG(ERROR) << "Failed to get hostname due to " << strerror(errno);
    return "";
  }

  hostname[MAXHOSTNAMELEN - 1] = '\0';
  return hostname;
}

static const char kUUIDChars[] = "0123456789ABCDEF";
string UUID() {
  std::srand(time(nullptr) + clock());
  string uuid;
  for (int i = 0; i < 16; ++i) {
    uuid.push_back(kUUIDChars[std::rand() & 0x0F]);
  }

  return uuid;
}

TEST(Utils, EnvHostName) {
  EXPECT_EQ(Env::HostName, _GetHostName());
  LOG(INFO) << Env::HostName;
}

TEST(Utils, Timestamp) {
  double timestamp = Timestamp();
  EXPECT_LE(0.0, timestamp);
  LOG(INFO) << "Current timestamp: " << timestamp;
}

TEST(Utils, PathAndMakeDirs) {
  string dir = "tmp_";
  dir += UUID();
  EXPECT_FALSE(IsExisted(dir));
  EXPECT_LE(0, MakeDirs(dir));
  EXPECT_TRUE(IsExisted(dir));

  dir = "tmp_not_existed/tmp";
  EXPECT_FALSE(IsExisted("tmp_not_existed"));
  EXPECT_FALSE(IsExisted(dir));
  EXPECT_GT(0, MakeDirs(dir, false));
  EXPECT_FALSE(IsExisted(dir));

  dir = "tmp_not_existed_to_mk_";
  dir += UUID();
  EXPECT_FALSE(IsExisted(dir));

  dir += "/tmp";
  EXPECT_FALSE(IsExisted(dir));
  EXPECT_LE(0, MakeDirs(dir, true));
  EXPECT_TRUE(IsExisted(dir));
}

}  // namespace nlptk
