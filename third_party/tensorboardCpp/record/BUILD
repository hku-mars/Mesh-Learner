# Copyright (c) 2023 Mininglamp Tech. Inc. (Liang Zhao)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])


cc_library(
  name = "record",
  srcs = [
    "async_file_writer.cc",
    "crc.cc",
    "crc.h",
    "file_writer.cc",
    "recorder.cc",
    "summary.cc",
    "summary.h",
    "utils.cc",
    "utils.h",
    "writer.cc",
  ],
  hdrs = [
    "async_file_writer.h",
    "file_writer.h",
    "recorder.h",
    "writer.h",
  ],
  deps = [
    "//proto:tensorboard_interface",
    "//utils:image",
    "@glog//:glog",
  ],
)

cc_test(
  name = "unittest",
  srcs = [
    "crc_test.cc",
    "recorder_test.cc",
    "utils_test.cc",
  ],
  data = [
    "//assets:recorder_test_data",
  ],
  deps = [
    ":record",
    "@gtest//:gtest_main",
  ]
)
