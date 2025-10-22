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

workspace(name = "tensorboard.cpp")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

new_git_repository(
  name = "stb",
  remote = "https://github.com/nothings/stb.git",
  commit = "5736b15f7ea0ffb08dd38af21067c314d6a3aae9",
  shallow_since = "1675018027 -0800",
  build_file = "//third_party/stb:BUILD.bazel",
)

http_archive(
  name = "glog",
  sha256 = "8a83bf982f37bb70825df71a9709fa90ea9f4447fb3c099e1d720a439d88bad6",
  strip_prefix = "glog-0.6.0",
  urls = [
    "https://github.com/google/glog/archive/v0.6.0.tar.gz"
  ],
)

http_archive(
  name = "com_github_gflags_gflags",
  sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
  strip_prefix = "gflags-2.2.2",
  urls = [
    "https://mirror.bazel.build/github.com/gflags/gflags/archive/v2.2.2.tar.gz",
    "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"
  ],
)

http_archive(
  name = "gtest",
  sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
  strip_prefix = "googletest-release-1.12.1",
  urls = [
    "https://github.com/google/googletest/archive/release-1.12.1.tar.gz"
  ],
)

http_archive(
  name = "com_google_protobuf",
  sha256 = "c29d8b4b79389463c546f98b15aa4391d4ed7ec459340c47bffe15db63eb9126",
  strip_prefix = "protobuf-3.21.3",
  urls = [
    "https://github.com/protocolbuffers/protobuf/archive/v3.21.3.tar.gz"
  ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()
