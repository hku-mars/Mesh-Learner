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

load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

proto_library(
  name = "tensorboard_interface_pb",
  deps = [
    ":event_pb",
    ":graph_pb",
    ":onnx_pb",
  ],
)

proto_library(
  name = "event_pb",
  srcs = [
    "event.proto",
    "summary.proto",

  ],
  deps = [
    ":tensor_pb",
  ]
)

proto_library(
  name = "onnx_pb",
  srcs = [
    "onnx.proto",
  ]
)

proto_library(
  name = "graph_pb",
  srcs = [
    "graph.proto",
    "node_def.proto",
    "versions.proto",
    "attr_value.proto",
  ],
  deps = [
    ":tensor_pb",
  ]
)

proto_library(
  name = "tensor_pb",
  srcs = [
    "tensor.proto",
    "resource_handle.proto",
    "tensor_shape.proto",
    "types.proto",
  ],
)

cc_proto_library(
  name = "tensorboard_interface",
  deps = [
    ":tensorboard_interface_pb",
  ],
)

cc_proto_library(
  name = "event",
  deps = [
    ":event_pb"
  ],
)

cc_proto_library(
  name = "onnx",
  deps = [
    ":onnx_pb"
  ],
)

cc_proto_library(
  name = "graph",
  deps = [
    ":graph_pb"
  ],
)
