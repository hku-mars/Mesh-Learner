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


#include <string>

#include "examples/add_event.h"
#include "glog/logging.h"
#include "record/recorder.h"

using nlptk::Recorder;

using std::string;

int main(int argc, char* argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // set the dir to store events for tensorboard
  string dir = argc > 1 ? argv[1] : "runs";

  // Init recoder with default file writer
  Recorder recorder(dir);

  if (!recorder.Ready()) {
    LOG(ERROR) << "Failed to initialize tensorboard recorder. Exit!";
    return -1;
  }

  // Add Scalar
  AddScalar(recorder);

  // Add Scalars
  AddScalars(recorder);

  // Add Histogram
  AddHistogram(recorder);

  // Add Image
  AddImage(recorder);

  // Add Images
  AddImages(recorder);

  // Add Audio
  AddAudio(recorder);

  // Add Text
  AddText(recorder);

  // Add Projector
  AddProjector(recorder);

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
