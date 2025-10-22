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

#include "examples/add_event.h"

#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "record/utils.h"
#include "utils/image.h"

using nlptk::Image;
using nlptk::Recorder;
using nlptk::StringUtil;

using std::string;
using std::vector;

string ReadData(const string& filename) {
  std::ifstream fin(filename, std::ios::binary);
  if (!fin.is_open() || fin.fail()) {
    LOG(ERROR) << "Failed to open file " << filename;
    fin.close();
    return "";
  }

  std::ostringstream ss;
  ss << fin.rdbuf();
  fin.close();
  return ss.str();
}

void AddScalar(const nlptk::Recorder& recorder) {
  std::default_random_engine generator;
  std::normal_distribution<float> normal(0, 0.1);
  for (int64_t i = 0; i < 10; ++i) {
    recorder.AddScalar("scalar", normal(generator), i);
    recorder.AddScalar("scalar/v1", normal(generator), i);
    recorder.AddScalar("scalar/v2", normal(generator), i);
  }
}

void AddScalars(nlptk::Recorder& recorder) {    // NOLINT(runtime/references)
  std::default_random_engine generator;
  std::normal_distribution<float> normal(0, 0.1);
  for (int64_t i = 0; i < 10; ++i) {
    auto x = normal(generator);
    recorder.AddScalars("scalars", {{"x", x}, {"y", x + 1.0}}, i);
  }
}

void AddHistogram(const nlptk::Recorder& recorder) {
  std::default_random_engine generator;
  for (uint64_t i = 0; i < 10; ++i) {
    std::normal_distribution<double> normal(i * 0.1, 1.0);
    vector<double> values;
    for (int j = 0; j < 10000; ++j) {
      values.push_back(normal(generator));
    }

    recorder.AddHistogram("histogram", values, i);
  }
}

void AddImage(const nlptk::Recorder& recorder) {
  auto scalar = ReadData("assets/screenshot.png");
  recorder.AddImage("image/screenshot", scalar, {1027, 1913, 3}, 0);
}

void AddImages(const nlptk::Recorder& recorder) {
  vector<string> images;
  uint32_t w = 420, h = 320, c = 4;
  for (int i = 0; i < 11; ++i) {
    auto filename = StringUtil::Format("assets/img%02d.png", i);
    auto image = Image::Load(filename);
    if (w != image->Width() || h != image->Height() || c != image->Channel()) {
      LOG(ERROR) << "Skip invalid image from '" << filename << "' with"
                 << " width=" << image->Width()
                 << " height=" << image->Height()
                 << " channel=" << image->Channel();
      continue;
    }

    images.push_back(image->Data());
    delete image;
  }

  recorder.AddImages("images", images, {420, 320, 4}, 1);
}

void AddAudio(const nlptk::Recorder& recorder) {
  auto audio = ReadData("assets/piano.mp3");
  recorder.AddAudio("audio/piano", audio,
                    {1, 48000 * 56, 48000., "audio/mp3"}, 1);
}

void AddText(const nlptk::Recorder& recorder) {
  for (int64_t i = 0; i < 10; ++i) {
    recorder.AddText("text", StringUtil::Format("text-%d", i), i);
  }
}

void AddProjector(const nlptk::Recorder& recorder) {
  size_t N = 100, D = 5;
  vector<float> mat;
  std::default_random_engine generator;
  std::normal_distribution<float> normal(0, 1);
  vector<string> labels;
  for (size_t i = 0; i < N; ++i) {
    labels.push_back(StringUtil::Format("L-%d", i));
    for (size_t j = 0; j < D; ++j) {
      mat.push_back(normal(generator));
    }
  }

  recorder.AddEmbedding(mat, N, D, labels, 0, "embedding");
}
