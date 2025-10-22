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

#include "record/recorder.h"

#include <fstream>
#include <random>
#include <sstream>
#include <utility>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "record/async_file_writer.h"
#include "record/utils.h"
#include "utils/image.h"

using std::ifstream;
using std::ostringstream;
using std::string;
using std::vector;

using nlptk::AsyncFileWriter;
using nlptk::Image;
using nlptk::Recorder;
using nlptk::StringUtil;
using nlptk::Writer;

string ReadBinaryFile(const string& filename) {
  ifstream fin(filename, std::ios::binary);
  if (!fin.is_open() || fin.fail()) {
    LOG(ERROR) << "Failed to open file " << filename;
    fin.close();
    return "";
  }

  ostringstream ss;
  ss << fin.rdbuf();
  fin.close();
  return ss.str();
}

TEST(Recorder, Init) {
  string dir = "runs";
  Recorder recorder(dir);

  ASSERT_TRUE(recorder.Ready());
  EXPECT_TRUE(nlptk::IsExisted(dir));
}

TEST(Recorder, AddScalar) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  string dir = "runs";
  Recorder recorder(dir);
  ASSERT_TRUE(recorder.Ready());

  std::default_random_engine generator;
  std::normal_distribution<float> normal(0, 0.1);
  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_LT(0, recorder.AddScalar("scalar", normal(generator), i));
    EXPECT_LT(0, recorder.AddScalar("scalar/v1", normal(generator), i));
    EXPECT_LT(0, recorder.AddScalar("scalar/v2", normal(generator), i));
  }
}

TEST(Recorder, AddScalars) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  string dir = "runs";
  Recorder recorder(dir);
  ASSERT_TRUE(recorder.Ready());

  std::default_random_engine generator;
  std::normal_distribution<float> normal(0, 0.1);
  for (int64_t i = 0; i < 10; ++i) {
    auto x = normal(generator);
    EXPECT_LT(0, recorder.AddScalars("scalars", {{"x", x}, {"y", x + 1.0}}, i));
  }

  // google::protobuf::ShutdownProtobufLibrary();
}

TEST(Recorder, AddHistogram) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  string dir = "runs";
  Recorder recorder(dir);
  ASSERT_TRUE(recorder.Ready());

  std::default_random_engine generator;
  for (uint64_t i = 0; i < 10; ++i) {
    std::normal_distribution<double> normal(i * 0.1, 1.0);
    vector<double> values;
    for (int j = 0; j < 10000; ++j) {
      values.push_back(normal(generator));
    }

    EXPECT_LE(0, recorder.AddHistogram("histogram", values, i));
  }
}

TEST(Recorder, AddImage) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  string dir = "runs";
  Recorder recorder(dir);
  ASSERT_TRUE(recorder.Ready());

  auto shot = ReadBinaryFile("assets/screenshot.png");
  auto scalars = ReadBinaryFile("assets/scalars.png");

  EXPECT_LT(0, recorder.AddImage("image/screenshot", shot, {1027, 1913, 3}, 0));
  EXPECT_LT(0, recorder.AddImage("image/scalars", scalars, {1026, 1915, 3}, 1));
}

TEST(Recorder, AddImages) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  string dir = "runs";
  Recorder recorder(dir);
  ASSERT_TRUE(recorder.Ready());

  vector<string> images;
  uint32_t w = 420, h = 320, c = 4;
  for (int i = 0; i < 11; ++i) {
    auto image = Image::Load(StringUtil::Format("assets/img%02d.png", i));
    ASSERT_EQ(w, image->Width());
    ASSERT_EQ(h, image->Height());
    ASSERT_EQ(c, image->Channel());
    images.push_back(image->Data());
    delete image;
  }

  EXPECT_LE(0, recorder.AddImages("images", images, {420, 320, 4}, 1));
}

TEST(Recorder, AddAudio) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  string dir = "runs";
  Recorder recorder(dir);
  ASSERT_TRUE(recorder.Ready());

  auto audio = ReadBinaryFile("assets/piano.mp3");
  EXPECT_LE(0, recorder.AddAudio("audio/piano", audio,
                                 {1, 48000 * 56, 48000., "audio/mp3"}, 1));
}

TEST(Recorder, AddText) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  string dir = "runs";
  Recorder recorder(dir);
  ASSERT_TRUE(recorder.Ready());

  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_LT(0, recorder.AddText("text", StringUtil::Format("text-%d", i), i));
  }
}

TEST(Recorder, AddEmbedding) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  string dir = "runs";
  Recorder recorder(dir);
  ASSERT_TRUE(recorder.Ready());

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

  EXPECT_LT(0, recorder.AddEmbedding(mat, N, D, labels, 0, "embedding"));
}

TEST(Recorder, AsyncFileWriter) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  string dir = "runs";
  Recorder recorder(dir, [](const string& p) -> Writer* {
                            return new AsyncFileWriter(p);
                         });
  ASSERT_TRUE(recorder.Ready());

  EXPECT_LT(0, recorder.AddText("async_text", "Here is the text message!", 0));

  std::default_random_engine gen;
  std::normal_distribution<float> normal(0, 0.5);
  for (int64_t i = 0; i < 1000; ++i) {
    EXPECT_LT(0, recorder.AddScalar("async_scalar", normal(gen) + 0.01 * i, i));
  }
}
