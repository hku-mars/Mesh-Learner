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

#ifndef RECORD_RECORDER_H_
#define RECORD_RECORDER_H_

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "record/writer.h"

namespace nlptk {

class ImageMetadata {
 public:
  ImageMetadata(int width, int height, int colorspace);

 public:
  int32_t     width;
  int32_t     height;
  int32_t     colorspace;  // 1: grayscale; 2: grayscale + alpha; 3: RGB
                           // 4: RGBA; 5: DIGITAL_YUV; 6: BGRA
};

class AudioMetadata {
 public:
  AudioMetadata(int64_t num_channels, int64_t length_frames,
                float sample_rate = 44100.0,
                const std::string& content_type = "audio/wav");

 public:
  int64_t       num_channels;
  int64_t       length_frames;
  float         sample_rate   = 44100.0;
  std::string   content_type  = "audio/wav";
};

class Recorder {
 public:
  using WriterMaker = std::function<Writer*(const std::string&)>;

  explicit Recorder(const std::string& log_dir, WriterMaker maker = Default);

  ~Recorder();

  bool Ready() const;

  int AddScalar(const std::string& tag, float scalar_value,
                int64_t global_step = -1) const;

  int AddScalars(const std::string& main_tag,
                 const std::map<std::string, float>& tag_scalar_dict,
                 int64_t global_step = -1);

  int AddHistogram(const std::string& tag,
                   const std::vector<double>& values,
                   int64_t global_step = -1,
                   const std::string& bins = "tensorflow") const;

  int AddHistogramRaw(const std::string& tag, double min, double max,
                      double num, double sum, double sum_squares,
                      const std::vector<double>& bucket_limits,
                      const std::vector<double>& bucket_counts,
                      int64_t global_step) const;

  int AddImage(const std::string& tag, const std::string& image_data,
               const ImageMetadata& image_metadata, int64_t global_step) const;

  int AddImages(const std::string& tag,
                const std::vector<std::string>& image_data,
                const ImageMetadata& image_metadata, int64_t global_step) const;

  int AddAudio(const std::string& tag, const std::string& audio_data,
               const AudioMetadata& audio_metadata, int64_t global_step) const;

  int AddText(const std::string& tag, const std::string& text_string,
              int64_t global_step = -1) const;

  int AddEmbedding(const std::vector<float>& mat, size_t N, size_t D,
                   const std::vector<std::string>& metadata,
                   int64_t global_step = 0,
                   const std::string& tag = "default") const;

 protected:
  static const WriterMaker Default;

 private:
  int AddProjectConfig(const std::string& tag, const std::string& dir,
                       const std::string& metadata_filename,
                       const std::string& label_img_filename,
                       int64_t global_step) const;

 private:
  std::string                       log_dir_;
  WriterMaker                       make_writer_;
  mutable Writer*                   writer_{nullptr};
  std::map<std::string, Writer*>    writers_;
};

}  // namespace nlptk

#endif  // RECORD_RECORDER_H_
