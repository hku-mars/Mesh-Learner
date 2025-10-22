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

#include "record/summary.h"

#include <algorithm>
#include <climits>

#include "glog/logging.h"
#include "record/utils.h"
#include "utils/image.h"

namespace nlptk {

using std::string;
using std::vector;

using tensorboard::Summary;
using tensorboard::SummaryMetadata;
using tensorboard::TensorProto;

vector<double> GenerateDefaultBins() {
  vector<double> pos_buckets, neg_buckets;

  double v = 1E-12;
  while (v < 1E20) {
    pos_buckets.push_back(v);
    neg_buckets.push_back(-v);
    v *= 1.1;
  }

  vector<double> buckets;
  buckets.insert(buckets.end(), neg_buckets.rbegin(), neg_buckets.rend());
  buckets.push_back(0.0);
  buckets.insert(buckets.end(), pos_buckets.begin(), pos_buckets.end());

  return buckets;
}

static const vector<double> DefaultBins = GenerateDefaultBins();

string CleanTag(const string& tag) {
  return tag;
  string name;
  const char* cur = tag.c_str();
  while (*cur == '/' && *cur != '\0') {
    ++cur;
  }

  for (; *cur != '\0'; ++cur) {
    if (std::isalnum(*cur) || *cur == '.' || *cur == '_' || *cur == '-' ||
        *cur == '/') {
      name.push_back(*cur);
    } else {
      name.push_back('_');
    }
  }

  if (tag != name) {
    LOG(INFO) << "Summary name " << tag << " is illegal; using " << name
              << " instead.";
  }

  return name;
}

Summary* Scalar(const string& name, float value) {
  auto tag = CleanTag(name);
  auto summary = new Summary();
  auto v = summary->add_value();
  v->set_tag(tag);
  v->set_simple_value(value);

  return summary;
}

Summary* Histogram(const string& name, const vector<double>& values,
                   const string& bins) {
  const auto& bucket_limit = DefaultBins;
  vector<size_t> counts(bucket_limit.size(), 0);
  double min = 0.0, max = 0.0;
  double sum = 0.0;
  double sum_square = 0.0;

  if (!values.empty()) {
    max = min = values[0];
  }

  for (const auto& v : values) {
    auto iter = std::lower_bound(bucket_limit.begin(), bucket_limit.end(), v);
    counts[iter - bucket_limit.begin()]++;
    sum += v;
    sum_square += v * v;
    if (v > max) {
      max = v;
    } else if (v < min) {
      min = v;
    }
  }

  auto tag = CleanTag(name);
  auto histo = new tensorboard::HistogramProto();
  histo->set_min(min);
  histo->set_max(max);
  histo->set_num(values.size());
  histo->set_sum(sum);
  histo->set_sum_squares(sum_square);
  for (size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] > 0) {
      histo->add_bucket_limit(bucket_limit[i]);
      histo->add_bucket(counts[i]);
    }
  }

  auto summary = new Summary();
  auto v = summary->add_value();
  v->set_tag(tag);
  v->set_allocated_histo(histo);

  return summary;
}

Summary* HistogramRaw(const std::string& name, double min, double max,
                      double num, double sum, double sum_squares,
                      const std::vector<double>& bucket_limits,
                      const std::vector<double>& bucket_counts) {
  auto tag = CleanTag(name);
  auto histo = new tensorboard::HistogramProto();
  histo->set_min(min);
  histo->set_max(max);
  histo->set_num(num);
  histo->set_sum(sum);
  histo->set_sum_squares(sum_squares);
  for (const auto& v : bucket_limits) {
    histo->add_bucket_limit(v);
  }

  for (const auto& c : bucket_counts) {
    histo->add_bucket(c);
  }

  auto summary = new Summary();
  auto v = summary->add_value();
  v->set_tag(tag);
  v->set_allocated_histo(histo);

  return summary;
}

Summary* Image(const string& name, const string& encoded_image, int32_t height,
               int32_t width, int32_t colorspace) {
  if (0 >= colorspace || 6 < colorspace || height <= 0 || width <= 0) {
    LOG(ERROR) << "Invalid image colorspace: " << colorspace;
    return nullptr;
  }

  if (encoded_image.empty()) {
  // if (encoded_image.size() != height * width * colorspace) {
    LOG(ERROR) << "Incompleted image data, got " << encoded_image.size()
               << ", expected " << height * width * colorspace;
    return nullptr;
  }

  auto img = new Summary::Image();
  img->set_height(height);
  img->set_width(width);
  img->set_colorspace(colorspace);
  img->set_encoded_image_string(encoded_image);

  auto tag = CleanTag(name);
  auto summary = new Summary();
  auto v = summary->add_value();
  v->set_tag(tag);
  v->set_allocated_image(img);

  return summary;
}

Summary* Images(const string& name, const vector<string>& encoded_images,
                int32_t height, int32_t width, int32_t colorspace,
                uint32_t max_cols) {
  if (0 >= colorspace || 6 < colorspace || height <= 0 || width <= 0) {
    LOG(ERROR) << "Invalid image colorspace: " << colorspace;
    return nullptr;
  }

  if (encoded_images.empty()) {
    LOG(ERROR) << "Empty image data";
    return nullptr;
  }

  if (encoded_images[0].size() !=
      static_cast<uint32_t>(height * width * colorspace)) {
    LOG(ERROR) << "Incompleted image data, got " << encoded_images[0].size()
               << ", expected " << height * width * colorspace;
    return nullptr;
  }

  for (size_t i = 1; i < encoded_images.size(); ++i) {
    if (encoded_images[0].size() != encoded_images[i].size()) {
      LOG(ERROR) << "Not equal image shape at " << i;
      return nullptr;
    }
  }

  // NHWC -> H'W'C
  const uint32_t cnt = encoded_images.size();
  const uint32_t ncols = cnt < max_cols ? cnt : max_cols;
  const uint32_t nrows = (cnt + ncols - 1) / ncols;
  string data(ncols * nrows * encoded_images[0].size(), '\0');
  const size_t lz = width * colorspace;
  for (size_t h = 0; h < nrows; ++h) {
    for (int i = 0; i < height; ++i) {
      for (size_t w = 0; w < ncols; ++w) {
        if (h * ncols + w >= cnt) {
          break;
        }

        auto cur = data.begin() + ((h * height + i) * ncols + w) * lz;
        const auto& from = encoded_images[h * ncols + w].begin() + i * lz;
        std::copy(from, from + lz, cur);
      }
    }
  }

  string encoded_image;
  if (Image::Write(data.data(), width * ncols, height * nrows, colorspace,
                   &encoded_image) < 0) {
    LOG(ERROR) << "Failed to encode image!";
    return nullptr;
  }

  auto img = new Summary::Image();
  img->set_height(height);
  img->set_width(width);
  img->set_colorspace(colorspace);
  img->set_encoded_image_string(encoded_image);

  auto tag = CleanTag(name);
  auto summary = new Summary();
  auto v = summary->add_value();
  v->set_tag(tag);
  v->set_allocated_image(img);

  return summary;
}

Summary* Audio(const string& name, const string& encoded_audio,
               float sample_rate, int64_t num_channels, int64_t length_frames,
               const string& content_type) {
  if (encoded_audio.empty()) {
    LOG(ERROR) << "Empty audio data!";
    return nullptr;
  }

  auto audio = new Summary::Audio();
  audio->set_sample_rate(sample_rate);
  audio->set_num_channels(num_channels);
  audio->set_length_frames(length_frames);
  audio->set_encoded_audio_string(encoded_audio);
  audio->set_content_type(content_type);

  auto tag = CleanTag(name);
  auto summary = new Summary();
  auto v = summary->add_value();
  v->set_tag(tag);
  v->set_allocated_audio(audio);

  return summary;
}

Summary* Text(const string& name, const string& text) {
  auto smd = new SummaryMetadata();
  auto plugin_data = smd->add_plugin_data();
  plugin_data->set_plugin_name("text");

  TensorProto* tensor = new TensorProto();
  tensor->set_dtype(tensorboard::DataType::DT_STRING);
  tensor->add_string_val(text);

  auto tensor_shape = new tensorboard::TensorShapeProto();
  auto shape = tensor_shape->add_dim();
  shape->set_size(1);
  tensor->set_allocated_tensor_shape(tensor_shape);

  auto tag = CleanTag(name + "/text_summary");
  auto summary = new Summary();
  auto v = summary->add_value();
  v->set_tag(tag);
  v->set_allocated_tensor(tensor);
  v->set_allocated_metadata(smd);

  return summary;
}

}  // namespace nlptk
