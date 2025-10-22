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

#ifndef RECORD_SUMMARY_H_
#define RECORD_SUMMARY_H_

#include <string>
#include <vector>

#include "proto/summary.pb.h"

namespace nlptk {

tensorboard::Summary* Scalar(const std::string& name, float value);

tensorboard::Summary* Histogram(const std::string& name,
                                const std::vector<double>& values,
                                const std::string& bins = "tensorflow");

tensorboard::Summary* HistogramRaw(const std::string& name, double min,
                                   double max, double num, double sum,
                                   double sum_squares,
                                   const std::vector<double>& bucket_limits,
                                   const std::vector<double>& bucket_counts);

tensorboard::Summary* Image(const std::string& name,
                            const std::string& encoded_image, int32_t height,
                            int32_t width, int32_t colorspace);

tensorboard::Summary* Images(const std::string& name,
                             const std::vector<std::string>& encoded_images,
                             int32_t height, int32_t width, int32_t colorspace,
                             uint32_t max_cols = 8);

tensorboard::Summary* Audio(const std::string& name,
                            const std::string& encoded_audio, float sample_rate,
                            int64_t num_channels, int64_t length_frames,
                            const std::string& content_type);

tensorboard::Summary* Text(const std::string& name, const std::string& text);

template <class T>
int MakeHistogram(const std::vector<T>& data, double* min, double* max,
                  double* num, double* sum, double* sum_squares,
                  std::vector<double>* bucket_limits,
                  std::vector<double>* bucket_counts,
                  const std::vector<T>& bins) {
  if (data.empty() || bins.empty()) {
    return -1;
  }

  *min = *max = data[0];
  *num = *sum = *sum_squares = 0.0;
  bucket_counts->assign(bins.size(), 0.0);
  for (const auto& v : data) {
    auto iter = std::lower_bound(bins.begin(), bins.end(), v);
    bucket_counts->at(iter - bins.begin()) += 1;
    *sum += v;
    *sum_squares += v * v;
    if (v > *max) {
      *max = v;
    } else if (v < *min) {
      *min = v;
    }
  }

  int cur = 0;
  bucket_limits->clear();
  for (size_t i = 0; i < bucket_counts->size(); ++i) {
    if (bucket_counts->at(i) > 0.0) {
      bucket_limits->push_back(bins[i]);
      if (i != cur) {
        bucket_counts->at(cur) = bucket_counts->at(i);
      }

      ++cur;
    }
  }

  bucket_counts->resize(cur);
  return cur;
}

}  // namespace nlptk

#endif  // RECORD_SUMMARY_H_
