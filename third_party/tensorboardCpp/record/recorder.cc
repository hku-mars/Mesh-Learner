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

#include <sys/stat.h>

#include <algorithm>
#include <cassert>
#include <ctime>
#include <utility>
#include <vector>
#include <fstream>

#include "glog/logging.h"
#include "proto/summary.pb.h"
#include "record/file_writer.h"
#include "record/summary.h"
#include "record/utils.h"

namespace nlptk {

using std::map;
using std::string;
using std::vector;

using tensorboard::Event;
using tensorboard::Summary;

ImageMetadata::ImageMetadata(int w, int h, int c)
    : width(w), height(h), colorspace(c) {
}

AudioMetadata::AudioMetadata(int64_t n, int64_t l, float s, const string& c)
    : num_channels(n), length_frames(l), sample_rate(s), content_type(c) {
}

int AddEvent(Writer* writer, Summary* summary, int step) {
  assert(writer);
  assert(summary);

  Event event;
  double wall_time = Timestamp();
  event.set_wall_time(wall_time);
  event.set_allocated_summary(summary);
  if (step >= 0) {
    event.set_step(step);
  }

  return writer->Write(std::move(event));
}

string Replace(const string& txt, char old_value, char new_value) {
  if (txt.find(old_value)) {
    auto new_txt = txt;
    std::replace(new_txt.begin(), new_txt.end(), old_value, new_value);
    return new_txt;
  }

  return txt;
}

const Recorder::WriterMaker Recorder::Default = [](const string& p) -> Writer* {
  return new FileWriter(p);
};

Recorder::Recorder(const string& log_dir, WriterMaker maker)
    : log_dir_(log_dir), make_writer_(maker) {
  if (StringUtil::EmptyOrSpaces(log_dir_)) {
    char buf[32];
    auto now = time(nullptr);
    tm info;
    strftime(buf, 80, "%b%d_%H-%M-%S", localtime_r(&now, &info));
    log_dir_ = StringUtil::Format("runs/%s_%s", buf, Env::HostName.c_str());
  }

  if (IsExisted(log_dir_)) {
    if (!IsDirectory(log_dir_)) {
      log_dir_ = GetDirectoryPath(log_dir_, true);
    }
  } else {
    if (MakeDirs(log_dir_) < 0) {
      LOG(ERROR) << "Failed to create tensorboard log dir: " << log_dir_;
    } else {
      LOG(INFO) << "Created tensorboard log dir: " << log_dir_;
    }
  }

  if (IsExisted(log_dir_)) {
    writer_ = make_writer_(JoinPath(log_dir_, "events"));
    writers_[log_dir_] = writer_;
  }
}

Recorder::~Recorder() {
  for (auto& item : writers_) {
    item.second->Close();
    delete item.second;
    item.second = nullptr;
  }

  writer_ = nullptr;
}

bool Recorder::Ready() const {
  return (nullptr != writer_) && writer_->Ready();
}

int Recorder::AddScalar(const string& tag, float value, int64_t step) const {
  if (nullptr == writer_) {
    return -1;
  }

  auto summary = Scalar(tag, value);
  if (nullptr == summary) {
    return -1;
  }

  return AddEvent(writer_, summary, step);
}

int Recorder::AddScalars(const string& main_tag,
                         const map<string, float>& tag_values,
                         int64_t global_step) {
  int ret = 0;
  for (const auto& item : tag_values) {
    auto dir = log_dir_ + "/" + Replace(main_tag, '/', '_') + "_" + item.first;
    auto iter = writers_.find(dir);
    if (iter == writers_.end()) {
      if (IsExisted(dir)) {
        if (!IsDirectory(dir)) {
          dir = GetDirectoryPath(dir, true);
          iter = writers_.find(dir);
        }
      } else {
        if (MakeDirs(dir) < 0) {
          LOG(ERROR) << "Failed to create tensorboard log dir: " << dir;
        } else {
          LOG(INFO) << "Created tensorboard log dir: " << dir;
        }
      }

      if (iter == writers_.end() && IsExisted(dir)) {
        auto writer = make_writer_(JoinPath(dir, "events"));
        iter = writers_.emplace(dir, writer).first;
      }
    }

    if (iter != writers_.end()) {
      auto summary = Scalar(main_tag, item.second);
      auto cnt = AddEvent(iter->second, summary, global_step);
      if (cnt < 0 || ret < 0) {
        ret = -1;
      } else {
        ret += cnt;
      }
    } else {
      ret = -1;
    }
  }

  return ret;
}

int Recorder::AddHistogram(const string& tag, const vector<double>& values,
                           int64_t global_step, const string& bins) const {
  if (nullptr == writer_) {
    return -1;
  }

  auto summary = Histogram(tag, values, bins);
  if (nullptr == summary) {
    return -1;
  }

  return AddEvent(writer_, summary, global_step);
}

int Recorder::AddHistogramRaw(const string& tag, double min, double max,
                              double num, double sum, double sum_squares,
                              const vector<double>& bucket_limits,
                              const vector<double>& bucket_counts,
                              int64_t global_step) const {
  if (nullptr == writer_) {
    return -1;
  }

  auto summary = HistogramRaw(tag, min, max, num, sum, sum_squares,
                              bucket_limits, bucket_counts);
  if (nullptr == summary) {
    return -1;
  }

  return AddEvent(writer_, summary, global_step);
}

int Recorder::AddImage(const string& tag, const string& img,
                       const ImageMetadata& meta, int64_t global_step) const {
  if (nullptr == writer_) {
    return -1;
  }

  auto summary = Image(tag, img, meta.height, meta.width, meta.colorspace);
  if (nullptr == summary) {
    return -1;
  }

  return AddEvent(writer_, summary, global_step);
}

int Recorder::AddImages(const string& tag, const vector<string>& imgs,
                        const ImageMetadata& meta, int64_t global_step) const {
  if (nullptr == writer_) {
    return -1;
  }

  auto summary = Images(tag, imgs, meta.height, meta.width, meta.colorspace);
  if (nullptr == summary) {
    return -1;
  }

  return AddEvent(writer_, summary, global_step);
}

int Recorder::AddAudio(const string& tag, const string& audio,
                       const AudioMetadata& amd, int64_t global_step) const {
  if (nullptr == writer_) {
    return -1;
  }

  auto summary = Audio(tag, audio, amd.sample_rate, amd.num_channels,
                       amd.length_frames, amd.content_type);
  if (nullptr == summary) {
    return -1;
  }

  return AddEvent(writer_, summary, global_step);
}

int Recorder::AddText(const string& tag, const string& text, int64_t s) const {
  if (nullptr == writer_) {
    return -1;
  }

  auto summary = Text(tag, text);
  if (nullptr == summary) {
    return -1;
  }

  return AddEvent(writer_, summary, s);
}

int Recorder::AddEmbedding(const vector<float>& mat, size_t N, size_t D,
                           const vector<string>& metadata, int64_t global_step,
                           const string& tag) const {
  if (nullptr == writer_) {
    return -1;
  }

  if (mat.size() != N * D) {
    LOG(ERROR) << "Ivalid data size: " << mat.size() << " != " << N << '*' << D;
    return -1;
  }

  if (global_step < 0) {
    global_step = 0;
  }

  // TODO(Liang Zhao): encode tag by replace below chars
  // '%' -> '%25'
  // '/' -> '%2f'
  // '\' -> '%5c'
  auto subdir = StringUtil::Format("%05d/%s", global_step, tag.c_str());
  auto save_path = JoinPath(log_dir_, subdir);
  if (MakeDirs(save_path) < 0) {
    return -1;
  }

  if (!metadata.empty()) {
    if (metadata.size() != N) {
      LOG(ERROR) << "#labels should equal with #data points";
      return -1;
    }

    std::ofstream fout(save_path + "/metadata.tsv", std::ios::binary);
    if (!fout.is_open() || fout.fail()) {
      LOG(ERROR) << "Failed to create metadata file: " << save_path
                 << "/metadata.tsv";
      fout.close();
      return -1;
    }

    // TODO(Liang Zhao): support metadata_header
    // assert len(meta_header) == len(meta[0]),
    //        "len of header must be equal to the number of columns in meta"
    // meta = ["\t".join(str(e) for e in l) for l in [meta_header] + meta]
    for (const auto& label : metadata) {
      fout << label << '\n';
    }

    fout.close();
  }

  // TODO(Liang Zhao): add label_img
  if (true) {
    std::ofstream fout(save_path + "/tensors.tsv", std::ios::binary);
    if (!fout.is_open() || fout.fail()) {
      LOG(ERROR) << "Failed to create tensor file: " << save_path
                 << "/tensors.tsv";
      fout.close();
      return -1;
    }

    auto cur = mat.data();
    for (size_t i = 0; i < N; ++i) {
      fout << *cur++;
      for (size_t j = 1; j < D; ++j) {
        fout << '\t' << *cur++;
      }

      fout << '\n';
    }

    fout.close();
  }

  return AddProjectConfig(tag, subdir,
                          metadata.empty() ? "" : "metadata.tsv",
                          "",
                          global_step);
}

int Recorder::AddProjectConfig(const string& tag, const string& dir,
                               const string& mfn, const string& lifn,
                               int64_t step) const {
  auto path = log_dir_ + "/projector_config.pbtxt";
  std::ofstream fout(path, std::ios::app);
  if (!fout.is_open() || fout.fail()) {
    LOG(ERROR) << "Failed to create projector config: " << path;
    fout.close();
    return -1;
  }

  string txt = "embeddings {\n";
  StringUtil::AppendFormat(&txt, "  tensor_name: \"%s:%05d\"\n",
                           tag.c_str(), step);
  StringUtil::AppendFormat(&txt, "  tensor_path: \"%s/tensors.tsv\"\n",
                           dir.c_str());
  if (!mfn.empty()) {
    StringUtil::AppendFormat(&txt, "  metadata_path: \"%s/%s\"\n",
                             dir.c_str(), mfn.c_str());
  }

  // TODO(Liang Zhao): support label_img
  if (!lifn.empty()) {
    /*
    StringUtil::AppendFormat(&txt, "  sprite {\n    image_path: \"%s/%s\"\n",
                             dir.c_str(), lifn.c_str());
    StringUtil::AppendFormat(&txt, "    single_image_dim: \"%d\"\n",
                             img.size(3));
    StringUtil::AppendFormat(&txt, "    single_image_dim: \"%d\"\n  }\n",
                             img.size(2));
    */
  }

  txt.append("}\n");
  fout << txt;
  fout.close();
  return txt.size();
}

}  // namespace nlptk
