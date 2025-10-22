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

#include "record/file_writer.h"

#include <ctime>

#include "glog/logging.h"
#include "record/crc.h"
#include "record/utils.h"

namespace nlptk {

using std::string;

FileWriter::FileWriter(const string& prefix, bool resume) {
  auto path = StringUtil::Format("%s.out.tfevents.%f.%s", prefix.c_str(),
                                 Timestamp(), Env::HostName.c_str());
  fout_.open(path, std::ios::out | std::ios::binary |
                       (resume ? std::ios::app : std::ios::trunc));
  if (!fout_.is_open() || fout_.fail()) {
    LOG(ERROR) << "Failed create record file '" << path << "'";
    fout_.close();
  }
}

FileWriter::~FileWriter() {
  if (fout_.is_open()) {
    fout_.close();
  }
}

int FileWriter::Write(tensorboard::Event&& event) {
  if (!fout_.is_open()) {
    return -1;
  }

  string buf;
  event.SerializeToString(&buf);
  return Write(buf);
}

int FileWriter::Flush() {
  if (!fout_.is_open()) {
    return -1;
  }

  fout_.flush();
  return 0;
}

int FileWriter::Close() {
  if (!fout_.is_open()) {
    return -1;
  }

  fout_.close();
  return 0;
}

int FileWriter::Ready() const {
  return fout_.is_open() && fout_.good();
}

int FileWriter::Write(const std::string& data) {
  if (!fout_.is_open() || data.empty()) {
    return -1;
  }

  uint64_t header = data.size();
  fout_.write(reinterpret_cast<const char*>(&header), sizeof(header));
  auto crc = MaskedCRC32c(reinterpret_cast<const char*>(&header),
                          sizeof(header));
  fout_.write(reinterpret_cast<const char*>(&crc), sizeof(crc));

  fout_.write(data.data(), data.size());
  crc = MaskedCRC32c(data.data(), data.size());
  fout_.write(reinterpret_cast<const char*>(&crc), sizeof(crc));

  fout_.flush();

  return data.size();
}

}  // namespace nlptk
