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

#include "record/async_file_writer.h"

#include <ctime>

#include "glog/logging.h"
#include "record/crc.h"
#include "record/utils.h"

namespace nlptk {

using std::string;

AsyncFileWriter::AsyncFileWriter(const string& prefix, size_t flush, bool app)
    : flush_secs_(flush) {
  auto path = StringUtil::Format("%s.out.tfevents.%f.%s", prefix.c_str(),
                                 Timestamp(), Env::HostName.c_str());
  fout_.open(path, std::ios::out | std::ios::binary |
                       (app ? std::ios::app : std::ios::trunc));
  if (!fout_.is_open() || fout_.fail()) {
    LOG(ERROR) << "Failed create record file '" << path << "'";
    fout_.close();
    stop_.store(true);
  } else {
    worker_ = std::thread(&AsyncFileWriter::AsyncFlush, this);
  }
}

AsyncFileWriter::~AsyncFileWriter() {
  if (!(stop_)) {
    stop_.store(true);

    if (fout_.is_open() && fout_.good()) {
      fout_.close();
    }
  }

  if (worker_.joinable()) {
    worker_.join();
  }
}

int AsyncFileWriter::Write(tensorboard::Event&& event) {
  if (!(stop_)) {
    string buf;
    event.SerializeToString(&buf);

    uint64_t header = buf.size();
    fout_.write(reinterpret_cast<const char*>(&header), sizeof(header));
    auto crc =
        MaskedCRC32c(reinterpret_cast<const char*>(&header), sizeof(header));
    fout_.write(reinterpret_cast<const char*>(&crc), sizeof(crc));

    fout_.write(buf.data(), buf.size());
    crc = MaskedCRC32c(buf.data(), buf.size());
    fout_.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
    return buf.size();
  }

  return -1;
}

int AsyncFileWriter::Flush() {
  if (!(stop_)) {
    std::lock_guard<std::mutex> lock{locker_};
    if (fout_.is_open() && fout_.good()) {
      fout_.flush();
    }
  }

  return 0;
}

int AsyncFileWriter::AsyncFlush() {
  auto period = std::chrono::seconds(flush_secs_);
  auto next_flush_time = std::chrono::high_resolution_clock::now() + period;

  while (!(stop_)) {
    if (std::chrono::high_resolution_clock::now() < next_flush_time) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      continue;
    }

    std::lock_guard<std::mutex> lock{locker_};
    if (fout_.is_open() && fout_.good()) {
      fout_.flush();
    }

    next_flush_time = std::chrono::high_resolution_clock::now() + period;
  }

  return 0;
}

int AsyncFileWriter::Close() {
  if (!(stop_)) {
    stop_.store(true);
    std::lock_guard<std::mutex> lock{locker_};
    if (fout_.is_open() && fout_.good()) {
      fout_.flush();
      fout_.close();
    }
  }

  return 0;
}

int AsyncFileWriter::Ready() const {
  return !(stop_);
}

}  // namespace nlptk
