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

#ifndef RECORD_ASYNC_FILE_WRITER_H_
#define RECORD_ASYNC_FILE_WRITER_H_

#include <atomic>
#include <fstream>
#include <mutex>                  // NOLINT(build/c++11)
#include <string>
#include <thread>                 // NOLINT(build/c++11)

#include "record/writer.h"

namespace nlptk {

class AsyncFileWriter : public Writer {
 public:
  explicit AsyncFileWriter(const std::string& path_prefix,
                           size_t flush_secs = 120, bool resume = false);

  ~AsyncFileWriter();

  int Write(tensorboard::Event&& event) override;

  int Flush() override;

  int Close() override;

  int Ready() const override;

 protected:
  int AsyncFlush();

 private:
  std::atomic<bool>   stop_{false};
  size_t              flush_secs_;
  std::ofstream       fout_;
  std::thread         worker_;
  std::mutex          locker_{};
};

}  // namespace nlptk

#endif  // RECORD_ASYNC_FILE_WRITER_H_
