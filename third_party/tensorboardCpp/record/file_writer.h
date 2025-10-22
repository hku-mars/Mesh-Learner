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

#ifndef RECORD_FILE_WRITER_H_
#define RECORD_FILE_WRITER_H_

#include <fstream>
#include <string>

#include "record/writer.h"

namespace nlptk {

class FileWriter : public Writer {
 public:
  explicit FileWriter(const std::string& path_prefix, bool resume = false);

  ~FileWriter();

  int Write(tensorboard::Event&& event) override;

  int Flush() override;

  int Close() override;

  int Ready() const override;

 protected:
  int Write(const std::string& data);

 private:
  std::ofstream   fout_;
};

}  // namespace nlptk

#endif  // RECORD_FILE_WRITER_H_
