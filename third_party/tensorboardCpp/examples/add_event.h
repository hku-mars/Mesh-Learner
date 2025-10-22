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

#ifndef EXAMPLES_ADD_EVENT_H_
#define EXAMPLES_ADD_EVENT_H_

#include "record/recorder.h"

void AddScalar(const nlptk::Recorder& recorder);

void AddScalars(nlptk::Recorder& recorder);   // NOLINT(runtime/references)

void AddHistogram(const nlptk::Recorder& recorder);

void AddImage(const nlptk::Recorder& recorder);

void AddImages(const nlptk::Recorder& recorder);

void AddAudio(const nlptk::Recorder& recorder);

void AddText(const nlptk::Recorder& recorder);

void AddProjector(const nlptk::Recorder& recorder);

#endif  // EXAMPLES_ADD_EVENT_H_
