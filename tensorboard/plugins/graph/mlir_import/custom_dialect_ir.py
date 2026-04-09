# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedValue:
    name: str
    type_text: str = ''


@dataclass
class ParsedBlock:
    arguments: list[ParsedValue] = field(default_factory=list)
    operations: list['ParsedOperation'] = field(default_factory=list)


@dataclass
class ParsedRegion:
    blocks: list[ParsedBlock] = field(default_factory=list)


@dataclass
class ParsedOperation:
    name: str
    result_names: list[str] = field(default_factory=list)
    result_count: int = 0
    attributes_text: str = ''
    type_text: str = ''
    operand_text: str = ''
    regions: list[ParsedRegion] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    result_types: list[Any] = field(default_factory=list)


@dataclass
class ParsedFunction:
    name: str
    arguments: list[ParsedValue] = field(default_factory=list)
    result_types: list[str] = field(default_factory=list)
    body: ParsedBlock = field(default_factory=ParsedBlock)


@dataclass
class ParsedModule:
    functions: list[ParsedFunction] = field(default_factory=list)
