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

from dataclasses import dataclass

from tensorboard.plugins.graph.mlir_import.custom_dialect_tokenizer import (
    split_top_level,
)


@dataclass
class ParsedType:
    raw_text: str
    kind: str = ''
    element_type: str = ''
    shape: list[str] | None = None
    memory_space: str = ''


def parse_type(type_text: str) -> ParsedType:
    text = type_text.strip()
    if not text:
        return ParsedType(raw_text='')

    kind = text.split('<', 1)[0].strip() if '<' in text else text
    element_type = ''
    shape: list[str] | None = None
    if '<' in text and '>' in text:
        inner = text[text.index('<') + 1:text.rfind('>')]
        segments = split_top_level(inner, ',')
        if segments:
            shape_and_element = segments[0]
            parts = [part for part in shape_and_element.split('x') if part]
            if parts:
                element_type = parts[-1]
                if len(parts) > 1:
                    shape = parts[:-1]

    memory_space = ''
    if '#dlgpu<' in text:
        memory_space = text[text.index('#dlgpu<'):]

    return ParsedType(
        raw_text=text,
        kind=kind,
        element_type=element_type,
        shape=shape,
        memory_space=memory_space,
    )
