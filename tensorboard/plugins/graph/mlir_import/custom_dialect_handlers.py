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

from tensorboard.plugins.graph.mlir_import.custom_dialect_tokenizer import (
    split_top_level,
)


RECOGNIZED_METADATA_KEYS = [
    'opName',
    'sym_name',
    'cluster',
    'ggType',
    'gtgType',
    'hashKey',
    'size',
    'memLoc',
    'dynInfo',
]


def _find_top_level_colon(text: str) -> int:
    stack: list[str] = []
    delimiters = {'(': ')', '[': ']', '{': '}', '<': '>'}
    for index, ch in enumerate(text):
        if ch in delimiters:
            stack.append(delimiters[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
        elif ch == ':' and not stack:
            return index
    return -1


def _normalize_value(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]

    colon = _find_top_level_colon(value)
    if colon != -1 and not value.startswith('{') and not value.startswith('#'):
        return value[:colon].strip()
    return value


def extract_operation_metadata(op_name: str, attributes_text: str) -> dict[str, str]:
    del op_name
    metadata: dict[str, str] = {}
    if not attributes_text:
        return metadata

    for entry in split_top_level(attributes_text, ','):
        if '=' not in entry:
            continue
        key, raw_value = entry.split('=', 1)
        key = key.strip()
        if key in RECOGNIZED_METADATA_KEYS:
            metadata[key] = _normalize_value(raw_value)
    return metadata
