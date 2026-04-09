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

import os

from tensorboard.compat.proto import graph_pb2
from tensorboard.plugins.graph.mlir_import.custom_dialect_parser import (
    parse_mlir_text,
)
from tensorboard.plugins.graph.mlir_import.mlir_to_graphdef import (
    convert_parsed_module_to_graphdef,
)


class MlirImportError(ValueError):
    pass


def load_mlir_graphdef(path: str) -> graph_pb2.GraphDef:
    if not path:
        raise MlirImportError("MLIR path is required")

    if not os.path.exists(path):
        raise MlirImportError(f"MLIR file '{path}' does not exist")

    if not os.path.isfile(path):
        raise MlirImportError(f"MLIR path '{path}' is not a file")

    try:
        with open(path, "r", encoding="utf-8") as f:
            mlir_text = f.read()
    except OSError as e:
        raise MlirImportError(f"failed to read MLIR file '{path}': {e}") from e

    try:
        parsed_module = parse_mlir_text(mlir_text)
    except ValueError as e:
        raise MlirImportError(f"failed to parse MLIR file '{path}': {e}") from e

    return convert_parsed_module_to_graphdef(parsed_module)
