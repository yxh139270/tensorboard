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

import re

from tensorboard.compat.proto import graph_pb2
from tensorboard.plugins.graph.mlir_import.custom_dialect_ir import ParsedModule
from tensorboard.plugins.graph.mlir_import.custom_dialect_tokenizer import (
    split_top_level,
)


_VALID_NODE_NAME_RE = re.compile(r"[^A-Za-z0-9_./]+")
_RETURN_OP_NAMES = {"return", "func.return", '"func.return"'}


def _normalize_value_name(name: str) -> str:
    if name.startswith("%"):
        name = name[1:]
    return name.strip().strip('"')


def _sanitize_name(name: str) -> str:
    sanitized = _VALID_NODE_NAME_RE.sub("_", name)
    return sanitized.strip("_") or "unnamed"


def _make_unique_name(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    suffix = 1
    while True:
        candidate = f"{base_name}_{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1


def _parse_operands(operand_text: str) -> list[str]:
    operands: list[str] = []
    for operand in split_top_level(operand_text, ","):
        value_name = operand.strip()
        if not value_name:
            continue
        if ":" in value_name:
            value_name = value_name.split(":", 1)[0].strip()
        if value_name:
            operands.append(value_name)
    return operands


def convert_parsed_module_to_graphdef(module: ParsedModule) -> graph_pb2.GraphDef:
    graph_def = graph_pb2.GraphDef()
    used_node_names: set[str] = set()

    for function in module.functions:
        function_prefix = _make_unique_name(
            _sanitize_name(function.name),
            used_node_names,
        )
        value_to_producer: dict[str, str] = {}
        placeholder_by_value: dict[str, str] = {}
        pending_inputs: list[tuple[graph_pb2.NodeDef, list[str]]] = []

        for argument in function.arguments:
            arg_name = _sanitize_name(_normalize_value_name(argument.name))
            node = graph_def.node.add()
            node.name = _make_unique_name(
                f"{function_prefix}/arg_{arg_name}",
                used_node_names,
            )
            node.op = "Placeholder"
            value_to_producer[argument.name] = node.name

        op_index = 0
        for operation in function.body.operations:
            op_name = operation.name.strip('"')
            if operation.name in _RETURN_OP_NAMES or op_name in _RETURN_OP_NAMES:
                continue

            sanitized_op_name = _sanitize_name(op_name)
            node = graph_def.node.add()
            node.name = _make_unique_name(
                f"{function_prefix}/op_{op_index:04d}_{sanitized_op_name}",
                used_node_names,
            )
            node.op = op_name
            op_index += 1

            pending_inputs.append(
                (node, _parse_operands(operation.operand_text))
            )

            for result_index, result_name in enumerate(operation.result_names):
                if result_index == 0:
                    value_to_producer[result_name] = node.name
                else:
                    value_to_producer[result_name] = f"{node.name}:{result_index}"

        for node, operands in pending_inputs:
            for operand_name in operands:
                producer = value_to_producer.get(operand_name)
                if producer is None:
                    producer = placeholder_by_value.get(operand_name)
                if producer is None:
                    placeholder_name = _sanitize_name(
                        _normalize_value_name(operand_name)
                    )
                    placeholder = graph_def.node.add()
                    placeholder.name = _make_unique_name(
                        f"{function_prefix}/placeholder_{placeholder_name}",
                        used_node_names,
                    )
                    placeholder.op = "Placeholder"
                    producer = placeholder.name
                    placeholder_by_value[operand_name] = producer
                node.input.append(producer)

    return graph_def
