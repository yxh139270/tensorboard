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

import unittest

from tensorboard.plugins.graph.mlir_import.custom_dialect_ir import (
    ParsedBlock,
    ParsedFunction,
    ParsedModule,
    ParsedOperation,
    ParsedRegion,
    ParsedValue,
)
from tensorboard.plugins.graph.mlir_import.mlir_to_graphdef import (
    convert_parsed_module_to_graphdef,
)


class MlirToGraphDefTest(unittest.TestCase):
    def test_convert_builds_nodes_and_edges(self):
        module = ParsedModule(
            functions=[
                ParsedFunction(
                    name="main",
                    arguments=[ParsedValue(name="%arg0")],
                    body=ParsedBlock(),
                )
            ]
        )
        module.functions[0].body.operations = [
            ParsedOperation(
                name='"test.identity"',
                result_names=["%0"],
                operand_text="%arg0",
            ),
            ParsedOperation(
                name='"test.add"',
                result_names=["%1"],
                operand_text="%0, %arg0",
            ),
            ParsedOperation(name="return", operand_text="%1"),
        ]

        graph_def = convert_parsed_module_to_graphdef(module)
        nodes_by_name = {node.name: node for node in graph_def.node}

        self.assertIn("main/arg_arg0", nodes_by_name)
        self.assertIn("main/op_0000_test.identity", nodes_by_name)
        self.assertIn("main/op_0001_test.add", nodes_by_name)
        self.assertEqual(nodes_by_name["main/arg_arg0"].op, "Placeholder")
        self.assertEqual(
            nodes_by_name["main/op_0000_test.identity"].input,
            ["main/arg_arg0"],
        )
        self.assertEqual(
            nodes_by_name["main/op_0001_test.add"].input,
            ["main/op_0000_test.identity", "main/arg_arg0"],
        )
        self.assertEqual(len(graph_def.node), 3)

    def test_convert_creates_placeholder_for_unknown_operand(self):
        module = ParsedModule(
            functions=[
                ParsedFunction(
                    name="main",
                    body=ParsedBlock(),
                )
            ]
        )
        module.functions[0].body.operations = [
            ParsedOperation(
                name='"test.use"',
                result_names=["%0"],
                operand_text="%missing",
            )
        ]

        graph_def = convert_parsed_module_to_graphdef(module)
        nodes_by_name = {node.name: node for node in graph_def.node}

        self.assertIn("main/placeholder_missing", nodes_by_name)
        self.assertEqual(nodes_by_name["main/placeholder_missing"].op, "Placeholder")
        self.assertEqual(
            nodes_by_name["main/op_0000_test.use"].input,
            ["main/placeholder_missing"],
        )

    def test_convert_resolves_forward_reference_to_real_producer(self):
        module = ParsedModule(
            functions=[
                ParsedFunction(
                    name="main",
                    body=ParsedBlock(),
                )
            ]
        )
        module.functions[0].body.operations = [
            ParsedOperation(
                name='"test.consume"',
                result_names=["%0"],
                operand_text="%late",
            ),
            ParsedOperation(
                name='"test.produce"',
                result_names=["%late"],
                operand_text="",
            ),
        ]

        graph_def = convert_parsed_module_to_graphdef(module)
        nodes_by_name = {node.name: node for node in graph_def.node}

        self.assertEqual(
            nodes_by_name["main/op_0000_test.consume"].input,
            ["main/op_0001_test.produce"],
        )
        self.assertNotIn("main/placeholder_late", nodes_by_name)

    def test_convert_prefixes_nodes_per_function(self):
        module = ParsedModule(
            functions=[
                ParsedFunction(
                    name="func_a",
                    arguments=[ParsedValue(name="%arg0")],
                    body=ParsedBlock(),
                ),
                ParsedFunction(
                    name="func_b",
                    arguments=[ParsedValue(name="%arg0")],
                    body=ParsedBlock(),
                ),
            ]
        )
        module.functions[0].body.operations = [
            ParsedOperation(
                name='"test.identity"',
                result_names=["%0"],
                operand_text="%arg0",
            )
        ]
        module.functions[1].body.operations = [
            ParsedOperation(
                name='"test.identity"',
                result_names=["%0"],
                operand_text="%arg0",
            )
        ]

        graph_def = convert_parsed_module_to_graphdef(module)
        node_names = {node.name for node in graph_def.node}

        self.assertIn("func_a/arg_arg0", node_names)
        self.assertIn("func_b/arg_arg0", node_names)
        self.assertIn("func_a/op_0000_test.identity", node_names)
        self.assertIn("func_b/op_0000_test.identity", node_names)

    def test_convert_maps_multi_result_outputs_to_distinct_ports(self):
        module = ParsedModule(
            functions=[
                ParsedFunction(
                    name="main",
                    arguments=[ParsedValue(name="%arg0")],
                    body=ParsedBlock(),
                )
            ]
        )
        module.functions[0].body.operations = [
            ParsedOperation(
                name='"test.split"',
                result_names=["%left", "%right"],
                operand_text="%arg0",
            ),
            ParsedOperation(
                name='"test.use_left"',
                result_names=["%0"],
                operand_text="%left",
            ),
            ParsedOperation(
                name='"test.use_right"',
                result_names=["%1"],
                operand_text="%right",
            ),
        ]

        graph_def = convert_parsed_module_to_graphdef(module)
        nodes_by_name = {node.name: node for node in graph_def.node}

        split_node_name = "main/op_0000_test.split"
        self.assertEqual(
            nodes_by_name["main/op_0001_test.use_left"].input,
            [split_node_name],
        )
        self.assertEqual(
            nodes_by_name["main/op_0002_test.use_right"].input,
            [f"{split_node_name}:1"],
        )

    def test_convert_makes_unique_names_when_sanitize_collides(self):
        module = ParsedModule(
            functions=[ParsedFunction(name="main", body=ParsedBlock())]
        )
        module.functions[0].body.operations = [
            ParsedOperation(
                name='"test.use"',
                result_names=["%0"],
                operand_text="%a-b, %a_b",
            )
        ]

        graph_def = convert_parsed_module_to_graphdef(module)
        placeholder_nodes = [
            node for node in graph_def.node if node.op == "Placeholder"
        ]

        self.assertEqual(len(placeholder_nodes), 2)
        self.assertEqual(
            len({node.name for node in placeholder_nodes}),
            2,
        )

    def test_convert_strips_quotes_from_node_op(self):
        module = ParsedModule(
            functions=[ParsedFunction(name="main", body=ParsedBlock())]
        )
        module.functions[0].body.operations = [
            ParsedOperation(
                name='"test.identity"',
                result_names=["%0"],
                operand_text="",
            )
        ]

        graph_def = convert_parsed_module_to_graphdef(module)
        nodes_by_name = {node.name: node for node in graph_def.node}

        self.assertEqual(
            nodes_by_name["main/op_0000_test.identity"].op,
            "test.identity",
        )

    def test_convert_expands_region_block_operations_with_hierarchical_names(self):
        module = ParsedModule(
            functions=[
                ParsedFunction(
                    name="main",
                    arguments=[ParsedValue(name="%arg0")],
                    body=ParsedBlock(),
                )
            ]
        )
        module.functions[0].body.operations = [
            ParsedOperation(
                name='"test.if"',
                result_names=["%0"],
                operand_text="%arg0",
                regions=[
                    ParsedRegion(
                        blocks=[
                            ParsedBlock(
                                operations=[
                                    ParsedOperation(
                                        name='"test.inner"',
                                        result_names=["%inner"],
                                        operand_text="%arg0",
                                    )
                                ]
                            )
                        ]
                    )
                ],
            )
        ]

        graph_def = convert_parsed_module_to_graphdef(module)
        nodes_by_name = {node.name: node for node in graph_def.node}

        self.assertIn("main/op_0000_test.if", nodes_by_name)
        self.assertIn(
            "main/op_0000_test.if/region_0/block_0/op_0000_test.inner",
            nodes_by_name,
        )

    def test_convert_nested_region_operations_keep_ssa_edges(self):
        module = ParsedModule(
            functions=[
                ParsedFunction(
                    name="main",
                    arguments=[ParsedValue(name="%arg0")],
                    body=ParsedBlock(),
                )
            ]
        )
        module.functions[0].body.operations = [
            ParsedOperation(
                name='"test.produce"',
                result_names=["%p"],
                operand_text="%arg0",
            ),
            ParsedOperation(
                name='"test.if"',
                result_names=["%0"],
                operand_text="%p",
                regions=[
                    ParsedRegion(
                        blocks=[
                            ParsedBlock(
                                operations=[
                                    ParsedOperation(
                                        name='"test.inner_use"',
                                        result_names=["%inner"],
                                        operand_text="%p",
                                    )
                                ]
                            )
                        ]
                    )
                ],
            ),
        ]

        graph_def = convert_parsed_module_to_graphdef(module)
        nodes_by_name = {node.name: node for node in graph_def.node}
        self.assertEqual(
            nodes_by_name[
                "main/op_0001_test.if/region_0/block_0/op_0000_test.inner_use"
            ].input,
            ["main/op_0000_test.produce"],
        )


if __name__ == "__main__":
    unittest.main()
