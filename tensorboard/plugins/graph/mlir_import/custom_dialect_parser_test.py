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

from tensorboard.plugins.graph.mlir_import.custom_dialect_parser import (
    parse_mlir_text,
)


class CustomDialectParserTest(unittest.TestCase):
    def test_parse_smoke(self):
        mlir_text = """
          func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
            %0 = \"test.identity\"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
            return %0 : tensor<1xf32>
          }
        """

        parsed_module = parse_mlir_text(mlir_text)

        self.assertEqual(len(parsed_module.functions), 1)
        self.assertEqual(parsed_module.functions[0].name, "main")
        self.assertEqual(len(parsed_module.functions[0].body.operations), 2)

    def test_invalid_function_header_raises(self):
        mlir_text = """
          func.func not_a_valid_header {
            return
          }
        """

        with self.assertRaisesRegex(ValueError, "invalid function header"):
            parse_mlir_text(mlir_text)

    def test_parse_nested_region_block_structure(self):
        mlir_text = """
          func.func @nested_region() {
            \"test.region_holder\"() {} {
            ^bb0(%arg0: i32):
              %0 = \"test.inner\"(%arg0) : (i32) -> i32
              \"test.yield\"(%0) : (i32) -> ()
            } : () -> ()
            return
          }
        """

        parsed_module = parse_mlir_text(mlir_text)

        self.assertGreaterEqual(len(parsed_module.functions), 1)
        function = parsed_module.functions[0]
        self.assertEqual(function.name, "nested_region")
        self.assertEqual(len(function.body.operations), 2)

        region_holder = function.body.operations[0]
        self.assertEqual(region_holder.name, '"test.region_holder"')
        self.assertEqual(len(region_holder.regions), 1)
        self.assertEqual(len(region_holder.regions[0].blocks), 1)
        self.assertEqual(len(region_holder.regions[0].blocks[0].operations), 2)

    def test_unmatched_parenthesis_in_operation_raises(self):
        mlir_text = """
          func.func @bad_paren(%arg0: tensor<1xf32>) -> tensor<1xf32> {
            %0 = \"test.identity\"(%arg0 : (tensor<1xf32>) -> tensor<1xf32>
            return %0 : tensor<1xf32>
          }
        """

        with self.assertRaisesRegex(
            ValueError, r"unmatched delimiter starting at \d+"
        ):
            parse_mlir_text(mlir_text)


if __name__ == "__main__":
    unittest.main()
