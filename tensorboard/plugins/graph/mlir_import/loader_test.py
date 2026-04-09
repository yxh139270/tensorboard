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

import os
import tempfile
import unittest

from tensorboard.plugins.graph.mlir_import.loader import (
    MlirImportError,
    load_mlir_graphdef,
)


class MlirLoaderTest(unittest.TestCase):
    def test_load_mlir_graphdef_success(self):
        mlir_text = """
          func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
            %0 = \"test.identity\"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
            return %0 : tensor<1xf32>
          }
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(mlir_text)
            file_path = f.name

        try:
            graph_def = load_mlir_graphdef(file_path)
        finally:
            os.remove(file_path)

        nodes_by_name = {node.name: node for node in graph_def.node}
        self.assertIn("main/arg_arg0", nodes_by_name)
        self.assertIn("main/op_0000_test.identity", nodes_by_name)

    def test_load_mlir_graphdef_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = os.path.join(temp_dir, "missing.mlir")
            with self.assertRaisesRegex(MlirImportError, "does not exist"):
                load_mlir_graphdef(missing_path)

    def test_load_mlir_graphdef_invalid_mlir_raises_parse_error(self):
        invalid_mlir_text = """
          func.func not_a_valid_header {
            return
          }
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(invalid_mlir_text)
            file_path = f.name

        try:
            with self.assertRaisesRegex(MlirImportError, "failed to parse"):
                load_mlir_graphdef(file_path)
        finally:
            os.remove(file_path)


if __name__ == "__main__":
    unittest.main()
