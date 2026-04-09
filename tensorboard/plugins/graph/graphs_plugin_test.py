# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Integration tests for the Graphs Plugin."""


import collections.abc
import math
import os.path
import tempfile
import types

import tensorflow as tf
from werkzeug import test as werkzeug_test
from werkzeug import wrappers

from google.protobuf import text_format
from tensorboard import context
from tensorboard import errors
from tensorboard.backend.event_processing import data_provider
from tensorboard.backend.event_processing import (
    plugin_event_multiplexer as event_multiplexer,
)
from tensorboard.compat.proto import config_pb2
from tensorboard.plugins import base_plugin
from tensorboard.plugins.graph import graphs_plugin
from tensorboard.util import test_util

tf.compat.v1.disable_v2_behavior()


# TODO(stephanwlee): Move more tests into the base class when v2 test
# can write graph and metadata with a TF public API.


_RUN_WITH_GRAPH_WITH_METADATA = ("_RUN_WITH_GRAPH_WITH_METADATA", True, True)
_RUN_WITHOUT_GRAPH_WITH_METADATA = (
    "_RUN_WITHOUT_GRAPH_WITH_METADATA",
    False,
    True,
)
_RUN_WITH_GRAPH_WITHOUT_METADATA = (
    "_RUN_WITH_GRAPH_WITHOUT_METADATA",
    True,
    False,
)
_RUN_WITHOUT_GRAPH_WITHOUT_METADATA = (
    "_RUN_WITHOUT_GRAPH_WITHOUT_METADATA",
    False,
    False,
)


class GraphsPluginBaseTest:

    _METADATA_TAG = "secret-stats"
    _MESSAGE_PREFIX_LENGTH_LOWER_BOUND = 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plugin = None

    def setUp(self):
        super().setUp()

    def generate_run(
        self, logdir, run_name, include_graph, include_run_metadata
    ):
        """Create a run."""
        raise NotImplementedError("Please implement generate_run")

    def load_plugin(self, run_specs, mlir_file=None):
        logdir = self.get_temp_dir()
        for run_spec in run_specs:
            self.generate_run(logdir, *run_spec)
        multiplexer = event_multiplexer.EventMultiplexer()
        multiplexer.AddRunsFromDirectory(logdir)
        multiplexer.Reload()
        provider = data_provider.MultiplexerDataProvider(multiplexer, logdir)
        flags = types.SimpleNamespace(mlir_file=mlir_file)
        ctx = base_plugin.TBContext(
            logdir=logdir,
            multiplexer=multiplexer,
            data_provider=provider,
            flags=flags,
        )
        return graphs_plugin.GraphsPlugin(ctx)

    def testRoutesProvided(self):
        """Tests that the plugin offers the correct routes."""
        plugin = self.load_plugin(
            [_RUN_WITH_GRAPH_WITH_METADATA, _RUN_WITHOUT_GRAPH_WITH_METADATA]
        )
        routes = plugin.get_plugin_apps()
        self.assertIsInstance(routes["/graph"], collections.abc.Callable)
        self.assertIsInstance(routes["/run_metadata"], collections.abc.Callable)
        self.assertIsInstance(routes["/info"], collections.abc.Callable)


class GraphsPluginV1Test(GraphsPluginBaseTest, tf.test.TestCase):
    def generate_run(
        self, logdir, run_name, include_graph, include_run_metadata
    ):
        """Create a run with a text summary, metadata, and optionally a
        graph."""
        tf.compat.v1.reset_default_graph()
        k1 = tf.constant(math.pi, name="k1")
        k2 = tf.constant(math.e, name="k2")
        result = (k1**k2) - k1
        expected = tf.constant(20.0, name="expected")
        error = tf.abs(result - expected, name="error")
        message_prefix_value = "error " * 1000
        true_length = len(message_prefix_value)
        assert (
            true_length > self._MESSAGE_PREFIX_LENGTH_LOWER_BOUND
        ), true_length
        message_prefix = tf.constant(
            message_prefix_value, name="message_prefix"
        )
        error_message = tf.strings.join(
            [message_prefix, tf.as_string(error, name="error_string")],
            name="error_message",
        )
        summary_message = tf.compat.v1.summary.text(
            "summary_message", error_message
        )

        sess = tf.compat.v1.Session()
        writer = test_util.FileWriter(os.path.join(logdir, run_name))
        if include_graph:
            writer.add_graph(sess.graph)
        options = tf.compat.v1.RunOptions(
            trace_level=tf.compat.v1.RunOptions.FULL_TRACE
        )
        run_metadata = config_pb2.RunMetadata()
        s = sess.run(
            summary_message, options=options, run_metadata=run_metadata
        )
        writer.add_summary(s)
        if include_run_metadata:
            writer.add_run_metadata(run_metadata, self._METADATA_TAG)
        writer.close()

    def _get_graph(self, plugin, run=None, **kwargs):
        """Set up runs, then fetch and return the graph as a proto."""
        (graph_pbtxt, mime_type) = plugin.graph_impl(
            context.RequestContext(),
            run if run is not None else _RUN_WITH_GRAPH_WITH_METADATA[0],
            **kwargs,
        )
        self.assertEqual(mime_type, "text/x-protobuf")
        return text_format.Parse(graph_pbtxt, tf.compat.v1.GraphDef())

    def test_info(self):
        plugin = self.load_plugin(
            [
                _RUN_WITH_GRAPH_WITH_METADATA,
                _RUN_WITH_GRAPH_WITHOUT_METADATA,
                _RUN_WITHOUT_GRAPH_WITH_METADATA,
                _RUN_WITHOUT_GRAPH_WITHOUT_METADATA,
            ]
        )
        expected = {
            "_RUN_WITH_GRAPH_WITH_METADATA": {
                "run": "_RUN_WITH_GRAPH_WITH_METADATA",
                "run_graph": True,
                "tags": {
                    "secret-stats": {
                        "conceptual_graph": False,
                        "profile": True,
                        "tag": "secret-stats",
                        "op_graph": False,
                    },
                },
            },
            "_RUN_WITH_GRAPH_WITHOUT_METADATA": {
                "run": "_RUN_WITH_GRAPH_WITHOUT_METADATA",
                "run_graph": True,
                "tags": {},
            },
            "_RUN_WITHOUT_GRAPH_WITH_METADATA": {
                "run": "_RUN_WITHOUT_GRAPH_WITH_METADATA",
                "run_graph": False,
                "tags": {
                    "secret-stats": {
                        "conceptual_graph": False,
                        "profile": True,
                        "tag": "secret-stats",
                        "op_graph": False,
                    },
                },
            },
        }

        actual = plugin.info_impl(context.RequestContext(), "eid")
        self.assertEqual(expected, actual)

    def test_graph_simple(self):
        plugin = self.load_plugin([_RUN_WITH_GRAPH_WITH_METADATA])
        graph = self._get_graph(
            plugin,
            tag=None,
            is_conceptual=False,
            experiment="eid",
        )
        node_names = set(node.name for node in graph.node)
        self.assertEqual(
            {
                "k1",
                "k2",
                "pow",
                "sub",
                "expected",
                "sub_1",
                "error",
                "message_prefix",
                "error_string",
                "error_message",
                "summary_message",
                "summary_message/tag",
                "summary_message/serialized_summary_metadata",
            },
            node_names,
        )

    def test_nonexistent(self):
        plugin = self.load_plugin([_RUN_WITH_GRAPH_WITH_METADATA])
        with self.assertRaises(errors.NotFoundError):
            graph = self._get_graph(
                plugin,
                run="nope",
                tag=None,
                is_conceptual=False,
                experiment="eid",
            )

    def test_graph_large_attrs(self):
        plugin = self.load_plugin([_RUN_WITH_GRAPH_WITH_METADATA])
        key = "o---;;-;"
        graph = self._get_graph(
            plugin,
            tag=None,
            is_conceptual=False,
            experiment="eid",
            limit_attr_size=self._MESSAGE_PREFIX_LENGTH_LOWER_BOUND,
            large_attrs_key=key,
        )
        large_attrs = {
            node.name: list(node.attr[key].list.s)
            for node in graph.node
            if key in node.attr
        }
        self.assertEqual({"message_prefix": [b"value"]}, large_attrs)

    def test_run_metadata(self):
        plugin = self.load_plugin([_RUN_WITH_GRAPH_WITH_METADATA])
        ctx = context.RequestContext()
        result = plugin.run_metadata_impl(
            ctx, "123", _RUN_WITH_GRAPH_WITH_METADATA[0], self._METADATA_TAG
        )
        (metadata_pbtxt, mime_type) = result
        self.assertEqual(mime_type, "text/x-protobuf")
        text_format.Parse(metadata_pbtxt, config_pb2.RunMetadata())
        # If it parses, we're happy.

    def test_is_active(self):
        plugin = self.load_plugin([_RUN_WITH_GRAPH_WITHOUT_METADATA])
        self.assertFalse(plugin.is_active())

    def test_info_includes_mlir_run_when_flag_set(self):
        mlir_text = """
          func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
            %0 = \"test.identity\"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
            return %0 : tensor<1xf32>
          }
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(mlir_text)
            mlir_file = f.name

        try:
            plugin = self.load_plugin([], mlir_file=mlir_file)
            actual = plugin.info_impl(context.RequestContext(), "eid")
        finally:
            os.remove(mlir_file)

        self.assertIn(graphs_plugin.MLIR_IMPORT_RUN_NAME, actual)
        self.assertEqual(
            {
                "run": graphs_plugin.MLIR_IMPORT_RUN_NAME,
                "run_graph": True,
                "tags": {},
            },
            actual[graphs_plugin.MLIR_IMPORT_RUN_NAME],
        )

    def test_graph_returns_mlir_graphdef_when_mlir_run_requested(self):
        mlir_text = """
          func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
            %0 = \"test.identity\"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
            return %0 : tensor<1xf32>
          }
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(mlir_text)
            mlir_file = f.name

        try:
            plugin = self.load_plugin([], mlir_file=mlir_file)
            graph = self._get_graph(
                plugin,
                run=graphs_plugin.MLIR_IMPORT_RUN_NAME,
                tag=None,
                is_conceptual=False,
                experiment="eid",
            )
        finally:
            os.remove(mlir_file)

        node_names = set(node.name for node in graph.node)
        self.assertIn("main/arg_arg0", node_names)
        self.assertIn("main/op_0000_test.identity", node_names)

    def test_graph_mlir_parse_error_raises_value_error(self):
        invalid_mlir_text = """
          func.func not_a_valid_header {
            return
          }
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(invalid_mlir_text)
            mlir_file = f.name

        try:
            plugin = self.load_plugin([], mlir_file=mlir_file)
            with self.assertRaisesRegex(ValueError, "failed to parse MLIR file"):
                plugin.graph_impl(
                    context.RequestContext(),
                    graphs_plugin.MLIR_IMPORT_RUN_NAME,
                    tag=None,
                    is_conceptual=False,
                    experiment="eid",
                )
        finally:
            os.remove(mlir_file)

    def test_graph_route_mlir_parse_error_returns_bad_request(self):
        invalid_mlir_text = """
          func.func not_a_valid_header {
            return
          }
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(invalid_mlir_text)
            mlir_file = f.name

        try:
            plugin = self.load_plugin([], mlir_file=mlir_file)
            info = plugin.info_impl(context.RequestContext(), "eid")
            run = next(iter(info))
            client = werkzeug_test.Client(plugin.graph_route, wrappers.Response)
            response = client.get("/?run=%s" % run)
        finally:
            os.remove(mlir_file)

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"failed to parse MLIR file", response.get_data())

    def test_info_uses_unique_mlir_run_name_on_conflict(self):
        mlir_text = """
          func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
            %0 = \"test.identity\"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
            return %0 : tensor<1xf32>
          }
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(mlir_text)
            mlir_file = f.name

        try:
            plugin = self.load_plugin(
                [(graphs_plugin.MLIR_IMPORT_RUN_NAME, False, False)],
                mlir_file=mlir_file,
            )
            info = plugin.info_impl(context.RequestContext(), "eid")
            mlir_run_names = [
                run
                for run, run_data in info.items()
                if run.startswith(graphs_plugin.MLIR_IMPORT_RUN_NAME)
                and run_data["run_graph"]
            ]
            mlir_graph = self._get_graph(
                plugin,
                run=mlir_run_names[0],
                tag=None,
                is_conceptual=False,
                experiment="eid",
            )
            with self.assertRaises(errors.NotFoundError):
                self._get_graph(
                    plugin,
                    run=graphs_plugin.MLIR_IMPORT_RUN_NAME,
                    tag=None,
                    is_conceptual=False,
                    experiment="eid",
                )
        finally:
            os.remove(mlir_file)

        self.assertIn(graphs_plugin.MLIR_IMPORT_RUN_NAME, info)
        self.assertLen(mlir_run_names, 1)
        self.assertNotEqual(mlir_run_names[0], graphs_plugin.MLIR_IMPORT_RUN_NAME)
        node_names = set(node.name for node in mlir_graph.node)
        self.assertIn("main/op_0000_test.identity", node_names)

    def test_graph_mlir_two_calls_with_different_limit_attr_size(self):
        mlir_text = """
          func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
            %0 = \"test.identity\"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
            return %0 : tensor<1xf32>
          }
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(mlir_text)
            mlir_file = f.name

        try:
            plugin = self.load_plugin([], mlir_file=mlir_file)
            first_graph = self._get_graph(
                plugin,
                run=graphs_plugin.MLIR_IMPORT_RUN_NAME,
                tag=None,
                is_conceptual=False,
                experiment="eid",
                limit_attr_size=1,
                large_attrs_key="_large_attrs",
            )
            second_graph = self._get_graph(
                plugin,
                run=graphs_plugin.MLIR_IMPORT_RUN_NAME,
                tag=None,
                is_conceptual=False,
                experiment="eid",
                limit_attr_size=2048,
                large_attrs_key="_large_attrs",
            )
        finally:
            os.remove(mlir_file)

        self.assertGreater(len(first_graph.node), 0)
        self.assertGreater(len(second_graph.node), 0)


if __name__ == "__main__":
    tf.test.main()
