# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Collection of first-party plugins for MLIR-only runtime."""

from tensorboard.plugins.core import core_plugin
from tensorboard.plugins.graph import graphs_plugin


# Ordering matters. The order in which these lines appear determines the
# ordering of tabs in TensorBoard's GUI.
_PLUGINS = [
    core_plugin.CorePluginLoader(include_debug_info=True),
    graphs_plugin.GraphsPlugin,
]


def get_plugins():
    """Returns a list specifying all known TensorBoard plugins.

    This includes both first-party, statically bundled plugins and
    dynamic plugins.

    This list can be passed to the `tensorboard.program.TensorBoard` API.

    Returns:
      The list of default first-party plugins.
    """
    return get_static_plugins() + get_dynamic_plugins()


def get_static_plugins():
    """Returns a list specifying TensorBoard's default first-party plugins.

    Plugins are specified in this list either via a TBLoader instance to load the
    plugin, or the TBPlugin class itself which will be loaded using a BasicLoader.

    This list can be passed to the `tensorboard.program.TensorBoard` API.

    Returns:
      The list of default first-party plugins.

    :rtype: list[Type[base_plugin.TBLoader] | Type[base_plugin.TBPlugin]]
    """

    return _PLUGINS[:]


def get_dynamic_plugins():
    """Returns dynamically loaded plugins.

    MLIR-only runtime does not load dynamic plugins.
    """
    return []
