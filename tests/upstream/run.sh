#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Run conda's own __cuda tests with this plugin installed, to check that replacing conda's
# built-in __cuda provider does not break behavior that conda tests.
#
# The conda submodule must be checked out at the tag matching the installed conda, and conda's
# test requirements (tests/upstream/conda/tests/requirements*.txt) must be installed:
#
#   git submodule update --init
#   git -C tests/upstream/conda fetch --depth 1 origin tag "$(python -m conda --version | cut -d' ' -f2)"
#   git -C tests/upstream/conda checkout "$(python -m conda --version | cut -d' ' -f2)"
#
# Tests that call conda.plugins.virtual_packages.cuda directly only test conda's own detector,
# which this plugin replaces; the remaining tests go through the plugin manager and so exercise
# this plugin.
set -euo pipefail

# conda's expected solver messages assume the default channel priority, not a user's condarc.
export CONDA_CHANNEL_PRIORITY=flexible

cd "$(dirname "$0")/conda"
exec python -m pytest -o addopts="" -p no:cacheprovider \
  tests/plugins/test_virtual_packages.py \
  tests/core/test_index.py::test_supplement_index_with_system_cuda \
  tests/core/test_solve.py \
  -k "cuda or subdir_override or virtual_package" \
  "$@"
