# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
mkdir -p ${PREFIX}/bin
echo "#! /bin/bash" > ${PREFIX}/bin/hello.sh
echo "set -ex" >> ${PREFIX}/bin/hello.sh
echo "echo \"This package was built with CUDAARCHS=${CUDAARCHS}\"" >> ${PREFIX}/bin/hello.sh
chmod u+x ${PREFIX}/bin/hello.sh
