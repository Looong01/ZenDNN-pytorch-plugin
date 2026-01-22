# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import torch


def optimize(model, dtype=torch.bfloat16):
    """
    NOTE: This API has been deprecated. Please refer to the README for
    instructions on running generative models with V.
    """
    raise DeprecationWarning(
        "The 'zentorch.llm.optimize' API has been deprecated. "
        "Please refer to the README for instructions on running generative models with vLLM."
    )
    sys.exit(0)
