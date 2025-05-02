# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
try:
    from .extension import *
except ModuleNotFoundError as e:
    # TODO Check if the error is due to isaacsim not being installed
    print('\033[91m' + "ISAAC SIM may not to be installed. Please install it to use full GO4R functionality.")
    import traceback, sys
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print('\033[0m')



