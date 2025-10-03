#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#   This code is part of the training material and, therefore, part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
from svlearn.config.configuration import ConfigurationMixin
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Get the project root directory (where config.yaml is located)
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "config.yaml"

config = ConfigurationMixin().load_config(str(config_path))
