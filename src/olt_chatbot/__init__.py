"""A demo chatbot."""

import os

from olt_chatbot.constants import settings as config

__all__ = ["config"]

# Set umask to 0o002 to ensure group write permissions
os.umask(0o002)
