import os
import pathlib

from dotenv import load_dotenv

load_dotenv()

RESOURCES_DIR = pathlib.Path(os.environ.get("RESOURCES_DIR", "resources"))
