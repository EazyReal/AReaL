# setup.py for train-tool-server
#
# Runtime-only deps are pinned in requirements.txt (single source of truth).
# This file mirrors them for `pip install .` so the package can be consumed
# without a separate `pip install -r requirements.txt` step.
from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent

__version__ = (HERE / "VERSION").read_text().strip()
long_description = (HERE / "README.md").read_text(encoding="utf-8")

install_requires = [
    # Tool server runtime
    "fire",
    "uvicorn",
    "fastapi",
    "pydantic",
    "httpx",
    "httptools",
    "colorlog",
    # Distributed orchestration
    "ray",
    # Vision / preprocessing
    "pillow",
    "numpy",
    "timm",
    "iopath",
    "ftfy",
    "qwen_omni_utils",
    # Misc helpers
    "tensordict",
    "codetiming",
    "nvitop",
    "regex",
    "tqdm",
    "pyyaml",
    # Model loaders (pinned in requirements.txt for exact versions)
    "vllm",
    "transformers",
]

extras_require = {
    "test": ["pytest", "pytest-asyncio", "pytest-rerunfailures"],
}

setup(
    name="train-tool-server",
    version=__version__,
    description="HTTP tool server for RL training - launches geo_edit agent backends + tool-aware router.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_packages(where=".", include=["train_tool_server", "train_tool_server.*"]),
    package_dir={"": "."},
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.10",
    include_package_data=True,
)
