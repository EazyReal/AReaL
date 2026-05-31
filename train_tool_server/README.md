# train-tool-server

HTTP tool server for RL training pipelines. Boots a set of per-agent backends + a
tool-aware FastAPI router that the rollout loop talks to via `POST /get_observation`.

Forked from [verl-tool](https://github.com/volcengine/verl) and stripped down to
just the geo_edit tool subset (paddleocr / sam3 / grounding_dino + CPU image ops).

## Layout

```
train_tool_server/
├── LICENSE
├── README.md                # this file
├── pyproject.toml           # version pulled from VERSION
├── setup.py                 # mirrors install_requires for `pip install .`
├── VERSION                  # single source of truth for version string
├── requirements.txt         # pinned runtime deps (vllm, transformers commit, etc.)
├── scripts/
│   ├── setup_env.sh         # `pip install -r requirements.txt` + editable installs
│   └── launch_tool_server.sh
└── train_tool_server/       # Python package (flat - no servers/ subdir)
    ├── __init__.py
    ├── server.py            # backend entry: `python -m train_tool_server.server`
    ├── router.py            # tool-aware router (router_factory for uvicorn)
    ├── ray_manager.py       # Ray actor manager (used when --use_ray True)
    ├── utils.py             # hash_requests, etc.
    └── tools/
        ├── __init__.py
        ├── base.py          # BaseTool + register_tool + auto-scans ALL_TOOLS
        ├── finish.py        # cleanup tool, always loaded
        ├── geo_edit_base.py # GeoEditToolBase + GeoEditAgentToolBase
        ├── geo_edit_function.py  # CPU image ops (crop, label, draw, bbox, highlight)
        ├── geo_paddleocr.py # OCR via PaddleOCR-VL (Ray + GPU)
        ├── geo_sam3.py      # segmentation via SAM 3.1 (Ray + GPU)
        └── geo_grounding_dino.py  # detection (Ray + GPU)
```

## Setup (per node)

The repo expects a sibling `geo_edit/` repo at `../geo_edit` for tool implementations
and model paths. Set those up first, then:

```bash
cd train_tool_server
bash scripts/setup_env.sh
```

This will:
1. `pip install -r requirements.txt` (vllm, pinned transformers, fire, uvicorn, ray, ...)
2. `pip install -e . --no-deps` (this repo, editable)
3. `pip install -e ../geo_edit --no-deps` (sibling, editable)

## Launch

```bash
# Start a Ray head node first (single node example):
ray start --head --port=6379

# Then launch all 4 default agents + router on the same machine:
bash scripts/launch_tool_server.sh

# Or a subset:
bash scripts/launch_tool_server.sh geo_edit_function geo_paddleocr

# Custom router port:
PORT=30888 bash scripts/launch_tool_server.sh
```

Logs land in `tool-server-logs/` (one file per agent + `router.log`).

Tool server endpoint: `http://<host>:<PORT>/get_observation`

Stop everything: `pkill -f "train_tool_server"`

## Required model weights

Defaults assume these paths on the host filesystem:

| Tool | Path |
|---|---|
| geo_paddleocr | `/storage/openpsi/models/PaddleOCR-VL-1.5` |
| geo_sam3 | `/storage/openpsi/models/sam3.1/sam3.1_multiplex.pt` |
| geo_grounding_dino | `/storage/openpsi/models/grounding-dino-base` |

Edit `geo_edit/tool_definitions/agents/{paddleocr_tool,sam3,grounding_dino}.py` to
change.

## How requests are routed

`launch_tool_server.sh` boots one tool backend per agent (ports `PORT+1`, `PORT+2`,
...) plus a router on `PORT`. The router parses incoming
`<action>{"name": ...}</action>` strings and forwards to the backend that owns the
tool. The function-to-backend mapping is hard-coded in
`train_tool_server/router.py` (`TOOL_TYPE_FUNCTIONS`).

`finish` requests fan out to every backend so per-trajectory env caches get
cleaned up.

## Development

```bash
# Verify the install:
python -m train_tool_server.server --help

# Lint:
ruff check train_tool_server/
```

## License

Apache-2.0 (inherited from verl-tool upstream). See `LICENSE`.
