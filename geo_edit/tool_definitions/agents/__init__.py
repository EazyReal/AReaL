"""Agent Tools Registry - kept-tool subset only.

This module exports static agent definitions (declarations, configs, prompts).
Execute functions are created dynamically by ToolRouter to avoid circular dependencies.

Tool surface (minimized):
  - paddleocr : text_ocr, table_ocr, formula_ocr, text_spotting, map_text_ocr
  - sam3      : auto_segment, bbox_segment, text_segment, exemplar_segment,
                concept_count, presence_check
  - grounding_dino : grounding_dino
"""

from typing import Dict, Type

from geo_edit.tool_definitions.agents import sam3, grounding_dino, paddleocr_tool

AGENT_DECLARATIONS: Dict[str, dict] = {
    "grounding_dino": grounding_dino.DECLARATION,
}

AGENT_RETURN_TYPES: Dict[str, str] = {
    "grounding_dino": grounding_dino.RETURN_TYPE,
}

AGENT_CONFIGS: Dict[str, dict] = {
    "sam3": sam3.agent_config,
    "grounding_dino": grounding_dino.agent_config,
    "paddleocr": paddleocr_tool.agent_config,
}

AGENT_SYSTEM_PROMPTS: Dict[str, str] = {
    "sam3": sam3.SYSTEM_PROMPT,
    "grounding_dino": grounding_dino.SYSTEM_PROMPT,
    "paddleocr": paddleocr_tool.SYSTEM_PROMPT,
}

AGENT_ACTOR_CLASSES: Dict[str, Type] = {
    "sam3": sam3.ACTOR_CLASS,
    "grounding_dino": grounding_dino.ACTOR_CLASS,
    "paddleocr": paddleocr_tool.ACTOR_CLASS,
}

MULTI_TOOL_DECLARATIONS: Dict[str, Dict] = {}

if hasattr(paddleocr_tool, 'DECLARATIONS'):
    for tool_name, decl in paddleocr_tool.DECLARATIONS.items():
        MULTI_TOOL_DECLARATIONS[tool_name] = {
            "declaration": decl,
            "base_agent": "paddleocr",
            "actor_class": paddleocr_tool.ACTOR_CLASS,
            "agent_config": paddleocr_tool.agent_config,
            "system_prompt": paddleocr_tool.SYSTEM_PROMPT,
        }

if hasattr(sam3, 'DECLARATIONS'):
    for tool_name, decl in sam3.DECLARATIONS.items():
        MULTI_TOOL_DECLARATIONS[tool_name] = {
            "declaration": decl,
            "base_agent": "sam3",
            "actor_class": sam3.ACTOR_CLASS,
            "agent_config": sam3.agent_config,
            "system_prompt": sam3.SYSTEM_PROMPT,
        }


def get_actor_class(agent_name: str) -> Type:
    return AGENT_ACTOR_CLASSES[agent_name]
