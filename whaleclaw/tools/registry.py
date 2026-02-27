"""Tool registry — manages available tools and generates LLM schemas."""

from __future__ import annotations

from whaleclaw.providers.base import ToolSchema
from whaleclaw.tools.base import Tool, ToolDefinition


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """Return definitions for all registered tools."""
        return [t.definition for t in self._tools.values()]

    def to_llm_schemas(self, include_names: set[str] | None = None) -> list[ToolSchema]:
        """Convert all tools to LLM-native ``ToolSchema`` objects.

        These are passed via the ``tools`` API parameter — not injected
        into the system prompt.
        """
        schemas: list[ToolSchema] = []
        for tool in self._tools.values():
            defn = tool.definition
            if include_names is not None and defn.name not in include_names:
                continue
            properties: dict[str, object] = {}
            required: list[str] = []
            for p in defn.parameters:
                prop: dict[str, object] = {"type": p.type, "description": p.description}
                if p.enum:
                    prop["enum"] = p.enum
                properties[p.name] = prop
                if p.required:
                    required.append(p.name)
            schemas.append(
                ToolSchema(
                    name=defn.name,
                    description=defn.description,
                    input_schema={
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                )
            )
        return schemas

    def to_prompt_fallback(self) -> str:
        """Generate a text description for providers without native tools support."""
        if not self._tools:
            return ""
        lines: list[str] = []
        for tool in self._tools.values():
            defn = tool.definition
            params_parts: list[str] = []
            for p in defn.parameters:
                req = "" if p.required else ", optional"
                enum_hint = f", enum={p.enum}" if p.enum else ""
                params_parts.append(
                    f"    - {p.name} ({p.type}{req}{enum_hint}): {p.description}"
                )
            lines.append(f"### {defn.name}\n{defn.description}")
            if params_parts:
                lines.append("参数:\n" + "\n".join(params_parts))
        return "\n\n".join(lines)
