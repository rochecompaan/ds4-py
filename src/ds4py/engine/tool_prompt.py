"""Generate model-family-native prompts from caller-owned tool definitions."""
from collections.abc import Iterable
from dataclasses import dataclass
import json
from typing import Mapping


@dataclass(frozen=True)
class ToolDefinition:
    """Caller-owned tool schema; names, descriptions, and parameters are untrusted."""
    name: str
    description: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class ToolPromptSpan:
    """A template span with an explicit trusted-rendering boundary."""
    text: str
    rendered: bool


class ToolPromptGenerator:
    """Generate executable family examples and schemas without native calls."""
    def __init__(self, family: str) -> None:
        self.family = family

    def generate(self, definitions: Iterable[ToolDefinition]) -> str:
        """Return the full prompt text for ordered tool definitions."""
        return "".join(span.text for span in self.spans(definitions))

    def spans(self, definitions: Iterable[ToolDefinition]) -> list[ToolPromptSpan]:
        """Separate trusted template markup from plain serialized definitions."""
        schemas = [
            json.dumps(
                {"name": item.name, "description": item.description, "parameters": item.parameters},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for item in definitions
        ]
        if self.family in ("deepseek4", "deepseek41"):
            return self._deepseek(schemas, v41=self.family == "deepseek41")
        if self.family == "glm":
            return self._glm(schemas)
        if self.family == "qwen":
            return self._qwen(schemas)
        raise ValueError(f"unknown tool prompt family: {self.family}")

    @staticmethod
    def _schema_spans(
        schemas: Iterable[str], prefix: str, suffix: str, *, qwen: bool = False,
    ) -> list[ToolPromptSpan]:
        spans = [ToolPromptSpan(prefix, True)]
        for index, schema in enumerate(schemas):
            if index:
                spans.append(ToolPromptSpan("\n", True))
            if qwen:
                spans.append(ToolPromptSpan('{"type":"function","function":', True))
            spans.append(ToolPromptSpan(schema, False))
            if qwen:
                spans.append(ToolPromptSpan("}", True))
        spans.append(ToolPromptSpan(suffix, True))
        return spans

    @classmethod
    def _deepseek(cls, schemas: Iterable[str], *, v41: bool) -> list[ToolPromptSpan]:
        calls = " calls" if v41 else "tool_calls"
        invoke = " invoke" if v41 else "invoke"
        parameter = " parameter" if v41 else "parameter"
        closing = f"</｜DSML｜{parameter}>"
        escaped = "&lt;" + closing[1:]
        doubly_escaped = "&amp;lt;" + closing[1:]
        suffix = (
            "</tools>\n\n"
            "Call example:\n"
            f"<｜DSML｜{calls}>\n"
            f"<｜DSML｜{invoke} name=\"read\">\n"
            f"<｜DSML｜{parameter} name=\"path\" string=\"true\">README.md</｜DSML｜{parameter}>\n"
            f"</｜DSML｜{invoke}>\n"
            f"</｜DSML｜{calls}>\n\n"
            "Use string=\"false\" for JSON values.\n"
            f"Escape literal {closing} as {escaped}; write {escaped} literally as {doubly_escaped}. "
            "Other entities stay unchanged.\n"
        )
        return cls._schema_spans(schemas, "# Tools\n\n<tools>", suffix)

    @classmethod
    def _glm(cls, schemas: Iterable[str]) -> list[ToolPromptSpan]:
        suffix = (
            "</tools>\n\n"
            "Call example:\n"
            "<tool_call>read<arg_key>path</arg_key><arg_value>README.md</arg_value></tool_call>\n\n"
            "Escape literal </arg_value> as &lt;/arg_value>; write &lt;/arg_value> literally as "
            "&amp;lt;/arg_value>. Other entities stay unchanged.\n"
        )
        return cls._schema_spans(schemas, "# Tools\n\n<tools>", suffix)

    @classmethod
    def _qwen(cls, schemas: Iterable[str]) -> list[ToolPromptSpan]:
        suffix = (
            "</tools>\n\n"
            "Call example:\n"
            "<tool_call>\n<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n"
            "</function>\n</tool_call>\n\n"
            "Escape literal </parameter> as &lt;/parameter>; write &lt;/parameter> literally as "
            "&amp;lt;/parameter>. Other entities stay unchanged.\n"
        )
        return cls._schema_spans(schemas, "# Tools\n\n<tools>", suffix, qwen=True)
