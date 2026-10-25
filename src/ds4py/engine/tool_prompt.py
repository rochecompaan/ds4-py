"""Generate model-family-native prompts from caller-owned tool definitions."""
from dataclasses import dataclass
import json
from typing import Mapping


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, object]


class ToolPromptGenerator:
    def __init__(self, family: str):
        self.family = family

    def generate(self, definitions) -> str:
        schemas = [
            json.dumps(
                {"name": item.name, "description": item.description, "parameters": item.parameters},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for item in definitions
        ]
        if self.family in ("deepseek4", "deepseek41"):
            return self._deepseek("\n".join(schemas), v41=self.family == "deepseek41")
        if self.family == "glm":
            return self._glm("\n".join(schemas))
        if self.family == "qwen":
            wrapped = "\n".join('{"type":"function","function":' + schema + "}" for schema in schemas)
            return self._qwen(wrapped)
        raise ValueError(f"unknown tool prompt family: {self.family}")

    @staticmethod
    def _deepseek(schemas: str, *, v41: bool) -> str:
        calls = " calls" if v41 else "tool_calls"
        invoke = " invoke" if v41 else "invoke"
        parameter = " parameter" if v41 else "parameter"
        closing = f"</｜DSML｜{parameter}>"
        escaped = "&lt;" + closing[1:]
        doubly_escaped = "&amp;lt;" + closing[1:]
        return (
            "# Tools\n\n"
            f"<tools>{schemas}</tools>\n\n"
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

    @staticmethod
    def _glm(schemas: str) -> str:
        return (
            "# Tools\n\n"
            f"<tools>{schemas}</tools>\n\n"
            "Call example:\n"
            "<｜tool▁call▁begin｜>read<｜tool▁sep｜>path=\"...\" ...<｜tool▁call▁end｜>\n\n"
            "Escape literal </arg_value> as &lt;/arg_value>; write &lt;/arg_value> literally as "
            "&amp;lt;/arg_value>. Other entities stay unchanged.\n"
        )

    @staticmethod
    def _qwen(schemas: str) -> str:
        return (
            "# Tools\n\n"
            f"<tools>{schemas}</tools>\n\n"
            "Call example:\n"
            "<tool_call>\n<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n"
            "</function>\n</tool_call>\n\n"
            "Escape literal </parameter> as &lt;/parameter>; write &lt;/parameter> literally as "
            "&amp;lt;/parameter>. Other entities stay unchanged.\n"
        )
