from typing import Any

import pytest

from ds4py.engine.tool_prompt import ToolDefinition, ToolPromptGenerator

READ_SCHEMA = '{"name":"read","description":"Read a file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}'
EDIT_SCHEMA = '{"name":"edit","description":"Edit ✓","parameters":{"type":"object","properties":{"path":{"type":"string"},"new_string":{"type":"string"},"whole":{"type":"boolean"}},"required":["path","new_string"]}}'
SCHEMAS = READ_SCHEMA + "\n" + EDIT_SCHEMA
DEFINITIONS = [
    ToolDefinition(
        "read",
        "Read a file.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    ),
    ToolDefinition(
        "edit",
        "Edit ✓",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "new_string": {"type": "string"},
                "whole": {"type": "boolean"},
            },
            "required": ["path", "new_string"],
        },
    ),
]


def deepseek_expected(schemas: Any, *, v41: Any = False) -> str:
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
        f'<｜DSML｜{invoke} name="read">\n'
        f'<｜DSML｜{parameter} name="path" string="true">README.md</｜DSML｜{parameter}>\n'
        f"</｜DSML｜{invoke}>\n"
        f"</｜DSML｜{calls}>\n\n"
        'Use string="false" for JSON values.\n'
        f"Escape literal {closing} as {escaped}; write {escaped} literally as {doubly_escaped}. "
        "Other entities stay unchanged.\n"
    )


def glm_expected(schemas: Any) -> str:
    return (
        "# Tools\n\n"
        f"<tools>{schemas}</tools>\n\n"
        "Call example:\n"
        "<tool_call>read<arg_key>path</arg_key><arg_value>README.md</arg_value></tool_call>\n\n"
        "Escape literal </arg_value> as &lt;/arg_value>; write &lt;/arg_value> literally as "
        "&amp;lt;/arg_value>. Other entities stay unchanged.\n"
    )


def qwen_expected(schemas: Any) -> str:
    return (
        "# Tools\n\n"
        f"<tools>{schemas}</tools>\n\n"
        "Call example:\n"
        "<tool_call>\n<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n"
        "</function>\n</tool_call>\n\n"
        "Escape literal </parameter> as &lt;/parameter>; write &lt;/parameter> literally as "
        "&amp;lt;/parameter>. Other entities stay unchanged.\n"
    )


def test_deepseek_prompt_is_exact() -> None:
    assert ToolPromptGenerator("deepseek4").generate(DEFINITIONS) == deepseek_expected(SCHEMAS)


def test_deepseek41_prompt_uses_spaced_dsml_tag_names() -> None:
    prompt = ToolPromptGenerator("deepseek41").generate(DEFINITIONS)
    assert prompt == deepseek_expected(SCHEMAS, v41=True)
    assert "</｜DSML｜ calls>" in prompt
    assert "</｜DSML｜ invoke>" in prompt
    assert "</｜DSML｜ parameter>" in prompt


def test_glm_prompt_is_exact() -> None:
    prompt = ToolPromptGenerator("glm").generate(DEFINITIONS)
    assert prompt == glm_expected(SCHEMAS)


def test_qwen_prompt_is_exact_and_wraps_each_schema() -> None:
    wrapped = (
        '{"type":"function","function":' + READ_SCHEMA + "}\n" + '{"type":"function","function":' + EDIT_SCHEMA + "}"
    )
    prompt = ToolPromptGenerator("qwen").generate(DEFINITIONS)
    assert prompt == qwen_expected(wrapped)
    assert "Edit ✓" in prompt
    assert "\\u2713" not in prompt


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("deepseek4", deepseek_expected("")),
        ("deepseek41", deepseek_expected("", v41=True)),
        ("glm", glm_expected("")),
        ("qwen", qwen_expected("")),
    ],
)
def test_empty_definitions_keep_the_fixed_structure(family: Any, expected: Any) -> None:
    prompt = ToolPromptGenerator(family).generate(())
    assert prompt == expected
    assert "<tools></tools>" in prompt


def test_nested_schema_insertion_order_is_preserved() -> None:
    definition = ToolDefinition("ordered", "Ordered.", {"z": 0, "a": 1})
    prompt = ToolPromptGenerator("qwen").generate([definition])
    assert '"parameters":{"z":0,"a":1}' in prompt


def test_generator_does_not_copy_c_tool_contracts_or_edit_upto() -> None:
    for family in ("deepseek4", "deepseek41", "glm", "qwen"):
        prompt = ToolPromptGenerator(family).generate(DEFINITIONS)
        for forbidden in ("[upto]", "edit-upto", "bash", "google_search", "visit_page"):
            assert forbidden not in prompt


def test_unknown_family_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown tool prompt family: other"):
        ToolPromptGenerator("other").generate(DEFINITIONS)


@pytest.mark.parametrize("family", ["deepseek4", "deepseek41", "glm", "qwen"])
def test_every_generated_example_executes_the_intended_call(family: Any) -> None:
    from ds4py.engine.tool_syntax.dsml import Dsml
    from ds4py.engine.tool_syntax.glm import Glm
    from ds4py.engine.tool_syntax.qwen import Qwen

    prompt = ToolPromptGenerator(family).generate([])
    examples = prompt.split("Call example:\n", 1)[1].split("\n\n", 1)[0]
    factory = {"glm": Glm, "qwen": Qwen}.get(family, lambda: Dsml(v41=family == "deepseek41"))
    # No prose or unsupported teaching stanza may be silently skipped.
    examples = examples.splitlines() if family == "glm" else [examples]
    assert examples
    for example in examples:
        parser = factory()
        parser.feed(example.encode())
        result = parser.finish()
        assert result.state == "DONE"
        assert [(call.name, dict(call.args)) for call in result.calls] == [("read", {"path": "README.md"})]
