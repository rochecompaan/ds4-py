from typing import Any
import json
from pathlib import Path

import pytest

from ds4py.engine.tool_syntax.dsml import Dsml
from ds4py.engine.tool_syntax.glm import Glm
from ds4py.engine.tool_syntax.qwen import Qwen


def load_cases() -> Any:
    cases = []
    fixture_root = Path(__file__).parents[2] / "fixtures" / "tool_syntax"
    for syntax in ("dsml", "glm", "qwen"):
        for index, case in enumerate(json.loads((fixture_root / f"{syntax}.json").read_text())):
            item = dict(case, syntax=syntax, id=f"{syntax}-{index}:{case['source']}")
            cases.append(item)
    for syntax in ("dsml", "glm", "qwen"):
        valid = next(case for case in cases if case["syntax"] == syntax and case["state"] == "DONE")
        cases.append(dict(
            valid,
            id=f"{syntax}-think-marker-splits",
            input="<think>reason</think>" + valid["input"],
        ))
    return tuple(cases)


def new_parser(case: Any) -> Any:
    if case["syntax"] == "dsml":
        return Dsml(v41=case.get("v41", False))
    if case["syntax"] == "glm":
        return Glm()
    return Qwen()


def run_chunks(case: Any, chunks: Any) -> Any:
    parser = new_parser(case)
    streamed = None
    for chunk in chunks:
        streamed = parser.feed(chunk)
    if streamed is None:
        streamed = parser.feed(b"")
    greedy_before_finish = parser.wants_greedy_sampling()
    finished = parser.finish()
    return {
        "feed_state": streamed.state,
        "state": finished.state,
        "error": finished.error,
        "greedy_before_finish": greedy_before_finish,
        "greedy": parser.wants_greedy_sampling(),
        "raw": finished.raw,
        "calls": tuple((call.name, tuple(call.args.items())) for call in finished.calls),
        "call_raw": tuple(call.raw for call in finished.calls),
        "param_close_prefix": finished.param_close_prefix,
        "in_think": finished.in_think,
        "dsml_in_think": finished.dsml_in_think,
    }


def simple_case(syntax: Any, v41: Any=False) -> dict[str, Any]:
    return {"syntax": syntax, "v41": v41, "id": f"simple-{syntax}-{v41}"}


def simple_stanza(syntax: Any, v41: Any=False, value: Any="abc") -> str:
    if syntax == "glm":
        return f"<tool_call>echo<arg_key>value</arg_key><arg_value>{value}</arg_value></tool_call>"
    if syntax == "qwen":
        return f"<tool_call><function=echo><parameter=value>{value}</parameter></function></tool_call>"
    tags = (" calls", " invoke", " parameter") if v41 else ("tool_calls", "invoke", "parameter")
    calls, invoke, parameter = tags
    return (f"<｜DSML｜{calls}><｜DSML｜{invoke} name=\"echo\">"
            f"<｜DSML｜{parameter} name=\"value\" string=\"true\">{value}</｜DSML｜{parameter}>"
            f"</｜DSML｜{invoke}></｜DSML｜{calls}>")


@pytest.fixture(params=load_cases(), ids=lambda case: case["id"])
def parser_case(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture
def parse_chunks() -> Any:
    return run_chunks
