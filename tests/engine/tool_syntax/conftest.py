import json
from pathlib import Path

import pytest

from ds4py.engine.tool_syntax.dsml import Dsml
from ds4py.engine.tool_syntax.glm import Glm
from ds4py.engine.tool_syntax.qwen import Qwen


def load_cases():
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


def new_parser(case):
    if case["syntax"] == "dsml":
        return Dsml(v41=case.get("v41", False))
    if case["syntax"] == "glm":
        return Glm()
    return Qwen()


def run_chunks(case, chunks):
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
        "param_close_prefix": finished.param_close_prefix,
        "in_think": finished.in_think,
        "dsml_in_think": finished.dsml_in_think,
    }


def simple_case(syntax, v41=False):
    return {"syntax": syntax, "v41": v41, "id": f"simple-{syntax}-{v41}"}


def simple_stanza(syntax, v41=False, value="abc"):
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
def parser_case(request):
    return request.param


@pytest.fixture
def parse_chunks():
    return run_chunks
