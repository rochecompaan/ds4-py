"""Default ds4-py agent composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Instrumentation, LocalWorkspace
from pydantic_ai_harness import (
    AskUser,
    FileSystem,
    RepoContext,
    Shell,
    Skills,
    ToolOutputLimits,
)
from pydantic_ai_harness.ask_user import Answerer as QuestionAnswerer

from ds4py.capabilities.run_record import DS4RunRecord
from ds4py.capabilities.shell_jobs import DS4ShellJobs
from ds4py.capabilities.system_notes import DS4SystemNotes
from ds4py.model import DS4Model


class AgentOptions(TypedDict, total=False):
    capabilities: list[AbstractCapability]
    instructions: str


@dataclass
class AgentComponents:
    agent: Agent
    components: list[AbstractCapability]


def build_agent_components(
    model: DS4Model,
    project_dir: str | Path,
    *,
    read_only: bool = False,
    system: str | None = None,
    interactive: bool = False,
    question_answerer: QuestionAnswerer | None = None,
    session_dir: str | Path | None = None,
) -> AgentComponents:
    """Build the default capability list and the harness agent."""
    project_dir = Path(project_dir)

    notes = DS4SystemNotes(model.session)
    model.note_provider = notes

    components: list[AbstractCapability] = [
        LocalWorkspace(project_dir, read_only=read_only),
        FileSystem(root_dir=project_dir, cwd=project_dir, content_hashes=True, read_only=False),
        Shell(cwd=project_dir),
        DS4ShellJobs(model.session, project_dir),
        RepoContext(workspace_dir=project_dir),
        Skills(
            directories=[
                project_dir / ".claude" / "skills",
                project_dir / ".agents" / "skills",
                Path.home() / ".claude" / "skills",
            ],
            missing_directories="skip",
        ),
        ToolOutputLimits(),
        DS4RunRecord(model.session, Path(session_dir or project_dir) / "events.jsonl"),
        Instrumentation(),
        notes,
    ]
    if interactive:
        components.append(AskUser(answerer=question_answerer))

    options: AgentOptions = {"capabilities": components}
    if system is not None:
        options["instructions"] = system
    return AgentComponents(agent=Agent(model, **options), components=components)
