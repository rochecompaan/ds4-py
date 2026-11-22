"""Default ds4-py agent composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Instrumentation, LocalWorkspace
from pydantic_ai_harness import (
    AskUser,
    RepoContext,
    Shell,
    Skills,
    ToolOutputLimits,
)
from pydantic_ai_harness.ask_user import Answerer as QuestionAnswerer

from ds4py.agent_prompt import DEFAULT_CODING_AGENT_INSTRUCTIONS
from ds4py.agent_prompt_pi import PI_CODING_AGENT_INSTRUCTIONS
from ds4py.capabilities.compaction import LLMCompaction
from ds4py.capabilities.file_system import DS4FileSystem
from ds4py.capabilities.harness_events import HarnessEventBridge
from ds4py.capabilities.pi_profile import PiProfile, resolve_tool_profile
from ds4py.capabilities.run_record import DS4RunRecord
from ds4py.capabilities.shell_jobs import DS4ShellJobs
from ds4py.capabilities.system_notes import DS4SystemNotes
from ds4py.capabilities.tool_errors import ToolErrorRecovery

if TYPE_CHECKING:
    from ds4py.agent_composition import ToolProfile
    from ds4py.capabilities.context import ContextComposition
    from ds4py.model import DS4Model


class AgentOptions(TypedDict, total=False):
    capabilities: list[AbstractCapability]
    instructions: str


@dataclass
class AgentComponents:
    agent: Agent
    components: list[AbstractCapability]
    compactor: LLMCompaction


def build_agent_components(  # noqa: PLR0913 - Agent composition mirrors the runtime feature set.
    model: DS4Model,
    project_dir: str | Path,
    *,
    read_only: bool = False,
    system: str | None = None,
    interactive: bool = False,
    question_answerer: QuestionAnswerer | None = None,
    session_dir: str | Path | None = None,
    context: ContextComposition | None = None,
    tool_profile: ToolProfile = "auto",
) -> AgentComponents:
    """Build the default capability list and the harness agent."""
    project_dir = Path(project_dir)
    compactor = LLMCompaction(model.binder)
    owner = context.owner if context is not None and context.owner is not None else compactor
    model.context_owner = owner

    notes = DS4SystemNotes(model.session)
    model.note_provider = notes
    family = getattr(getattr(getattr(model.session, "engine", None), "info", None), "family", "")
    tool_profile = resolve_tool_profile(tool_profile, family=family)

    components: list[AbstractCapability] = [LocalWorkspace(project_dir, read_only=read_only)]
    if tool_profile == "pi":
        components.extend([PiProfile(project_dir), RepoContext(workspace_dir=project_dir)])
        default_instructions = PI_CODING_AGENT_INSTRUCTIONS
    else:
        components.extend(
            [
                DS4FileSystem(
                    tools=(
                        "read_file",
                        "write_file",
                        "edit_file",
                        "search_files",
                        "find_files",
                        "create_directory",
                        "file_info",
                    ),
                    root_dir=project_dir,
                    cwd=project_dir,
                    content_hashes=True,
                    read_only=False,
                    max_read_lines=160,
                    max_read_chars=50_000,
                ),
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
            ]
        )
        default_instructions = DEFAULT_CODING_AGENT_INSTRUCTIONS
    components.extend(
        [
            ToolErrorRecovery(),
            HarnessEventBridge(model.session),
            DS4RunRecord(model.session, Path(session_dir or project_dir) / "events.jsonl"),
            Instrumentation(),
            notes,
            owner,
        ]
    )
    if interactive and tool_profile != "pi":
        components.append(AskUser(answerer=question_answerer))

    options: AgentOptions = {
        "capabilities": components,
        "instructions": system if system is not None else default_instructions,
    }
    return AgentComponents(agent=Agent(model, **options), components=components, compactor=compactor)
