"""Default ds4-py agent composition."""

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation, LocalWorkspace
from pydantic_ai_harness import (
    AskUser,
    FileSystem,
    RepoContext,
    Shell,
    Skills,
    ToolOutputLimits,
)

from ds4py.capabilities.approvals import DS4Approvals
from ds4py.capabilities.run_record import DS4RunRecord
from ds4py.capabilities.shell_jobs import DS4ShellJobs
from ds4py.capabilities.system_notes import DS4SystemNotes


@dataclass
class AgentComponents:
    agent: Agent
    components: list


def build_agent_components(
    model,
    project_dir,
    *,
    read_only=False,
    yolo=False,
    system=None,
    interactive=False,
    answerer=None,
    session_dir=None,
):
    """Build the default capability list and the harness agent."""
    project_dir = Path(project_dir)

    notes = DS4SystemNotes(model.session)
    model.note_provider = notes

    components = [
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
        DS4RunRecord(model.session, (session_dir or project_dir) / "events.jsonl"),
        Instrumentation(),
        notes,
        DS4Approvals(answerer, yolo=yolo, project_dir=project_dir),
    ]
    if interactive:
        components.append(AskUser(answerer=answerer))

    options = {"capabilities": components}
    if system is not None:
        options["instructions"] = system
    return AgentComponents(agent=Agent(model, **options), components=components)
