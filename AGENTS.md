# Instructions for coding agents

Run commands from the repository root. Keep the existing devenv + uv workflow canonical on macOS and Linux. Do not install a second Python environment or a second hook runner.

## Set up the environment

Enter devenv before every project command:

```bash
devenv shell
```

Then synchronize the locked dependencies and install the Git hooks once:

```bash
uv sync --frozen --extra agent
uv run prek install
```

`devenv.lock` pins the Nix inputs. `uv.lock` pins Python tools and packages. Do not let uv download another interpreter.

## Run checks

Run the complete static gate after relevant changes:

```bash
uv run --frozen --extra agent prek run --all-files --show-diff-on-failure
```

This includes Ruff lint and format, ty, TOML, YAML, JSON, GitHub workflow schema checks, actionlint, and the `TypeGuard` ban. Do not run only changed files when a change can affect types or formatting elsewhere.

Run the portable regression suite:

```bash
uv run --frozen --extra agent pytest -m "not build and not live" -q
```

For packaging or CI changes, also run:

```bash
uv build --wheel
unzip -l dist/*.whl | grep -q 'ds4py/abi/Makefile.lib'
actionlint .github/workflows/ci.yml
```

Portable tests do not prove native inference or live replay parity. See [docs/development.md](docs/development.md) for native builds, backend selection, and live checks.

## Typing and lint policy

- Use `TypeIs`, never `TypeGuard`, for custom type-narrowing validators.
- Prefer precise domain types, `TypedDict`, `Protocol`, and generics. Use `object` for unknown values that will be narrowed. Keep `Any` only at reviewed dynamic boundaries such as ctypes.
- Use lowercase `typing_extensions.sentinel` and the sentinel object itself in unions. Do not use `Literal[sentinel]`.
- Use standard-library `TypeIs`, `override`, `Required`, `NotRequired`, and `ReadOnly`. Use `typing_extensions.TypedDict` for `closed` and `extra_items`.
- Use PEP 695 `type` aliases and generic syntax. Use `Never` only for non-returning or unreachable paths.
- Keep native `Any` exceptions as small as possible. Validate dynamic values at the boundary instead of casting them throughout ordinary modules.
- Use safe fixes only. Never use `--unsafe-fixes`, blanket suppression comments, or broad per-file ignores to make a check pass.
- Every suppression must be narrow, specific, and justified by a real checker limitation or project contract. Do not change public behavior only to satisfy a linter.

## Tests

Write a focused failing test before changing runtime behavior. Run it, watch it fail, implement the smallest change, and run it again. Keep mechanical formatting and behavioral changes in separate commits.

Run focused native, engine, agent, ABI, or benchmark tests for the subsystem you change, then run the complete static gate and portable suite. Existing tests must not be weakened to make a refactor pass.

Some developer machines load personal `~/.claude/skills` catalogs into the agent tests. If tests fail with unexpected skill instructions, run them with an empty isolated home:

```bash
mkdir -p /tmp/ds4-quality-empty-home
HOME=/tmp/ds4-quality-empty-home uv run --frozen --extra agent pytest -m "not build and not live" -q
```

Do not commit generated build output, native libraries, traces, virtual environments, or local `.devenv` state.
