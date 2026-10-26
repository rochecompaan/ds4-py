# Development environment

See [Testing ds4-py](testing.md) for backend selection, native checks, and live replay instructions.

Install [Nix](https://nixos.org/download/) and [devenv](https://devenv.sh/getting-started/).
From this checkout, enter the environment:

```sh
devenv shell
```

The environment provides Python 3.11, uv, a C compiler, GNU Make, Git, curl,
GNU coreutils, unzip, and actionlint. `devenv.lock` pins the Nix inputs.
uv uses the Nix Python and does not download another interpreter.
It stores the project's virtual environment in `.devenv/state/venv`.
Shell startup does not create a Python project or install its dependencies.

For automatic activation, install direnv and enable its hook in your shell.
Then run `direnv allow` in this checkout. The included `.envrc` loads devenv.

## M1 commands

The M1 native-layer work starts with the Python project scaffold.
Run these commands inside `devenv shell` when you start that task:

```sh
uv init --package --name ds4py --python ">=3.11"
uv add --dev pytest
```

pytest stays in the project's development dependencies, not the Nix environment.
After the scaffold and tests exist, run:

```sh
uv sync --dev
uv run pytest -m "not build and not live" -v
uv build --wheel
unzip -l dist/*.whl
```

After the CI workflow exists, run `actionlint .github/workflows/ci.yml`.

## Native and live tests

The environment provides the CPU build tools. It does not download ds4 source
trees, model files, or GPU toolkits. The M1 plan pins these source revisions:

| Source | Repository | Commit |
| --- | --- | --- |
| Upstream | https://github.com/antirez/ds4.git | `5379c1c9ac315f1540371b06a3db88491d4059ff` |
| Fork | https://github.com/pauleveritt/ds4.git | `cfb2faf5c03199790db7c22166c4c71b549f1e66` |

Clone each tree and check out its pinned commit. Set the paths to those checkouts
before you run the build tests:

```sh
export DS4_SRC_UPSTREAM=/path/to/upstream
export DS4_SRC_FORK=/path/to/fork
uv run pytest -m build -v
```

Live tests also need a GGUF model file and a built `ds4-agent` executable:

```sh
export DS4_TEST_MODEL=/path/to/model.gguf
export DS4_AGENT=/path/to/ds4-agent
uv run pytest -m live -v
```

Build and live tests skip when their required variables are unset.
CUDA requires a separate CUDA toolkit. Metal requires macOS and its SDK.
ROCm is not supported by the planned `ds4py build` command.
