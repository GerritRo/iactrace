# Contributing to IACTrace

## Development Setup

```bash
# Fork on GitHub/GitLab, then:
git clone https://github.com/<your-username>/iactrace.git
cd iactrace
git remote add upstream https://github.com/GerritRo/iactrace.git

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Verify everything works
pytest
```

For GPU development add `pip install -e ".[dev,gpu]"`.

---

## Workflow (Gitflow)

We use a **Gitflow** model: `main` holds tagged releases, `dev` is the
integration branch. All work happens on short-lived branches off `dev`.

| Branch prefix      | Purpose                 | Target  |
|--------------------|-------------------------|---------|
| `feature/<name>`   | New features            | `dev`   |
| `bugfix/<name>`    | Bug fixes               | `dev`   |
| `hotfix/<x.y.z>`   | Urgent fixes in prod    | `main`  |
| `release/<x.y.z>`  | Release prep            | `main`  |

**Typical contribution flow:**

```bash
git fetch upstream && git checkout -b feature/my-change upstream/dev

# ... develop, commit, test ...

git push origin feature/my-change
# Open a PR targeting dev
```

Keep branches short-lived. Rebase on `upstream/dev` before opening a PR.

---

## Committing with Commitizen

We use [Commitizen](https://commitizen-tools.github.io/commitizen/) to enforce
[Conventional Commits](https://www.conventionalcommits.org/) and generate
changelogs automatically. Use `cz commit` instead of `git commit`:

```bash
cz commit
```

This walks you through an interactive prompt:

```
? Select the type of change you are committing: (Use arrow keys)
 » fix: A bug fix
   feat: A new feature
   docs: Documentation only changes
   refactor: A code change that neither fixes a bug nor adds a feature
   perf: A code change that improves performance
   test: Adding missing or correcting existing tests
   build: Changes that affect the build system or dependencies
   ci: Changes to CI configuration files and scripts
   chore: Other changes that don't modify src or test files

? What is the scope of this change? (press enter to skip)
  core, sensors, telescope, io, viz, utils

? Write a short, imperative description of the change:
  > handle NaN in aspheric surface intersection

? Provide additional contextual information (press enter to skip):
  > The Newton-Raphson solver produced NaN for near-parallel rays

? Is this a BREAKING CHANGE?  No
? Footer (press enter to skip, e.g. "Closes #42"):
  > Fixes #87
```

Result: `fix(core): handle NaN in aspheric surface intersection`

You can also write commit messages manually. The format is:

```
<type>(<scope>): <description>
```

Breaking changes use `!` after the type: `feat(io)!: require explicit integrator`

**Why this matters:** `CHANGELOG.md` is generated directly from these commit
messages at release time via `cz bump --changelog`.

---

## Pull Requests

- **Title** follows Conventional Commits format (it becomes the merge commit message)
- **Target branch** is `dev` (unless it's a hotfix targeting `main`)
- **PR checklist:**
  - [ ] Tests pass (`pytest`)
  - [ ] Lint passes (`ruff check iactrace`)
  - [ ] Types pass (`mypy iactrace`)
  - [ ] New code has tests

---

## Code Quality

### Quick reference

```bash
ruff check iactrace                          # Lint
ruff check --fix iactrace                    # Lint + auto-fix
ruff format iactrace                         # Format
mypy iactrace                                # Type check
pytest                                       # Tests (excludes notebooks)
pytest -m notebooks -v -o addopts=""         # Notebook tests
pytest --cov=iactrace --cov-report=html      # Coverage report
```

### Benchmarks (ASV)

If your change touches performance-critical code:

```bash
asv machine --machine local --yes
asv continuous origin/dev HEAD --show-stderr --factor 1.1
```

Benchmark classes live in `benchmarks/benchmarks.py`.

---

## Documentation

```bash
pip install -e ".[docs]"
cd docs && make html
# Open docs/_build/html/index.html
```

- **Example notebooks** go in `docs/examples/` and must be added to `docs/examples/index.rst`

---

## Release Process

Maintainers only. Uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).

1. Create `release/x.y.z` from `dev`
2. `cz bump --changelog` to bump version + generate changelog
3. Run full test suite + benchmarks + doc build
4. Merge into `main`, tag `vx.y.z`, backmerge into `dev`

---

## Getting Help

- [GitHub Issues](https://github.com/GerritRo/iactrace/issues) for bugs and feature requests
- [Documentation](https://gerritro.github.io/iactrace/)
- Email: gerrit.roellinghoff@fau.de

If unsure about a change, open an issue first to discuss the approach.