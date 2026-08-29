# V-Link Paper README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task.

**Goal:** Replace the inherited root README with an original V-Link paper and
RoboTwin2 reproduction page.

**Architecture:** Keep the root README concise and paper-facing. Delegate the
full dataset schema and operational override reference to the existing
`examples/RoboTwin2/README.md` guide.

**Tech Stack:** Markdown, Bash command examples, pytest contract tests.

---

### Task 1: Replace and verify the root README

**Files:**

- Modify: `README.md`
- Reference: `examples/RoboTwin2/README.md`
- Test: `tests/gr00t/model/test_robotwin2_c2_recipe.py`

- [ ] Replace all inherited branding and generic upstream documentation with
  the V-Link paper title, authors, method, reported results, release scope,
  reproduction commands, license, and citation.
- [ ] Link the packaged RoboTwin training dataset in GitHub Releases.
- [ ] Place paper results after Evaluation and before License.
- [ ] Verify the method statements and results against
  `/mnt/data/yehao/V-Link-arXiv.pdf`.
- [ ] Verify setup and workflow commands against the retained RoboTwin2 shell
  scripts and detailed guide.
- [ ] Run
  `/opt/conda/envs/gr00t_n1.6/bin/python -m pytest -q tests/gr00t/model/test_robotwin2_c2_recipe.py`
  and expect all tests to pass.
- [ ] Run `bash -n examples/RoboTwin2/*.sh` and expect all scripts to pass.
- [ ] Commit the documentation and push `main` to the configured GitHub remote.
