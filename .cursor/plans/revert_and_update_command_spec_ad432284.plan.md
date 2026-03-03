---
name: Revert and update command spec
overview: Revert the command specification file to its intended purpose (instructions for the command) and update it so expansion output is always generated as a notebook in a separate file.
todos: []
isProject: false
---

# Revert command spec and require notebook output in separate file

## 1. Revert [show-exploded-call-hierarchy.md](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis.cursor\commands\show-exploded-call-hierarchy.md)

The file currently contains **expansion output** from a single invocation (the `LabRecorderXDF.init_from_lab_recorder_xdf_file` hierarchy). It should be restored to a **command specification** that describes how the command behaves and how the agent should implement it.

**Target content (command spec):**

- **Purpose:** Describe the "show-exploded-call-hierarchy" Cursor command.
- **Behavior (from your original description):**
  - Takes a selection of Python code from a Python file (e.g. a single line or expression).
  - Follows the call hierarchy of that selection recursively (resolve the selected call, then each function/method it calls, then their callees, etc.).
  - Presents the result so the user can see the consequence of the code.
- **No example output:** The spec file will not contain the expansion output for any particular selection; that belongs in the separate output file.

## 2. Update the spec to require output format and separate file

Add to the command specification:

- **Output format:** The expansion must always be generated in **Jupyter notebook format** (`.ipynb`), with:
  - Markdown cells for narrative and section headers.
  - Code cells for the selected line and for representative code snippets at each level (with Python syntax highlighting).
  - Clickable file links using workspace-relative paths and `#L<line>` fragments where supported (e.g. `[label](../../../RepoName/path/to/file.py#L123)` from a notebook under `.cursor/commands/`).
- **Output location:** The expansion output must be written to a **separate file**, not into the command spec file. For example:
  - A new notebook in the same directory, e.g. `exploded-call-hierarchy.ipynb`, or a timestamped/selection-derived name so multiple runs don’t overwrite each other (e.g. `exploded-call-hierarchy-<brief-hash-or-name>.ipynb`).
  - The spec file ([show-exploded-call-hierarchy.md](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis.cursor\commands\show-exploded-call-hierarchy.md)) must never be overwritten with expansion output; it remains the stable command specification only.

## 3. Suggested spec wording (for the .md file)

Replace the current file body with something like:

```markdown
# show-exploded-call-hierarchy

Takes a selection of Python code from a Python file (e.g. a single line or expression). Follows the call hierarchy of that selection recursively: resolve the selected call, then each function/method it calls, then their callees, and so on. Presents the result so the user can see the consequence of the code.

## Output requirements

- **Format:** Always generate the expansion as a **Jupyter notebook** (`.ipynb`) with markdown cells for narrative and code cells for the selected line and for representative snippets at each level (Python syntax highlighting). Use workspace-relative clickable file links with `#L<line>` where applicable.
- **Location:** Write the expansion to a **separate file** (e.g. `exploded-call-hierarchy.ipynb` or a similar name in the same directory). **Do not** write expansion output into this command specification file; this file describes the command only.
```

Optional: add a one-line note that line numbers in jump links should be verified against the current codebase when generating the notebook.

## Summary


| Action          | File                                                                                                                                                               | Change                                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Revert + update | [.cursor/commands/show-exploded-call-hierarchy.md](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis.cursor\commands\show-exploded-call-hierarchy.md) | Replace current (expansion output) with the short command spec above, including output-format and separate-file requirements. |
| No change       | [.cursor/commands/exploded-call-hierarchy.ipynb](c:\Users\pho\repos\EmotivEpoc\ACTIVE_DEV\PhoOfflineEEGAnalysis.cursor\commands\exploded-call-hierarchy.ipynb)     | Keep as the example output for the previous invocation; future runs will create/update a separate notebook per the spec.      |


No edits to the notebook or other files are required for this plan; only the command spec `.md` is modified.