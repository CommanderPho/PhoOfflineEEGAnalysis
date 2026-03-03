# show-exploded-call-hierarchy

Takes a selection of Python code from a Python file (e.g. a single line or expression). Follows the call hierarchy of that selection recursively: resolve the selected call, then each function/method it calls, then their callees, and so on. Presents the result so the user can see the consequence of the code.

## Output requirements

- **Format:** Always generate the expansion as a **Jupyter notebook** (`.ipynb`) with markdown cells for narrative and code cells for the selected line and for representative snippets at each level (Python syntax highlighting). Use workspace-relative clickable file links with `#L<line>` where applicable.
- **Location:** Write the expansion to a **separate file** (e.g. `exploded-call-hierarchy.ipynb` or a similar name in the same directory). **Do not** write expansion output into this command specification file; this file describes the command only.

When generating the notebook, verify line numbers in jump links against the current codebase so links resolve correctly.
