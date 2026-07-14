# Doc prompts

## Readthedocs

## Please add the necessary files to expose the docs/ in llc4320-native-grid-preprocessing Repository on readthedocs.

## Mermaid diagram

1. Please generate a Mermaid diagram that shows the flow of data through the pipeline.  You may need to make separate ones for the the cutouts, global, and tiles.  Put these in the docs/ directory in a file named `mermaid_diagrams.md`.  If you have any questions, put them under the Q&A section.  Log your work.

## Q&A

- **Mermaid rendering in the docs build.** The diagrams live in
  `docs/mermaid_diagrams.md` as fenced ```mermaid blocks (GitHub renders
  these natively). For the Read the Docs / Sphinx build to render them, the
  build needs `sphinxcontrib-mermaid` (add to `docs/requirements.txt` and to
  `extensions` in `docs/conf.py`) or, if using MyST-Markdown, the `mermaid`
  fence directive enabled. Want me to wire that in as part of the Read the
  Docs setup item above? (Left out for now since that's a separate prompt.)
- **Scope.** I produced three diagrams as suggested (global, cutouts,
  tiles) and added `mermaid_diagrams.md` to the `docs/` toctree
  (`docs/index.md`). The global diagram also folds in the one-time ingest
  (`transfer_llc4320`) and grid-generation setup, since those produce the
  raw inputs. Say the word if you'd prefer those split into a separate
  "ingest" diagram.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs