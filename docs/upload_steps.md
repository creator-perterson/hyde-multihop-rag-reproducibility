# Maintaining the public JIIS package

This repository already exists. Preserve its history and use a normal commit/push after inspecting the exact diff; do not reinitialize it or force-push to replace history.

The current publication allowlist is recorded in `release/github_publication_manifest.json`. Keep the current manuscript under `paper/jiis/`, numerical packages under `reproducibility/` and versioned downloads under `release/assets/`. Add only reviewed paths. Do not use a whole-checkout add operation.

Before publishing changed numeric inputs, rerun their capsule commands and rejection tests in a clean directory. If a frozen file changes, establish a new version and update the associated manifest instead of silently rewriting the old archive. Rebuild manuscript sources when their text or figures change.

Do not add temporary/cache/virtual-environment files, corpora/questions/gold text, raw predictions/generations, private endpoints or credentials, model weights/indexes, submission cover letters, internal reviewer/author records or entire historical submission bundles. `.gitignore` does not remove files already tracked: inspect both the staged diff and existing tracked paths.

Only record a public commit URL, tag or DOI after the corresponding operation succeeds. No archival DOI currently exists. Current model-running code and local configuration candidates require separate scope/portability checks; successful capsule execution proves only its declared numerical scope.
