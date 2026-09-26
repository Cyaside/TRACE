# TRACE working rules

- Do not stage, commit, create a remote, or push unless the user explicitly requests that action. Completing implementation does not authorize a commit.
- `.local/` contains confidential working material. Keep draft guides, dated reports, test execution output, environment snapshots, scratch work and backups there.
- Public downloads and generated datasets in `data/raw/`, `data/canonical/` and `data/prepared/` stay local and ignored. Track download/preparation code and source references instead. Publishing dataset files requires a new explicit user instruction.
- The user has chosen trained model exports in `artifacts/models/` to be eligible for Git LFS. Eligibility is not permission to stage, commit or upload.
- Keep resume bundles in ignored `artifacts/runs/`; they include internal provenance and optimizer state. Publish model weights through the export command, which excludes that evidence.
- Do not copy local material into release files or force-add ignored paths without explicit approval for the named material. A request to commit source code does not authorize publishing local evidence.
- Keep synthetic test source in `tests/`; save dated test output under `.local/reports/YYYY-MM-DD/tests/`.
- Preserve historical manifests, checkpoints and ledger entries. Record path relocations separately instead of altering recorded hashes or provenance.
- Tell the user before substantial training or sustained compute. Structural work and checks must not start training.
