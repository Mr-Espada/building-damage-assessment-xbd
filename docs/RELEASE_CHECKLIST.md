# Before public release

Completed preparation: small candidate structure, original source preservation, notebook sanitation, data/weight exclusions, documented results and limitations, explicit-path commands, dependency recipe, changelog and references.

Required publication decisions/checks:

- [x] Author confirmed code ownership and delegated license selection; MIT LICENSE, author attribution and package metadata added.
- [x] Exclude full report PDFs, data, weights and qualitative imagery from this candidate. Review separate permissions if adding those materials later.
- [x] Keep final README metrics labeled “reported” and “not independently reproduced”; retain plausible-checkpoint classification.
- [x] Run release validation and inspect the local Git staged-file list for this candidate; exact file hashes and exclusion checks are recorded privately.
- [x] Initialize local Git on `main` **inside this release directory** and stage only the reviewed public files.
- [x] Create the initial local Git commit using the configured Git identity after verifying the reviewed public file set.
- [x] Publish the reviewed repository under [Mr-Espada/building-damage-assessment-xbd](https://github.com/Mr-Espada/building-damage-assessment-xbd) and verify anonymous public access and the remote file set.
- [x] Issue the requested collaborator invitation with write access; repository-management permissions do not change academic authorship.
- [ ] Collaborator accepts the GitHub invitation. This is an external action; invitation delivery is verified, while active access must be checked after acceptance.

For later changes, re-run the release checker and review the actual Git file list before pushing.

Optional evidence recovery and later work:

- Search for final history, split/array manifests, environment records or a historical checkpoint digest if available. Keep missing-evidence disclosures if they cannot be recovered; this does not require retraining before an honest archival release.
- Decide whether alternatives merit a separately attributed public archive; they remain local.
- Any explicitly authorized new experiment must keep its data/run record and results separate.

Second-pass provenance classification: **Plausible final-checkpoint candidate**. See [historical HPC provenance](HISTORICAL_HPC_PROVENANCE.md). Exact historical reproduction is currently incomplete. An honest archival release can state that fact, but must not advertise benchmark reproducibility that has not been established.
