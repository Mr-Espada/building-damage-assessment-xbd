# Representative results and provenance

`reported_metrics.json` transcribes values from the authoritative final report, printed page 24 / PDF page 33. `reported_metrics.png` is a new chart of those unchanged numbers, labeled as reported and not independently reproduced. It is not a newly trained result or generated prediction example.

The report describes 105 completed epochs out of 150 planned. The recovered code/output artifacts do not establish which candidate checkpoint produced those values. No final 105-epoch history or exact final arrays were recovered. Do not replace these values with alternative-run logs or new evaluations.

The report includes qualitative Figures 13–17, including failure examples and pre-disaster/training-set illustrations. Those figures are preserved in the original PDF; original PNGs remain local. They are excluded from this candidate until scene/checkpoint association and image attribution are reviewed. This avoids presenting untraceable images as freshly reproduced evidence.

For exact source identity, the authoritative report SHA-256 is `7f3852f5fa2f5897f1f4cc2de7c08a7064a20e06e1fc540d0753234c0466fdf3`. The 24-page PDF with the same name in the original capstone folder is an earlier write-up.

Candidate weights were restored without original imagery. Candidate06 is a **plausible final-checkpoint candidate**, with stronger evidence of selection for local inference from a hidden notebook's saved outputs. Both candidates' optimizer counts fit report-sized batches conditionally; best-only selection could explain an earlier saved state in a 105-epoch run, but no final history establishes that explanation. Saved LR approximately 1e-5 does not establish initial LR.

Report Figure 13 has same-site correspondence to a different saved pre-disaster example; Figure 17's source/actual mask corresponds to a graph without a prediction panel. No recovered complete prediction output is authenticated to a report figure. These partial links are not score validation. See [forensic evidence and limits](../docs/HISTORICAL_HPC_PROVENANCE.md) and [checkpoint metadata/checksums](checkpoint_candidates.json).
