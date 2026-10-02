# Limitations

- Report metrics have written provenance but incomplete executable provenance. Exact arrays, preparation driver, historical environment and the checkpoint's 105-epoch association are not recovered.
- Validation combines xBD hold and test. There is no separately preserved independent test score, and no recovered analysis of event/geographic overlap between splits.
- Class imbalance is substantial. The report discusses confusion between undamaged/minor and other neighboring damage levels and imprecise building boundaries. Aggregate five-class scores include background and cannot establish minority-class performance.
- The final model uses post-disaster imagery. Pre-disaster inference examples do not establish a trained change-detection method. Barlow Twins and paired-image segmentation are future proposals in the report.
- Labels are rasterized to pixels. The implementation does not establish instance-level building damage scoring or operational emergency-response accuracy. Legacy unclassified/missing labels map to undamaged, introducing possible label noise.
- No recovered confidence intervals, per-class confusion matrices, cross-event generalization study or deployment validation supports operational reliability claims.
- Earlier resize pipelines, arbitrary directory-order subsets and categorical-mask interpolation can alter experimental results. They are archived concerns, not silently corrected historical results.
- Later SAM fine-tuning attempts detach predictions from the model's gradient graph and misuse multimask proposals as damage classes. Checkpoint filenames alone do not prove successful learning.
- Training the report-described scale requires considerable hardware. Packaging and synthetic checks do not establish full-scale CPU/GPU performance or reproduce the original compute conditions.

This project demonstrates undergraduate work in data analysis, geospatial label processing and semantic segmentation experimentation. Its public description should reflect those contributions and the evidence that is actually available.
