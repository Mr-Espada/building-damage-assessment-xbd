# Safe application and repository wording

Prepared 1 October 2026 for Mohammad Hachim Eraissouni. These statements distinguish the original report from independently reproduced evidence. Use the metrics only with the report attribution. They are validation values, not an independent test score or official xView2 benchmark score.

## Academic CV

Developed a TensorFlow/Keras U-Net with a ResNet101 encoder for building-damage segmentation using xBD satellite imagery; the undergraduate capstone report recorded validation IoU 0.62743 and F1 0.6746.

## Master's motivation letter

My undergraduate capstone investigated building-damage mapping from disaster satellite imagery using xBD and a U-Net–ResNet101 model. I subsequently recovered the implementation and prepared a research archive that distinguishes the original reported findings from remaining reproducibility gaps.

## Public GitHub README

Recovered checkpoint `27112023184806` is explicitly selected in a historical inference notebook with saved prediction outputs. It is the strongest candidate for the weights used locally after training and a plausible candidate for the final HPC-trained model. Its association with the report's 105-epoch run and validation IoU 0.62743 / F1 0.6746 remains unverified.

## Interview: are the original metrics reproducible?

The original report records those validation results. The recovered model loads successfully, and a saved notebook links the selected checkpoint path to local inference, but I no longer have the final training logs or exact prepared validation split. I have preserved the reported numbers without claiming independent reproduction. The code archive and compatibility checks can be inspected; exact historical score reproduction remains unverified.

If asked about HPC, distinguish recollection explicitly: “I recall training the final model on HPC and retaining the weights for local testing. The report describes four A40 GPUs, but I no longer have machine logs that authenticate the recovered checkpoint to that run.”

See [the forensic findings](HISTORICAL_HPC_PROVENANCE.md) for the supporting evidence and its limits.
