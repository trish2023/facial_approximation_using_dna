# DNA Facial Approximation QC Report

## Executive Summary

Traffic-light classification: **GREEN**
Overall reliability score: **0.7797**

## Ancestry Results

| Field | Value |
| --- | --- |
| Predicted ancestry label | IndoAryan-proxy |
| ANI proportion | 0.7600 |
| ASI proportion | 0.2400 |
| Ancestry confidence | 0.7600 |
| Ancestry certainty | 0.5200 |
| Model type | RandomForestClassifier |
| Reference populations | IndoAryan-proxy, Dravidian-proxy |

## Sex Prediction

| Field | Value |
| --- | --- |
| Predicted sex | male |
| Confidence | 0.9900 |
| X chromosome variants | NA |
| Y chromosome variants | NA |

## Template Selection

| Field | Value |
| --- | --- |
| Selected template | indoaryan_male |
| Template path | data/reference/face_templates/indoaryan_male.jpg |
| Ancestry label | IndoAryan-proxy |
| Sex | male |
| Template source | Gemini AI |

## Pigmentation Results

| Field | Value |
| --- | --- |
| Eye colour | brown |
| Hair colour | blond |
| Skin tone | dark_to_black |
| SNP coverage (%) | 78.0500 |
| Confidence score | 0.9457 |
| Imputed SNPs | 9 |
| Eye mask area | 94 |
| Hair mask area | 153717 |
| Skin mask area | 67045 |
| Mean colour shift | 31.8176 |
| Maximum colour shift | 57.9483 |

Predictions represent probabilistic genetic inference and may not fully capture environmental or population-specific variation.

## PRS Summary

| Trait | PRS | Z-score | Direction | Magnitude | Confidence |
| --- | --- | --- | --- | --- | --- |
| xiong2025_Chin_34_35 | -0.14606000000000002 | -1.3577954533421708 | NA | NA | NA |
| xiong2025_Chin_34_36 | -0.14843 | -1.1850282530794252 | NA | NA | NA |
| xiong2025_Chin_35_36 | -0.2144 | -1.7888440948669517 | NA | NA | NA |
| xiong2025_Lowercheek_37_38 | 0.0 | 0.5600029499679515 | NA | NA | NA |

## Morphology Results

| Trait | Z-score | Direction | Magnitude | Confidence | SNP count |
| --- | --- | --- | --- | --- | --- |
| xiong2025_Chin_35_36 | -1.7888 | Decreased | Moderate | 4.0000 | 5 |
| xiong2025_Chin_34_35 | -1.3578 | Decreased | Moderate | 2.7156 | 4 |
| xiong2025_Chin_34_36 | -1.1850 | Decreased | Moderate | 2.6498 | 5 |
| xiong2025_Lowercheek_37_38 | 0.5600 | Increased | Mild | 0.5600 | 1 |

| Field | Value |
| --- | --- |
| Landmarks modified | 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 |
| Maximum displacement | 15.3285 |
| Mean displacement | 8.7871 |

## Reconstruction Quality

| Field | Value |
| --- | --- |
| Landmark count | 76 |
| Triangle count | 140 |
| Maximum displacement | 15.3285 |
| Mean displacement | 1.5031 |
| P95 displacement | 11.8635 |
| Unrealistic warp | False |
| Boundary smoothing applied | True |

## Reliability Assessment

| Measure | Value |
| --- | --- |
| Pigmentation reliability | 0.8301 |
| Morphology reliability | 0.8283 |
| Ancestry reliability | 0.6640 |
| Overall reliability | 0.7797 |
| Traffic-light classification | GREEN |

## Limitations

- Missing upstream artifacts are reported as unavailable rather than inferred.
- Pigmentation predictions represent probabilistic genetic inference and may not capture all environmental or population-specific variation.
- Morphology reliability is limited to the currently available lower-face GWAS traits.
- Reconstruction quality reflects the current template and warp stages only; it is not a full forensic validation.
