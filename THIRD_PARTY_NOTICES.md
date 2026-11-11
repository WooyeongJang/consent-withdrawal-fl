# Third-party notices and data attribution

## Licenses in this repository

- Code and smart contracts: MIT License (`LICENSE`).
- Data-derived tables, configurations, consent scenarios, results, and the Supplementary Document: Creative Commons Attribution 4.0 International (`LICENSE-data`). This covers `site_check/`, `c2021/`, `results/`, and the supplementary PDF.
- The third-party files below keep their own licenses.

## Data attribution (CC BY 4.0)

The files in `site_check/`, `c2021/excluded_*.csv`, `results/consent/`, and the demonstration results contain patient, record, fold, and site identifiers derived from the following datasets. No signals or diagnostic statements are redistributed. Changes: records were grouped into institutions, filtered, and assigned to withdrawal scenarios as described in the paper and in each folder's README.

- PTB-XL v1.0.3. P. Wagner, N. Strodthoff, R. Bousseljot, W. Samek, and T. Schaeffter, PhysioNet, 2022, https://doi.org/10.13026/kfzx-aw45. Original publication: P. Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset," *Scientific Data*, 2020, https://doi.org/10.1038/s41597-020-0495-6. License: CC BY 4.0.
- PhysioNet/Computing in Cardiology Challenge 2021 v1.0.3. M. Reyna et al., "Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021," PhysioNet, 2022, https://doi.org/10.13026/34va-7q14. Original publication: Computing in Cardiology 2021, https://doi.org/10.23919/CinC53138.2021.9662687. License: CC BY 4.0.
- PhysioNet: T. Pollard et al., "PhysioNet as a global platform for biomedical research," *Nature Health*, vol. 1, no. 8, pp. 792–795, 2026, https://doi.org/10.1038/s44360-026-00096-z.

## Challenge 2021 code tables (not included)

`c2021/dx_mapping_scored.csv` and `c2021/dx_mapping_unscored.csv` (SNOMED CT code tables of https://github.com/physionetchallenges/evaluation-2021, BSD-2-Clause) were used as reference when building `c2021/mapping_superclass_v1.csv`. They are not redistributed here. `scripts/fetch_c2021_maps.sh` downloads them from commit e3a9eb494181cf074648cf964a0be3841da27fc3 and verifies their SHA-256. They are not needed to build the training caches.

## `tests/test_fedosd_official.py` (functions `official_*`)

The computation parts of `tfedplat/algorithm/unlearning/FedOSD.py` and `UnlearnAlgorithm.py` are copied from https://github.com/zibinpan/FedOSD, commit 41cc10635c6f2396795d0d9c6239ea181b7c9ee6.

```
MIT License

Copyright (c) 2025 zibinpan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
