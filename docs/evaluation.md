# Model evaluation: 5-fold cross-validation

Each listing is predicted by a model fitted on the other 4 folds, so no
listing is ever predicted by a model that saw it. All variants use the same
folds (seed 0). Listings are 2010-2020 model years.

Each cell is **median % error / coverage**:

- **Median % error**: the middle value of |predicted - actual| / actual, as a
  percent. Lower is better. It is a median, so a few absurd listings don't
  dominate it.
- **Coverage**: the share of actual prices that fall inside the model's 95%
  prediction interval. It should be about 95%. Lower means the intervals are
  too narrow (overconfident); higher means they are wider than they need to be.
- **Average**: the plain mean of the six cars' values.

Outliers were removed from the **training folds only**. Test listings are never
dropped or changed. The two sets differ only in which test listings are scored:

- **all**: every listing, including bad ones such as $500 down-payment ads.
- **plausible**: only the listings kept when `drop_outliers` is run on the whole
  car's listings (3 passes, no features). That filter chooses which rows are
  scored; it is not used for training. It did see every listing, so the
  plausible scores are somewhat flattering to every variant.

"Week 3 style" fits one mileage-only model per model year, so a listing whose
model year has fewer than 30 training listings gets no prediction and is left
out of that variant's scores (counts are listed below the tables).

## All listings

| Variant | Camry | Accord | Civic | Tacoma | F-150 | Silverado 1500 | Average |
|---|---|---|---|---|---|---|---|
| Week 3 style (mileage only, own year) | 9.2% / 92.8% | 11.9% / 93.2% | 11.8% / 94.0% | 11.8% / 94.2% | 13.9% / 94.0% | 11.3% / 95.1% | 11.6% / 93.9% |
| Mileage + age, no outlier removal | 10.4% / 94.1% | 12.2% / 94.4% | 12.4% / 94.9% | 11.7% / 95.1% | 13.7% / 95.0% | 12.1% / 95.4% | 12.1% / 94.8% |
| Mileage + age, 1 pass | 10.1% / 88.8% | 11.6% / 91.1% | 11.9% / 90.5% | 11.7% / 92.2% | 13.3% / 90.4% | 11.8% / 92.6% | 11.7% / 90.9% |
| Mileage + age, 3 passes | 9.9% / 87.9% | 11.7% / 90.8% | 12.1% / 89.2% | 11.6% / 92.0% | 13.4% / 90.2% | 11.9% / 92.5% | 11.7% / 90.4% |
| + clean title | 9.3% / 86.5% | 11.3% / 91.2% | 11.2% / 90.8% | 11.3% / 91.8% | 13.3% / 90.4% | 11.6% / 92.6% | 11.3% / 90.6% |
| + clean title + 4WD | 9.3% / 86.5% | 11.3% / 91.2% | 11.2% / 90.8% | 10.0% / 92.0% | 12.7% / 90.7% | 11.6% / 92.6% | 11.0% / 90.6% |
| + clean title + 4WD, 1 pass | 9.5% / 89.1% | 11.2% / 91.5% | 11.3% / 91.6% | 10.0% / 91.9% | 12.7% / 91.3% | 11.8% / 93.2% | 11.1% / 91.4% |

## Plausible listings

| Variant | Camry | Accord | Civic | Tacoma | F-150 | Silverado 1500 | Average |
|---|---|---|---|---|---|---|---|
| Week 3 style (mileage only, own year) | 8.4% / 97.7% | 11.1% / 97.0% | 11.4% / 97.1% | 11.5% / 96.0% | 13.2% / 97.7% | 11.0% / 97.0% | 11.1% / 97.1% |
| Mileage + age, no outlier removal | 9.7% / 98.9% | 11.5% / 98.4% | 11.7% / 98.1% | 11.4% / 97.0% | 13.1% / 98.7% | 11.7% / 97.9% | 11.5% / 98.2% |
| Mileage + age, 1 pass | 9.4% / 94.8% | 10.9% / 95.2% | 11.1% / 94.4% | 11.3% / 94.0% | 12.8% / 94.1% | 11.5% / 95.0% | 11.2% / 94.6% |
| Mileage + age, 3 passes | 9.1% / 93.9% | 10.9% / 95.0% | 11.4% / 93.0% | 11.3% / 93.8% | 12.7% / 93.9% | 11.5% / 95.0% | 11.1% / 94.1% |
| + clean title | 8.7% / 92.2% | 10.8% / 95.1% | 10.8% / 94.0% | 11.1% / 93.6% | 12.7% / 94.0% | 11.4% / 95.1% | 10.9% / 94.0% |
| + clean title + 4WD | 8.7% / 92.2% | 10.8% / 95.1% | 10.8% / 94.0% | 9.8% / 93.7% | 12.1% / 94.3% | 11.4% / 95.0% | 10.6% / 94.1% |
| + clean title + 4WD, 1 pass | 8.8% / 95.0% | 10.5% / 95.4% | 10.9% / 94.8% | 9.8% / 93.5% | 12.1% / 94.9% | 11.6% / 95.6% | 10.6% / 94.9% |

## Listings scored

| Car | All listings | Plausible listings |
|---|---|---|
| Camry | 1,129 | 1,057 |
| Accord | 1,078 | 1,031 |
| Civic | 1,205 | 1,156 |
| Tacoma | 1,193 | 1,170 |
| F-150 | 4,165 | 4,001 |
| Silverado 1500 | 2,524 | 2,458 |

## Listings that could not be predicted

- Week 3 style (mileage only, own year): Camry 30 of 1,129, Accord 12 of 1,078, Tacoma 13 of 1,193
