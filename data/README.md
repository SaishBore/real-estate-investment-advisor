# Data folder

The data files are not stored in this repository because they are large (the raw dataset is about 40 MB, and the generated files are up to 83 MB each).

## How to get them

1. Download `india_housing_prices.csv` from the dataset link given in the project brief.
2. Save it in this folder as `data/india_housing_prices.csv`.
3. From the project root, run:

```bash
python src/step1.py
```

This creates the other files in this folder:

| File | What it is |
|---|---|
| `cleaned_housing.csv` | Readable cleaned data (used by the EDA notebook and the app's Market insights tab) |
| `processed_housing.csv` | Encoded data with the two targets added |
| `train.csv` | Training set (80%), scaled |
| `test.csv` | Test set (20%), scaled |

## About the dataset

250,000 Indian property listings with 23 columns (location, size, price, neighborhood and ownership details). It has no missing values or duplicates, but it appears to be synthetic: prices, sizes and locations are independent of one another. See `docs/Project_Documentation.pdf` for details.
