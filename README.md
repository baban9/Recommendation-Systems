# Recommendation Systems

Book recommendation engine comparing popularity-based, content-based, and collaborative filtering approaches to reduce choice overload.

## Problem

Users face too many book options with no personalized guidance. A recommender should surface relevant titles from prior ratings and content similarity.

## Approach

1. **Popularity baseline**: Rank by aggregate ratings
2. **Content-based**: TF-IDF or feature similarity on book metadata
3. **Collaborative filtering**: User-item matrix factorization or neighborhood methods

Implementation lives in `Book recommendation_ Book_recommendation.ipynb` with supporting presentation materials.

## Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `Book recommendation_ Book_recommendation.ipynb` and run top-down.

## Tech stack

Python 3, Jupyter, pandas, scikit-learn

## Evaluation mindset

Compare recommenders on:

- Coverage (catalog breadth)
- Personalization (non-popularity bias)
- Offline precision@k on holdout ratings

## Limitations and next steps

- Extract recommender classes into `src/recommenders/`
- Add offline evaluation harness with fixed train/test split
- Serve top-k recommendations via a small Flask API
