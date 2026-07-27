# Input data specification

The unified runner expects preprocessed ML-1M files. Raw data and derived safety
annotations are not redistributed.

## Interaction files

For `--dataset ml-1m_safe` and `--data-dir /path/to/Data`:

```text
/path/to/Data/ml-1m_safe.train.rating
/path/to/Data/ml-1m_safe.valid.rating
/path/to/Data/ml-1m_safe.test.rating
```

Each non-empty line must begin with integer `user_id` and `item_id`. Additional
rating/timestamp columns are permitted. Whitespace, tab, comma, and `::`
separators are supported by the included readers.

Optional sampled-evaluation files are detected automatically:

```text
ml-1m_safe.valid.negative
ml-1m_safe.test.negative
```

## Profiles JSONL

`--profiles` must contain one JSON object per user. Required information:

```json
{
  "user_id": 0,
  "user_info": {
    "is_minor": true,
    "age_group": "Under 18"
  },
  "profile": "Structured or textual user profile"
}
```

The builders support additional historical/preference fields used by the
original profile-generation pipeline.

## Training safety CSV

`--train-safe` contains observed training interactions and item metadata. The
reader detects common aliases, but the following columns are recommended:

```text
inner_user_id, inner_item_id, rating, timestamp, title, genres, overview,
sex_code, violence_code, profanity_code, drug_code, intense_code, isAdult
```

## Item safety CSV

`--item-safe` contains one row per item:

```text
inner_item_id, title, genres, overview,
sex_code, violence_code, profanity_code, drug_code, intense_code, isAdult
```

Risk codes are expected to be numeric. The reference thresholds are 3 for minor
users and 4 for adult users.

## User information CSV

`--user-info` must contain a user identifier and minor flag:

```text
inner_user_id,is_minor
0,1
1,0
```

Additional tolerance/statistical fields are allowed.
