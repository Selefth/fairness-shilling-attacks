
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem
from datetime import datetime, timezone

"""
Dataset loading and preprocessing utilities for various recommendation datasets.
Includes logic for filtering, transforming implicit feedback, and splitting users into groups.
"""

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

AGE_YOUNGER, AGE_OLDER = 'y', 'o'
GENDER_MALE, GENDER_FEMALE = 'm', 'f'

def apply_min_items_threshold(df, threshold, item_col="Item_id", user_col="User_id"):
    """Apply a threshold filter to ensure a minimum number of items per user."""
    preprocessor = DataFramePreprocessor(item_col, user_col)
    min_items_filter = MinItemsPerUser(threshold, item_col, user_col)
    preprocessor.add_filter(min_items_filter)
    return min_items_filter.apply(df)

def apply_min_users_threshold(df, threshold, item_col="Item_id", user_col="User_id"):
    """Apply a threshold filter to ensure a minimum number of users per item."""
    preprocessor = DataFramePreprocessor(item_col, user_col)
    min_users_filter = MinUsersPerItem(threshold, item_col, user_col)
    preprocessor.add_filter(min_users_filter)
    return min_users_filter.apply(df)

def transform_implicit_to_explicit(df, user_col="User_id", interaction_col="Interaction", target_range=(1, 5)):
    """
    Transform implicit feedback (e.g., play counts, check-ins) to explicit-style ratings.
    Applies a log transformation to reduce skewness in interaction data and then scales the values to a specified 
    range (default: [1, 5]) using Min-Max scaling on a per-user basis, rounding to the nearest integer.
    """
    df[interaction_col] = np.log1p(df[interaction_col])

    # Apply MinMax scaling per user and round to the nearest integer
    def scale_user_interactions(user_interactions):
        scaler = MinMaxScaler(feature_range=target_range)
        scaled_values = scaler.fit_transform(user_interactions.values.reshape(-1, 1)) # reshape to 2D array of shape (n_samples, 1)
        return scaled_values.flatten().round()
    
    # Transform each user's interactions separately
    df[interaction_col] = df.groupby(user_col)[interaction_col].transform(scale_user_interactions)
        
    return df

def load_ml_1m():
    ratings_path = os.path.join(BASE_PATH, "ml-1m", "ratings.dat"); users_path = os.path.join(BASE_PATH, "ml-1m", "users.dat")
    
    ratings = pd.read_csv(ratings_path, sep="::", header=None, usecols=[0,1,2,3], names=["User_id","Item_id","Interaction","Timestamp"], engine="python")

    # Convert Unix timestamps to ISO 8601 format
    ratings["Timestamp"] = ratings["Timestamp"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))

    users = pd.read_csv(users_path, sep="::", header=None, usecols=[0, 1, 2], names=["User_id", "Gender", "Age"], engine="python", encoding="latin-1")
    users = users.dropna(subset=["Age", "Gender"])
    users["Gender"] = users["Gender"].str.lower()  # Convert gender labels to lowercase
    df = pd.merge(ratings, users, on="User_id")

    # users' partitioning based on gender
    groups_gender = {gender: df[df["Gender"] == gender]["User_id"].unique().tolist() for gender in [GENDER_MALE, GENDER_FEMALE]}

    # users' partitioning based on age
    a_thres = 45
    groups_age = {
        AGE_YOUNGER: df[df["Age"] < a_thres]["User_id"].unique().tolist(),
        AGE_OLDER: df[df["Age"] >= a_thres]["User_id"].unique().tolist()
    }
    
    return df, groups_gender, groups_age

def load_ml_100k():
    ratings_path = os.path.join(BASE_PATH, "ml-100k", "u.data"); users_path = os.path.join(BASE_PATH, "ml-100k", "u.user")

    ratings = pd.read_csv(ratings_path, sep='\t', header=None, usecols=[0,1,2,3], names=["User_id", "Item_id", "Interaction", "Timestamp"])

    # Convert Unix timestamps to ISO 8601 format
    ratings["Timestamp"] = ratings["Timestamp"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))

    users = pd.read_csv(users_path, sep='|', header=None, usecols=[0, 1, 2], names=["User_id", "Age", "Gender"], engine="python")
    users = users.dropna(subset=["Age", "Gender"])
    users["Gender"] = users["Gender"].str.lower()  # Convert gender labels to lowercase
    df = pd.merge(ratings, users, on="User_id")
    
    # users' partitioning based on gender
    groups_gender = {gender: df[df["Gender"] == gender]["User_id"].unique().tolist() for gender in [GENDER_MALE, GENDER_FEMALE]}

    # users' partitioning based on age
    a_thres = 41
    groups_age = {
        AGE_YOUNGER: df[df["Age"] < a_thres]["User_id"].unique().tolist(),
        AGE_OLDER: df[df["Age"] >= a_thres]["User_id"].unique().tolist()
    }
    
    return df, groups_gender, groups_age

def load_lastfm_1k():
    plays_path = os.path.join(BASE_PATH, "lastfm-1k", "userid-timestamp-artid-artname-traid-traname.tsv"); users_path = os.path.join(BASE_PATH, "lastfm-1k", "userid-profile.tsv")

    plays = pd.read_csv(plays_path, sep="\t", header=None, usecols=[0, 1, 2], names=["User_id", "Timestamp", "Item_id"], on_bad_lines="skip")

    # Group interactions (plays) by User_id, Item_id (artist), and keep the latest timestamp
    plays = plays.groupby(["User_id", "Item_id"]).agg(
        Interaction=("User_id", "size"),
        Timestamp=("Timestamp", "max")
    ).reset_index()

    users = pd.read_csv(users_path, sep="\t", usecols=[0, 1, 2], names=["User_id", "Gender", "Age"], skiprows=1)
    users = users.dropna(subset=["Age", "Gender"])
    df = plays.merge(users, on="User_id", how="inner")

    # Preprocess data
    df = df[(df["Age"] >= 10) & (df["Age"] <= 90)] # remove outliers
    df = transform_implicit_to_explicit(df, "User_id", "Interaction") # transform implicit feedback to explicit feedback
    df = apply_min_items_threshold(df, 20) # apply a threshold filter to ensure a minimum number of items per user
    
    # users' partitioning
    groups_gender = {gender: df[df["Gender"] == gender]["User_id"].unique().tolist() for gender in [GENDER_MALE, GENDER_FEMALE]}

    # users' partitioning based on age
    a_thres = 27
    groups_age = {
        AGE_YOUNGER: df[df["Age"] < a_thres]["User_id"].unique().tolist(),
        AGE_OLDER: df[df["Age"] >= a_thres]["User_id"].unique().tolist()
    }
    
    return df, groups_gender, groups_age

def load_foursquare(city):
    checkins_path = os.path.join(BASE_PATH, "foursquare", "dataset_TIST2015_Checkins.txt")
    checkins = pd.read_csv(
        checkins_path, 
        sep='\t', 
        header=None, 
        usecols=[0, 1, 2], 
        names=["User_id", "Venue_id", "UTC_Time"]
    )

    profile_path = os.path.join(BASE_PATH, f"foursquare/dataset_UbiComp2016_UserProfile_{city}.txt")
    profiles = pd.read_csv(
        profile_path, 
        sep='\t', 
        header=None, 
        usecols=[0, 1], 
        names=["User_id", "Gender"]
    )

    # Map gender values to 'm' and 'f' for consistency
    profiles["Gender"] = profiles["Gender"].map({"male": GENDER_MALE, "female": GENDER_FEMALE})

    # Aggregate interactions by user-venue pairs, keeping the latest timestamp and counting interactions
    df = checkins.groupby(["User_id", "Venue_id"]).agg(
        Interaction=("User_id", "size"),
        Timestamp=("UTC_Time", "max")
    ).reset_index()

    # Convert NL timestamps to ISO 8601 format, skipping errors
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format='%a %b %d %H:%M:%S %z %Y', errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    df = df.dropna(subset=["Timestamp"])  # Drop rows with parsing errors

    df = df.rename(columns={"Venue_id": "Item_id"})
    df = pd.merge(df, profiles, on="User_id", how="inner")

    # Preprocess data
    df = transform_implicit_to_explicit(df, "User_id", "Interaction") # transform implicit feedback to explicit feedback

    # Preprocessing for minimum items per user and users per item thresholds based on 
    # "Fair Augmentation for Graph Collaborative Filtering" (Boratto et al., RecSys '24).
    df = apply_min_users_threshold(df, 20)
    df = apply_min_items_threshold(df, 20)

    groups_gender = {
        gender: df[df["Gender"] == gender]["User_id"].unique().tolist()
        for gender in [GENDER_MALE, GENDER_FEMALE]
    }

    return df, groups_gender, {}

def load_dataset_by_name(dataset_name):
    datasets = {
        "ml-100k": load_ml_100k,
        "ml-1m": load_ml_1m,
        "lastfm-1k": load_lastfm_1k,
        "fnyc": lambda: load_foursquare("NYC"),
        "ftky": lambda: load_foursquare("TKY")
    }

    if dataset_name in datasets:
        return datasets[dataset_name]()
    raise ValueError(f"Dataset '{dataset_name}' not recognized. Available datasets are: {list(datasets.keys())}")
