"""
MongoDB Data Ingestion Script — SpamHam AI Platform
Author: Abhishek Bisht <abhiishekbishtt@gmail.com>

Usage:
    1. Set MONGODB_URL in your .env file or environment variable:
       MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority"
    2. Run: python3 upload_data_mongodb.py
"""

import os
import sys
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def main():
    # 1. Obtain MongoDB URI from environment variables
    mongo_uri = os.getenv("MONGODB_URL") or os.getenv("MONGO_DB_URL")
    
    if not mongo_uri:
        print("❌ Error: MONGODB_URL environment variable is not set!")
        print("Please set MONGODB_URL in your .env file or export MONGODB_URL='mongodb+srv://...'")
        sys.exit(1)

    # 2. Locate CSV Dataset
    csv_file_path = "spamham.csv"
    if not os.path.exists(csv_file_path):
        csv_file_path = os.path.join("notebooks", "spamham.csv")

    if not os.path.exists(csv_file_path):
        print(f"❌ Error: Dataset file '{csv_file_path}' not found!")
        sys.exit(1)

    print(f"📦 Loading dataset from '{csv_file_path}'...")
    df = pd.read_csv(csv_file_path)
    print(f"✅ Loaded {len(df):,} records into DataFrame.")

    # 3. Connect to MongoDB Atlas / Local Database
    print("🔌 Connecting to MongoDB...")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

    # 4. Target Database & Collection
    db_name = os.getenv("MONGO_DATABASE_NAME", "spamham_db")
    collection_name = os.getenv("MONGO_COLLECTION_NAME", "spam_ham")
    
    db = client[db_name]
    collection = db[collection_name]

    # 5. Ingest Records
    records = df.to_dict(orient="records")
    existing_count = collection.count_documents({})
    
    if existing_count > 0:
        print(f"ℹ️ Collection '{db_name}.{collection_name}' already contains {existing_count:,} documents.")
        user_choice = input("Overwrite existing collection? (y/N): ").strip().lower()
        if user_choice == 'y':
            collection.delete_many({})
            print("🗑️ Dropped old documents.")
        else:
            print("Skipping insertion.")
            sys.exit(0)

    print(f"🚀 Inserting {len(records):,} documents into MongoDB collection '{db_name}.{collection_name}'...")
    result = collection.insert_many(records)
    print(f"🎉 Successfully inserted {len(result.inserted_ids):,} documents into MongoDB!")

if __name__ == "__main__":
    main()
