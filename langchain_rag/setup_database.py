#!/usr/bin/env python3
"""
Database setup script for RAG Chatbot with PostgreSQL
This script creates the necessary database and tables for the chat history feature.
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

# Database configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_chatbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")

def create_database():
    """Create the database if it doesn't exist."""
    try:
        # Connect to PostgreSQL server (not to specific database)
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database="postgres"  # Connect to default postgres database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{POSTGRES_DB}'")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database '{POSTGRES_DB}'...")
            cursor.execute(f"CREATE DATABASE {POSTGRES_DB}")
            print(f"✅ Database '{POSTGRES_DB}' created successfully!")
        else:
            print(f"✅ Database '{POSTGRES_DB}' already exists.")
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"❌ Error creating database: {e}")
        return False
    
    return True

def test_connection():
    """Test connection to the specific database."""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Connected to PostgreSQL: {version[0]}")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"❌ Error connecting to database: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up PostgreSQL database for RAG Chatbot...")
    print(f"Host: {POSTGRES_HOST}:{POSTGRES_PORT}")
    print(f"Database: {POSTGRES_DB}")
    print(f"User: {POSTGRES_USER}")
    print("-" * 50)
    
    if create_database():
        if test_connection():
            print("\n🎉 Database setup completed successfully!")
            print("\nNext steps:")
            print("1. Make sure your .env file has the correct database credentials")
            print("2. Run the FastAPI application: uvicorn rag_gemini:app --reload")
            print("3. The tables will be created automatically when the app starts")
        else:
            print("\n❌ Database setup failed. Please check your credentials.")
    else:
        print("\n❌ Database setup failed.")
