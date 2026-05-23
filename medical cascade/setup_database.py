import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

db_user = os.getenv("DB_USER", "postgres")
db_password = os.getenv("DB_PASSWORD", "mannu")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "ads")

def create_database():
    try:
        # Connect to the default 'postgres' database
        conn = psycopg2.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database '{db_name}'...")
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created successfully!")
        else:
            print(f"Database '{db_name}' already exists.")
            
        cursor.close()
        conn.close()
        
        # Now initialize tables using the existing db_utils
        from db_utils import init_db, seed_side_effects
        print("Initializing tables...")
        init_db()
        print("Tables initialized.")
        
        if os.path.exists('side_effects.csv'):
            print("Seeding side effects data...")
            seed_side_effects('side_effects.csv')
            print("Seeding complete.")
            
    except Exception as e:
        print(f"Error during database setup: {e}")
        print("\nTIP: Make sure your PostgreSQL password is correct in the .env file.")

if __name__ == "__main__":
    create_database()
