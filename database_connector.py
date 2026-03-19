import mysql.connector
from mysql.connector import Error

# --- IMPORTANT ---
# Replace these placeholders with your actual MySQL credentials.
DB_USER = "root"  # e.g., "root"
DB_PASSWORD = "Pokemon#123"
DB_HOST = "localhost"
DB_NAME = "attendance_system_db"

def get_db_connection():
    """
    Establishes a connection to the MySQL database.
    Returns the connection object or None if connection fails.
    """
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        if connection.is_connected():
            return connection
    except Error as e:
        # Print a more user-friendly error message
        print(f"Error: Could not connect to the database '{DB_NAME}'. Please check your credentials and ensure the MySQL server is running.")
        print(f"MySQL Error: {e}")
        return None

if __name__ == "__main__":
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)

        # 1) Show tables
        cursor.execute("SHOW TABLES;")
        print("Tables in DB:", [row['Tables_in_attendance_system_db'] for row in cursor.fetchall()])

        # 2) Test admin login
        username = "admin"
        password = "password123"
        cursor.execute("SELECT * FROM admins WHERE username=%s AND password=%s", (username, password))
        result = cursor.fetchone()

        if result:
            print("✅ Admin login successful!")
            print(result)
        else:
            print("❌ Invalid login")

        cursor.close()
        conn.close()
    else:
        print("❌ Could not connect to database")
