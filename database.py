import pyodbc

SERVER = "localhost"
DATABASE = "VisionDB"
DRIVER = "ODBC Driver 18 for SQL Server"

def get_connection():
    conn_str = (
        f"DRIVER={{{DRIVER}}};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        f"Trusted_Connection=yes;"
        f"Encrypt=no;"
    )
    return pyodbc.connect(conn_str)
