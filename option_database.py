import sqlite3

class OptionPricingDatabase:
    def __init__(self, db_name="options_data.db"):
        """Initialize database connection and create tables if they don't exist."""
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        """Create inputs and outputs tables if they don't exist."""
        # Store base inputs for option pricing
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS inputs (
                calculation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_price REAL,
                strike_price REAL,
                time_to_expiry REAL,
                interest_rate REAL,
                volatility REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Store option values based on shocks (linked to inputs via calculation_id)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                calculation_id INTEGER,
                option_type TEXT,
                stock_shock REAL,
                volatility_shock REAL,
                option_price REAL,
                FOREIGN KEY (calculation_id) REFERENCES inputs (calculation_id)
            )
        """)
        self.conn.commit()

    def insert_input(self, stock_price, strike_price, time_to_expiry, interest_rate, volatility):
        """Insert user input values and return the calculation ID."""
        self.cursor.execute("""
            INSERT INTO inputs (stock_price, strike_price, time_to_expiry, interest_rate, volatility)
            VALUES (?, ?, ?, ?, ?)
        """, (stock_price, strike_price, time_to_expiry, interest_rate, volatility))
        self.conn.commit()
        return self.cursor.lastrowid  # Get the newly created calculation_id

    def insert_output(self, calculation_id, option_type, stock_shock, volatility_shock, option_price):
        """Insert calculated option values with shocks linked to an input ID."""
        self.cursor.execute("""
            INSERT INTO outputs (calculation_id, option_type, stock_shock, volatility_shock, option_price)
            VALUES (?, ?, ?, ?, ?)
        """, (calculation_id, option_type, stock_shock, volatility_shock, option_price))
        self.conn.commit()

    def fetch_inputs(self):
        """Retrieve all stored input records."""
        self.cursor.execute("SELECT * FROM inputs ORDER BY timestamp DESC")
        return self.cursor.fetchall()

    def fetch_outputs(self, calculation_id):
        """Retrieve stored outputs for a given calculation ID."""
        self.cursor.execute("SELECT * FROM outputs WHERE calculation_id = ?", (calculation_id,))
        return self.cursor.fetchall()

    def close_connection(self):
        """Close database connection."""
        self.conn.close()
