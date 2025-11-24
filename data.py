import sqlite3
import requests
import pandas as pd
import time
import os
import json
import threading
from dotenv import load_dotenv
from binance.client import Client
from websocket import WebSocketApp

import asyncio
import httpx
import aiosqlite

class BinanceClient:
    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self, symbols, db_path="data/historical_data.db"):
        """Initialize with a list of symbols."""
        self.symbols = [symbol.upper() for symbol in symbols]  # Standardize symbols
        self.db_path = db_path

    async def get_historical_data(self, symbol, interval="1m", limit=1000):
        """Fetch historical OHLCV data asynchronously for a given symbol."""
        url = f"{self.BASE_URL}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            if response.status_code != 200:
                print(f"❌ Error fetching {symbol}: {response.text}")
                return None

            data = response.json()

        # Convert raw Binance response to list of tuples for SQLite
        formatted_data = [
            (
                entry[0],  # Timestamp (milliseconds)
                float(entry[1]),  # Open
                float(entry[2]),  # High
                float(entry[3]),  # Low
                float(entry[4]),  # Close
                float(entry[5])   # Volume
            )
            for entry in data
        ]

        return symbol, formatted_data

    async def save_to_db(self, symbol, data):
        """Store fetched historical data into SQLite asynchronously."""
        table_name = symbol

        async with aiosqlite.connect(self.db_path) as db:
            # Create table dynamically if not exists
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp INTEGER PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL
                )
            """)
            await db.commit()

            # Insert data in bulk
            query = f"""
                INSERT OR IGNORE INTO {table_name} (timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            await db.executemany(query, data)
            await db.commit()

        print(f"✅ Data saved for {symbol} in {self.db_path}")

    async def fetch_and_store_all(self, interval="1m", limit=1000):
        """Fetch and store historical data for all symbols in parallel."""
        tasks = [self.get_historical_data(symbol, interval, limit) for symbol in self.symbols]
        results = await asyncio.gather(*tasks)

        # Filter out None results (failed fetches)
        valid_results = [res for res in results if res is not None]

        # Save each valid dataset
        save_tasks = [self.save_to_db(symbol, data) for symbol, data in valid_results]
        await asyncio.gather(*save_tasks)


# Example usage
async def main():
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Add more symbols as needed
    binance_client = BinanceClient(symbols)
    await binance_client.fetch_and_store_all(interval="1m", limit=100)

# Run the async function
asyncio.run(main())



    