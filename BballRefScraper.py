import requests
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
import pickle

def perGameStats():
    # Define the base URL
    base_url = "https://www.basketball-reference.com/leagues/"

    # Define the range of seasons you want to scrape
    start_year = 2000
    end_year = 2022

    # Create an empty DataFrame to store the scraped data
    data = pd.DataFrame()

    # Iterate over each season
    for year in range(start_year, end_year+1):
        # Construct the URL for the season's per game stats page
        url = f"{base_url}NBA_{year}_per_game.html"
        
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table containing the per game stats
        table = soup.find('table', id='per_game_stats')
        
        # Convert the HTML table to a DataFrame
        df = pd.read_html(str(table))[0]
        
        # Remove rows with duplicate players within the same season, retaining 'TOT' row
        df = df.groupby('Player', as_index=False).first()
        
        # Add a column for the season
        df['Season'] = year
        
        # Append the DataFrame to the main data DataFrame
        data = data._append(df, ignore_index=True)

    # Save the data to a CSV file
    data.to_csv('per_game_stats1.csv', index=False)

