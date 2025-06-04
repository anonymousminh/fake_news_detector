import os
import json
import pandas as pd
from pathlib import Path

class FakeNewsNetLoader:
    def __init__(self, dataset_path):
        """
        Initialize the FakeNewsNet dataset loader.
        
        Args:
            dataset_path: Path to the directory containing the CSV datasets
        """
        self.dataset_path = Path(dataset_path)
        self.sources = ['politifact', 'gossipcop']
        self.categories = ['fake', 'real']
        
    def load_news_content(self):
        """
        Load news content from CSV files in the dataset.
        
        Returns:
            DataFrame with columns: id, news_url, title, tweets_ids, news_source, label
        """
        dataframes = []
        
        for source in self.sources:
            for category in self.categories:
                # Build CSV file path: e.g. politifact_fake.csv located in the "dataset" subfolder
                csv_file = self.dataset_path / "dataset" / f"{source}_{category}.csv"
                
                if not csv_file.exists():
                    print(f"Warning: File {csv_file} does not exist")
                    continue
                
                try:
                    df_part = pd.read_csv(csv_file)
                    
                    # Rename 'url' column to 'news_url' if present
                    if 'url' in df_part.columns:
                        df_part = df_part.rename(columns={'url': 'news_url'})
                    
                    # Rename 'tweet_ids' to 'tweets_ids' if needed, or create the column if missing
                    if 'tweet_ids' in df_part.columns and 'tweets_ids' not in df_part.columns:
                        df_part = df_part.rename(columns={'tweet_ids': 'tweets_ids'})
                    if 'tweets_ids' not in df_part.columns:
                        df_part['tweets_ids'] = None
                        
                    # Add extra columns to track the source and category label
                    df_part['news_source'] = source
                    df_part['label'] = category
                    
                    dataframes.append(df_part)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
        
        if dataframes:
            df = pd.concat(dataframes, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        return df
    
    def save_processed_dataset(self, output_path='data/datasets/processed'):
        """
        Process the dataset and save it to CSV.
        
        Args:
            output_path: Path to save the processed dataset
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Load and process data
        df = self.load_news_content()
        
        # Save to CSV
        output_file = os.path.join(output_path, 'fakenewsnet_processed.csv')
        df.to_csv(output_file, index=False)
        
        print(f"Processed dataset saved to {output_file}")
        print(f"Dataset statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Dataset columns: {list(df.columns)}")
        
        return output_file

if __name__ == "__main__":
    # Example usage: update the path to match where your CSV files are stored.
    loader = FakeNewsNetLoader('data/datasets/FakeNewsNet/dataset')
    loader.save_processed_dataset()
