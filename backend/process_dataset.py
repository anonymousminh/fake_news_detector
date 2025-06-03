from app.data_processing.dataset_loader import FakeNewsNetLoader
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process FakeNewsNet dataset')
    # Update the default dataset_path relative to the backend folder.
    parser.add_argument('--dataset_path', type=str, default='../data/datasets/FakeNewsNet',
                        help='Path to the FakeNewsNet dataset')
    # Update output path if needed.
    parser.add_argument('--output_path', type=str, default='../data/datasets/processed',
                        help='Path to save the processed dataset')
    
    args = parser.parse_args()
    
    print(f"Processing FakeNewsNet dataset from {args.dataset_path}")
    loader = FakeNewsNetLoader(args.dataset_path)
    output_file = loader.save_processed_dataset(args.output_path)
    
    print(f"Dataset processing complete. Output saved to {output_file}")

if __name__ == "__main__":
    main()
