#!/usr/bin/env python3
"""
Script to download and prepare STS datasets for the STS similarity system.
"""

import os
import pandas as pd
import requests
import zipfile
import csv
from pathlib import Path

def download_file(url, filename):
    """Download a file from URL."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")

def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def parse_sts_file(file_path, has_scores=True):
    """Parse STS file with proper error handling for malformed lines."""
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        
        for line_num, line in enumerate(f, 2):  # Start from line 2 (after header)
            line = line.strip()
            if not line:
                continue
                
            # Split by tab and handle extra fields
            parts = line.split('\t')
            
            if has_scores:
                # STS-B format with scores: index genre filename year old_index source1 source2 sentence1 sentence2 score
                # We need: sentence1 (index 7), sentence2 (index 8), score (index 9)
                if len(parts) >= 10:
                    try:
                        sentence1 = parts[7]     # Sentence1 is at index 7
                        sentence2 = parts[8]     # Sentence2 is at index 8
                        score = float(parts[9])  # Score is at index 9
                        
                        # Clean sentences (remove quotes if present)
                        sentence1 = sentence1.strip('"')
                        sentence2 = sentence2.strip('"')
                        
                        data.append({
                            'sentence1': sentence1,
                            'sentence2': sentence2,
                            'score': score
                        })
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping malformed line {line_num}: {e}")
                        continue
                else:
                    print(f"Warning: Skipping line {line_num} with insufficient fields (found {len(parts)}, need 10)")
                    continue
            else:
                # STS-B format without scores: index genre filename year old_index source1 source2 sentence1 sentence2
                # We need: sentence1 (index 7), sentence2 (index 8), dummy score
                if len(parts) >= 9:
                    try:
                        sentence1 = parts[7]     # Sentence1 is at index 7
                        sentence2 = parts[8]     # Sentence2 is at index 8
                        
                        # Clean sentences (remove quotes if present)
                        sentence1 = sentence1.strip('"')
                        sentence2 = sentence2.strip('"')
                        
                        data.append({
                            'sentence1': sentence1,
                            'sentence2': sentence2,
                            'score': 0.0  # Dummy score for test set
                        })
                    except (IndexError) as e:
                        print(f"Warning: Skipping malformed line {line_num}: {e}")
                        continue
                else:
                    print(f"Warning: Skipping line {line_num} with insufficient fields (found {len(parts)}, need 9)")
                    continue
    
    return data

def prepare_sts_data():
    """Download and prepare STS-B dataset from official source."""
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Official STS-B dataset URL
    sts_b_url = "https://dl.fbaipublicfiles.com/glue/data/STS-B.zip"
    zip_filename = "data/STS-B.zip"
    
    try:
        # Download the zip file
        download_file(sts_b_url, zip_filename)
        
        # Extract the zip file
        extract_zip(zip_filename, 'data')
        
        # Convert to required format
        print("Converting to required format...")
        
        # Process train data
        train_file = 'data/STS-B/train.tsv'
        if os.path.exists(train_file):
            print("Parsing training data...")
            train_data = parse_sts_file(train_file, has_scores=True)
            
            with open('train.txt', 'w', encoding='utf-8') as f:
                f.write('index\tsentence1\tsentence2\tscore\n')
                for idx, item in enumerate(train_data):
                    f.write(f"{idx+1}\t{item['sentence1']}\t{item['sentence2']}\t{item['score']}\n")
            print(f"Created train.txt with {len(train_data)} pairs")
        
        # Process dev data
        dev_file = 'data/STS-B/dev.tsv'
        if os.path.exists(dev_file):
            print("Parsing development data...")
            dev_data = parse_sts_file(dev_file, has_scores=True)
            
            with open('dev.txt', 'w', encoding='utf-8') as f:
                f.write('index\tsentence1\tsentence2\tscore\n')
                for idx, item in enumerate(dev_data):
                    f.write(f"{idx+1}\t{item['sentence1']}\t{item['sentence2']}\t{item['score']}\n")
            print(f"Created dev.txt with {len(dev_data)} pairs")
        
        # Process test data - check if it has scores
        test_file = 'data/STS-B/test.tsv'
        if os.path.exists(test_file):
            print("Checking test data format...")
            # Check first few lines to see if test has scores
            with open(test_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    first_data_line = lines[1].strip().split('\t')
                    has_scores = len(first_data_line) >= 10  # If 10+ columns, it has scores
            
            if has_scores:
                print("Parsing test data (with scores)...")
                test_data = parse_sts_file(test_file, has_scores=True)
            else:
                print("Test data has no scores, using dev data for test...")
                test_data = dev_data  # Use dev data as test since test has no scores
            
            with open('test.txt', 'w', encoding='utf-8') as f:
                f.write('index\tsentence1\tsentence2\tscore\n')
                for idx, item in enumerate(test_data):
                    f.write(f"{idx+1}\t{item['sentence1']}\t{item['sentence2']}\t{item['score']}\n")
            print(f"Created test.txt with {len(test_data)} pairs")
        else:
            # If test.tsv doesn't exist, use dev as test
            print("Test file not found, using dev data as test data...")
            with open('test.txt', 'w', encoding='utf-8') as f:
                f.write('index\tsentence1\tsentence2\tscore\n')
                for idx, item in enumerate(dev_data):
                    f.write(f"{idx+1}\t{item['sentence1']}\t{item['sentence2']}\t{item['score']}\n")
            print(f"Created test.txt with {len(dev_data)} pairs (from dev data)")
        
        # Clean up zip file
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
            print("Cleaned up zip file")
        
        print("Dataset preparation completed!")
        print("Files created: train.txt, dev.txt, test.txt")
        
        # Print dataset statistics
        if os.path.exists('train.txt') and os.path.exists('dev.txt') and os.path.exists('test.txt'):
            with open('train.txt', 'r') as f:
                train_lines = len(f.readlines()) - 1  # Subtract header
            with open('dev.txt', 'r') as f:
                dev_lines = len(f.readlines()) - 1  # Subtract header
            with open('test.txt', 'r') as f:
                test_lines = len(f.readlines()) - 1  # Subtract header
            print(f"Dataset statistics:")
            print(f"- Training pairs: {train_lines}")
            print(f"- Development pairs: {dev_lines}")
            print(f"- Test pairs: {test_lines}")
        
    except Exception as e:
        print(f"Error downloading STS-B dataset: {e}")
        return False
    
    return True

def download_alternative_dataset():
    """Download an alternative STS dataset if the main one fails."""
    
    print("Trying alternative dataset source...")
    
    # Alternative: SICK dataset from a different source
    sick_url = "https://raw.githubusercontent.com/facebookresearch/SentEval/master/data/downstream/SICK/SICK.txt"
    
    try:
        download_file(sick_url, 'data/sick.txt')
        
        # Convert SICK format to our format
        with open('data/sick.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header
        data_lines = [line.strip().split('\t') for line in lines[1:] if line.strip()]
        
        # Split into train/test
        train_size = int(len(data_lines) * 0.8)
        train_data = data_lines[:train_size]
        test_data = data_lines[train_size:]
        
        # Write train file
        with open('train.txt', 'w', encoding='utf-8') as f:
            f.write('index\tsentence1\tsentence2\tscore\n')
            for idx, row in enumerate(train_data):
                if len(row) >= 4:
                    f.write(f"{idx+1}\t{row[1]}\t{row[2]}\t{row[3]}\n")
        
        # Write test file
        with open('test.txt', 'w', encoding='utf-8') as f:
            f.write('index\tsentence1\tsentence2\tscore\n')
            for idx, row in enumerate(test_data):
                if len(row) >= 4:
                    f.write(f"{idx+1}\t{row[1]}\t{row[2]}\t{row[3]}\n")
        
        print("Created train.txt and test.txt from SICK dataset")
        return True
        
    except Exception as e:
        print(f"Failed to download alternative dataset: {e}")
        return False

def create_sample_dataset():
    """Create a small sample dataset for testing."""
    
    print("Creating sample dataset for testing...")
    
    sample_data = [
        ("A man is playing guitar.", "A person is playing a musical instrument.", 4.2),
        ("The cat is sleeping.", "A dog is running.", 1.1),
        ("The weather is nice today.", "It's a beautiful day.", 4.5),
        ("I love pizza.", "I hate vegetables.", 0.8),
        ("The movie was great.", "The film was excellent.", 4.8),
        ("She is reading a book.", "He is watching TV.", 1.5),
        ("The car is red.", "The vehicle is blue.", 3.2),
        ("I'm going to the store.", "I'm heading to the shop.", 4.1),
        ("The baby is crying.", "The child is laughing.", 2.3),
        ("The computer is broken.", "The laptop needs repair.", 3.9),
        ("The sun is shining.", "It's raining outside.", 1.0),
        ("The food tastes delicious.", "The meal is very good.", 4.3),
        ("He is running fast.", "She is walking slowly.", 2.1),
        ("The book is interesting.", "The novel is boring.", 1.8),
        ("The house is big.", "The home is large.", 4.7)
    ]
    
    # Split into train/test
    train_size = int(len(sample_data) * 0.7)
    train_data = sample_data[:train_size]
    test_data = sample_data[train_size:]
    
    # Write train file
    with open('train.txt', 'w', encoding='utf-8') as f:
        f.write('index\tsentence1\tsentence2\tscore\n')
        for idx, (s1, s2, score) in enumerate(train_data):
            f.write(f"{idx+1}\t{s1}\t{s2}\t{score}\n")
    
    # Write test file
    with open('test.txt', 'w', encoding='utf-8') as f:
        f.write('index\tsentence1\tsentence2\tscore\n')
        for idx, (s1, s2, score) in enumerate(test_data):
            f.write(f"{idx+1}\t{s1}\t{s2}\t{score}\n")
    
    print("Created sample train.txt and test.txt files")
    print("This is a small dataset for testing your system")

def main():
    """Main function to download and prepare dataset."""
    
    print("STS Dataset Downloader")
    print("=" * 30)
    print("Downloading official STS-B dataset from Facebook Research...")
    
    try:
        # Try to download STS-B dataset from official source
        success = prepare_sts_data()
        
        if success and os.path.exists('train.txt') and os.path.exists('dev.txt') and os.path.exists('test.txt'):
            print("\n✅ Success! Official STS-B dataset downloaded and prepared:")
            print("- train.txt")
            print("- dev.txt")
            print("- test.txt")
            print("\nYou can now run: python sts_similarity.py")
        else:
            print("\n❌ Main dataset download failed, trying alternative...")
            alt_success = download_alternative_dataset()
            
            if not alt_success:
                print("\n❌ Alternative dataset also failed, creating sample dataset...")
                create_sample_dataset()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Creating sample dataset instead...")
        create_sample_dataset()
    
    print("\nDataset preparation completed!")

if __name__ == "__main__":
    main() 