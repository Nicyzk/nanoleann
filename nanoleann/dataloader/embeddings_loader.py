import argparse
import pickle

import numpy as np

embedding_files="data/embeddings.pkl"

def retrieve_arrays(n: int) -> np.ndarray:
    with open(embedding_files, 'rb') as pickle_file:   
        emb = pickle.load(pickle_file)
    return emb[n]

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Process a number and return a list of NumPy arrays."
    )
    
    # Add the argument for the number
    # type=int ensures the input is converted to an integer automatically
    parser.add_argument(
        "number", 
        type=int, 
        help="The input number used to determine how many arrays to retrieve"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call your function
    result = retrieve_arrays(args.number)

    # Output for demonstration purposes
    print(f"Successfully find the embedding with len: {result.size}")

if __name__ == "__main__":
    main()
