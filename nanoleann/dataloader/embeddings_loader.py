import argparse
import struct
import sys

import numpy as np

embedding_files = "data/embedding.npy"

def retrieve_arrays(n: int) -> np.ndarray:
    # mmap_mode='r' ensures we don't load the entire file into RAM,
    # just the slice we need.
    emb = np.load(embedding_files, mmap_mode='r')
    return emb[n]

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Process a number, retrieve the tensor, and save it to a file."
    )
    
    # 1. Argument for the index
    parser.add_argument(
        "number", 
        type=int, 
        help="The input number used to determine which array to retrieve"
    )

    # 2. Argument for the output file
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="The file path to write the result tensor to (e.g., result.bin)"
    )

    # Parse the arguments
    args = parser.parse_args()

    try:
        # Call your function
        result = retrieve_arrays(args.number)

        # 3. Write to custom binary format
        # Format: [4 bytes for item_size] + [Raw Data Bytes...]
        with open(args.output, "wb") as f:
            # Write itemsize (e.g., 8 for float64)
            f.write(struct.pack('<I', result.itemsize))
            # Write raw data
            f.write(result.tobytes())

        sys.exit(0)

    except IndexError:
        sys.exit(1)
    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()