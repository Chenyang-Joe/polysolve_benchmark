import numpy as np
import struct

file_1 = "/mnt/hdd1/chenyang/benchmark_data/matrix_resource/solver-mat-0906/trash-compactor-shapes/5_55_A.bin" 
file_2 = "/mnt/hdd1/chenyang/benchmark_data/matrix_resource/solver-mat-0906/trash-compactor-shapes/5_55_b.bin"

# I want to check if these two files inlcude any NaN or Inf values
# because I got the error:
# ERROR detected by Hypre ...  BEGIN
# ERROR -- hypre_PCGSolve: INFs and/or NaNs detected in input.
# User probably placed non-numerics in supplied A or x_0.
# Returning error flag += 101.  Program not terminated.
# ERROR detected by Hypre ...  END

def read_binary_matrix(filename):
    """Read binary matrix file and return as numpy array"""
    with open(filename, 'rb') as f:
        # Read all data as doubles
        data = np.fromfile(f, dtype=np.float64)
    return data

def check_for_non_finite(filename, label):
    """Check if binary file contains NaN or Inf values"""
    print(f"\n{'='*60}")
    print(f"Checking {label}: {filename}")
    print('='*60)
    
    try:
        data = read_binary_matrix(filename)
        
        # Check for NaN
        nan_count = np.isnan(data).sum()
        nan_indices = np.where(np.isnan(data))[0]
        
        # Check for Inf
        inf_count = np.isinf(data).sum()
        inf_indices = np.where(np.isinf(data))[0]
        
        # Statistics
        print(f"Total elements: {len(data)}")
        print(f"NaN values found: {nan_count}")
        if nan_count > 0:
            print(f"  First few NaN indices: {nan_indices[:min(10, len(nan_indices))]}")
        
        print(f"Inf values found: {inf_count}")
        if inf_count > 0:
            print(f"  First few Inf indices: {inf_indices[:min(10, len(inf_indices))]}")
        
        # Show value range for finite values
        finite_data = data[np.isfinite(data)]
        if len(finite_data) > 0:
            print(f"\nFinite value statistics:")
            print(f"  Min: {finite_data.min()}")
            print(f"  Max: {finite_data.max()}")
            print(f"  Mean: {finite_data.mean()}")
            print(f"  Std: {finite_data.std()}")
        
        # Overall result
        if nan_count == 0 and inf_count == 0:
            print(f"\n✓ {label} is clean - no NaN or Inf values detected")
            return True
        else:
            print(f"\n✗ {label} contains non-finite values!")
            return False
            
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

# Check both files
print("\nChecking binary files for NaN and Inf values...")
result_A = check_for_non_finite(file_1, "Matrix A")
result_b = check_for_non_finite(file_2, "Vector b")

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"Matrix A: {'CLEAN' if result_A else 'CONTAINS NaN/Inf'}")
print(f"Vector b: {'CLEAN' if result_b else 'CONTAINS NaN/Inf'}")
print('='*60)