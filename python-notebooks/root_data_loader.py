# Cluster Energy Resolution Analysis
# Structured with reusable functions

import os
import glob
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_root_file(file_pattern, tree_name, branches, n_events=None, entry_start=0):
    """
    Load branches from a ROOT file.
    
    Parameters:
    -----------
    file_pattern : str
        Glob pattern to locate ROOT file(s)
    tree_name : str
        Name of the tree in the ROOT file
    branches : list of str
        List of branch names to load
    n_events : int, optional
        Number of events to read. If None, reads all events.
    entry_start : int, optional
        Starting entry index (default: 0)
    
    Returns:
    --------
    awkward.Array
        Array containing the requested branches
    """
    root_files = glob.glob(file_pattern, recursive=True)
    if not root_files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    file_path = root_files[0]
    print(f"Loading file: {file_path}")
    
    entry_stop = entry_start + n_events if n_events else None
    
    with uproot.open(file_path) as f:
        tree = f[tree_name]
        arr = tree.arrays(branches, library="ak", entry_start=entry_start, entry_stop=entry_stop)
    
    print(f"Loaded {len(arr)} events")
    return arr


def validate_cluster_counts(arr, branch_names):
    """
    Validate that all branches have the same per-event cluster counts.
    
    Parameters:
    -----------
    arr : awkward.Array
        Array containing the branches
    branch_names : list of str
        List of branch names to validate
    
    Raises:
    -------
    ValueError
        If per-event cluster counts differ between branches
    """
    counts = [ak.num(arr[branch]) for branch in branch_names]
    
    # Check all counts are equal
    all_equal = True
    for i in range(1, len(counts)):
        if not ak.all(counts[0] == counts[i]):
            all_equal = False
            break
    
    if not all_equal:
        raise ValueError(f"Per-event cluster counts differ between branches {branch_names}")
    
    print(f"✓ All branches have consistent per-event counts")


def flatten_to_dataframe(arr, branch_mapping=None):
    """
    Flatten awkward array and convert to pandas DataFrame.
    
    Parameters:
    -----------
    arr : awkward.Array
        Array containing the branches
    branch_mapping : dict, optional
        Dictionary mapping branch names to new column names.
        If None, uses original branch names.
    
    Returns:
    --------
    pd.DataFrame
        Flattened DataFrame with one row per cluster
    """
    if branch_mapping is None:
        branch_mapping = {name: name for name in arr.fields}
    
    z = ak.zip({new_name: arr[old_name] for old_name, new_name in branch_mapping.items()})
    flat = ak.flatten(z)
    df = pd.DataFrame(ak.to_list(flat))
    
    print(f"Flattened to {len(df)} clusters")
    return df


def load_root_file_chunked(file_pattern, tree_name, branches, chunk_size=10000, n_events=None):
    """
    Load branches from a ROOT file in chunks to avoid memory overload.
    
    Parameters:
    -----------
    file_pattern : str
        Glob pattern to locate ROOT file(s)
    tree_name : str
        Name of the tree in the ROOT file
    branches : list of str
        List of branch names to load
    chunk_size : int
        Number of events per chunk (default: 10000)
    n_events : int, optional
        Total number of events to read. If None, reads all events.
    
    Yields:
    -------
    awkward.Array
        Array chunk containing the requested branches
    """
    root_files = glob.glob(file_pattern, recursive=True)
    if not root_files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    file_path = root_files[0]
    print(f"Loading file: {file_path}")
    
    with uproot.open(file_path) as f:
        tree = f[tree_name]
        total_events = tree.num_entries
        
        if n_events is None:
            n_events = total_events
        
        print(f"Total events available: {total_events}")
        print(f"Reading {n_events} events in chunks of {chunk_size}")
        
        entry_start = 0
        while entry_start < n_events:
            entry_stop = min(entry_start + chunk_size, n_events)
            arr = tree.arrays(branches, library="ak", 
                            entry_start=entry_start, 
                            entry_stop=entry_stop)
            print(f"  Loaded events {entry_start}-{entry_stop}")
            yield arr
            entry_start = entry_stop


def process_chunks(file_pattern, tree_name, branches, processor_func, 
                   chunk_size=10000, n_events=None):
    """
    Load ROOT file in chunks and process each chunk with a custom function.
    
    Parameters:
    -----------
    file_pattern : str
        Glob pattern to locate ROOT file(s)
    tree_name : str
        Name of the tree in the ROOT file
    branches : list of str
        List of branch names to load
    processor_func : callable
        Function that takes an awkward array and returns results.
        Can accumulate results across chunks.
    chunk_size : int
        Number of events per chunk
    n_events : int, optional
        Total number of events to read
    
    Returns:
    --------
    Results from the processor function
    """
    results = None
    
    for chunk_arr in load_root_file_chunked(file_pattern, tree_name, branches, 
                                            chunk_size, n_events):
        chunk_result = processor_func(chunk_arr)
        
        if results is None:
            results = chunk_result
        else:
            results = combine_results(results, chunk_result)
    
    return results


def combine_results(results1, results2):
    """
    Combine results from two chunks (helper for processing).
    Assumes results are dictionaries with lists/arrays as values.
    """
    if isinstance(results1, dict):
        combined = {}
        for key in results1.keys():
            if isinstance(results1[key], list):
                combined[key] = results1[key] + results2[key]
            elif isinstance(results1[key], np.ndarray):
                combined[key] = np.concatenate([results1[key], results2[key]])
        return combined
    return results1
