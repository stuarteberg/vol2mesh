import numpy as np

try:
    from dvidutils import remap_duplicates
    _dvidutils_available = False
except ImportError:
    _dvidutils_available = False

def first_occurrences(vertices):
    if _dvidutils_available:
        return remap_duplicates(vertices)
    else:
        return first_occurrences_pandas(vertices)

def first_occurrences_pandas(vertices):
    """
    For each of duplicate row in the given table,
    which earlier row is it a duplicate of?
    
    In other words, pd.duplicated(df, keep='first') tells you which rows
    are duplicates of earlier rows, but it doesn't tell you exactly where
    those earlier rows are.  This function tells you.

    This is the same as dvidutils.remap_duplicates(), but implemented
    in Python (using pandas). This is 10-100x slower than that function,
    but it doesn't require a compiled module, so it might be easier to
    deploy (e.g. on Windows).
    
    Otherwise, the description is the same as remap_duplicates():

    The goal of this function is to tell you how to remove duplicate
    rows from a list of vertices.  For each duplicate vertex we find,
    it tells you which row (earlier in the list) it is a duplicate of.
    If you have some array that refers to these vertices (e.g. mesh faces),
    you can use this mapping to relabel those references so that the
    'duplicates' are no longer needed.  At that point, you could drop the
    duplicate vertices from your list (as long as you renumber the face
    references accordingly).
    
    Given an array of vertices (N,3), find those vertices which are
    duplicates and return an index mapping that points only to the
    first occurrence of each duplicate vertex found.
    Non-duplicate vertexes are not included in the result,
    i.e. anything missing from the results is implicitly identity-mapped.
    
    Returns an array where each row is a duplicate index (D) and the first
    index it is a duplicate of (F):
    
      [[D,F],
       [D,F],
       [D,F],
       ...
      ]
    """
    import pandas as pd
    if isinstance(vertices, pd.DataFrame):
        df = vertices
    else:
        df = pd.DataFrame(vertices)

    # Pre-filter for the duplicates only.
    dupe_rows = df.duplicated(keep=False)
    if dupe_rows.sum() == 0:
        return np.ndarray((0,2), dtype=np.uint32)

    df_dupes_only = df[dupe_rows].copy()
    df_dupes_only['first_index'] = np.uint32(0)
    def set_first(group_df):
        group_df['first_index'] = group_df.index[0]
        return group_df
        
    cols = df.columns.tolist()
    firsts = df_dupes_only.groupby(cols).apply(set_first)[['first_index']]
    firsts['dupe_index'] = firsts.index.get_level_values(-1).astype(np.uint32)
    firsts.reset_index(drop=True, inplace=True)
    dupe_firsts = firsts.query('dupe_index != first_index')
    return dupe_firsts[['dupe_index', 'first_index']].values
