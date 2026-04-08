def preprocess_data(df, text_column, min_words):
    """Cleans data and filters out text too short for Binoculars."""
    print(f"Preprocessing {len(df)} records...")
    
    # Drop rows where the text column is completely missing
    df = df.dropna(subset=[text_column]).copy()
    
    # Ensure everything is a string
    df[text_column] = df[text_column].astype(str)
    
    # Filter out extremely short texts (Binoculars defaults to human if too short)
    df['temp_word_count'] = df[text_column].apply(lambda x: len(x.split()))
    df = df[df['temp_word_count'] >= min_words].copy()
    
    # Drop the temporary column to keep your original schema clean
    df = df.drop(columns=['temp_word_count'])
    
    print(f"Preprocessing complete. {len(df)} valid records remaining.")
    return df