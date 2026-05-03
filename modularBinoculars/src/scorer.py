from binoculars import Binoculars

def score_dataframe(df, text_column, batch_size, mode, threshold):
    print(f"Initializing Binoculars in '{mode}' mode...")
    # The library handles bfloat16 internally based on defaults
    bino = Binoculars(mode=mode) 
    
    scores = []
    predictions = []
    total_batches = (len(df) + batch_size - 1) // batch_size

    print(f"Starting inference across {total_batches} batches...")
    
    for i in range(0, len(df), batch_size):
        batch_texts = df[text_column].iloc[i:i+batch_size].tolist()
        
        try:
            # FIX 1: Only call compute_score to prevent double-inference
            batch_scores = bino.compute_score(batch_texts)
            
            # Handle scalar return if batch size happens to be 1
            if isinstance(batch_scores, (int, float)):
                batch_scores = [batch_scores]
                
            # FIX 2: Apply the threshold manually locally
            batch_preds = [
                "Most likely AI-generated" if s < threshold else "Most likely human-generated"
                for s in batch_scores
            ]
            
            scores.extend(batch_scores)
            predictions.extend(batch_preds)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            scores.extend([None] * len(batch_texts))
            predictions.extend(["Error"] * len(batch_texts))
            
        if (i // batch_size) % 10 == 0:
            print(f"Processed batch {i//batch_size} / {total_batches}")

    # Attach results to the dataframe
    result_df = df.copy()
    result_df['binoculars_score'] = scores
    result_df['binoculars_prediction'] = predictions
    return result_df