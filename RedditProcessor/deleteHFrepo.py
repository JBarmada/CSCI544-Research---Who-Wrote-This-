from huggingface_hub import HfApi
import os

HF_TOKEN   = os.environ.get("HF_TOKEN", "")                                   # your token
HF_REPO_ID = "validname/reddit-ai-detection-english-80k"  # your repo

api = HfApi()

# Delete the whole repo and recreate it clean
api.delete_repo(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
print("Repo deleted.")

# Now re-push properly
import pandas as pd
from datasets import Dataset, DatasetDict

pre_df  = pd.read_csv("reddit_output/reddit_pre_2022.csv")
post_df = pd.read_csv("reddit_output/reddit_post_2022.csv")

dataset = DatasetDict({
    "pre_2022":  Dataset.from_pandas(pre_df,  preserve_index=False),
    "post_2022": Dataset.from_pandas(post_df, preserve_index=False),
})

dataset.push_to_hub(HF_REPO_ID, token=HF_TOKEN, private=False)
print("Done.")