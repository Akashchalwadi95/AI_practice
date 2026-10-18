import pandas as pd
import numpy as np

raw_eval_data = {
    "eval_id": [101, 102, 103, 104, 105, 106],
    "prompt": [
        "Explain quantum computing", 
        "How to cook pasta", 
        "Write a binary search in C++", 
        "Summarize World War 2", 
        "What is FastAPI?", 
        "Draft an email to manager"
    ],
    "category": ["science", "lifestyle", "coding", "history", "coding", "business"],
    "faithfulness_score": [0.95, 0.80, 0.99, np.nan, 0.92, 0.85],
    "answer_relevance": [0.90, 0.75, 0.98, 0.88, 0.90, 0.80],
    "latency_ms": [320, 150, 450, 600, 280, 210]
}

df = pd.DataFrame(raw_eval_data)

print("---Initial dataframe---")
print(df)

df["faithfulness_score"] = df["faithfulness_score"].fillna(df["faithfulness_score"].mean())

df["overall_quality"] = (df["faithfulness_score"] + df["answer_relevance"])/2

quality = df[df["overall_quality"] >= 0.90]
print(quality)

print("---performance---")
summary = df.groupby("category").agg(
    latency=("latency_ms", "mean"),
    quality=("overall_quality", "mean")
)

print(summary)