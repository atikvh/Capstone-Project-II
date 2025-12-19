import pandas as pd
from rouge_score import rouge_scorer

def compute_rouge(reference, system):
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(reference, system)
    return (
        scores["rouge1"].precision,
        scores["rouge1"].recall,
        scores["rouge1"].fmeasure,
    )

if __name__ == "__main__":
    df = pd.read_csv(
        "datasets/rouge_dataset.csv",
        quoting=1,
        quotechar='"',
        escapechar='\\',
        on_bad_lines='skip'
    )

    results = []

    for idx, row in df.iterrows():
        ref = row["reference"]
        sys = row["system"]  

        p, r, f = compute_rouge(ref, sys)

        results.append({
            "doc_id": row["doc_id"],
            "ROUGE-1 Precision": round(p, 4),
            "ROUGE-1 Recall": round(r, 4),
            "ROUGE-1 F1": round(f, 4)
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv("rouge_scores_output.csv", index=False)

    print("\nROUGE-1 evaluation complete! Saved as rouge_scores_output.csv")
    print(result_df)