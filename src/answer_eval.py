"""Answer-level evaluation: runs every test question through the chatbot."""
import re
import pandas as pd
from chatbot import HybridFaqChatbot, REFUSAL

DATA_DIR = "../data"
OUT_CSV = "../target/answer_eval.csv"
NOT_COVERED = "The knowledge base does not cover this yet."
STOP = set("the a an and or of to in on for with is are be it that this as at by from can may should you your".split())

# off-topic questions the bot should decline (edit or extend this list)
OFF_TOPIC = [
    "What is the capital of France?",
    "Give me a recipe for banana bread.",
    "Write an essay about the Roman Empire.",
    "How do I install Python on Windows?",
    "Who won the last football world cup?",
    "Make a table of the best laptops.",
    "What is the price of Bitcoin?",
    "Explain quantum physics to me.",
]

def words(text):
    # content words only, so common words don't inflate the score
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3 and w not in STOP}

def status_of(reply):
    if reply == REFUSAL:
        return "refused"
    return "unanswered" if reply == NOT_COVERED else "answered"

def main():
    bot = HybridFaqChatbot()
    topics = pd.read_csv(f"{DATA_DIR}/topics.csv")
    gt = pd.read_csv(f"{DATA_DIR}/groundtruth.csv")
    gt = gt[gt["relevance_judgment"] >= 1]
    relevant = gt.groupby("topic_id")["passage_id"].apply(set).to_dict()

    rows = []
    for qid, tid, q in topics[["question_id", "topic_id", "question"]].values:
        reply, passages = bot.answer(q)
        row = {"question_id": qid, "question": q, "status": status_of(reply), "reply": reply}
        if row["status"] == "answered":
            ids = [d for d, _ in passages]
            rel = relevant.get(tid, set())
            cited = {int(n) for n in re.findall(r"\[(\d)\]", reply)}
            ctx = words(" ".join(p.passage for _, p in passages))
            rw = words(reply)
            row["self_cited"] = not bot.auto_cited                      # model cited without help
            row["context_hit"] = any(d in rel for d in ids)             # a correct passage was retrieved
            row["cited_relevant"] = any(ids[n - 1] in rel for n in cited if 1 <= n <= len(ids))
            row["support"] = len(rw & ctx) / len(rw) if rw else 0.0     # faithfulness proxy
            row["manual_faithful"] = ""                                 # fill by hand on a sample
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    ans = df[df["status"] == "answered"]
    print(f"\nIn-scope questions: {len(df)}")
    for s in ("answered", "unanswered", "refused"):
        print(f"  {s:<11}: {(df['status'] == s).mean() * 100:5.1f}%")
    if len(ans):
        print(f"Among {len(ans)} answered:")
        print(f"  model cited by itself : {ans['self_cited'].mean() * 100:5.1f}%")
        print(f"  correct passage found : {ans['context_hit'].mean() * 100:5.1f}%")
        print(f"  cited a correct one   : {ans['cited_relevant'].mean() * 100:5.1f}%")
        print(f"  avg word support      : {ans['support'].mean() * 100:5.1f}%")

    # off-topic questions: both a refusal and "not covered" count as correctly declined
    declined = sum(status_of(bot.answer(q)[0]) != "answered" for q in OFF_TOPIC)
    print(f"Off-topic correctly declined: {declined}/{len(OFF_TOPIC)}")

if __name__ == "__main__":
    main()