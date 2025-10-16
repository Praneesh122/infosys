import pandas as pd
import time
from openai import OpenAI

client = OpenAI(api_key="sk-proj-uor-key")
df = pd.read_csv("reviews.csv")

text_col = None
for col in df.columns:
    if df[col].dtype == "object" and col.lower() not in ["date"]:
        text_col = col
        break
p
print(f"Using column for sentiment analysis: {text_col}")

sentiments = []
batch_size = 3  # safer for free-tier

for i in range(0, len(df), batch_size):
    batch = df[text_col].astype(str).iloc[i:i+batch_size].tolist()

    reviews_text = "\n".join([f"{j+1}. {review}" for j, review in enumerate(batch)])
    prompt = f"""
Classify the sentiment of the following reviews as Positive, Neutral, or Negative.
Respond in the format: <number>: <sentiment>

Reviews:
{reviews_text}
"""

    success = False
    while not success:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a sentiment classifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            output = response.choices[0].message.content.strip()
            for line in output.splitlines():
                if ":" in line:
                    sentiments.append(line.split(":")[1].strip())
            success = True
        except Exception as e:
            print(f" Rate limit hit, waiting 30s... ({e})")
            time.sleep(30)  # wait longer, then retry

# Match lengths
df = df.iloc[:len(sentiments)]
df["sentiment"] = sentiments
df.to_csv("reviews_with_sentiment.csv", index=False)

print(" Sentiment analysis complete! Saved as reviews_with_sentiment.csv")
print(df[[text_col, "sentiment"]].head(10))
print("\nSentiment counts:")
print(df["sentiment"].value_counts())
