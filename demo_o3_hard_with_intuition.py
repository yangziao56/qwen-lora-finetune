# import os
# from openai import AzureOpenAI

# client = AzureOpenAI(
#     azure_endpoint = os.getenv("OPENAI_API_BASE"),
#     api_key        = os.getenv("OPENAI_API_KEY"),
#     api_version    = "2025-04-01-preview",
#     timeout        = 30,                    # 秒
# )

# resp = client.chat.completions.create(
#     model = "o3",
#     messages = [
#         {"role": "user",
#          "content": "The quick brown fox jumps over the lazy dog. 这句话有什么特别？"}
#     ]
# )

# print(resp.choices[0].message.content)




import os
import pandas as pd
import time
import re
import json
from openai import AzureOpenAI

# Azure OpenAI client initialization
client = AzureOpenAI(
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2025-04-01-preview",
    timeout=30,
)

def get_headline_score(headline: str) -> tuple[int | None, str | None]:
    """
    Call the o3 model to rate a single headline (1 to 10).
    Returns an integer score or None if the response is invalid.
    """
    if not isinstance(headline, str) or not headline.strip():
        return None, None
    try:
        # add system prompt for email‐marketing copy evaluation
        system_prompt = (
            "You are an email headline expert.\n\n"
            "Hard-fail rules\n"
            "• If the headline has ANY of these issues, overall_score = 2 or 1 as noted:\n"
            "  – spam_caps   : ≥2 exclamation marks   → 1\n"
            "  – grammar     : obvious grammar / spelling error        → 2\n"
            "  – mislead     : claim unrelated to email content        → 1\n"
            "  – too_long    : >60 characters OR >12 words             → 2\n\n"
            "Otherwise, judge overall quality on your best professional intuition:\n"
            "10 = outstanding,\n"
            "9  = excellent,\n"
            "8  = very good,\n"
            "7  = good,\n"
            "6  = above average,\n"
            "5  = average,\n"
            "4  = below average,\n"
            "3  = poor,\n"
            "2  = very poor,\n"
            "1  = terrible.\n\n"
            "Return JSON only:\n"
            "{\n"
            "  \"overall_score\": <1-10>,\n"
            "  \"hard_trigger\": \"<none | spam_caps | grammar | mislead | too_long>\",\n"
            "  \"one_sentence_why\": \"<≤ 30 words>\"\n"
            "}"
        )
        response = client.chat.completions.create(
            model="o3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Headline: \"{headline}\""}
            ],
            #temperature=0,
        )
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            score = data.get("overall_score")
            reason = data.get("one_sentence_why")
        except json.JSONDecodeError:
            print(f"Invalid JSON for '{headline}': {content}")
            return None, None
        # validate score
        if isinstance(score, int) and 1 <= score <= 10:
            return score, reason
        return None, reason
    except Exception as e:
        print(f"Error processing '{headline}': {e}")
        return None, None

def main():
    input_csv_path = "output3/merged_headlines_cleaned.csv"
    output_csv_path = "output3/headlines_with_scores.csv"

    df = pd.read_csv(input_csv_path)
    total = len(df)

    # remove old output
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    for idx, row in df.iterrows():
        headline = row["headline"]
        score, reason = get_headline_score(headline)
        print(f"[{idx+1}/{total}] \"{headline}\" -> Score: {score}, Reason: {reason}")
        time.sleep(1)

        # append this single result
        pd.DataFrame([{"headline": headline, "score": score, "reason": reason}]) \
          .to_csv(
              output_csv_path,
              mode="a",
              header=(idx == 0),  # write header only on first iteration
              index=False,
              encoding="utf-8-sig"
          )

    print(f"Done! Results saved to: {output_csv_path}")

if __name__ == "__main__":
    # 确保您已安装 pandas: pip install pandas
    main()