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
    Call the o3 model to rate a single headline (1 to 5).
    Returns an integer score or None if the response is invalid.
    """
    if not isinstance(headline, str) or not headline.strip():
        return None, None
    try:
        # add system prompt for email‐marketing copy evaluation
        system_prompt = (
            "You are an email-marketing copy expert.\n\n"
            "For ONE advertising email headline, mark each of the 7 dimensions as PASS or FAIL.\n"
            "A dimension passes only if the description is fully met.\n\n"
            "Dimensions\n"
            "1. attractiveness – compelling hook that draws immediate interest\n"
            "2. relevance_accuracy – headline truthfully reflects email content\n"
            "3. brevity – ≤ 45 characters AND ≤ 9 words\n"
            "4. action_urgency – clear, genuine CTA or real time/scarcity signal\n"
            "5. persona_fit – wording & emotion suit the given persona and feel personalised\n"
            "6. clarity_fluency – natural grammar, easy to read\n"
            "7. credibility_originality – no spam traits, clichés, clickbait, ALL-CAPS abuse\n\n"
            "Scoring\n"
            "• Let **n_pass** be the number of dimensions that pass.\n"
            "• overall_score = 5 if n_pass ≥ 6 4 if n_pass = 5 3 if n_pass = 3–4 2 if n_pass = 2 1 if n_pass ≤ 1\n"
            "• If credibility_originality fails, overall_score must not exceed 2.\n\n"
            "Return **JSON only**:\n"
            "{\n"
            "  \"overall_score\": <1-5>,\n"
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
        if isinstance(score, int) and 1 <= score <= 5:
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