import re
import csv

def clean_headlines(input_file, output_file):
    """
    Cleans the headline data from the input CSV file and writes it to the output CSV file.

    Cleaning steps:
    1. Removes lines with specific unwanted patterns or instructional text.
    2. Removes lines containing first-person pronouns.
    3. Strips leading/trailing whitespace.
    4. Removes ALL double quotes and Markdown bold markers (**).
    5. Skips empty lines.
    """
    # Patterns that identify entire lines to be removed.
    unwanted_patterns = [
        r'\|',  # Contains a vertical bar
        r'^\s*\.{3,}\s*$',  # Line consists only of '...'
        r'\[Brand\]',  # Contains a placeholder like [Brand]
        r'so need to format accordingly',
        r'Do NOT use first-person words',
        r'The suspense is tasty'
    ]

    # Regex to find first-person pronouns as whole words.
    first_person_regex = re.compile(r"\b(I|we|my|our|I'm|I've|I'll|I'd)\b", re.IGNORECASE)

    cleaned_rows = []

    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            
            # Read and store the header
            header = next(reader)
            cleaned_rows.append(header)

            for row in reader:
                if not row:
                    continue # Skip empty rows

                headline = row[0]
                
                # 1. Check if the line should be skipped entirely
                if any(re.search(pattern, headline) for pattern in unwanted_patterns):
                    continue
                
                if first_person_regex.search(headline):
                    continue

                # 2. Clean the content of the line
                # Strip leading/trailing whitespace
                cleaned_headline = headline.strip()
                # Remove ALL double quotes
                cleaned_headline = cleaned_headline.replace('"', '')
                # Remove markdown bold markers
                cleaned_headline = cleaned_headline.replace('**', '')

                # 3. Add the cleaned row to our list
                if cleaned_headline: # Ensure it's not empty after cleaning
                    cleaned_rows.append([cleaned_headline])

        # Write to a new CSV, but disable quoting to prevent re-adding quotes
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, quoting=csv.QUOTE_NONE, escapechar='\\')
            writer.writerows(cleaned_rows)

        print(f"Cleaning complete. Cleaned data saved to '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_csv = 'all_headlines_shuffled.csv'
    output_csv = 'all_headlines_cleaned.csv'
    clean_headlines(input_csv, output_csv)