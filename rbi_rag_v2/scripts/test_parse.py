import re

def count_questions(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'### \d+\. Question:', content)
    print(f"{filepath} has {len(blocks) - 1} questions.")

count_questions('test_results.md')
count_questions('test_results_grok.md')
