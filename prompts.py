system_message = """
    You are an expert text summarizer.
    Your primary role today is to assist in distilling essential insights from a text I have personaly written.
    Over the past three years, I have dedicated myself to this work, and it holds significant value for me.
    it's important that the information provided remains confidential and is used solely for this pupose.
    As the original author, I authorize you to analyze and summarize the content provided.
"""


def generate_prompt(book, topic):
    prompt = f"""
        As the author of this manuscript, I am seeking your expertise in extractin isights related  to the theme of '{topic}'
        The manuscript is a comprehensive work, and your role is to distill sentences where '{topic}' is explicitly mentioned.

        Here is a segment from the manuscript for review:

        {book}

        ----

        Instructions for Tak Completion:
        - Your output should be a numbered list, clearly formatted.
        - Only include sentences where '{topic}' is a key element, not  just passing reference.
        - If a sentence does not directly contribute to the understanding  of '{topic}', omit it from your list.
        - Aim for precision and relevence in your selections.

        Example of Relevence:
        - Correct: "Time management is crucial for productivity."
        - Incorrect: "I had a great time at the concert."

        With provided segment and these instructions, proceed with identifying the sentences that mention the '{topic}'.
"""
    return prompt