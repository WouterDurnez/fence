FAQ_TEMPLATE = '''Below, delimited by triple backticks, is a text. Summarize the text in 2 or more sentences. Do not add newlines or bullet points.

Next, formulate {{ number_of_questions }} open and general question about the given text above. The questions will be used in an FAQ section.

Give an answer to all of your questions using the knowledge from the text above. The answers should be concise. It should be no more than two sentences. Do not add newlines or bullet points.

Importantly, you will be penalized if the answer to the question cannot be retrieved directly from the text. Any made up additional information will be penalized as well.

The output should be valid TOML in triple backticks, Here's an example (for 2 questions, adjust the number of questions accordingly):

```
summary = """This is a summary of the content."""

[[qa_pairs]]
question = """Is this a first question?"""
answer = """Yes, and this is a first answer"""

[[qa_pairs]]
question = """Is this a second question?"""
answer = """Yes, and this is a second answer"""
```

Do not deviate from the example response format! Importantly, you MUST use triple quotes for the summary and answers, and using the TOML list syntax for the questions and answers. You will be penalized for deviating from the format.
 
Make the summary and 3 questions if possible! All the questions together should cover all the information in the text. The summary should cover all the information as well.

This is the text to be summarized and questioned:

```{{ state }}```

Remember, stick to the TOML output format! I will tip generously if you make it sound like a standalone text.
'''

SUMMARY_TEMPLATE = """Below, delimited by triple backticks, is a a series of summaries, generated over consecutive chunks of text. Your task is to generate a new summary that covers all the information in the summaries below. Make it sound as a standalone text. You will be penalized for anything that makes it sound like the new text is a summary of a larger text. Return the summary in triple backticks.

Here are the summaries:

```
{% for summary in summaries %}
{{ summary }}
{% endfor %}
```

Now please generate a new summary that covers the essence of the summaries above. You are not allowed to exceed 3 sentences! You will be penalized for adding more.
 
Return the summary in triple backticks. I will tip generously if you make it sound like a standalone text.
"""

if __name__ == '__main__':

    from fence.src.llm.templates import PromptTemplate

    summary_template = PromptTemplate(template=SUMMARY_TEMPLATE, input_variables=['summaries'])

    print(summary_template.render({'summaries': ['This is the first summary.', 'This is the second summary.']}))

