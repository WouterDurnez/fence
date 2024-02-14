FAQ_TEMPLATE = '''You are a helpful assistant, in charge of generating FAQs that can easily be parsed to be used in an online platform.

Below, delimited by triple backticks, is a text. Summarize the text in 2 or more sentences. Do not add newlines or bullet points.

Next, formulate {{ number_of_questions }} open and general question about the given text above. The questions will be used in an FAQ section.

Give an answer to all of your questions using the knowledge from the text above. The answers should be concise. It should be no more than two sentences. Do not add newlines or bullet points.

Also provide the part of the text that contains the answer to the question.

Importantly, you will be penalized if the answer to the question cannot be retrieved directly from the text. Any made up additional information will be penalized as well.

The output should be valid TOML in triple backticks, Here's an example (for 2 questions, adjust the number of questions accordingly):

```
summary = """This is a summary of the content."""

[[qa_pairs]]
question = """Is this a first question?"""
answer = """Yes, and this is a first answer"""
context = """This is the passage in the text that answers the first question."""

[[qa_pairs]]
question = """Is this a second question?"""
answer = """Yes, and this is a second answer"""
context = """This is the passage in the text that answers the first question."""
```

- Do not deviate from the example response format! Importantly, you MUST use triple quotes for the summary and answers, and using the TOML list syntax for the questions and answers. You will be penalized for deviating from the format.

- Make the summary and {{ number_of_questions }} questions if possible! All the questions together should cover all the information in the text. The summary should cover all the information as well.

- Do not start answers with references to the text, such as "According to the text,". The answers should be standalone. You will be penalized for adding such references.

- Wrap ALL VALUES in triple quotes, even if they are single words or numbers. You will be penalized for not doing so. This helps to ensure that the output is valid TOML, and that the parser can handle the output.

For example, if the text is:

```
The sky is blue.
```

This would be a bad answer:

```
[[qa_pairs]]
question = """What color is the sky?"""
answer = """The text mentions that the sky is blue."""
context = """The sky is blue."""
```
The answer is bad because it says 'The text mentions', which does not make it sound standalone. This would be a good answer:
```
[[qa_pairs]]
question = """What color is the sky?"""
answer = """The color of the sky is blue."""
context = """The sky is blue."""
```

This is the text to be summarized and questioned:

```{{ state }}```

Remember, stick to the TOML output format! 
'''

SUMMARY_TEMPLATE = """You are a helpful assistant in charge of generating high-level meta summaries.

Below, delimited by triple backticks, is a a series of summaries, generated over consecutive chunks of text. Your task is to generate a new summary that covers all the information in the summaries below. Make it sound as a standalone text. You will be penalized for anything that makes it sound like the new text is a summary of a larger text. Return the summary in triple backticks.

Here are the summaries:

```
{% for summary in summaries %}
{{ summary }}
{% endfor %}
```

Now please generate a new summary that covers the essence of the summaries above. You are not allowed to exceed 3 sentences! You will be penalized for adding more.
 
Return the summary in triple backticks. I will tip generously if you make it sound like a standalone text.
"""

if __name__ == "__main__":
    from fence.src.llm.templates import PromptTemplate

    summary_template = PromptTemplate(
        template=SUMMARY_TEMPLATE, input_variables=["summaries"]
    )

    print(
        summary_template.render(
            {"summaries": ["This is the first summary.", "This is the second summary."]}
        )
    )
