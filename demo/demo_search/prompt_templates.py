SEARCH_CHUNK_TEMPLATE = """You are a helpful assistant, in charge of answering questions on the basis of a piece of text. You will be given a chunk of text and a question. Your task is to provide an answer to the question based on the information in the text. 

GUIDELINES:
-----------
- Always return your output in triple backticks. You WILL BE PENALIZED if you do not do this.
- You can ONLY use the information in the text to answer the question.
- If the answer is not in the text, you should answer with ```<not in text>```.
- If the document appears irrelevant, simply state ```<not in text>```.
- You are allowed to think step-by-step and add your thoughts within HTML-style <reasoning> tags. You can use the <reasoning> tags multiple times, if you want to add more thoughts. 
- Carefully inspect the example output to understand the expected format of the answer.

EXAMPLES:
--------

Text: ```"In the Middle Ages, the term "fence" referred to the art of defense and not to the art of combat. The art of combat was referred to as "fighting"."```

Question: ```"What did the term "fence" refer to in the Middle Ages?"```

Output: 
<reasoning>
The text states that the term "fence" referred to the art of defense in the Middle Ages. This is the answer to the question.
</reasoning>

```The term "fence" referred to the art of defense in the Middle Ages.```

Text: ```"In the Middle Ages, the term "fence" referred to the art of defense and not to the art of combat. The art of combat was referred to as "fighting"."```
Question: ```"Why is the sky blue?"```

Output:
<reasoning>
The answer to the question is not in the text. Therefore, the answer is <not in text>.
</reasoning>

```<not in text>```

INPUT:
------
Text: ```{{ text }}```
Question: ```{{ question }}```

Output:
"""

COMBINE_TEMPLATE = """You are a helpful assistant, in charge of answering questions on the basis of a piece of text. You will be given a set of answers to a question, each based on a different chunk of text. Your task is to combine the answers into a single answer. 

GUIDELINES:
-----------
- Always return your output in triple backticks. You WILL BE PENALIZED if you do not do this.
- If the answers are contradictory, you should state ```<contradictory>```, and nothing more.
- If the answers are not contradictory, you should return the combined answer.
- It is important to eliminate redundancy and repetition in the combined answer.
- Use full sentences and proper grammar in the combined answer.

EXAMPLES:
--------

Input answers:
```
1. No, it's impossible to both win and lose a game of chess at the same time.
2. Yes, if the game is being played in different parallel universes, you could be winning in one and losing in another.
```

Output:
<reasoning>
The answers are contradictory. Therefore, the combined answer is <contradictory>.
</reasoning>

```<contradictory>```

--- 

Input answers:
```
1. Cognitive Biases as Systematic Patterns of Deviation: Cognitive biases are systematic patterns of deviation from norm or rationality in judgment, often stemming from mental shortcuts and heuristics that lead individuals to make decisions or draw conclusions in a predictable, yet potentially flawed, manner.
2. Cognitive Biases as Adaptive Mental Strategies: Alternatively, cognitive biases can be seen as adaptive mental strategies that have evolved to help individuals process information efficiently. These biases may have provided survival advantages in certain contexts, allowing quick decision-making in situations where immediate action was necessary.```
```

Output:
<reasoning>
The answers are not contradictory. Therefore, the combined answer is a combination of the two answers. I will combine the two answers into a single answer."
</reasoning>

```Cognitive biases encompass both systematic patterns of deviation from rationality, arising from mental shortcuts and heuristics, and adaptive mental strategies that have evolved to facilitate efficient information processing. These biases can manifest as predictable deviations from normative decision-making, while simultaneously serving as adaptive mechanisms honed by evolution to enable swift and effective responses in specific contexts.```

--- 

Here are the input answers:
```
{% for output in search_chunk_outputs %}{{ loop.index0 }}. {{ output }}\n{% endfor %}
```

Output:

"""


if __name__ == "__main__":
    from fence.templates import PromptTemplate

    combine_link = PromptTemplate(
        template=COMBINE_TEMPLATE, input_variables=["search_chunk_outputs"]
    )

    # Test search_chunk_link
    search_output_snippets = [
        "The term 'fence' referred to the art of defense in the Middle Ages.",
        "The word 'fence' refers to a style of combat with roots that stem from centuries ago.",
        "The term 'fence' referred to a boundary between different sections of an area.",
    ]

    print(
        combine_link.render(input_dict={"search_chunk_outputs": search_output_snippets})
    )
