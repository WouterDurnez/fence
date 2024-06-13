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
Text: ```{text}```
Question: ```{question}```

Output:
"""

COMBINE_TEMPLATE = """You are a helpful assistant, in charge of answering questions on the basis of a piece of text. You will be given a set of answers to a question, each based on a different chunk of text. Your task is to combine the answers into a refined single answer.

GUIDELINES:
-----------
- Always return your output in triple backticks. You WILL BE PENALIZED if you do not do this.
- If the answers are contradictory, you should state ```<contradictory>```, and nothing more.
- If the answers are not contradictory, you should return the combined answer.
- It is important to eliminate redundancy and repetition in the combined answer.
- Make sure the answer is concise and to the point. Do not return more than a few sentences. Summarize the answers if necessary.
- Use full sentences and proper grammar in the combined answer.
- Do not refer to things like 'the text', 'the document', or 'the passage' in the combined answer. The combined answer should be a standalone response to the question, as if you answered it directly.
- It is possible the answer list only contains one answer. In this case, rewrite the answer in a way that matches the guidelines, which includes making it a standalone response.

EXAMPLES:
--------

Input answers:
```
- No, it's impossible to both win and lose a game of chess at the same time.
- Yes, if the game is being played in different parallel universes, you could be winning in one and losing in another.
```

Output:
<reasoning>
The answers are contradictory. Therefore, the combined answer is <contradictory>.
</reasoning>

```<contradictory>```

---

Input answers:
```
- Cognitive Biases as Systematic Patterns of Deviation: Cognitive biases are systematic patterns of deviation from norm or rationality in judgment, often stemming from mental shortcuts and heuristics that lead individuals to make decisions or draw conclusions in a predictable, yet potentially flawed, manner.
- Cognitive Biases as Adaptive Mental Strategies: Alternatively, cognitive biases can be seen as adaptive mental strategies that have evolved to help individuals process information efficiently. These biases may have provided survival advantages in certain contexts, allowing quick decision-making in situations where immediate action was necessary.```
```

Output:
<reasoning>
The answers are not contradictory. Therefore, the combined answer is a combination of the two answers. I will combine the two answers into a single answer."
</reasoning>

```Cognitive biases encompass both systematic patterns of deviation from rationality, arising from mental shortcuts and heuristics, and adaptive mental strategies that have evolved to facilitate efficient information processing. These biases can manifest as predictable deviations from normative decision-making, while simultaneously serving as adaptive mechanisms honed by evolution to enable swift and effective responses in specific contexts.```

---

Input answers:
```
- The term "fence" referred to the art of defense in the Middle Ages.
```

Output:
<reasoning>
There is only one answer. It is concise and to the point. I don't need to combine or rewrite it.
</reasoning>

```The term "fence" referred to the art of defense in the Middle Ages.```

---

Input answers:
- The text mentions plans for 2024 include iterating on 2023 features, replacing the outdated UI/UX based on research, and becoming a leader in the Revenue Enablement wave as per the Forrester Wave.
- Alignment with WebApp UI, Experiences Library, Dashboard/Home, Admin settings, and AI were mentioned as topics that are candidates for the roadmap in 2024 H1 (first half of 2024).
```

Output:
<reasoning>
There are two answers. I will remove any references to any unseen text or document to make the answer stand on its own. I will condense the information to make it more concise while still conveying the main points effectively.
</reasoning>

```In 2024, plans involve iterating on features from 2023, updating the UI/UX based on research findings, and aiming to lead in Revenue Enablement according to Forrester Wave. Focus areas for the first half of 2024 include WebApp UI, Experiences Library, Dashboard/Home, Admin settings, and AI integration.```

---

EXAMPLES OF WHAT NOT TO DO:
--------------------------
Input answers:
```
- According to the text, the sky is blue because of the way the atmosphere scatters sunlight.
- The sky is blue because of the way the atmosphere scatters sunlight.
```

Output:
<reasoning>
The answers are redundant. I will combine them into a single answer.
</reasoning>

```According to the text, the sky is blue because of the way the atmosphere scatters sunlight.```

This  final output is bad because it contains a reference to the text. The combined answer should be standalone and not refer to the text. The correct output could be:

```The sky is blue due to the way the atmosphere scatters sunlight.```

---
Here are the input answers:
```
{search_results}
```

Output:

```
"""
