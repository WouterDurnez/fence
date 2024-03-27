"""
Base templates for the LLM that can be used to build more complex, modular prompts, Links and Chains.
"""


##########
# Common #
##########

CONTEXTS = {
    "email-template-composer": "The text you are given is an email. The text might contain product names or specific concepts. If you identify them as such, do not change them - keep them as is.",
    "shared-space-comment": "The text you are given is a comment.",
    "shared-space-invitation": "The text you are given is an invitation message.",
}

GUIDELINES = """
- You may see placeholder tags, such as <placeholder0>, <placeholder1>, etc., in the input. These MUST be kept in the output. Do not remove or alter these tags if they were present already! Do NOT add any new ones! I will tip generously if you do a good job!
- Make sure to use proper grammar, punctuation, and spelling.
"""

####################
# Recipe Templates #
####################

TONE_TEMPLATE = """CONTEXT: You are an assistant that helps with text revision. {{ context }}

Rewrite the following text to sound more {{ modifier }}. Then return the reviewed text in triple backticks, like so:

```
This is the reviewed text.
```

GUIDELINES:
- You must keep the text around the same length! You will be penalized for adding or removing too many words!
{{ guidelines }}

Remember, the tone of the reviewed text should be {{ modifier }}.

This is the text to be reviewed, delimited by triple backticks:

```{{ state }}```

REMEMBER: Make sure placeholders, i.e. <placeholder0>, <placeholder1>, etc., are RETAINED in the reviewed text!!! You will be penalized if you remove or alter them!

"""

VERBOSITY_TEMPLATE = """CONTEXT: You are an assistant that helps with text revision. {{ context }}

Adjust the length of the following text. {% if modifier == 'shorter' %}You must reduce the text to half its length, while keeping the tone and meaning of the text intact as much as possible.{% elif modifier == 'longer' %}You must extend the text with about half the original length. Do NOT make anything up just to make it longer! If the information was not in the original text, it should not be added to the reviewed text. That includes dates, locations, etc.{% endif %}

You are allowed to reason, before you start, about the text and the changes you are about to make. This will help you to keep the tone and meaning of the text intact. This reasoning should be included in the response. Then return the final, revised text in triple backticks, like so:

```
This is the revised text.
```

GUIDELINES:
- You must keep the tone of the text intact! You will be penalized for changing the tone too much!
{{ guidelines }}

Do not forget to make the reviewed text about two times {{ modifier }}. Do not overdo this! You will be penalized if you do not adhere to this instruction! I will tip generously if you do a good job!

EXAMPLE:
{% if modifier == 'shorter' %}
Input: ```Good day, sir. I hope this message finds you well. I am writing to inform you that the meeting has been postponed to next week.```
Output: Reasoning: I will carefully examine the original text to identify redundant phrases, unnecessary details, and areas where sentences can be combined without losing the main idea. By doing so, I aim to maintain the essence of the message while reducing the overall length by half.
Revised text: ```Good day, sir. The meeting has been postponed to next week.```
{% elif modifier == 'longer' %}
Input: ```Good day, sir. The meeting has been postponed to next week.```
Output: Reasoning: I will carefully examine the original text to identify areas where sentences can be expanded without losing the main idea. By doing so, I aim to maintain the essence of the message while extending the overall length by half.
Revised text: ```Good day, sir. I am writing to inform you that the meeting has been postponed to next week. I hope this message finds you well.```
{% endif %}

EXAMPLES OF WHAT NOT TO DO:
{% if modifier == 'shorter' %}
Input: ```Good day, sir. I hope this message finds you well. I am writing to inform you that the meeting has been postponed to next week.```
Output: Revised text: ```Good day, sir. I hope this message finds you well. I am writing to inform you that the meeting has been postponed to next week.```

This is not a good example of a shorter text. The text is not reduced to half its length.
{% elif modifier == 'longer' %}
Input: ```Good day, sir. The meeting has been postponed to next week.```
Output: Revised text: ```Good day, sir. The meeting on <placeholder0> has been postponed to next week. I hope this message finds you well.```

The text contains a placeholder tag that was not present in the original text.
{% endif %}


This is the text to be reviewed:

```{{ state }}```

REMEMBER: Make sure placeholders, i.e. <placeholder0>, <placeholder1>, etc., are RETAINED in the reviewed text!!! You will be penalized if you remove or alter them!

"""

POLICY_TEMPLATE = """CONTEXT: You are an marketing assistant that helps with text revision. You review text to make sure it is in line with company branding and styling policies. {{ context }}

Below, you will find a list of policies you should check the input against. Review the text to ensure it adheres to the policies. If it does not, make the necessary changes to bring it in line with the policies. However, if it does adhere to the policies, do not make any changes! You should always be able to justify your changes.

Once you are done, return the reviewed text in triple backticks, like so:

```
This is the reviewed text.
```

Here is the structure of the policies:

<policy>
This is the guideline or policy you should check the input against.

<positive_examples>
How it should be done: an optional list of examples that are in line with the policy.
</positive_examples>

<negative_examples>
How it should not be done: an optional list of examples that do not adhere to the policy.
</negative_examples>
</policy>

EXAMPLE:

If the input is: ```Last night, a grand gala was held in New York, organized by a renowned charity. Lavish decorations adorned the venue, and guests were served exquisite cuisine. Throughout the evening, speeches were given, and substantial donations were made. Memories were made, and the gala's success was celebrated by all.``` and the policies are:

<policy>
Use active voice.

<positive_examples>
- The team completed the project.
</positive_examples>

<negative_examples>
- The project was completed by the team.
</negative_examples>

</policy>

You should notice that the input uses passive voice. You should revise the text to use active voice. The revised text should be: ```Last night, the renowned charity organized a grand gala in New York. They adorned the venue with lavish decorations and served exquisite cuisine to the guests. Throughout the evening, speakers delivered speeches, and attendees made substantial donations. Everyone celebrated the gala's success and made lasting memories.```

Here are the policies that currently apply.

POLICIES:

{{ modifier }}

ADDITIONAL GUIDELINES:
{{ guidelines }}

Now it's time to review the text against the current list of policies. You are allowed to reason, before you start, about the text and the changes you are about to make. This will help you to keep the tone and meaning of the text intact. This reasoning should be included in the response, in <reasoning></reasoning> tags. Then return the final, revised text in triple backticks.

This is the text to be reviewed, delimited by triple backticks:

```{{ state }}```

Your reviewed version of the input, altered only where necessary:

"""

CORRECTION_TEMPLATE = '''CONTEXT: You are an assistant that helps with text revision. {{ context }}

Review the following text for spelling and grammar mistakes. If there are any, correct them. Return the corrected text AND the mistakes in as TOML, delimited by triple backticks. The corrected text should be under a 'state' key, and the mistakes should be under an 'errors' key. 

Make sure to put the corrected text in triple quotes, even if it is only one line long. This makes it easier to copy-paste the text into the TOML response. 

For example, if the input was: "Thsi is the orignal text.", the output should be:

```
state = """This is the original text."""
errors = ["Thsi", "orignal"]
```

If the input has no spelling or grammar mistakes, for instance in the case of "This is the original text.", the output should be:

```
state = """This is the original text."""
errors = []
```

Again, your response should be valid TOML in triple backticks, like so:

```
state= """*the reviewed text*"""
errors = [*list of mistakes*]
```

GUIDELINES:
- You must keep the text around the same length! You will be penalized for adding or removing too many words!
- You must keep the tone of the text intact! You will be penalized for changing the tone too much!
{{ guidelines }}

This is the text to be reviewed, delimited by triple backticks:

```{{ state }}```

REMEMBER: Make sure placeholders, i.e. <placeholder0>, <placeholder1>, etc., are RETAINED in the reviewed text!!! You will be penalized if you remove or alter them!

'''
