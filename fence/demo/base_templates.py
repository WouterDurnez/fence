"""
Base templates for the LLM that can be used to build more complex, modular prompts, Links and Chains.
"""

BASE_TEMPLATE = """
Review the following text according to the instructions that follow. Then return the reviewed text in triple backticks, like so:

```
This is the reviewed text.
```

The instructions are as follows:
{% if 'flavor' in recipe %}
    - Rewrite the text to sound more {{ recipe['flavor'] }}.
{% endif %}

{% if 'verbosity' in recipe %}
    - Make the text {{ recipe['verbosity'] }} for better understanding. {% if recipe['verbosity'] == 'shorter' %}Reduce the number of sentences, while keeping the meaning of the text intact as much as possible.{% elif recipe['verbosity'] == 'longer' %}Expand on the message and go into detail when possible.{% endif %}
{% endif %}

{% if 'spelling' in recipe and recipe['spelling'] %}
    - Ensure there are no spelling mistakes in the text. If there are, correct them, and add the mistakes and corrections to the TOML response under the 'errors' key.
{% endif %}

{% if 'policy' in recipe %}
    - Make sure you adhere to the following policy: \n<policy>{{ recipe['policy'] }}</policy>
{% endif %}

This is the text to be reviewed:\n

<input>{{ state }}</input>

Again, your response should be in triple backticks, like so:

```
This is the reviewed text.
```

Do not add any other tags, such as the <input> and </input> tags!

{% if 'verbosity' in recipe %}
Do not forget to {{ recipe['verbosity'] }} the text as much as possible!
{% endif %}

{% if 'flavor' in recipe %}
Remember, the tone of the next text should be {{ recipe['flavor'] }}.
{% endif %}

{% if 'policy' in recipe %}
As a final reminder, it is VITAL that you stick to EVERY POLICY : \n<policy>\n{{ recipe['policy'] }}\n</policy>\n\n
{% endif %}
"""

FLAVOR_TEMPLATE = """
Rewrite the following text to sound more {{ recipe['flavor'] }}. Then return the reviewed text in triple backticks, like so:

```
This is the reviewed text.
```

This is the text to be reviewed:

<input>{{ state }}</input>

Again, your response should be in triple backticks, like so:

```
This is the reviewed text.
```


Additional guidelines:
- You must keep the text around the same length! You will be penalized for adding or removing too many words!
- Do not add any other tags, such as the <input> and </input> tags! 

{% if 'flavor' in recipe %}
Remember, the tone of the reviewed text should be {{ recipe['flavor'] }}.
{% endif %}

"""

VERBOSITY_TEMPLATE = """
Review the following text for better understanding. {% if recipe['verbosity'] == 'shorter' %}Strongly the number of sentences, while keeping the tone and meaning of the text intact as much as possible.{% elif recipe['verbosity'] == 'longer' %}Expand on the message and go into detail when possible.{% endif %}

Then return the reviewed text in triple backticks, like so:

```
This is the reviewed text.
```

This is the text to be reviewed:

<input>{{ state }}</input>

Again, your response should be in triple backticks, like so:

```
This is the reviewed text.
```

Additional guidelines:
- You must keep the tone of the text intact! You will be penalized for changing the tone too much!
- Do not add any other tags, such as the <input> and </input> tags! 


{% if 'flavor' in recipe %}
Do not forget to make the reviewed text significantly {{ recipe['verbosity'] }}!
{% endif %}
"""

POLICY_TEMPLATE = """
Review the following text to make it adhere to the policy or policies that follow. Then return the reviewed text in triple backticks, like so:

```
This is the reviewed text.
```

These are the policies you need to adhere to:

{% for policy in recipe['policy'] %}
    - {{ policy }}
{% endfor %}

This is the text to be reviewed:

<input>{{ state }}</input>

Again, your response should be in triple backticks, like so:

```
This is the reviewed text.
```


Additional guidelines:
- You must keep the text around the same length! You will be penalized for adding or removing too many words!
- You must keep the tone of the text intact! You will be penalized for changing the tone too much!
- Do not add any other tags, such as the <input> and </input> tags! 


As a final reminder, it is VITAL that you stick to EVERY POLICY!
"""

SPELLING_TEMPLATE = '''
Review the following text for spelling mistakes. If there are any, correct them. Return the corrected text AND the spelling mistakes in as TOML, delimited by triple backticks. The corrected text should be under an 'output' key, and the spelling mistakes should be under an 'errors' key. 

Make sure to put the corrected text in triple quotes, even if it is only one line long. This makes it easier to copy-paste the text into the TOML response. 

For example, if the input was: "Thsi is the orignal text.", the output should be:

```
output = """This is the original text."""
errors = ["Thsi", "orignal"]
```

If the input has no spelling mistakes, for instance in the case of "This is the original text.", the output should be:

```
output = """This is the original text."""
errors = []
```

This is the text to be reviewed:

<input>{{ state }}</input>

Again, your response should be valid TOML in triple backticks, like so:

```
output= """*the reviewed text*"""
errors = [*list of spelling mistakes*]
```

Additional guidelines:
- You must keep the text around the same length! You will be penalized for adding or removing too many words!
- You must keep the tone of the text intact! You will be penalized for changing the tone too much!
- Do not add any other tags, such as the <input> and </input> tags!
'''

BASE_TEMPLATE_TOML = '''
Review the following text and return the reviewed text in TOML format, under a 'reviewed' key. Repeat the original text in the response under an 'input' key.
  
{% if 'flavor' in recipe %}
    Rewrite the text to sound more {{ recipe['flavor'] }}.
{% endif %}

{% if 'verbosity' in recipe %}
    Make the text {{ recipe['verbosity'] }} for better understanding. {% if recipe['verbosity'] == 'shorter' %}Reduce the number of sentences, while keeping the meaning of the text intact as much as possible.{% elif recipe['verbosity'] == 'longer' %}Expand on the message and go into detail when possible.{% endif %}
{% endif %}

{% if 'spelling' in recipe and recipe['spelling'] %}
    Ensure there are no spelling mistakes in the text. If there are, correct them, and add the mistakes and corrections to the TOML response under the 'errors' key.
{% endif %}

{% if 'policy' in recipe %}
    Make sure you adhere to the following policy: \n<policy>{{ recipe['policy'] }}</policy>\n\n Importantly, add any part
    of the text that does not adhere to the policy to the json response under the 'policy_violations' key.
{% endif %}

This is the text to be reviewed:\n

<input>{{ state }}</input>

Make sure not to include the <input> and </input> delimiters in the output. Again, your response should be in TOML format, with the reviewed text under the 'reviewed' key, as such:

```
input = """This is the original text, not including the <input> and </input> delimiters."""
reviewed = """This is the reviewed text, not including the <input> and </input> delimiters."""
errors = [*list of spelling mistakes*]
policy_violations = [*list of sentences or phrases that violate the policy*]
```

ALWAYS put strings in triple quotes, even if they are only one line long. This makes it easier to copy-paste the text into the TOML response.

{% if 'flavor' in recipe %}
Do not forget to {{ recipe['verbosity'] }} the text as much as possible!
{% endif %}

{% if 'flavor' in recipe %}
Remember, the tone of the next text should be {{ recipe['flavor'] }}.
{% endif %}

{% if 'policy' in recipe %}
As a final reminder, it is VITAL that you stick to EVERY POLICY : \n<policy>\n{{ recipe['policy'] }}\n</policy>\n\n
{% endif %}
'''
