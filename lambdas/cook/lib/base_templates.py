base_template = '''
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

<input>{{ input }}</input>

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
