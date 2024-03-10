"""
Base templates for the LLM that can be used to build more complex, modular prompts, Links and Chains.
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
