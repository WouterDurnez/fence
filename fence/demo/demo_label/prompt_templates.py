"""
Base templates for the LLM that can be used to build more complex, modular prompts, Links and Chains.
"""

POLICY_TEMPLATE = """
Review the following filename to make it adhere to the policy or policies that follow. Then return the reviewed filename in triple backticks, like so:

```
*reviewed_filename*
```

These are the policies you need to adhere to:

{% for policy in recipe['policy'] %}
    - {{ policy }}
{% endfor %}

Additional guidelines:
- Be conservative with your changes. Only make the necessary changes to the filename.

As a final reminder, it is VITAL that you stick to EVERY POLICY!

This is the filename to be reviewed:

```{{ state }}```

"""
