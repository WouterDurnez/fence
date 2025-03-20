<img src="https://github.com/WouterDurnez/fence/blob/main/docs/logo.png?raw=true" alt="tests" height="200"/>

[![Python](https://img.shields.io/pypi/pyversions/fence-llm)](https://pypi.org/project/fence-llm/)
[![Test Status](https://github.com/WouterDurnez/fence/actions/workflows/ci-pipeline.yaml/badge.svg)](https://github.com/WouterDurnez/fence/actions)
[![codecov](https://codecov.io/gh/WouterDurnez/fence/branch/main/graph/badge.svg?token=QZQZQZQZQZ)](https://codecov.io/gh/WouterDurnez/fence)
[![PyPI version](https://badge.fury.io/py/fence-llm.svg)](https://badge.fury.io/py/fence-llm)
[![Documentation Status](https://readthedocs.org/projects/fence-llm/badge/?version=latest)](https://fence-llm.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

# 🤺 Fence

`Fence` is a simple, lightweight library for LLM communication. A lot of the functionality was inspired by/derived of LangChain (the OG LLM package) basics, since that's how the package was born - as a stripped down version of LangChain functionality, with cooler names.

## 🤔 Raison d'être

The simple answer: by accident. The slightly longer answer: LangChain used to be (is?) a pretty big package with a ton of dependencies. The upside is that it's powerful for PoC purposes, because it has it all.

The downsides:

- It's **_big_**. It takes up a lot of space (which can be an issue in some environments/runtimes), often for functionality that isn't needed.
- It's fairly **_complex_**. It's a big package with a lot of functionality, which can be overwhelming for new users.
- It **_wasn't exactly dependable_** in an industrial setting before. Version jumps were common, and the package was often broken after a new release.

As a result, many developers (particularly those working in large production environments) have advocated for more lightweight, custom functionality that favors stability and robustness.

### Circling back: why Fence?

Since our work was in a production environment, mostly dealing with Bedrock, we just started building some **basic components** from scratch. We needed a way to communicate with our models, which turned out to as the `Link` class (_wink wink_).
Then, some other things were added left and right, and this eventually turned into a miniature package. Not in small part because it was fun to go down this road. But mostly because it strikes the right balance between convenience and flexiblity.

Naturally, it's nowhere as powerful as, for instance, LangChain. If you want to build a quick PoC with relatively complex logic, maybe go for the OG instead. If you want to be set on your way with a simple, lightweight package that's easy to understand and extend, Fence might be the way to go.

## 🛠️ How do I use it?

Fence just has a few basic components. See the [notebooks](notebooks) for examples on how to use them. Documentation is coming soon, but for now, you can check out the [source code](fence) for more details.

## 📦 Installation

You can install Fence from PyPI:

```bash
pip install fence-llm
```

## 👋 Look ma, no dependencies (kinda)!

Here's a hello world example:

```python
from fence.links import Link
from fence.templates.string import StringTemplate
from fence.models.openai import GPT4omini

# Create a link
link = Link(
    model=GPT4omini(),
    template=StringTemplate("Write a poem about the value of a {topic}!"),
    name='hello_world_link'
)

# Run the link
output = link.run(topic='fence')['state']
print(output)
```

This will output something like:

```bash
[2024-10-04 17:45:15] [ℹ️ INFO] [links.run:203]              Executing <hello_world_link> Link
Sturdy wood and nails,
Boundaries draw peace and calm,
Guarding hearts within.
```

Much wow, very next level. There's more in the [notebook](notebooks) section, with a _lot_ more to cover!

## 💪 Features

### What can I do with Fence?

- **Uniform interface for `LLMs`**. Since our main use case was Bedrock, we built Fence to work with Bedrock models. However, it also has openAI support, and it's easy to extend to other models (contributors welcome!)
- **Links and Chains** help you build complex pipelines with multiple models. This is a feature that's been around since LangChain, and it's still here. You can parametrize templates, and pass the output of one model to another.
- **Template classes** that handle the basics, and that work across models (e.g., a MessageTemplate can be sent to a Bedrock Claude3 model, _or_ to an openAI model - system/user/assistant formatting is handled under the hood).
- **Agents** to move on to the sweet, sweet next level of LLM orchestration. Built using the ReAct pattern.
- **Basic utils on board** for typical tasks like retries, parallelization, logging, output parsers, etc.

## 🤝 Contributing

We welcome contributions! Check out the [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
