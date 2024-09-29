<img src="https://github.com/WouterDurnez/fence/blob/main/docs/logo.png?raw=true" alt="tests" height="200"/>

[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)
](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
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

## 💪 Features

### What can I do with Fence?

- **Uniform interface for `LLMs`**. Since our main use case was Bedrock, we built Fence to work with Bedrock models. However, it also has openAI support, and it's easy to extend to other models (contributors welcome!)
- **Links and Chains** help you build complex pipelines with multiple models. This is a feature that's been around since LangChain, and it's still here. You can parametrize templates, and pass the output of one model to another.
- **Template classes** that handle the basics, and that work across models (e.g., a MessageTemplate can be sent to a Bedrock Claude3 model, _or_ to an openAI model - system/user/assistant formatting is handled under the hood).
- **Basic utils on board** for typical tasks like retries, parallelization, logging, output parsers, etc.

### What can't I do with Fence?

It's obviously not as powerful as some of the other packages out there, that hold tons more of features. We're also not trying to fall into the trap of building 'yet another framework' (insert [XKCD](https://xkcd.com/927/) here), so we're trying to guard our scope. If you need a lot of bells and whistles, you might want to look at any of these:

- [`LangChain`](https://www.langchain.com/)

The OG, no explanation needed.

- [`Griptape`](https://www.griptape.ai)

A more recent package, with a lot of cool features! Great for building PoCs, too. Built by ex-AWS folks, and promises to be a lot more industry-oriented.

## 🛠️ How do I use it?

Fence just has a few basic components. See the [notebooks](notebooks) for examples on how to use them. Documentation is coming soon, but for now, you can check out the [source code](fence) for more details.

## 📦 Installation

You can install Fence from PyPI:

```bash
pip install fence-llm
```

## 🗺️ Roadmap

- [ ] Add more models
- [ ] Add more tests
- [ ] Add some basic design patterns (e.g., CoT, MapReduce)
- [ ] Add more tutorials
- [ ] Add more user-friendly auth methods (e.g., SigV4, OpenAI keys as params/config, etc.)

## 🤝 Contributing

We welcome contributions! Check out the [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## 🙋🏻‍♂️ Open issues or obstacles

- [ ] Bedrock access depends on user logging in themselves ➡️ Can we make this more user-friendly?
