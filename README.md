<img src="docs/logo.png" alt="tests" height="200"/>

[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)
](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# 🤺 Fence

No, not that kind of fence. The once with the interlinked pieces of metal. Like a chain fence?

`Fence` is a simple, lightweight library for LLM communication. A lot of the functionality derived from LangChain (the OG LLM package) basics, since that's how the package was born - as a stripped down version of LangChain functionality, with cooler names.

## 🤔 Raison d'être

### Why does this exist?

The simple answer: by accident. The slightly longer answer: LangChain used to be (is?) a pretty big package with a ton of dependencies. The upside is that it's powerful for PoC purposes, because it has it all. The downsides:

- It's **_big_**. It takes up a lot of space (which can be an issue in some environments/runtimes), often for functionality that isn't needed.
- It's fairly **_complex_**. It's a big package with a lot of functionality, which can be overwhelming for new users.
- It **_wasn't exactly dependable_** in an industrial setting before. Version jumps were common, and the package was often broken after a new release.

### Circling back: why Fence?

Since our work was in a production environment, mostly dealing with Bedrock, we just started building some **basic components** from scratch. We needed a way to communicate with our models, which turned out to as the `Link` class (_wink wink_).
Then, some other things were added left and right, and this eventually turned into a miniature package. Not in small part because it was fun to go down this road. It's not as powerful as LangChain or some of the other things out there, but it does the job in a lightweight, simple way. It's also meant to be a lot more stable, since we're not adding new features all the time.

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

## 🗺️ How do I use it?

Fence just has a few basic components. See the [notebooks](notebooks) for examples on how to use them. Documentation is coming soon, but for now, you can check out the [source code](fence) for more details.

## 📦 Installation

You can install Fence via pip:

```bash
pip install fence
```

## ?? Contributing

We welcome contributions! Check out the [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
