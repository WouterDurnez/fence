<img src="docs/fence.jpg" alt="tests" height="200"/>

# 🤺 Fence

No, not that kind of fence. The once with the interlinked pieces of metal. Like a chain fence? 

Why? Well, I got fed up with LangChain being so bloated, and I wanted to make a quick little repo
that covers some of the basic functionality of LangChain, but in a much smaller package. Also, if
we're calling the things Chains now, then a single component should be a Link (I don't care that it's
confusing, it's a good name). Extrapolating the concept gives us Fence. 

## What is it?

Fence is a simple, lightweight library for LLM communication. A lot of the functionality derived from LangChain basics, since that's how the package was born - as a stripped down version of LangChain functionality, with cooler names.

## How do I use it?

Fence just has a few basic components.

1. First, we need [to hook up with our Large Language Models](fence/models). Fence contains classes for all models in the Claude family. Using these models automatically logs usage (invocations, token counts, etc.) in DataDog.
2. These models need input. Older Claude models used string-based input (with the infamous Human: ... Assistant:) formatting. Third generation Claude models (Haiku, Sonnet, and Opus) use a new, OpenAI inspired API