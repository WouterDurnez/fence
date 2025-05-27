import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px

from fence.benchmark.prompts import (
    HIGHLIGHT,
    LANGUAGE_NAME,
    SYSTEM_PROMPT,
    USER_MESSAGE,
)
from fence.links import Link
from fence.links import logger as link_logger
from fence.models.anthropic.claude import Claude3Haiku as AnthropicClaude3Haiku
from fence.models.anthropic.claude import Claude3Opus as AnthropicClaude3Opus
from fence.models.anthropic.claude import Claude4Opus as AnthropicClaude4Opus
from fence.models.anthropic.claude import Claude4Sonnet as AnthropicClaude4Sonnet
from fence.models.anthropic.claude import Claude35Haiku as AnthropicClaude35Haiku
from fence.models.anthropic.claude import Claude35Sonnet as AnthropicClaude35Sonnet
from fence.models.anthropic.claude import Claude37Sonnet as AnthropicClaude37Sonnet
from fence.models.base import LLM
from fence.models.bedrock.claude import (
    Claude4Sonnet,
    Claude35Sonnet,
    Claude37Sonnet,
    ClaudeHaiku,
    ClaudeInstant,
    ClaudeSonnet,
)
from fence.models.bedrock.nova import NovaLite, NovaMicro, NovaPro
from fence.models.gemini.gemini import Gemini1_5_Pro, GeminiFlash1_5, GeminiFlash2_0
from fence.models.openai.gpt import GPT4o, GPT4omini
from fence.templates.messages import MessagesTemplate
from fence.templates.models import Message, Messages
from fence.utils.base import logger, time_it
from fence.utils.logger import setup_logging
from fence.utils.optim import parallelize, retry

#########
# SETUP #
#########

# Set the plotly template to white minimal
px.defaults.template = "plotly_white"

# Setup logging
link_logger.setLevel(logging.CRITICAL)
setup_logging(log_level="info", are_you_serious=False)

#############
# BENCHMARK #
#############

# We'll call the model N_CALLS times, requesting a
# test question based on a highlight and language name
template = MessagesTemplate(
    source=Messages(
        system=SYSTEM_PROMPT,
        messages=[
            Message(role="user", content=USER_MESSAGE),
        ],
    )
)


# Single model run
@parallelize(max_workers=10)
@retry(max_retries=5, delay=20)
def test_model(index: int, model: LLM):
    start_time = time.time()
    link = Link(template=template, model=model)

    result = link.run(  # noqa
        input_dict={"highlight": HIGHLIGHT, "language_name": LANGUAGE_NAME}
    )
    total_time = time.time() - start_time
    return total_time


def benchmark(models: list, n_calls: int = 20):
    """
    Benchmark the models
    """
    timings = {}

    @parallelize(max_workers=10)
    @time_it(only_warn=True, threshold=10)
    def run_model_benchmark(model: LLM, n_calls: int = n_calls):
        logger.info(f"Testing model: {model.__class__.__name__}")

        # Gather timings across N_CALLS runs
        timings[model.model_name] = test_model(range(n_calls), model=model)

    run_model_benchmark(models, n_calls=n_calls)

    return timings


#######
# VIZ #
#######


def viz(timings: dict, target_folder: str | Path):
    """
    Plot the timings as a strip plot using plotly

    :param timings: dictionary containing model:[timings] mapping
    :param target_folder: path to folder to store figure in
    """
    # Prepare the data
    df = pd.DataFrame(
        [(model, time) for model, times in timings.items() for time in times],
        columns=["Model", "Time"],
    )

    # Add model family column
    def get_model_family(model_name):
        match model_name:
            case name if "Anthropic" in name:
                return "Anthropic"
            case name if "Claude" in name:
                return "Claude"
            case name if "GPT" in name:
                return "GPT"
            case name if "Nova" in name:
                return "Nova"
            case name if "Gemini" in name:
                return "Gemini"
            case _:
                return "Other"

    df["Family"] = df["Model"].apply(get_model_family)

    # Create the strip plot
    fig = px.strip(
        df,
        y="Time",
        x="Model",
        color="Family",
        labels={"Time": "Time (s)", "Model": "Model"},
        title="Model Benchmark Timings",
        color_discrete_map={True: "blue", False: "red"},
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Time (s)",
        legend_title="Model Type",
        font=dict(size=12),
        showlegend=False,
        # Set y-axis range to 0-35 seconds
        # yaxis=dict(range=[0, 35]),
    )

    # Adjust the size of the plot points
    fig.update_traces(marker=dict(size=8))

    # Show the plot
    fig.show()

    # Save the plot as an HTML file with timestamp
    if target_folder:
        target_folder = Path(target_folder)
        target_folder.mkdir(exist_ok=True, parents=True)
    save_name = f"model_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    save_path = Path(target_folder) / save_name if target_folder else save_name
    fig.write_html(str(save_path))


########
# MAIN #
########

if __name__ == "__main__":

    models = []

    # Add Bedrock models
    models += [
        model
        for model in (
            ClaudeHaiku(),
            ClaudeInstant(),
            ClaudeSonnet(),
            Claude35Sonnet(region="us-east-1"),
            Claude37Sonnet(cross_region="eu"),
            Claude4Sonnet(cross_region="us", region="us-east-1"),
            NovaPro(cross_region="eu"),
            NovaLite(cross_region="eu"),
            NovaMicro(cross_region="eu"),
        )
    ]

    # Add GPT models
    models += [model() for model in (GPT4o, GPT4omini)]

    # Add Gemini models
    models += [model() for model in (Gemini1_5_Pro, GeminiFlash1_5, GeminiFlash2_0)]

    # Add Anthropic models
    models += [
        model()
        for model in (
            AnthropicClaude35Haiku,
            AnthropicClaude35Sonnet,
            AnthropicClaude4Opus,
            AnthropicClaude4Sonnet,
            AnthropicClaude37Sonnet,
            AnthropicClaude3Opus,
            AnthropicClaude3Haiku,
        )
    ]

    # Add source to all models
    for model in models:
        model.source = "benchmark"

    # Run the benchmark for each model
    timings = benchmark(models, n_calls=3)

    # Visualize the results
    viz(timings, target_folder="figures")
