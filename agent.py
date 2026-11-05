from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from your .env file; must run before creating the client

import anthropic

from tools import tools, run_tool

client = anthropic.Anthropic()

MODEL = "claude-haiku-5-5"
MAX_STEPS = 6  # safety limit so a confused loop can't run (and bill) forever

SYSTEM_PROMPT = """You help people estimate the price of used vehicles. 
You have access to three tools: one that estimates the price of a vehicle given its mileage and other filters, 
one that creates a scatterplot of price against mileage for comparison listings along with a fitted price curve and its prediction interval, 
and one that creates a residual plot of the same data. 
You can call these tools as many times as you need to answer the user's question. Follow these rules:

- Always get prices by calling the estimate_price tool. Never guess or quote a price from memory.
- If the user asks for a price estimate, they must give at least a manufacturer and a mileage. If either is missing, ask for it.
- If the user asks for a plot, they must give at least a manufacturer. If it's missing, ask for it.
- If the user doesn't specify which plot to create, create both.
- Only set filters the user actually mentioned.
- If reporting an estimate, report the estimate together with its prediction interval, and say how many listings it is based on.
- If reporting a plot, say how many listings it is based on and where the image file was saved. Do not describe what the plot shows.
- If the result says 'extrapolated' is true, warn that the mileage is outside the range of the data.
- If the tool returns an error, explain it in plain words and suggest loosening the filters."""


def answer(question, history=None, verbose=False):
    """Send a question to Claude, run any tools it asks for, and return its final text.

    `history` is an optional list of earlier messages, so follow-up questions work.
    It is updated in place, so pass the same list back in on the next call.
    """
    messages = history if history is not None else []
    messages.append({"role": "user", "content": question})

    for _ in range(MAX_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        # Keep Claude's whole reply (text and tool requests) in the conversation.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Claude is finished: join its text blocks into the final answer.
            return "".join(block.text for block in response.content if block.type == "text")

        # Claude asked for one or more tools: run each and collect the results.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    print(f"[tool call] {block.name}({block.input})")
                result = run_tool(block.name, block.input)
                if verbose:
                    print(f"[tool result] {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # must match the request's id
                    "content": result,
                })

        # Results go back as a user message, then the loop calls Claude again.
        messages.append({"role": "user", "content": tool_results})

    return "Sorry, I couldn't finish that request. Try rephrasing it."
