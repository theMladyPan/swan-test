import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import openai
import dotenv
import os
import time
import logfire


dotenv.load_dotenv()

app = FastAPI()
aclient = openai.AsyncClient(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4.1-mini"

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    send_to_logfire="if-token-present",
)

logfire.instrument_fastapi(app)
logfire.instrument_openai(aclient)
logfire.instrument_httpx()


async def stream_response(history: list[dict[str, str]]):
    response = await aclient.chat.completions.create(
        model=MODEL,
        messages=history,
        stream=True,
    )

    async for chunk in response:
        for choice in chunk.choices:
            if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                yield choice.delta.content
            elif hasattr(choice, "message") and hasattr(choice.message, "content"):
                yield choice.message.content


async def stream_sentences(history: list[dict[str, str]]):
    sentence = ""
    async for chunk in stream_response(history):
        if chunk is not None:
            sentence += chunk
            if sentence.endswith(".") or sentence.endswith("!") or sentence.endswith("?"):
                yield sentence + "\n"
                sentence = ""


@app.get("/stream")
async def stream(question: str):
    """
    Streams the generated response as a text response.
    """
    input = [
        {
            "role": "user",
            "content": question,
        },
        {
            "role": "assistant",
            "content": "Sure, let me check that for you.",
        },
        {
            "role": "developer",
            "content": "Please generate a full response to the user request. The response should be a \
                direct continuation of the short response, as it will be transcribed to a speech.",
        },
    ]

    return StreamingResponse(
        stream_sentences(input),
        media_type="text/event-stream",
        headers={"Content-Type": "text/event-stream"},
    )


async def get_short_and_long_response(question: str):
    result_quick = await aclient.responses.create(
        model=MODEL,
        input=question,
        instructions="Be concise. Based on the user message, generate very short message (one sentence) which I can play to the user while \
            we prepare long response. If you dont know the answer right away, be creative and generate something \
            interesting and engaging. \
            In this stage stop after one sentence. \
            Do not use any code, no markdown, no links, no emojis. \
            ",
    )
    response_quick = result_quick.output_text

    input = [
        {
            "role": "user",
            "content": question,
        },
        {
            "role": "assistant",
            "content": response_quick,
        },
        {
            "role": "user",
            "content": "continue...",
        },
    ]

    result_full_task = asyncio.create_task(
        aclient.responses.create(
            model=MODEL,
            input=input,
            instructions="Continue in the response generation without repeating previous message. \
            Prepare answer suitable for text to speech conversion. \
            No code, no markdown, plain human readable text.",
        )
    )

    async for chunk in generate_speech_stream(response_quick):
        yield chunk

    logfire.info("Serving full response")
    response_full = await result_full_task

    async for chunk in generate_speech_stream(response_full.output_text):
        yield chunk


async def generate_speech_stream(text: str):
    """
    Generates an asynchronous audio stream from the given text using OpenAI's TTS.
    """
    logfire.info("Generating speech stream", text=text)
    with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",  # Or your preferred model
        voice="alloy",
        input=text,
    ) as response:
        for chunk in response.iter_bytes(chunk_size=1024):
            yield chunk


@app.get("/stream_audio")
async def stream_audio(question: str):
    """
    Streams the generated audio as an MP3 response.
    """
    return StreamingResponse(get_short_and_long_response(question), media_type="audio/mpeg")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("src/stream.html", "r") as file:
        content = file.read()
    return HTMLResponse(content=content)
