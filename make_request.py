from pathlib import Path
import openai
import os
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "Please set the OPENAI_API_KEY environment variable."


parser = argparse.ArgumentParser(description="Make a request to OpenAI API.")
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="gpt-4o-mini-tts",
    help="Model to use for the request.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="speech.mp3",
    help="Output file name for the speech.",
)
parser.add_argument(
    "input",
    type=str,
    help="Input text to convert to speech.",
)
args = parser.parse_args()

speech_file_path = Path(__file__).parent / args.output

with openai.audio.speech.with_streaming_response.create(
    model=args.model,
    voice="alloy",
    input=input,
) as response:
    response.stream_to_file(speech_file_path)
