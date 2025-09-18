import os
from uuid import uuid4
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.eleven_labs import ElevenLabsTools
from agno.agent import Agent, RunResponse
from agno.utils.audio import write_audio_to_file
from agno.utils.log import logger
import streamlit as st
from dotenv import load_dotenv
import requests

# Streamlit Page Setup
st.set_page_config(page_title="📰 ➡️ 🎙️ Blog to Podcast Agent", page_icon="🎙️")
st.title("📰 ➡️ 🎙️ Blog to Podcast Agent")

# Load environment variables from .env if present
load_dotenv()


def scrape_with_firecrawl(url: str, api_key: str) -> str:
    """Scrape page content using Firecrawl REST API. Returns extracted text."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"url": url}
    try:
        resp = requests.post(
            "https://api.firecrawl.dev/v1/scrape", json=payload, headers=headers, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        # Firecrawl returns data in various shapes; try typical fields
        if isinstance(data, dict):
            # Some responses nest content under 'data' or return 'content'/'markdown'
            content = (
                data.get("content")
                or data.get("markdown")
                or (data.get("data") or {}).get("content")
                or (data.get("data") or {}).get("markdown")
            )
            if content:
                return str(content)
        return ""
    except Exception as e:
        logger.error(f"Firecrawl API error: {e}")
        return ""

# Sidebar: API Keys
st.sidebar.header("🔑 API Keys")

# Prefer values from environment if present
model_choice = st.sidebar.selectbox(
    "Model",
    options=[
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
    ],
    index=0,
    help="Use a lighter model to reduce credit usage on OpenRouter.",
)
openrouter_api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    type="password",
    value=os.getenv("OPENROUTER_API_KEY", ""),
)
elevenlabs_api_key = st.sidebar.text_input(
    "ElevenLabs API Key",
    type="password",
    value=os.getenv("ELEVEN_LABS_API_KEY", ""),
)
firecrawl_api_key = st.sidebar.text_input(
    "Firecrawl API Key",
    type="password",
    value=os.getenv("FIRECRAWL_API_KEY", ""),
)

# Check if all keys are provided
keys_provided = all([openrouter_api_key, elevenlabs_api_key, firecrawl_api_key])

# Input: Blog URL
url = st.text_input("Enter the Blog URL:", "")

# Button: Generate Podcast
generate_button = st.button("🎙️ Generate Podcast", disabled=not keys_provided)

if not keys_provided:
    st.warning("Please enter all required API keys to enable podcast generation.")

if generate_button:
    if url.strip() == "":
        st.warning("Please enter a blog URL first.")
    else:
        # Set API keys as environment variables for Agno and Tools
        # Configure OpenAI client to use OpenRouter gateway
        os.environ["OPENAI_API_KEY"] = openrouter_api_key
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        os.environ["ELEVEN_LABS_API_KEY"] = elevenlabs_api_key
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key

        with st.spinner("Processing... Scraping blog, summarizing and generating podcast 🎶"):
            try:
                # 1) Scrape content via Firecrawl REST API
                content = scrape_with_firecrawl(url, firecrawl_api_key)
                if not content or len(content.strip()) == 0:
                    st.error("Unable to extract content from the provided URL. Please try another blog/article URL.")
                    raise ValueError("Empty scrape result")

                blog_to_podcast_agent = Agent(
                    name="Blog to Podcast Agent",
                    agent_id="blog_to_podcast_agent",
                    # Route OpenAI requests via OpenRouter, using selected model
                    model=OpenAIChat(id=model_choice),
                    tools=[
                        ElevenLabsTools(
                            voice_id="JBFqnCBsd6RMkjVDRZzb",
                            model_id="eleven_multilingual_v2",
                            target_directory="audio_generations",
                        )
                    ],
                    description="You are an AI agent that can generate audio using the ElevenLabs API.",
                    instructions=[
                        "When the user provides a blog URL, you will receive the extracted content directly.",
                        "1. Create a concise summary of the blog content that is NO MORE than 3000 characters long.",
                        "2. The summary should capture the main points while being engaging and conversational.",
                        "3. Use the ElevenLabsTools to convert the summary to audio.",
                        "Ensure the summary is within the 3000 character limit to avoid ElevenLabs API limits.",
                    ],
                    markdown=True,
                    debug_mode=True,
                )

                # 2) Ask agent to summarize provided content and convert to audio
                prompt = (
                    "You are given the following article content. "
                    "Summarize it to a podcast script under 3000 characters, engaging and conversational, then convert to audio.\n\n"
                    f"<content>\n{content}\n</content>"
                )
                podcast: RunResponse = blog_to_podcast_agent.run(prompt)

                save_dir = "audio_generations"
                os.makedirs(save_dir, exist_ok=True)

                if podcast.audio and len(podcast.audio) > 0:
                    filename = f"{save_dir}/podcast_{uuid4()}.wav"
                    write_audio_to_file(
                        audio=podcast.audio[0].base64_audio,
                        filename=filename
                    )

                    st.success("Podcast generated successfully! 🎧")
                    audio_bytes = open(filename, "rb").read()
                    st.audio(audio_bytes, format="audio/wav")

                    st.download_button(
                        label="Download Podcast",
                        data=audio_bytes,
                        file_name="generated_podcast.wav",
                        mime="audio/wav"
                    )
                else:
                    st.error("No audio was generated. Please try again.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Streamlit app error: {e}")