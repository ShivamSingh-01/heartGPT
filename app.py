from agno.agent import Agent
from agno.models.groq import Groq
from agno.media import Image as AgnoImage
from agno.tools.duckduckgo import DuckDuckGoTools
import streamlit as st
from pathlib import Path
import tempfile
import os
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def initialize_agents(api_key: str):
    try:
        model = Groq(
            id="llama-3.1-8b-instant",
            api_key=api_key
        )

        therapist_agent = Agent(
            model=model,
            name="Therapist Agent",
            instructions=[
                "You are an empathetic therapist.",
                "Validate feelings and provide emotional support.",
                "Use gentle encouragement and warmth.",
                "Analyze emotional tone carefully."
            ],
            markdown=True
        )

        closure_agent = Agent(
            model=model,
            name="Closure Agent",
            instructions=[
                "Create heartfelt emotional closure messages.",
                "Help express unsent feelings honestly.",
                "Provide emotional release exercises."
            ],
            markdown=True
        )

        routine_planner_agent = Agent(
            model=model,
            name="Routine Planner Agent",
            instructions=[
                "Create a 7-day recovery plan.",
                "Include daily activities and self-care.",
                "Suggest productive distractions and growth tasks."
            ],
            markdown=True
        )

        brutal_honesty_agent = Agent(
            model=model,
            name="Brutal Honesty Agent",
            tools=[DuckDuckGoTools()],
            instructions=[
                "Give direct and objective relationship analysis.",
                "Provide growth insights.",
                "Be honest but constructive."
            ],
            markdown=True
        )

        return therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent

    except Exception as e:
        st.error(f"Error initializing agents: {str(e)}")
        return None, None, None, None


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(
    page_title="heartGPT🖤",
    page_icon="💔",
    layout="wide"
)

st.title(" heartGPT🖤")
st.markdown("### Your AI-powered breakup recovery team is here!")

# Sidebar API key input
with st.sidebar:
    st.header("🔑 API Configuration")

    api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",
        help="Get your free API key from https://console.groq.com"
    )

    if api_key:
        st.success("API Key provided! ✅")
    else:
        st.warning("Please enter your API key to proceed.")

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Share Your Feelings")
    user_input = st.text_area(
        "How are you feeling? What happened?",
        height=150,
        placeholder="Tell us your story..."
    )

with col2:
    st.subheader("Upload Chat Screenshots (Optional)")
    uploaded_files = st.file_uploader(
        "Upload screenshots",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            st.image(file, caption=file.name, use_container_width=True)

# Button
if st.button("Get Recovery Plan 💝", type="primary"):

    if not api_key:
        st.warning("Please enter your API key in the sidebar first!")
        st.stop()

    therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent = initialize_agents(api_key)

    if not all([therapist_agent, closure_agent, routine_planner_agent, brutal_honesty_agent]):
        st.error("Failed to initialize agents.")
        st.stop()

    if not user_input and not uploaded_files:
        st.warning("Please share your feelings or upload screenshots.")
        st.stop()

    st.header("Your Healing Journey Begins Here!")

    # Process Images
    def process_images(files):
        processed_images = []
        for file in files:
            try:
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"temp_{file.name}")

                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())

                agno_image = AgnoImage(filepath=Path(temp_path))
                processed_images.append(agno_image)

            except Exception as e:
                logger.error(f"Error processing image {file.name}: {str(e)}")

        return processed_images

    all_images = process_images(uploaded_files) if uploaded_files else []

    try:
        # Therapist
        with st.spinner("🤗 Getting emotional support..."):
            response = therapist_agent.run(user_input, images=all_images)
            st.subheader("🤗 Emotional Support")
            st.markdown(response.content)

        # Closure
        with st.spinner("✍️ Creating closure guidance..."):
            response = closure_agent.run(user_input, images=all_images)
            st.subheader("✍️ Finding Closure")
            st.markdown(response.content)

        # Recovery Plan
        with st.spinner("📅 Building recovery plan..."):
            response = routine_planner_agent.run(user_input, images=all_images)
            st.subheader("📅 7-Day Recovery Plan")
            st.markdown(response.content)

        # Honest Feedback
        with st.spinner("💪 Giving honest perspective..."):
            response = brutal_honesty_agent.run(user_input, images=all_images)
            st.subheader("💪 Honest Perspective")
            st.markdown(response.content)

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        st.error("Model failed. Please check your API key or try again.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>Built with ❤️ for Broken Hearts</div>",
    unsafe_allow_html=True
)
