import streamlit as st
import time

from src.Agents.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Research Desk",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# SESSION STATE
# =============================================================================

DEFAULTS = {
    "results": {},
    "running": False,
    "done": False,
    "research_topic": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =============================================================================
# LIGHTWEIGHT DESIGN
# Native Streamlit only — no HTML, CSS, or JavaScript.
# =============================================================================

st.markdown(
    """
# 🔎 Research Desk

**Research smarter. Get from a question to a reviewed report in one workflow.**

Search → Read → Write → Review
"""
)

st.divider()


# =============================================================================
# RESEARCH INPUT
# =============================================================================

with st.container(border=True):
    st.subheader("Start a research task")
    st.caption(
        "Enter a topic or question. Four specialized agents will work through it "
        "step by step."
    )

    topic = st.text_input(
        "Research topic",
        value=st.session_state.research_topic,
        placeholder="e.g. How will AI agents change software development?",
        disabled=st.session_state.running,
    )

    examples = [
        "Future of LLMs in software development",
        "AI agents in 2026",
        "Roadmap for AGI development",
    ]

    st.caption("Examples: " + "  •  ".join(examples))

    start_col, clear_col = st.columns([1, 5])

    with start_col:
        start_button = st.button(
            "🚀 Start Research",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.running,
        )

    with clear_col:
        clear_button = st.button(
            "Clear",
            use_container_width=False,
            disabled=st.session_state.running,
        )


# =============================================================================
# BUTTON ACTIONS
# =============================================================================

if clear_button:
    st.session_state.results = {}
    st.session_state.running = False
    st.session_state.done = False
    st.session_state.research_topic = ""
    st.rerun()

if start_button:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.research_topic = topic.strip()
        st.session_state.results = {}
        st.session_state.running = True
        st.session_state.done = False
        st.rerun()


# =============================================================================
# PROGRESS UI
# =============================================================================

STAGES = [
    ("search", "🔎", "Search Agent", "Find recent and relevant sources"),
    ("reader", "📖", "Reader Agent", "Read and extract useful information"),
    ("writer", "✍️", "Writer", "Turn the research into a clear report"),
    ("critic", "🧐", "Critic", "Review the report for quality and gaps"),
]


def render_progress(active=None, completed=None):
    """Render a simple native Streamlit progress dashboard."""

    completed = completed or []

    st.subheader("Research progress")

    if active == "done":
        st.success("Research completed successfully.")
        st.progress(1.0, text="4 of 4 stages completed")
    else:
        finished = len(completed)
        progress_value = finished / len(STAGES)

        st.progress(
            progress_value,
            text=f"{finished} of {len(STAGES)} stages completed",
        )

    for key, icon, name, description in STAGES:
        if key in completed:
            with st.container(border=True):
                left, middle, right = st.columns([0.5, 4, 1])

                with left:
                    st.write("✅")

                with middle:
                    st.write(f"**{name}**")
                    st.caption(description)

                with right:
                    st.caption("DONE")

        elif key == active:
            with st.container(border=True):
                left, middle, right = st.columns([0.5, 4, 1])

                with left:
                    st.write("🔵")

                with middle:
                    st.write(f"**{name}**")
                    st.caption(description)

                with right:
                    st.caption("WORKING")

        else:
            with st.container(border=True):
                left, middle, right = st.columns([0.5, 4, 1])

                with left:
                    st.write("⚪")

                with middle:
                    st.write(f"**{name}**")
                    st.caption(description)

                with right:
                    st.caption("WAITING")


# =============================================================================
# MULTI-AGENT PIPELINE
# =============================================================================

if st.session_state.running and not st.session_state.done:

    topic_val = st.session_state.research_topic
    results = {}

    progress_placeholder = st.empty()

    def update_progress(active, completed):
        with progress_placeholder.container():
            render_progress(active, completed)

    # -------------------------------------------------------------------------
    # 1. SEARCH
    # -------------------------------------------------------------------------

    update_progress("search", [])

    with st.spinner("Search Agent is finding reliable sources..."):
        search_agent = build_search_agent()

        search_result = search_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Find recent, reliable and detailed information about: {topic_val}",
                    )
                ]
            }
        )

    results["search"] = search_result["messages"][-1].content
    st.session_state.results = dict(results)

    # -------------------------------------------------------------------------
    # 2. READER
    # -------------------------------------------------------------------------

    update_progress("reader", ["search"])

    with st.spinner("Reader Agent is reading the most relevant source..."):
        reader_agent = build_reader_agent()

        reader_result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic_val}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{results['search'][:800]}",
                    )
                ]
            }
        )

    results["reader"] = reader_result["messages"][-1].content
    st.session_state.results = dict(results)

    # -------------------------------------------------------------------------
    # 3. WRITER
    # -------------------------------------------------------------------------

    update_progress("writer", ["search", "reader"])

    with st.spinner("Writer is preparing your research report..."):
        research_combined = (
            f"SEARCH RESULTS:\n{results['search']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
        )

        results["writer"] = writer_chain.invoke(
            {
                "topic": topic_val,
                "research": research_combined,
            }
        )

    st.session_state.results = dict(results)

    # -------------------------------------------------------------------------
    # 4. CRITIC
    # -------------------------------------------------------------------------

    update_progress("critic", ["search", "reader", "writer"])

    with st.spinner("Critic is reviewing the research report..."):
        results["critic"] = critic_chain.invoke(
            {
                "report": results["writer"],
            }
        )

    st.session_state.results = dict(results)

    # -------------------------------------------------------------------------
    # COMPLETE
    # -------------------------------------------------------------------------

    with progress_placeholder.container():
        render_progress(
            "done",
            ["search", "reader", "writer", "critic"],
        )

    st.session_state.running = False
    st.session_state.done = True

    time.sleep(0.5)
    st.rerun()


# =============================================================================
# RESULTS
# =============================================================================

results = st.session_state.results

if results:

    st.divider()

    st.header("📄 Research Report")

    if "writer" in results:

        with st.container(border=True):
            st.caption("FINAL REPORT")
            st.markdown(results["writer"])

        st.download_button(
            label="⬇️ Download Report",
            data=results["writer"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    # -------------------------------------------------------------------------
    # Supporting agent outputs
    # -------------------------------------------------------------------------

    st.subheader("Agent details")

    if "search" in results:
        with st.expander("🔎 Search Agent — Sources and findings"):
            st.write(results["search"])

    if "reader" in results:
        with st.expander("📖 Reader Agent — Extracted content"):
            st.write(results["reader"])

    if "critic" in results:
        with st.expander("🧐 Critic — Report review"):
            st.write(results["critic"])


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("Research Desk · Multi-Agent Research System")
