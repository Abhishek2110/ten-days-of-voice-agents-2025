import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ---------------------------------------------------------------------
# Load tutor content from JSON
# ---------------------------------------------------------------------

# agent.py is in backend/src; shared-data is at repo root
CONTENT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "shared-data"
    / "day4_tutor_content.json"
)


def _load_tutor_content() -> Dict[str, Dict[str, Any]]:
    try:
        with CONTENT_PATH.open("r", encoding="utf-8") as f:
            data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        logger.error(
            "day4_tutor_content.json not found at %s. Make sure the file exists.",
            CONTENT_PATH,
        )
        data = []
    except json.JSONDecodeError as e:
        logger.error("Failed to parse tutor content JSON: %s", e)
        data = []

    content_by_id: Dict[str, Dict[str, Any]] = {}
    for item in data:
        cid = item.get("id")
        if cid:
            content_by_id[cid] = item
    return content_by_id


TUTOR_CONTENT: Dict[str, Dict[str, Any]] = _load_tutor_content()

DEFAULT_CONCEPT_ID = next(iter(TUTOR_CONTENT.keys()), "variables")


# ---------------------------------------------------------------------
# Shared tools: list concepts and fetch concept details
# ---------------------------------------------------------------------


@function_tool()
async def list_concepts(context: RunContext) -> Dict[str, Any]:
    """
    List the available concepts for the Teach-the-Tutor coach.

    The LLM should call this when:
    - The user asks "what can I learn?"
    - The user doesn't specify a concept.
    """
    concepts = [
        {
            "id": cid,
            "title": c.get("title", ""),
            "sample_question": c.get("sample_question", ""),
        }
        for cid, c in TUTOR_CONTENT.items()
    ]
    return {
        "concepts": concepts,
        "default_concept_id": DEFAULT_CONCEPT_ID,
    }


@function_tool()
async def get_concept(context: RunContext, concept_id: str) -> Dict[str, Any]:
    """
    Fetch the summary and sample question for a given concept id.

    The LLM should:
    - Use `summary` when explaining in learn mode.
    - Use `sample_question` as the first quiz/teach-back prompt.
    """
    concept = TUTOR_CONTENT.get(concept_id)
    if not concept:
        return {
            "error": f"Unknown concept id '{concept_id}'",
            "available_ids": list(TUTOR_CONTENT.keys()),
        }
    return concept


# ---------------------------------------------------------------------
# Router / entry agent: greets user and hands off to a mode agent
# ---------------------------------------------------------------------


COMMON_TTS_KWARGS = dict(
    style="Conversation",
    tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
    text_pacing=True,
)


@function_tool()
async def switch_mode(
    context: RunContext,
    mode: str,
    concept_id: Optional[str] = None,
):
    """
    Switch the user between learning modes: 'learn', 'quiz', or 'teach_back'.

    The LLM should call this when:
    - The user picks a mode at the start.
    - The user says things like "switch to quiz", "I want to teach back", etc.

    Args:
      mode: one of 'learn', 'quiz', or 'teach_back'
      concept_id: optional; default is the global default concept.
    """
    mode_norm = (mode or "").strip().lower()
    cid = concept_id or DEFAULT_CONCEPT_ID

    if mode_norm == "learn":
        agent = LearnAgent(concept_id=cid)
        return agent, f"Connecting you to Matthew to help you learn the concept '{cid}'."
    elif mode_norm == "quiz":
        agent = QuizAgent(concept_id=cid)
        return agent, f"Connecting you to Alicia to quiz you on '{cid}'."
    elif mode_norm in ("teach_back", "teach-back", "teachback"):
        agent = TeachBackAgent(concept_id=cid)
        return agent, f"Connecting you to Ken so you can teach back '{cid}' in your own words."
    else:
        # Stay in router; tell LLM to ask again.
        return {
            "error": f"Unknown mode '{mode}'. Valid modes are: learn, quiz, teach_back."
        }


class Assistant(Agent):
    """
    Router agent for Day 4 Teach-the-Tutor.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are the main Teach-the-Tutor coordinator.

                You interact with the user via voice only (no rich formatting). Keep responses concise.

                Flow:
                1) Greet the user warmly and introduce yourself:
                   - Explain that you are their Active Recall Coach with three modes:
                     - Learn mode (Matthew): explains concepts briefly.
                     - Quiz mode (Alicia): asks short questions.
                     - Teach-back mode (Ken): listens to their explanation and gives short feedback.

                2) Ask two things:
                   - WHICH concept they want to focus on first.
                     - If they don't know, call list_concepts to see what is available,
                       and read the titles back in simple language.
                   - WHICH mode they want to start with:
                     - learn, quiz, or teach_back.

                3) Once you know the concept and mode:
                   - Call switch_mode(mode=..., concept_id=...) to hand the user off.
                   - The tool result includes a sentence like "Connecting you to Alicia...".
                   - Read that naturally as you hand them over.

                4) At any time later in the conversation:
                   - If the user says "I want to switch to quiz", "explain again", "let me teach it back", etc.
                   - The active mode agent should call switch_mode again.

                Guidelines:
                - Do NOT teach content yourself; your job is routing and clarifying options.
                - Keep your own turns brief and focused on next steps.
                """,
            tts=murf.TTS(
                voice="en-US-matthew",  # greeting voice; handoff will change to other agents
                **COMMON_TTS_KWARGS,
            ),
            tools=[list_concepts, get_concept, switch_mode],
        )

    async def on_enter(self) -> None:
        # Greet immediately when the room connects
        await self.session.generate_reply(
            instructions="""
                Greet the user as their Teach-the-Tutor Active Recall Coach.
                Briefly explain the three modes (learn, quiz, teach_back) in one or two short sentences.
                Ask them which concept they want to work on and which mode they want to start with.
            """
        )


# ---------------------------------------------------------------------
# Mode-specific agents: Learn, Quiz, Teach-back
# Each has its own Murf Falcon voice
# ---------------------------------------------------------------------


class LearnAgent(Agent):
    def __init__(self, concept_id: Optional[str] = None) -> None:
        self.current_concept_id = concept_id or DEFAULT_CONCEPT_ID

        super().__init__(
            instructions=f"""
                You are Matthew, the Learn coach in a Teach-the-Tutor active recall system.

                Goal: KEEP IT VERY SHORT. Do not go deep.

                Your behavior:
                - Focus on ONE concept at a time. The current concept_id is "{self.current_concept_id}".
                - Use get_concept to pull the official summary for the concept.
                - When explaining:
                  - Use at most ONE short sentence or two short sentences.
                  - Do NOT go into deep details, theory, or long paragraphs.
                - After giving this tiny explanation, immediately ask:
                  - If they want a quick recap,
                  - OR to switch to quiz mode,
                  - OR to switch to teach_back mode.

                Examples of follow-up questions:
                - "Should I quiz you on this now, or do you want to explain it back?"
                - "Do you want another very short explanation, or should we switch to quiz or teach-back?"

                Tool usage:
                - Use get_concept to get the summary, but compress your spoken explanation into one or two lines.
                - If the user clearly asks to switch (quiz or teach_back), call switch_mode with:
                  - mode = "quiz" or "teach_back"
                  - concept_id = the current concept id.
            """,
            tts=murf.TTS(
                voice="en-US-matthew",
                **COMMON_TTS_KWARGS,
            ),
            tools=[list_concepts, get_concept, switch_mode],
        )

    async def on_enter(self) -> None:
        # Auto-greet and give one tiny explanation, then propose switching
        await self.session.generate_reply(
            instructions=f"""
                You are Matthew in LEARN mode.
                1) Greet the user in one short sentence.
                2) Use get_concept to fetch the summary for '{self.current_concept_id}'.
                3) Compress that summary into ONE short spoken explanation sentence.
                4) Immediately ask if they want to:
                   - switch to quiz mode,
                   - switch to teach-back mode,
                   - or hear one more tiny explanation line.
            """
        )


class QuizAgent(Agent):
    def __init__(self, concept_id: Optional[str] = None) -> None:
        self.current_concept_id = concept_id or DEFAULT_CONCEPT_ID

        super().__init__(
            instructions=f"""
                You are Alicia, the Quiz coach in a Teach-the-Tutor active recall system.

                Goal: ONE QUESTION AT A TIME, THEN OFFER TO SWITCH. Keep it shallow, not deep.

                Your behavior:
                - Focus on ONE concept at a time. The current concept_id is "{self.current_concept_id}".
                - Use get_concept to retrieve the official sample_question for that concept.
                - Ask ONLY ONE question at a time.
                - After the user answers:
                  - Give very short feedback (1–2 sentences).
                  - Immediately ask if they want to:
                    - get another quick quiz question,
                    - switch to learn mode,
                    - or switch to teach_back mode.

                Examples of follow-ups:
                - "Nice. Do you want one more quick question, or should I switch you to learn mode or teach-back mode?"
                - "Good attempt. Should I ask another small question, or switch you to Matthew or Ken?"

                Do NOT:
                - Go into long theory.
                - Ask a long chain of questions without checking if they want to switch.

                Tool usage:
                - Use get_concept for guidance.
                - If they say "switch to learn" or "switch to teach back", call switch_mode with the right mode and current concept id.
            """,
            tts=murf.TTS(
                voice="en-US-alicia",
                **COMMON_TTS_KWARGS,
            ),
            tools=[list_concepts, get_concept, switch_mode],
        )

    async def on_enter(self) -> None:
        # Auto-greet and ask first short quiz question, then say you'll switch if they want
        await self.session.generate_reply(
            instructions=f"""
                You are Alicia in QUIZ mode.
                1) Greet the user in one short sentence.
                2) Use get_concept to fetch the sample_question for '{self.current_concept_id}'.
                3) Ask ONLY that one sample_question as your first quiz question.
                4) After they answer and you give brief feedback, you must ask if they want:
                   - another quick question,
                   - or to switch to learn mode,
                   - or to switch to teach-back mode.
            """
        )


class TeachBackAgent(Agent):
    def __init__(self, concept_id: Optional[str] = None) -> None:
        self.current_concept_id = concept_id or DEFAULT_CONCEPT_ID

        super().__init__(
            instructions=f"""
                You are Ken, the Teach-back coach in a Teach-the-Tutor active recall system.

                Your job:
                - Greet the user briefly as Ken.
                - Focus on ONE concept at a time. The current concept_id is "{self.current_concept_id}".
                - Use get_concept to understand the official summary and sample_question.
                - Then ask the user to explain the concept back to you in their own words.
                  Example:
                  - "In your own words, what is this concept and why is it useful?"

                Evaluation style:
                - Let the user give their full explanation without interrupting.
                - Then give concise qualitative feedback:
                  - 1–2 strengths ("You explained X clearly")
                  - 1–2 improvements ("Next time, also mention Y")
                - Keep it short, not deep.

                After giving feedback:
                - Ask if they want to:
                  - switch to quiz mode for a quick check,
                  - switch to learn mode for a quick recap,
                  - or move to another concept.

                Tool usage:
                - Use list_concepts if they want to switch topics.
                - Use get_concept as your reference for what a good explanation should include.
                - Use switch_mode when they clearly ask to switch.
            """,
            tts=murf.TTS(
                voice="en-US-ken",
                **COMMON_TTS_KWARGS,
            ),
            tools=[list_concepts, get_concept, switch_mode],
        )

    async def on_enter(self) -> None:
        # Auto-greet and request teach-back when handed off to
        await self.session.generate_reply(
            instructions=f"""
                You are Ken in TEACH-BACK mode.
                Greet the user briefly.
                Ask them to explain the concept '{self.current_concept_id}' in their own words.
                After they answer, give short feedback (strengths + one small suggestion),
                then ask if they want to switch to quiz mode, learn mode, or another concept.
            """
        )


# ---------------------------------------------------------------------
# Worker lifecycle & session setup
# ---------------------------------------------------------------------


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # STT + LLM on the session; TTS per Agent for per-mode voices.
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start session with the Router Assistant
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))