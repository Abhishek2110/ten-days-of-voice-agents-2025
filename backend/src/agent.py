import logging
import os
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional

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

WELLNESS_LOG_PATH = "wellness_log.json"


def _load_wellness_log() -> list:
    """Internal helper: load the JSON log as a list of entries."""
    if not os.path.exists(WELLNESS_LOG_PATH):
        return []
    try:
        with open(WELLNESS_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        logger.warning(f"Failed to read {WELLNESS_LOG_PATH}: {e}")
        return []


def _write_wellness_log(entries: list) -> None:
    """Internal helper: write the entire log list to disk."""
    try:
        with open(WELLNESS_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write {WELLNESS_LOG_PATH}: {e}")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a daily Health & Wellness Voice Companion.

The user's name is Abhishek. Always address Abhishek by name in greetings and natural conversation.
Example greetings:
- "Hi Abhishek, good to connect again."
- "Welcome back, Abhishek. How are you feeling today?"

When generating the agent_summary for save_daily_check_in:
- ALWAYS refer to the user as Abhishek.
- Write in third person.
- Example: "Abhishek felt a bit low on energy today but stayed motivated with simple goals."

High-level role:
- You do short, supportive daily check-ins about mood, energy, stress, and simple goals.
- You are NOT a doctor, therapist, or crisis counselor.
- You never give medical, diagnostic, or medication advice.
- If the user hints at self-harm, severe distress, or medical emergencies, gently encourage them to seek local professional or emergency help.

Conversation structure for a typical daily check-in:
1) Warm welcome and context
   - Briefly greet the user using their name, for example:
     - "Hi Abhishek, good to see you."
     - "Hi Abhishek, how are you doing today?"
   - If appropriate, call the `get_last_check_in` tool once to see previous data.
   - If prior data exists, naturally reference one specific detail, for example:
     - "Last time you mentioned feeling low on energy, Abhishek."
     - "Previously you were focused on getting more consistent with exercise."
   - Keep this reference short and supportive.

2) Ask about mood and energy (no diagnosis)
   - Ask open, gentle questions like:
     - "How are you feeling today, emotionally, Abhishek?"
     - "What is your energy like today?"
     - "Anything in particular stressing you out or on your mind?"
   - You may ask follow-up questions to understand mood and energy, but keep it brief.

3) Ask about intentions / objectives for the day
   - Ask for 1–3 simple, practical goals:
     - "What are 1 to 3 things you would like to get done today, Abhishek?"
     - "Is there anything you want to do just for yourself, like rest, movement, or a hobby?"
   - If goals are vague, help Abhishek gently make them more concrete and realistic.

4) Offer simple, realistic advice or reflections
   - Your advice should be:
     - Small, actionable, grounded in everyday life.
     - Non-medical and non-diagnostic.
   - Typical strategies:
     - Break large goals into smaller tasks.
     - Suggest short breaks, short walks, stretching, or 5-minute focus sprints.
     - Encourage basic self-kindness and realistic expectations.

5) Recap and confirm
   - Briefly summarize:
     - The overall mood and energy for today.
     - The 1–3 main objectives or intentions you heard from Abhishek.
   - Ask:
     - "Does this sound right?" or
     - "Would you like to adjust anything?"

6) Persist the check-in
   - After you have:
     - A mood description,
     - An approximate mood score from 1 to 5 (1 = very low, 5 = very good),
     - An energy description,
     - Any key stressors or notes,
     - A list of 1–3 objectives,
     - And your own short summary sentence (1–2 lines),
   - Call the tool `save_daily_check_in` exactly once near the end of the check-in.
   - Provide all relevant fields so that the backend can store them.
   - When generating the agent_summary, always mention Abhishek in third person.
     Example: "Abhishek felt a bit tired today but focused on one main task and a short walk."
   - Do not ask the user to confirm the JSON; treat it as an internal step.

Responding to history questions:
- If Abhishek asks things like:
  - "How has my mood been this week?"
  - "Did I follow through on my goals most days?"
- Call `get_mood_overview` with a suitable number of days (like 7 or 30).
- Then explain the trends in simple, supportive language:
  - Mention rough averages or patterns.
  - Stay non-judgmental and encouraging.

Style:
- Speak as if you are talking, not writing an essay.
- Use short, clear sentences and natural conversation.
- Do not use emojis, markdown, or special symbols.
- Be kind, grounded, and realistic. Avoid toxic positivity.
""",
        )

    # TOOL 1: Save today's check-in to wellness_log.json
    @function_tool
    async def save_daily_check_in(
        self,
        context: RunContext,
        mood_label: str,
        mood_score_1_to_5: int,
        energy_description: str,
        stressors: Optional[str],
        objectives: List[str],
        agent_summary: str,
    ) -> str:
        """
        Save a single daily wellness check-in to the JSON log.

        The agent should call this tool exactly once near the end of each daily check-in,
        after learning about the user's mood, energy, stressors, and main objectives.

        Args:
            mood_label: Short text like "stressed", "okay", "hopeful", or "tired but optimistic".
            mood_score_1_to_5: Integer from 1 (very low) to 5 (very good). The agent should approximate this.
            energy_description: User's general energy level in natural language.
            stressors: Optional short description of main stressors or concerns, or empty string if none.
            objectives: 1–3 goals or intentions for today, each as a short text string.
            agent_summary: A one or two sentence summary by the agent.
                           It should refer to the user as "Abhishek" in third person, for example:
                           "Abhishek felt a bit tired but motivated. Focus for today is finishing one key task and taking a short walk."
        """
        now = datetime.now(timezone.utc).isoformat()

        entry = {
            "timestamp": now,
            "mood_label": mood_label,
            "mood_score_1_to_5": int(mood_score_1_to_5),
            "energy_description": energy_description,
            "stressors": stressors or "",
            "objectives": objectives,
            "agent_summary": agent_summary,
        }

        entries = _load_wellness_log()
        entries.append(entry)
        _write_wellness_log(entries)

        logger.info("Saved daily check-in entry to wellness_log.json")
        return "Daily wellness check-in saved."

    # TOOL 2: Get the last check-in, so the agent can reference it
    @function_tool
    async def get_last_check_in(self, context: RunContext) -> str:
        """
        Return a short textual summary of the most recent check-in, if any.

        The agent should usually call this at the start of a conversation,
        so it can gently reference something from the last session.

        If no previous check-in exists, the tool will indicate that explicitly.
        """
        entries = _load_wellness_log()
        if not entries:
            return "no_previous_check_ins"

        last = entries[-1]

        # Keep this short and easy for the LLM to incorporate.
        # We intentionally provide both date and key fields.
        return (
            f"Last check-in on {last['timestamp']}. "
            f"Mood label: {last.get('mood_label', 'unknown')}, "
            f"mood score: {last.get('mood_score_1_to_5', 'unknown')}, "
            f"energy: {last.get('energy_description', 'unknown')}, "
            f"objectives: {', '.join(last.get('objectives', []))}. "
            f"Summary: {last.get('agent_summary', '')}"
        )

    # TOOL 3: Simple trend overview (advanced goal – weekly/monthly summary)
    @function_tool
    async def get_mood_overview(
        self,
        context: RunContext,
        last_n_days: int = 7,
    ) -> str:
        """
        Compute a simple, human-readable mood and goal-follow-through overview
        over the last N days.

        The agent should call this when the user asks things like:
        - "How has my mood been this week?"
        - "Have I been following through on my goals?"

        Args:
            last_n_days: Number of days to look back, default 7.
        """
        entries = _load_wellness_log()
        if not entries:
            return "There are no check-ins yet to summarize."

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=last_n_days)

        # Filter entries within the window
        window_entries = []
        for e in entries:
            ts_str = e.get("timestamp")
            try:
                ts = datetime.fromisoformat(ts_str)
            except Exception:
                continue
            if ts >= cutoff:
                window_entries.append(e)

        if not window_entries:
            return f"There are no check-ins in the last {last_n_days} days."

        # Basic aggregates
        mood_scores = []
        days_with_objectives = 0

        for e in window_entries:
            score = e.get("mood_score_1_to_5")
            try:
                score = int(score)
                mood_scores.append(score)
            except Exception:
                pass

            objectives = e.get("objectives", [])
            if isinstance(objectives, list) and len(objectives) > 0:
                days_with_objectives += 1

        num_days = len(window_entries)
        avg_mood = sum(mood_scores) / len(mood_scores) if mood_scores else None

        # Build a simple, supportive summary string
        summary_parts = [
            f"In the last {last_n_days} days, you have {num_days} recorded check-ins."
        ]

        if avg_mood is not None:
            summary_parts.append(
                f"Your average mood score has been about {avg_mood:.1f} on a 1 to 5 scale."
            )

        summary_parts.append(
            f"On roughly {days_with_objectives} of those days, you noted at least one objective or intention."
        )

        summary_parts.append(
            "Remember, this is just a rough pattern, not a judgment. Small, consistent steps still count."
        )

        return " ".join(summary_parts)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
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

    # Start the session, which initializes the voice pipeline
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
