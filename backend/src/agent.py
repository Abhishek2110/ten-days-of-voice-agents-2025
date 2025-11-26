import json
import logging
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

FAQ_FILE = Path("razorpay_faq.json")
LEADS_FILE = Path("razorpay_leads.json")


class Assistant(Agent):
    def __init__(self) -> None:
        system_instructions = """
You are a Sales Development Representative (SDR) for Razorpay.

Your goals:
1) Greet the user warmly.
2) Ask what brought them here and what they are working on.
3) Keep the conversation focused on understanding their needs around online payments, payouts, subscriptions, or business banking.
4) Answer questions about Razorpay's product, who it is for, and pricing.

Very important:
- For ANY product, company, or pricing question about Razorpay, you MUST first call the `lookup_razorpay_faq` tool with the user's question.
- Use ONLY the information returned by that tool when giving factual answers. Do NOT invent or guess details beyond that content.
- If the tool says the answer is not found, say: "I am not fully sure about that detail and do not want to guess. A human from our team can confirm this for you."

Lead qualification:
During the conversation, naturally collect these fields:
- Name
- Company
- Email
- Role
- Planned use case for Razorpay
- Team size
- Timeline to go live (now / soon / later)

End-of-call behavior:
- When the user indicates they are done (for example: "that's all", "I am done", "thank you", "thanks, that helps"):
  1) Give a short spoken summary:
     - Who they are
     - What their company does (if known)
     - Their use case
     - Team size
     - Timeline
  2) Then CALL the `save_razorpay_lead` tool ONCE with all the collected fields.
     - If any field is missing, pass an empty string for that field.
  3) After the tool call, thank them and close the conversation politely.

Style:
- Friendly, clear, concise sentences.
- No emojis or special formatting.
- Keep answers short and avoid long paragraphs.
- Always stay on-topic: Razorpay and the user's business needs.
"""
        super().__init__(instructions=system_instructions.strip())

    # ----------------- Tools -----------------

    @function_tool
    async def lookup_razorpay_faq(self, ctx: RunContext, query: str) -> str:
        """
        Look up answers about Razorpay's products, pricing, or company details.

        Always use this for:
        - "What does your product do?"
        - "Who is this for?"
        - "Do you have a free tier?"
        - "What is your pricing?"
        - "How much do you charge for UPI payments?"
        - Any other Razorpay-specific question.
        """
        logger.info(f"Looking up Razorpay FAQ for query: {query!r}")

        if not FAQ_FILE.exists():
            logger.error("razorpay_faq.json not found on disk.")
            return (
                "I could not load the Razorpay FAQ data properly. "
                "A teammate can confirm this detail for you."
            )

        try:
            faq_data: Dict[str, Any] = json.loads(FAQ_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to parse razorpay_faq.json.")
            return (
                "I could not load the Razorpay FAQ data properly. "
                "A teammate can confirm this detail for you."
            )

        faq_items: List[Dict[str, str]] = faq_data.get("faq", [])
        pricing: Dict[str, Any] = faq_data.get("pricing", {})

        query_lower = query.lower()

        # ---- Special handling: UPI pricing question ----
        if "upi" in query_lower and any(
            word in query_lower for word in ["price", "pricing", "charge", "fee", "cost"]
        ):
            pg_pricing = pricing.get("payment_gateway", {})
            upi_price = pg_pricing.get("upi")
            gst_info = pg_pricing.get("gst")
            if upi_price:
                # Example: "UPI payments are 0% per transaction. Additional 18% GST is applicable."
                extra = f" {gst_info}" if gst_info else ""
                return f"UPI payments are {upi_price}.{extra}"

        # ---- Generic FAQ matching (Q/A text) ----
        if not faq_items:
            logger.warning("FAQ list is empty in razorpay_faq.json.")
            return (
                "I could not find that exact detail in my Razorpay FAQ data. "
                "A teammate can confirm this for you."
            )

        best_score = 0
        best_answer: Optional[str] = None

        query_words = [w for w in query_lower.replace("?", " ").split() if len(w) > 2]

        for item in faq_items:
            text = (item.get("q", "") + " " + item.get("a", "")).lower()
            score = sum(1 for w in query_words if w in text)
            if score > best_score:
                best_score = score
                best_answer = item.get("a", "")

        if best_answer and best_score > 0:
            return best_answer

        return (
            "I could not find that exact detail in my Razorpay FAQ data. "
            "A teammate can confirm this for you."
        )

    @function_tool
    async def save_razorpay_lead(
        self,
        ctx: RunContext,
        name: str,
        company: str,
        email: str,
        role: str,
        use_case: str,
        team_size: str,
        timeline: str,
        notes: str = "",
    ) -> str:
        """
        Save a qualified Razorpay lead to a JSON file.

        The LLM should call this tool ONCE at the end of the conversation,
        after summarizing the lead to the user.
        """
        lead = {
            "name": name.strip(),
            "company": company.strip(),
            "email": email.strip(),
            "role": role.strip(),
            "use_case": use_case.strip(),
            "team_size": team_size.strip(),
            "timeline": timeline.strip(),
            "notes": notes.strip(),
        }

        logger.info(f"Saving Razorpay lead: {lead}")

        leads: List[Dict[str, Any]] = []
        if LEADS_FILE.exists():
            try:
                leads = json.loads(LEADS_FILE.read_text(encoding="utf-8"))
                if not isinstance(leads, list):
                    leads = []
            except Exception:
                logger.exception("Failed to read existing leads file, overwriting.")
                leads = []

        leads.append(lead)
        LEADS_FILE.write_text(json.dumps(leads, indent=2), encoding="utf-8")

        return "Lead saved successfully."


# ---------------------------------------------------------------------
# Prewarm: only VAD (FAQ is loaded inside the tool)
# ---------------------------------------------------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarm complete: VAD loaded.")


# ---------------------------------------------------------------------
# Entrypoint: set up the voice pipeline and start the agent session
# ---------------------------------------------------------------------

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

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

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
