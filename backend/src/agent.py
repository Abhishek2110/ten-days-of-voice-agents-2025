import logging
import json
import os
from datetime import datetime
from typing import List, Optional
import zoneinfo

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


class Assistant(Agent):
    def __init__(self) -> None:
        # Initialize order state for this assistant instance
        self.order = {
            "drinkType": None,
            "size": None,
            "milk": None,
            "extras": [],
            "name": None,
        }

        super().__init__(
            instructions=(
                "You are a friendly barista at Brew Bliss Coffee, taking voice orders.\n"
                "The user is speaking to you, but you see text.\n\n"
                "You must help the user place a drink order and maintain an internal order state with the fields:\n"
                "drinkType, size, milk, extras, name.\n\n"
                "Order state schema:\n"
                "{\n"
                '  \"drinkType\": \"string\",\n'
                '  \"size\": \"string\",\n'
                '  \"milk\": \"string\",\n'
                '  \"extras\": [\"string\"],\n'
                '  \"name\": \"string\"\n'
                "}\n\n"
                "Behavior:\n"
                "- Ask clarifying questions until all of these fields are filled.\n"
                "- Typical sizes are small, medium, large.\n"
                "- Milk can be whole, skim, oat, almond, soy, etc.\n"
                "- Extras can include things like extra shot, vanilla syrup, caramel, whipped cream.\n"
                "- Always capture the customer's name.\n"
                "- Use the tools update_order_state and finalize_order whenever the user provides or confirms details.\n"
                "- Once all fields are filled and confirmed, call finalize_order to save the order.\n"
                "- After finalize_order, give a concise text summary of the order and tell the user it has been placed.\n\n"
                "Style:\n"
                "- You are warm, efficient, and a bit playful, like a barista during a busy but fun morning.\n"
                "- Keep responses concise, to the point.\n"
                "- When the user greets you (hello, hi, hey, good morning, etc), always respond with a short friendly greeting followed by: 'Welcome to Brew Bliss Coffee. What can I get started for you today?'\n"
                "- Do not use complex formatting or punctuation such as emojis, asterisks, or other special symbols."
            ),
        )

    # TOOL 1: Update the order state
    @function_tool
    async def update_order_state(
        self,
        context: RunContext,
        drinkType: Optional[str] = None,
        size: Optional[str] = None,
        milk: Optional[str] = None,
        extras: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> dict:
        """
        Use this tool to update the current coffee order state.

        Call this whenever the user gives or changes details about their drink.
        Any argument that is not provided will be left unchanged.

        Args:
            drinkType: The drink type, for example latte, cappuccino, americano, cold brew.
            size: The drink size, for example small, medium, large.
            milk: The milk type, for example whole, skim, oat, almond, soy.
            extras: A list of extras, for example ["extra shot", "vanilla syrup", "whipped cream"].
            name: The customer's name for the order.

        Returns:
            A dict containing:
            - order: the full current order state
            - missing_fields: a list of fields that are still missing
        """

        logger.info("Updating order state")

        if drinkType is not None:
            self.order["drinkType"] = drinkType.strip()
        if size is not None:
            self.order["size"] = size.strip()
        if milk is not None:
            self.order["milk"] = milk.strip()
        if extras is not None:
            # Replace extras entirely if provided
            self.order["extras"] = [e.strip() for e in extras if e.strip()]
        if name is not None:
            self.order["name"] = name.strip()

        missing = [
            key
            for key, value in self.order.items()
            if value is None or (isinstance(value, list) and len(value) == 0)
        ]

        return {
            "order": self.order,
            "missing_fields": missing,
        }

    # TOOL 2: Finalize and save the order to JSON
    @function_tool
    async def finalize_order(self, context: RunContext) -> dict:
        """
        Use this tool when the order is fully specified and confirmed by the user.

        This tool:
        - Validates that drinkType, size, milk, and name are filled.
        - Saves the order as a JSON file on disk.
        - Returns a summary of the saved order and the file path.

        extras is optional. If the user does not want extras, it can stay empty.
        """

        logger.info("Finalizing order")

        # Only require these fields
        required_fields = ["drinkType", "size", "milk", "name"]

        missing = [
            key
            for key in required_fields
            if self.order.get(key) is None or self.order.get(key) == ""
        ]

        if missing:
            return {
                "success": False,
                "message": "Order is not complete. Missing fields.",
                "missing_fields": missing,
                "order": self.order,
            }

        import os, json
        from datetime import datetime

        os.makedirs("orders", exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe_name = (self.order["name"] or "customer").replace(" ", "_")
        filename = os.path.join("orders", f"order_{safe_name}_{timestamp}.json")

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.order, f, indent=2)

            logger.info(f"Saved order JSON to: {filename}")

        except Exception as e:
            logger.exception("Failed to save order JSON")
            return {
                "success": False,
                "message": f"Failed to save order file: {e}",
                "order": self.order,
            }

        summary = (
            f"Order for {self.order['name']}: "
            f"{self.order['size']} {self.order['drinkType']} with {self.order['milk']} milk"
        )
        if self.order.get("extras"):
            extras_str = ", ".join(self.order["extras"])
            summary += f", extras: {extras_str}"

        return {
            "success": True,
            "message": "Order saved successfully.",
            "order": self.order,
            "summary": summary,
            "file": filename,
        }


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Voice AI pipeline
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

    # Start the session with our barista assistant
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
