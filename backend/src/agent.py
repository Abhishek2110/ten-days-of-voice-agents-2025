import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

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

DB_PATH = Path(__file__).parent / "fraud_cases.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_sample_db() -> None:
    """
    Create the fraud_cases table and insert fake sample rows if empty.
    """
    conn = _get_conn()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fraud_cases (
                case_id INTEGER PRIMARY KEY,
                userName TEXT NOT NULL,
                securityIdentifier TEXT NOT NULL,
                maskedCard TEXT NOT NULL,
                cardEnding TEXT NOT NULL,
                transactionAmount REAL NOT NULL,
                currency TEXT NOT NULL,
                merchantName TEXT NOT NULL,
                transactionCategory TEXT NOT NULL,
                transactionSource TEXT NOT NULL,
                location TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                securityQuestion TEXT NOT NULL,
                securityAnswer TEXT NOT NULL,
                status TEXT NOT NULL,
                outcomeNote TEXT NOT NULL
            )
            """
        )

        cur = conn.execute("SELECT COUNT(*) AS cnt FROM fraud_cases")
        count = cur.fetchone()["cnt"]
        if count > 0:
            logger.info("fraud_cases table already initialized")
            return

        sample_cases: List[Dict[str, Any]] = [
            {
                "case_id": 1,
                "userName": "Rahul",
                "securityIdentifier": "AUSF-00123",
                "maskedCard": "**** **** **** 4242",
                "cardEnding": "4242",
                "transactionAmount": 4599.00,
                "currency": "INR",
                "merchantName": "BrightMart Online",
                "transactionCategory": "e-commerce",
                "transactionSource": "brightmart.example.com",
                "location": "Jaipur, India",
                "timestamp": "2025-11-20 14:32:10",
                "securityQuestion": "What is your favourite sport?",
                "securityAnswer": "cricket",
                "status": "pending_review",
                "outcomeNote": "",
            },
            {
                "case_id": 2,
                "userName": "Nikhil",
                "securityIdentifier": "AUSF-00456",
                "maskedCard": "**** **** **** 9911",
                "cardEnding": "9911",
                "transactionAmount": 12999.00,
                "currency": "INR",
                "merchantName": "Nova Travels",
                "transactionCategory": "travel",
                "transactionSource": "nova-travels.example.com",
                "location": "Singapore",
                "timestamp": "2025-11-21 09:10:45",
                "securityQuestion": "What is your favourite colour?",
                "securityAnswer": "blue",
                "status": "pending_review",
                "outcomeNote": "",
            },
            {
                "case_id": 3,
                "userName": "Arjun",
                "securityIdentifier": "AUSF-00789",
                "maskedCard": "**** **** **** 7788",
                "cardEnding": "7788",
                "transactionAmount": 850.50,
                "currency": "INR",
                "merchantName": "QuickBasket Groceries",
                "transactionCategory": "groceries",
                "transactionSource": "quickbasket.example.com",
                "location": "Mumbai, India",
                "timestamp": "2025-11-22 20:05:30",
                "securityQuestion": "In which city were you born?",
                "securityAnswer": "delhi",
                "status": "pending_review",
                "outcomeNote": "",
            },
        ]

        conn.executemany(
            """
            INSERT INTO fraud_cases (
                case_id,
                userName,
                securityIdentifier,
                maskedCard,
                cardEnding,
                transactionAmount,
                currency,
                merchantName,
                transactionCategory,
                transactionSource,
                location,
                timestamp,
                securityQuestion,
                securityAnswer,
                status,
                outcomeNote
            )
            VALUES (
                :case_id,
                :userName,
                :securityIdentifier,
                :maskedCard,
                :cardEnding,
                :transactionAmount,
                :currency,
                :merchantName,
                :transactionCategory,
                :transactionSource,
                :location,
                :timestamp,
                :securityQuestion,
                :securityAnswer,
                :status,
                :outcomeNote
            )
            """,
            sample_cases,
        )

        logger.info(f"Initialized SQLite DB at {DB_PATH} with sample fraud cases")


def _row_to_case(row: sqlite3.Row, include_secret: bool = False) -> Dict[str, Any]:
    """
    Convert a sqlite Row to a plain dict. Optionally hide securityAnswer.
    """
    data = dict(row)
    if not include_secret and "securityAnswer" in data:
        data.pop("securityAnswer", None)
    return data


def _select_case_by_id(case_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    with conn:
        row = conn.execute(
            "SELECT * FROM fraud_cases WHERE case_id = ?", (case_id,)
        ).fetchone()
    if row is None:
        return None
    return _row_to_case(row, include_secret=True)


class FraudAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a fraud alert voice agent for AU Small Finance Bank's fraud monitoring team.

                The user is talking to you over a phone-style voice call.

                Your name is Riya.  
                In the greeting you MUST always say:
                "Hello, this is Riya from AU Small Finance Bank's Fraud Monitoring Department."

                Your primary goal in each call:
                1. Greet the customer and introduce yourself exactly as:
                "Hello, this is Riya from AU Small Finance Bank's Fraud Monitoring Department."
                2. Explain that you are calling about a suspicious transaction.
                3. Ask for the customer's first name to look up the case.
                4. ALWAYS use the load_fraud_case tool with the name they give.
                5. If there is no active case, politely inform them and end the call.

                If a fraud case is found:
                - Use the securityQuestion from the case to verify the customer.
                - Ask the question clearly and let the user answer in their own words.
                - Then ALWAYS call verify_security_answer with the user's answer.
                - If verification fails twice, apologize, explain you cannot proceed,
                update the case as verification_failed, and end the call.

                If verification succeeds:
                1. Read out the suspicious transaction details in simple language:
                - merchantName
                - transactionAmount + currency
                - maskedCard (only masked form)
                - approximate time (timestamp) and location
                2. Ask a clear yes/no question like:
                "Did you make this transaction yourself?"
                3. If the user clearly says YES:
                - Treat it as legitimate.
                - Call update_fraud_case with status="confirmed_safe"
                    and a short outcome note like:
                    "Customer confirmed transaction as legitimate."
                - Reassure them that the card remains active.
                4. If the user clearly says NO:
                - Treat it as fraud.
                - Call update_fraud_case with status="confirmed_fraud"
                    and a short note like:
                    "Customer denied transaction. Card blocked and dispute raised. (mock)"
                - Explain in simple terms that:
                    - You are blocking the card as a precaution (demo only),
                    - A dispute or investigation is being raised (demo only).

                Important safety rules:
                - NEVER ask for full card number, CVV, PIN, net banking password, OTP,
                or any sensitive credentials.
                - If the user tries to share such details, stop them and say:
                "For your safety, please do not share your full card number, PIN, or OTP.
                The bank will never ask for these over a call."
                - Use only non-sensitive information that comes from the tools,
                such as security question, masked card, merchant, amount, time, location.

                Tone and style:
                - Calm, professional, and reassuring.
                - Short sentences.
                - No emojis or special formatting.
                """,
        )


    @function_tool
    async def load_fraud_case(self, context: RunContext, user_name: str) -> dict:
        """
        Load a pending fraud case for the given customer name from SQLite DB.

        Args:
            user_name: First name of the customer.

        Returns:
            {
              "found": bool,
              "case": {
                "case_id": int,
                "userName": str,
                "securityIdentifier": str,
                "maskedCard": str,
                "cardEnding": str,
                "transactionAmount": float,
                "currency": str,
                "merchantName": str,
                "transactionCategory": str,
                "transactionSource": str,
                "location": str,
                "timestamp": str,
                "securityQuestion": str,
                "status": str,
                "outcomeNote": str
              } | null
            }
        """
        logger.info(f"Loading fraud case from DB for user_name={user_name!r}")
        conn = _get_conn()
        name_lower = user_name.strip().lower()

        with conn:
            row = conn.execute(
                """
                SELECT *
                FROM fraud_cases
                WHERE LOWER(userName) = ?
                  AND status = 'pending_review'
                LIMIT 1
                """,
                (name_lower,),
            ).fetchone()

        if row is None:
            return {"found": False, "case": None}

        case_for_llm = _row_to_case(row, include_secret=False)
        return {"found": True, "case": case_for_llm}

    @function_tool
    async def verify_security_answer(
        self, context: RunContext, case_id: int, user_answer: str
    ) -> dict:
        """
        Check the customer's answer to the security question for a given case.

        Args:
            case_id: ID of the fraud case.
            user_answer: The answer given by the user.

        Returns:
            {"verified": bool}
        """
        logger.info(
            f"Verifying security answer for case_id={case_id}, answer={user_answer!r}"
        )
        conn = _get_conn()
        with conn:
            row = conn.execute(
                "SELECT securityAnswer FROM fraud_cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()

        if row is None:
            return {"verified": False}

        correct = str(row["securityAnswer"]).strip().lower()
        given = str(user_answer).strip().lower()
        verified = bool(correct and correct == given)
        return {"verified": verified}

    @function_tool
    async def update_fraud_case(
        self,
        context: RunContext,
        case_id: int,
        status: str,
        outcome_note: str,
    ) -> dict:
        """
        Update the status and outcome note of a fraud case in the SQLite DB.

        Args:
            case_id: ID of the fraud case.
            status: One of "confirmed_safe", "confirmed_fraud", "verification_failed".
            outcome_note: Short description of the outcome.

        Returns:
            {"success": bool, "updated_case": {...} | null}
        """
        logger.info(
            f"Updating fraud case {case_id} -> status={status}, note={outcome_note!r}"
        )
        conn = _get_conn()
        with conn:
            conn.execute(
                """
                UPDATE fraud_cases
                SET status = ?, outcomeNote = ?
                WHERE case_id = ?
                """,
                (status, outcome_note, case_id),
            )

            row = conn.execute(
                "SELECT * FROM fraud_cases WHERE case_id = ?", (case_id,)
            ).fetchone()

        if row is None:
            logger.warning(f"Fraud case {case_id} not found after update")
            return {"success": False, "updated_case": None}

        updated_case = _row_to_case(row, include_secret=False)
        logger.info(f"Updated fraud case {case_id}: {updated_case}")
        return {"success": True, "updated_case": updated_case}


def prewarm(proc: JobProcess):
    _init_sample_db()
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-Daisy",
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
        agent=FraudAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions=(
            "Greet the user as: 'Hello, this is Riya from AU Small Finance Bank's "
            "Fraud Monitoring Department.' Briefly explain that you are calling "
            "about a suspicious card transaction and then politely ask for their "
            "first name so you can look up their fraud case."
        ),
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))