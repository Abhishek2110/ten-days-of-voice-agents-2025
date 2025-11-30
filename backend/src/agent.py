import json
import logging
import os
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("ecommerce-agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Get the src directory path
# -------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Load Product Catalog from JSON
# -------------------------
PRODUCTS_FILE = os.path.join(SRC_DIR, "products.json")

def _load_catalog() -> List[Dict]:
    """Load products from products.json file."""
    try:
        if os.path.exists(PRODUCTS_FILE):
            with open(PRODUCTS_FILE, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"{PRODUCTS_FILE} not found. Using empty catalog.")
            return []
    except Exception as e:
        logger.error(f"Error loading {PRODUCTS_FILE}: {e}")
        return []

CATALOG = _load_catalog()

# -------------------------
# Orders Directory & File Setup
# -------------------------
ORDERS_DIR = os.path.join(SRC_DIR, "orders")
ORDERS_FILE = os.path.join(ORDERS_DIR, "orders.json")

if not os.path.exists(ORDERS_DIR):
    os.makedirs(ORDERS_DIR)

if not os.path.exists(ORDERS_FILE):
    with open(ORDERS_FILE, "w") as f:
        json.dump([], f)

# -------------------------
# Per-session Userdata (shopping-centric)
# -------------------------
@dataclass
class Userdata:
    player_name: Optional[str] = None  # retained name field (player -> customer)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    cart: List[Dict] = field(default_factory=list)  # list of {product_id, quantity, attrs}
    orders: List[Dict] = field(default_factory=list)  # orders placed in this session
    history: List[Dict] = field(default_factory=list)  # conversational actions for trace

# -------------------------
# Merchant-layer helpers (ACP-inspired mini layer)
# -------------------------

def _load_all_orders() -> List[Dict]:
    try:
        with open(ORDERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_order(order: Dict):
    orders = _load_all_orders()
    orders.append(order)
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2)


def list_products(filters: Optional[Dict] = None) -> List[Dict]:
    """Naive filtering by category, max_price, color, size substring, or query words.

    Improvements:
    - Accepts category synonyms (e.g., 'phone', 'mobile', 'phones' -> 'mobile').
    - Supports a flexible max_price and min_price (if provided in filters).
    - Matches category by substring if exact match fails.
    """
    filters = filters or {}
    results = []
    query = filters.get("q")
    category = filters.get("category")
    max_price = filters.get("max_price") or filters.get("to") or filters.get("max")
    min_price = filters.get("min_price") or filters.get("from") or filters.get("min")
    color = filters.get("color")
    size = filters.get("size")

    if category:
        cat = category.lower()
        if cat in ("phone", "phones", "mobile", "mobile phone", "mobiles"):
            category = "mobile"
        elif cat in ("tshirt", "t-shirts", "tees", "tee"):
            category = "tshirt"
        else:
            category = cat

    for p in CATALOG:
        ok = True
        if category:
            pcat = p.get("category", "").lower()
            if pcat != category and category not in pcat and pcat not in category:
                ok = False
        if max_price:
            try:
                if p.get("price", 0) > int(max_price):
                    ok = False
            except Exception:
                pass
        if min_price:
            try:
                if p.get("price", 0) < int(min_price):
                    ok = False
            except Exception:
                pass
        if color and p.get("color") and p.get("color") != color:
            ok = False
        if size and (not p.get("sizes") or size not in p.get("sizes")):
            ok = False
        if query:
            q = query.lower()
            if "phone" in q or "mobile" in q:
                if p.get("category") != "mobile":
                    ok = False
            else:
                if q not in p.get("name", "").lower() and q not in p.get("description", "").lower():
                    ok = False
        if ok:
            results.append(p)
    return results


def find_product_by_ref(ref_text: str, candidates: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Resolve references like 'second hoodie' or 'black hoodie' to a product dict.
    Heuristics improved:
    - Handle ordinals like 'first/second/third' within a filtered candidate list.
    - If ref mentions 'phone' or 'mobile' prefer mobile category products.
    - Match by id, color+category, name substring, or numeric index.
    """
    ref = (ref_text or "").lower().strip()
    cand = candidates if candidates is not None else CATALOG

    wants_mobile = any(w in ref for w in ("phone", "phones", "mobile", "mobiles"))
    filtered = cand
    if wants_mobile:
        filtered = [p for p in cand if p.get("category") == "mobile"]
        if not filtered:
            filtered = cand

    ordinals = {"first": 0, "second": 1, "third": 2, "fourth": 3}
    for word, idx in ordinals.items():
        if word in ref:
            if idx < len(filtered):
                return filtered[idx]

    for p in cand:
        if p["id"].lower() == ref:
            return p

    for p in cand:
        if p.get("color") and p["color"] in ref and p.get("category") and p["category"] in ref:
            return p

    for p in filtered:
        name = p["name"].lower()
        if all(tok in name for tok in ref.split() if len(tok) > 2):
            return p
    for p in cand:
        for tok in ref.split():
            if len(tok) > 2 and tok in p["name"].lower():
                return p

    for token in ref.split():
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(filtered):
                return filtered[idx]

    for word, idx in ordinals.items():
        if word in ref and idx < len(cand):
            return cand[idx]

    return None


@function_tool
async def show_catalog(
    ctx: RunContext[Userdata],
    q: Annotated[Optional[str], Field(description="Search query (optional)", default=None)] = None,
    category: Annotated[Optional[str], Field(description="Category (optional)", default=None)] = None,
    max_price: Annotated[Optional[int], Field(description="Maximum price (optional)", default=None)] = None,
    color: Annotated[Optional[str], Field(description="Color (optional)", default=None)] = None,
) -> str:
    """Return a short spoken summary of matching products (name, price, id).
    Improvements:
    - Recognize category synonyms like 'phones' and 'tees'.
    - Return up to 8 items and explicitly call out mobiles if present.
    """
    userdata = ctx.userdata
    if category:
        cat = category.lower()
        if cat in ("phone", "phones", "mobile", "mobile phone", "mobiles"):
            category = "mobile"
        elif cat in ("tshirt", "t-shirts", "tees", "tee"):
            category = "tshirt"
        else:
            category = cat
    if not category and q:
        if any(w in q.lower() for w in ("phone", "phones", "mobile", "mobiles")):
            category = "mobile"
        if any(w in q.lower() for w in ("tee", "tshirt", "t-shirts", "tees")):
            category = "tshirt"

    filters = {"q": q, "category": category, "max_price": max_price, "color": color}
    prods = list_products({k: v for k, v in filters.items() if v is not None})
    if not prods:
        return "Sorry — I couldn't find any items that match. Would you like to try another search?"
    lines = [f"Here are the top {min(8, len(prods))} items I found at Dr Abhishek Shop:"]
    for idx, p in enumerate(prods[:8], start=1):
        size_info = f" (sizes: {', '.join(p['sizes'])})" if p.get('sizes') else ""
        lines.append(f"{idx}. {p['name']} — {p['price']} {p['currency']} (id: {p['id']}){size_info}")
    lines.append("You can say: 'I want the second item in size M' or 'add mug-001 to my cart, quantity 2'.")
    if any(p.get('category') == 'mobile' for p in prods):
        lines.append("To buy a phone say: 'Add phone-002 to my cart' or 'I want the second phone, quantity 1'.")
    return "\n".join(lines)


def find_product_by_ref(ref_text: str, candidates: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Resolve references like 'second hoodie' or 'black hoodie' to a product dict.
    Very simple heuristic: look for ordinal words, color or exact id/name matching.
    """
    ref = (ref_text or "").lower().strip()
    cand = candidates if candidates is not None else CATALOG

    ordinals = {"first": 0, "second": 1, "third": 2}
    for word, idx in ordinals.items():
        if word in ref:
            if idx < len(cand):
                return cand[idx]

    for p in cand:
        if p["id"].lower() == ref:
            return p

    for p in cand:
        if p.get("color") and p["color"] in ref and p.get("category") and p["category"] in ref:
            return p

    for p in cand:
        if p["name"].lower() in ref or any(w in p["name"].lower() for w in ref.split()):
            return p

    for token in ref.split():
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(cand):
                return cand[idx]

    return None


def create_order_object(line_items: List[Dict], currency: str = "INR") -> Dict:
    """line_items: [{product_id, quantity, attrs}]
    Returns an order dict (id, items, total, currency, created_at)
    """
    items = []
    total = 0
    for li in line_items:
        pid = li.get("product_id")
        qty = int(li.get("quantity", 1))
        prod = next((p for p in CATALOG if p["id"] == pid), None)
        if not prod:
            raise ValueError(f"Product {pid} not found")
        line_total = prod["price"] * qty
        total += line_total
        items.append({
            "product_id": pid,
            "name": prod["name"],
            "unit_price": prod["price"],
            "quantity": qty,
            "line_total": line_total,
            "attrs": li.get("attrs", {}),
        })
    order = {
        "id": f"order-{str(uuid.uuid4())[:8]}",
        "items": items,
        "total": total,
        "currency": currency,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    _save_order(order)
    return order


def get_most_recent_order() -> Optional[Dict]:
    all_orders = _load_all_orders()
    if not all_orders:
        return None
    return all_orders[-1]


@function_tool
async def show_catalog(
    ctx: RunContext[Userdata],
    q: Annotated[Optional[str], Field(description="Search query (optional)", default=None)] = None,
    category: Annotated[Optional[str], Field(description="Category (optional)", default=None)] = None,
    max_price: Annotated[Optional[int], Field(description="Maximum price (optional)", default=None)] = None,
    color: Annotated[Optional[str], Field(description="Color (optional)", default=None)] = None,
) -> str:
    """Return a short spoken summary of matching products (name, price, id)."""
    userdata = ctx.userdata
    filters = {"q": q, "category": category, "max_price": max_price, "color": color}
    prods = list_products({k: v for k, v in filters.items() if v is not None})
    if not prods:
        return "Sorry — I couldn't find any items that match. Would you like to try another search?"
    # Summarize top 4
    lines = [f"Here are the top {min(4, len(prods))} items I found at Dr Abhishek Shop:"]
    for idx, p in enumerate(prods[:4], start=1):
        lines.append(f"{idx}. {p['name']} — {p['price']} {p['currency']} (id: {p['id']})")
    lines.append("You can say: 'I want the second item in size M' or 'add mug-001 to my cart, quantity 2'.")
    return "\n".join(lines)


@function_tool
async def add_to_cart(
    ctx: RunContext[Userdata],
    product_ref: Annotated[str, Field(description="Reference to product: id, name, or spoken ref")] ,
    quantity: Annotated[int, Field(description="Quantity", default=1)] = 1,
    size: Annotated[Optional[str], Field(description="Size (optional)", default=None)] = None,
) -> str:
    """Resolve a product and add to the session cart."""
    userdata = ctx.userdata
    # take recent catalog as candidates
    candidates = CATALOG
    prod = find_product_by_ref(product_ref, candidates)
    if not prod:
        return "I couldn't resolve which product you meant. Try using the item id or say 'show catalog' to hear options.'"
    userdata.cart.append({
        "product_id": prod["id"],
        "quantity": int(quantity),
        "attrs": {"size": size} if size else {},
    })
    userdata.history.append({
        "time": datetime.utcnow().isoformat() + "Z",
        "action": "add_to_cart",
        "product_id": prod["id"],
        "quantity": int(quantity),
    })
    return f"Added {quantity} x {prod['name']} to your cart. What would you like to do next?"


@function_tool
async def show_cart(
    ctx: RunContext[Userdata],
) -> str:
    userdata = ctx.userdata
    if not userdata.cart:
        return "Your cart is empty. You can say 'show catalog' to browse items.'"
    lines = ["Items in your cart:"]
    total = 0
    for li in userdata.cart:
        p = next((x for x in CATALOG if x["id"] == li["product_id"]), None)
        if not p:
            continue
        line_total = p["price"] * li.get("quantity", 1)
        total += line_total
        sz = li.get("attrs", {}).get("size")
        sz_text = f", size {sz}" if sz else ""
        lines.append(f"- {p['name']} x {li['quantity']}{sz_text}: {line_total} INR")
    lines.append(f"Cart total: {total} INR")
    lines.append("Say 'place my order' to checkout or 'clear cart' to empty the cart.")
    return "\n".join(lines)


@function_tool
async def clear_cart(
    ctx: RunContext[Userdata],
) -> str:
    userdata = ctx.userdata
    userdata.cart = []
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "clear_cart"})
    return "Your cart has been cleared. What would you like to do next?"


@function_tool
async def place_order(
    ctx: RunContext[Userdata],
    confirm: Annotated[bool, Field(description="Confirm order placement", default=True)] = True,
) -> str:
    """Create order from session cart and persist. Returns order summary."""
    userdata = ctx.userdata
    if not userdata.cart:
        return "Your cart is empty — nothing to place. Would you like to browse items?"
    # Build line_items
    line_items = []
    for li in userdata.cart:
        line_items.append({
            "product_id": li["product_id"],
            "quantity": li.get("quantity", 1),
            "attrs": li.get("attrs", {}),
        })
    order = create_order_object(line_items)
    userdata.orders.append(order)
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "place_order", "order_id": order["id"]})
    userdata.cart = []
    return f"Order placed. Order ID {order['id']}. Total {order['total']} {order['currency']}. What would you like to do next?"


@function_tool
async def last_order(
    ctx: RunContext[Userdata],
) -> str:
    ord = get_most_recent_order()
    if not ord:
        return "You have no past orders yet."
    lines = [f"Most recent order: {ord['id']} — {ord['created_at']}"]
    for it in ord['items']:
        lines.append(f"- {it['name']} x {it['quantity']}: {it['line_total']} {ord['currency']}")
    lines.append(f"Total: {ord['total']} {ord['currency']}")
    return "\n".join(lines)

# -------------------------
# The Agent (Amazon's AI Assistant Ken)
# -------------------------
class ECommerceAgent(Agent):
    def __init__(self):
        instructions = """
        You are 'Ken', the friendly voice assistant for Amazon.
        Universe: A online shopping platform selling mugs, hoodies, tees, laptops, mobile phones and many more.
        Tone: Warm, helpful, slightly jocular; keep sentences short for TTS clarity.
        Role: Help the customer browse the catalog, add items to cart, place orders, and review recent orders.

        Rules:
            - Use the provided tools to show the catalog, add items to cart, show the cart, place orders, show last order and clear the cart.
            - Keep continuity using the per-session userdata. Mention cart contents if relevant.
            - Drive short voice-first turns suitable for spoken delivery.
            - When presenting options, include product id and price (e.g. 'mug-001 — 299 INR').
        """
        super().__init__(
            instructions=instructions,
            tools=[show_catalog, add_to_cart, show_cart, clear_cart, place_order, last_order],
        )

def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n" + "🛍️" * 6)
    logger.info("🚀 STARTING VOICE E-COMMERCE AGENT (Amazon) — ")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-ken",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    await session.start(
        agent=ECommerceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))