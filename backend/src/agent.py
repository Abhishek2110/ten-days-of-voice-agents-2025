import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

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
        instructions = """
            You are FlipFresh, a friendly food & grocery ordering assistant.

            The user is talking to you via voice. Be concise, natural, and conversational.
            Do NOT use emojis or special formatting.

            FIRST MESSAGE BEHAVIOR
            - On your very first reply in a new conversation, ALWAYS introduce yourself by name.
            - Start your first response with something like:
            "Hi, I'm FlipFresh, your food and grocery ordering assistant. I can help you order groceries, snacks, and ingredients for simple meals. What would you like today?"
            - After the first message, you do NOT need to repeat your name unless the user asks.

            Your main job:
            - Help the user order groceries, snacks, and simple prepared foods.
            - You have tools to manage a shopping cart and place orders.
            - You can also add multiple items when the user asks for "ingredients for X"
            such as "peanut butter sandwich" or "pasta for two people".

            GENERAL BEHAVIOR
            - Greet the user and briefly say what you can do:
            e.g. "I can help you order groceries, snacks, and ingredients for simple meals."
            - Ask clarifying questions when needed:
            - If they just say "milk", ask about size or type if relevant.
            - If they don't specify quantity, default to 1 and tell them.
            - Always confirm cart operations out loud:
            - After adding: "I've added 1 Whole Wheat Bread to your cart."
            - After updating/removing: clearly explain what changed.
            - When user asks:
            - "What's in my cart?" → call the cart listing tool.
            - "Remove X" → call the remove item tool.
            - "Change X to 2" → call the update quantity tool.

            CATALOG & CART TOOLS
            You have tools for:
            - Searching the catalog by category or text query.
            - Adding, updating, removing items in the cart.
            - Listing the cart and computing totals.
            - Adding ingredients for simple meals (recipes).
            - Placing the final order and saving it to JSON.

            IMPORTANT:
            - Use the tools whenever you need to:
            - Inspect what items exist.
            - Change the cart.
            - Handle "ingredients for X".
            - Place the order.
            - Do NOT invent products that are not in the catalog.
            If something is not available, politely say so and suggest alternatives.

            INGREDIENTS / RECIPES
            For requests like:
            - "I need ingredients for a peanut butter sandwich."
            - "Get me what I need for pasta for two people."
            Always call the 'add_meal_ingredients' tool with:
            - meal_name set to the dish name (e.g. "peanut butter sandwich", "pasta").
            - servings set to how many people or servings the user mentioned (default 1).

            ORDER PLACEMENT
            - When the user says "that's all", "place my order", or "I'm done":
            1) Summarize the cart and total.
            2) Ask for their name and address if you don't have it.
            3) Call the 'place_order' tool with name and address.
            4) Tell the user their order has been placed and give the total and order id.
            """
        super().__init__(instructions=instructions)

        self.catalog: List[Dict[str, Any]] = self._load_catalog()
        self.cart: Dict[str, Dict[str, Any]] = {}

        self.recipes: Dict[str, Dict[str, Any]] = {

            "cheesy toast": {
                "items": [
                    {"id": "bread_wheat_400g", "quantity_per_serving": 0.25},
                    {"id": "cheese_mozzarella_200g", "quantity_per_serving": 0.25}
                ]
            },

            "movie night combo": {
                "items": [
                    {"id": "snacks_chips_masala_100g", "quantity_per_serving": 0.5},
                    {"id": "snacks_biscuits_dig_200g", "quantity_per_serving": 0.25}
                ]
            },
        }


    def _load_catalog(self) -> List[Dict[str, Any]]:
        """Load catalog.json from the data directory."""
        try:
            base_dir = Path(__file__).resolve().parent
            catalog_path = base_dir / "data" / "catalog.json"
            with catalog_path.open("r", encoding="utf-8") as f:
                catalog = json.load(f)
            logger.info(f"Loaded catalog with {len(catalog)} items from {catalog_path}")
            return catalog
        except Exception as e:
            logger.exception("Failed to load catalog.json")
            return []


    def _find_item_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        for item in self.catalog:
            if item.get("id") == item_id:
                return item
        return None

    def _normalize(self, s: str) -> str:
        # lowercase and replace punctuation with spaces
        return "".join(
            ch.lower() if ch.isalnum() or ch.isspace() else " "
            for ch in s
        )

    def _search_items(
        self,
        query: Optional[str],
        category: Optional[str],
    ) -> List[Dict[str, Any]]:
        results = self.catalog

        # Filter by category only if non-empty string
        if category:
            cat = category.strip().lower()
            if cat:
                results = [
                    i
                    for i in results
                    if i.get("category", "").lower() == cat
                ]

        if query:
            q = query.strip()
            if q:
                q_tokens = [t for t in self._normalize(q).split() if t]

                def matches(item: Dict[str, Any]) -> bool:
                    fields = [
                        item.get("name", ""),
                        item.get("brand", ""),
                        " ".join(str(tag) for tag in item.get("tags", [])),
                    ]
                    combined = self._normalize(" ".join(fields))
                    return all(tok in combined for tok in q_tokens)

                results = [i for i in results if matches(i)]

        logger.debug(
            f"search_catalog -> query={query!r}, category={category!r}, results_count={len(results)}"
        )
        return results


    def _compute_cart_totals(self) -> Dict[str, Any]:
        items_out: List[Dict[str, Any]] = []
        subtotal = 0.0
        for item_id, entry in self.cart.items():
            item = entry["item"]
            quantity = entry["quantity"]
            price = float(item.get("price", 0.0))
            line_total = price * quantity
            subtotal += line_total
            items_out.append(
                {
                    "id": item_id,
                    "name": item.get("name"),
                    "category": item.get("category"),
                    "unit_price": price,
                    "quantity": quantity,
                    "line_total": line_total,
                }
            )
        return {"items": items_out, "subtotal": subtotal}

    def _match_recipe(self, meal_name: str) -> Optional[str]:
        """Return the key of the recipe that best matches meal_name."""
        name = meal_name.lower()
        for recipe_name in self.recipes.keys():
            if recipe_name in name or name in recipe_name:
                return recipe_name
        return None

    @function_tool
    async def search_catalog(
        self,
        context: RunContext,
        query: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search the food & grocery catalog.

        Args:
            query: Free-text query like "peanut butter", "chips", "sandwich". Optional.
            category: Category filter such as "groceries", "snacks", "prepared_food". Optional.
        """
        results = self._search_items(query=query, category=category)
        return {
            "query": query,
            "category": category,
            "count": len(results),
            "items": results,
        }


    @function_tool
    async def add_item_to_cart(
        self,
        context: RunContext,
        item_id: str,
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """
        Add a specific catalog item to the cart.

        Use this when:
        - The user clearly asks for a specific product and quantity, e.g.:
          "Add 2 Whole Wheat Bread", "Add 1 pack of pasta_penne_500g".

        Args:
            item_id: The catalog item id (e.g. "bread_wheat_400g").
            quantity: How many units to add. Defaults to 1.

        Behavior:
            - If the item is already in the cart, increase its quantity.
            - Returns the updated cart summary and subtotal.
        """
        if quantity <= 0:
            quantity = 1

        item = self._find_item_by_id(item_id)
        if not item:
            return {
                "success": False,
                "message": f"Item with id '{item_id}' not found in catalog.",
            }

        if item_id not in self.cart:
            self.cart[item_id] = {"item": item, "quantity": 0}

        self.cart[item_id]["quantity"] += quantity

        cart_summary = self._compute_cart_totals()
        return {
            "success": True,
            "action": "add",
            "added_item": {
                "id": item_id,
                "name": item.get("name"),
                "quantity_added": quantity,
            },
            "cart": cart_summary,
        }

    @function_tool
    async def update_cart_item(
        self,
        context: RunContext,
        item_id: str,
        quantity: int,
    ) -> Dict[str, Any]:
        """
        Update the quantity of an item in the cart.

        Use this when:
        - User says things like:
          "Change the bread to 2", "Make the pasta 3 packs", "Set peanut butter to 0".

        Args:
            item_id: Catalog item id already present in the cart.
            quantity: New quantity. If <= 0, the item is removed.

        Returns:
            Updated cart with subtotal.
        """
        if item_id not in self.cart:
            return {
                "success": False,
                "message": f"Item with id '{item_id}' is not in the cart.",
            }

        if quantity <= 0:
            removed = self.cart.pop(item_id)
            cart_summary = self._compute_cart_totals()
            return {
                "success": True,
                "action": "remove",
                "removed_item": {
                    "id": item_id,
                    "name": removed["item"].get("name"),
                },
                "cart": cart_summary,
            }

        self.cart[item_id]["quantity"] = quantity
        cart_summary = self._compute_cart_totals()
        return {
            "success": True,
            "action": "update",
            "updated_item": {
                "id": item_id,
                "name": self.cart[item_id]["item"].get("name"),
                "quantity": quantity,
            },
            "cart": cart_summary,
        }

    @function_tool
    async def remove_cart_item(
        self,
        context: RunContext,
        item_id: str,
    ) -> Dict[str, Any]:
        """
        Remove an item completely from the cart.

        Use this when:
        - User says: "Remove the chips", "Delete the sandwich from my cart".

        Args:
            item_id: Catalog item id.

        Returns:
            Updated cart with subtotal.
        """
        if item_id not in self.cart:
            return {
                "success": False,
                "message": f"Item with id '{item_id}' is not in the cart.",
            }

        removed = self.cart.pop(item_id)
        cart_summary = self._compute_cart_totals()
        return {
            "success": True,
            "action": "remove",
            "removed_item": {
                "id": item_id,
                "name": removed["item"].get("name"),
            },
            "cart": cart_summary,
        }

    @function_tool
    async def list_cart(
        self,
        context: RunContext,
    ) -> Dict[str, Any]:
        """
        List all items currently in the cart, along with quantities and subtotal.

        Use this when:
        - User asks: "What's in my cart?", "Read out my order so far", "Show me my items".
        """
        cart_summary = self._compute_cart_totals()
        return {
            "success": True,
            "cart": cart_summary,
        }

    @function_tool
    async def add_meal_ingredients(
        self,
        context: RunContext,
        meal_name: str,
        servings: int = 1,
    ) -> Dict[str, Any]:
        """
        Add ingredients for a simple meal (recipe) to the cart.

        Use this when:
        - User says: "I need ingredients for a peanut butter sandwich."
        - Or: "Get me what I need for pasta for two people."

        Args:
            meal_name: Name of the meal (e.g. "peanut butter sandwich", "pasta").
            servings: Number of servings / people. Defaults to 1.

        Behavior:
            - Looks up a known recipe.
            - Multiplies quantities by the number of servings (rounded up to full packs).
            - Adds all required items to the cart.
        """
        if servings <= 0:
            servings = 1

        recipe_key = self._match_recipe(meal_name)
        if not recipe_key:
            return {
                "success": False,
                "message": f"No recipe found for '{meal_name}'.",
            }

        recipe = self.recipes[recipe_key]
        added_items: List[Dict[str, Any]] = []

        for entry in recipe["items"]:
            item_id = entry["id"]
            per_serving = float(entry.get("quantity_per_serving", 1.0))
            total_units = per_serving * float(servings)
            pack_quantity = max(1, int(round(total_units + 0.0001)))

            item = self._find_item_by_id(item_id)
            if not item:
                continue

            if item_id not in self.cart:
                self.cart[item_id] = {"item": item, "quantity": 0}
            self.cart[item_id]["quantity"] += pack_quantity

            added_items.append(
                {
                    "id": item_id,
                    "name": item.get("name"),
                    "quantity_added": pack_quantity,
                }
            )

        cart_summary = self._compute_cart_totals()
        return {
            "success": True,
            "meal": recipe_key,
            "servings": servings,
            "added_items": added_items,
            "cart": cart_summary,
        }

    @function_tool
    async def place_order(
        self,
        context: RunContext,
        customer_name: str,
        delivery_address: str,
    ) -> Dict[str, Any]:
        """
        Place the current order and save it to a JSON file.

        Use this when:
        - User says: "That's all", "Place my order", "I'm done".

        REQUIRED:
            - Before calling, make sure the user has confirmed the cart.
            - Pass the user's name and delivery address.

        Behavior:
            - Computes line totals and overall total.
            - Creates an order JSON with:
                - order_id, timestamp, customer info
                - items (id, name, quantity, unit_price, line_total)
                - total_amount, currency
            - Saves order under ./orders/order_<timestamp>.json
            - Clears the in-memory cart after saving.
        """
        if not self.cart:
            return {
                "success": False,
                "message": "Cannot place order: the cart is empty.",
            }

        summary = self._compute_cart_totals()
        items = summary["items"]
        total_amount = summary["subtotal"]

        order_id = f"QB-{int(datetime.utcnow().timestamp())}"
        timestamp = datetime.utcnow().isoformat()

        order_obj = {
            "order_id": order_id,
            "timestamp": timestamp,
            "status": "Order Placed",
            "customer": {
                "name": customer_name,
                "delivery_address": delivery_address,
            },
            "items": items,
            "total_amount": total_amount,
            "currency": "INR",
        }

        base_dir = Path(__file__).resolve().parent
        orders_dir = base_dir / "orders"
        orders_dir.mkdir(parents=True, exist_ok=True)

        filename_safe_ts = timestamp.replace(":", "-")
        order_path = orders_dir / f"order_{filename_safe_ts}.json"

        try:
            with order_path.open("w", encoding="utf-8") as f:
                json.dump(order_obj, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved order {order_id} to {order_path}")
        except Exception as e:
            logger.exception("Failed to save order JSON")
            return {
                "success": False,
                "message": "Failed to save order to disk.",
                "error": str(e),
            }

        self.cart.clear()

        return {
            "success": True,
            "order_id": order_id,
            "order_path": str(order_path),
            "total_amount": total_amount,
        }


def prewarm(proc: JobProcess):
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
            voice="en-US-ronnie",
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

    await session.say(
        "Hi, I'm FlipFresh, your food and grocery ordering assistant. "
        "You can tell me what you want to order, like bread, milk, or ingredients for a quick meal."
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))