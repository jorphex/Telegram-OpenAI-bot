import logging
import requests
import time
import asyncio
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, error as telegram_error
from telegram.constants import ChatAction
from telegram.ext import Application, MessageHandler, CallbackContext, filters, CommandHandler, CallbackQueryHandler
import telegramify_markdown
from openai import OpenAI

# --- CONFIGURATION ---
TELEGRAM_TOKEN = 'token'
OPENAI_API_KEY = 'key'
client = OpenAI(api_key=OPENAI_API_KEY)

# --- LOGGER SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- DATA CLASSES ---
@dataclass
class ModeConfig:
    """Configuration for a specific operational mode."""
    name: str
    model_id: str
    instructions: str
    emoji: str
    callback_data: str
    temperature: Optional[float] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None

@dataclass
class ChatState:
    """Holds the state for a single chat conversation."""
    mode_config: ModeConfig
    last_response_id: Optional[str] = None
    debounce_task: Optional[asyncio.Task] = None
    pending_messages: List[Dict[str, Any]] = field(default_factory=list)
    
# --- MODE DEFINITIONS ---
# Mode 1: Casual
MODE_CASUAL = ModeConfig(
    name="Casual",
    model_id="gpt-4o-mini",
    instructions="""instructions""",
    temperature=0.8,
    emoji="💬",
    callback_data="set_mode_casual"
)

# Mode 2: Standard
MODE_STANDARD = ModeConfig(
    name="Standard",
    model_id="gpt-4.1-mini",
    instructions="""instructions""",
    temperature=0.6,
    emoji="💭",
    callback_data="set_mode_standard"
)

# Mode 3: Sharp
MODE_SHARP = ModeConfig(
    name="Sharp",
    model_id="gpt-4.1",
    instructions="""instructions""",
    temperature=0.4,
    emoji="💡",
    callback_data="set_mode_sharp"
)
# Mode 4: Reasoning
MODE_REASONING = ModeConfig(
    name="Reasoning",
    model_id="o4-mini",
    instructions="Be nice.",
    reasoning_effort="high", # Set effort
    temperature=None, # Explicitly set temperature to None
    emoji="🤔",
    callback_data="set_mode_reasoning"
)

# Add the new mode to the lookup dictionary
MODES_BY_CALLBACK: Dict[str, ModeConfig] = {
    MODE_CASUAL.callback_data: MODE_CASUAL,
    MODE_STANDARD.callback_data: MODE_STANDARD,
    MODE_SHARP.callback_data: MODE_SHARP,
    MODE_REASONING.callback_data: MODE_REASONING,
}

# --- DEFAULT MODE ---
# Choose which mode is the default when a chat starts
DEFAULT_MODE = MODE_STANDARD 

# --- STATE MANAGEMENT ---
chat_states: Dict[int, ChatState] = {}

# --- HELPER FUNCTIONS ---
def split_text(text, max_length=4050):
    """Splits text into chunks suitable for Telegram."""
    if len(text) <= max_length:
        return [text]
    lines = text.split("\n")
    parts = []
    current_part = ""
    for line in lines:
        if len(current_part) + len(line) + 1 > max_length:
            if current_part:
                parts.append(current_part)
            current_part = line
            while len(current_part) > max_length:
                 parts.append(current_part[:max_length])
                 current_part = current_part[max_length:]
        else:
            if current_part:
                current_part += "\n" + line
            else:
                current_part = line
    if current_part:
        parts.append(current_part)

    final_parts = []
    for part in parts:
        if len(part) > max_length:
            for i in range(0, len(part), max_length):
                final_parts.append(part[i:i+max_length])
        else:
            final_parts.append(part)
    return final_parts


def query_assistant(input_messages: List[Dict[str, Any]],
                    model_id: str,
                    system_instructions: str = None,
                    previous_response_id: str = None,
                    temperature: Optional[float] = None,
                    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None):
    """
    Queries the OpenAI Responses API, handling temperature OR reasoning effort.

    Args:
        input_messages: List of message objects for API 'input'.
        model_id: The ID of the OpenAI model.
        system_instructions: System-level instructions.
        previous_response_id: ID of the previous response.
        temperature: Sampling temperature (0-2). Used if reasoning_effort is None.
        reasoning_effort: Reasoning effort ('low', 'medium', 'high'). Used if not None.

    Returns:
        tuple[str | None, str | None]: Reply text and response ID.
    """
    # --- Prepare the API request payload ---
    api_payload = {
        "model": model_id,
        "input": input_messages,
        "tools": [{"type": "web_search_preview"}],
        "previous_response_id": previous_response_id,
        "truncation": "auto",
        "instructions": system_instructions if system_instructions else None,
    }

    # Add EITHER reasoning effort OR temperature, not both.
    if reasoning_effort:
        api_payload["reasoning"] = {"effort": reasoning_effort}
        logger.debug(f"Using reasoning effort: {reasoning_effort}")
    elif temperature is not None:
        api_payload["temperature"] = temperature
        logger.debug(f"Using temperature: {temperature}")
    else:
        logger.debug("Neither temperature nor reasoning effort specified, using API defaults.")

    # Remove None values from payload (important for optional args like previous_response_id)
    api_payload = {k: v for k, v in api_payload.items() if v is not None}

    logger.debug(f"Sending payload to OpenAI Responses API: {api_payload}")

    try:
        # --- Call the OpenAI Responses API ---
        response = client.responses.create(**api_payload)
        logger.debug(f"Received response object from OpenAI: {response}")
        response_id_to_store = response.id
        reply_text = ""
        if response.output:
            for output_item in response.output:
                if output_item.type == "message" and output_item.role == "assistant":
                    if output_item.content:
                        for content_item in output_item.content:
                            if content_item.type == "output_text":
                                reply_text += content_item.text
                    break
        if not reply_text and response.status == "completed":
             logger.warning(f"OpenAI response completed for model {model_id} but no output_text found.")
             reply_text = ""
        elif response.status != "completed":
             logger.error(f"OpenAI response status was not 'completed' for model {model_id}: {response.status}, Error: {response.error}")
             error_msg = f"Error: OpenAI response issue (Status: {response.status})."
             if response.error and hasattr(response.error, 'message'): error_msg += f" Details: {response.error.message}"
             return error_msg, None
        return reply_text, response_id_to_store
    except Exception as e:
        logger.error(f"Error calling OpenAI Responses API with model {model_id}: {e}", exc_info=True)
        if hasattr(e, 'message'): return f"Error: Could not get response from OpenAI. ({e.message})", None
        else: return "Error: Could not get response from OpenAI.", None

# --- TELEGRAM ---
def _get_chat_state(chat_id: int) -> ChatState:
    """Gets or creates the ChatState for a given chat_id."""
    if chat_id not in chat_states:
        logger.info(f"Creating new chat state for chat_id {chat_id} with default mode '{DEFAULT_MODE.name}'.")
        chat_states[chat_id] = ChatState(mode_config=DEFAULT_MODE)
    return chat_states[chat_id]

async def handle_message(update: Update, context: CallbackContext):
    """Handles incoming messages, accumulates during debounce, schedules processing."""
    if not update.message or not update.message.chat or not update.message.from_user:
        logger.debug("Ignoring update without message, chat, or user.")
        return

    chat_id = update.message.chat.id
    user_id = update.message.from_user.id # Potential future use
    debounce_delay = 2.5 # Seconds for long messages automatically split by Telegram, can be shorter like 0.5

    # Get or create the state for this chat
    chat_state = _get_chat_state(chat_id)

    # --- Extract message content ---
    message_content_parts = []
    is_media = False

    if update.message.text:
        message_content_parts.append({"type": "input_text", "text": update.message.text})
        logger.info(f"Chat {chat_id}: Received text: '{update.message.text[:50]}...'")

    if update.message.photo:
        is_media = True
        photo = update.message.photo[-1]
        try:
            photo_file = await photo.get_file()
            image_url = photo_file.file_path
            message_content_parts.append({"type": "input_image", "image_url": image_url})
            logger.info(f"Chat {chat_id}: Received photo (caption: {update.message.caption})")
            if update.message.caption and not update.message.text:
                 message_content_parts.insert(0, {"type": "input_text", "text": update.message.caption})

        except Exception as e:
            logger.error(f"Chat {chat_id}: Error processing photo: {e}")
            await context.bot.send_message(chat_id=chat_id, text="⚠️ Sorry, I couldn't process the image.")

    # --- Accumulate content ---
    if not message_content_parts:
        logger.info(f"Chat {chat_id}: Ignoring message with no processable content (e.g., sticker).")
        return

    chat_state.pending_messages.extend(message_content_parts)
    logger.debug(f"Chat {chat_id}: Added content to pending. Total pending: {len(chat_state.pending_messages)}")

    # --- Debounce logic ---
    if chat_state.debounce_task and not chat_state.debounce_task.done():
        try:
            chat_state.debounce_task.cancel()
            logger.debug(f"Chat {chat_id}: Previous debounce task cancelled.")
        except Exception as e:
            logger.error(f"Chat {chat_id}: Error cancelling debounce task: {e}", exc_info=False)

    # Define the coroutine to run after the delay
    async def _schedule_processing(delay: float, target_chat_id: int):
        try:
            await asyncio.sleep(delay)
            logger.info(f"Chat {target_chat_id}: Debounce delay finished. Processing accumulated messages.")
            await _process_accumulated_messages(target_chat_id, context)
        except asyncio.CancelledError:
            logger.debug(f"Chat {target_chat_id}: Debounce task cancelled (new message arrived).")
        except Exception as e:
            logger.error(f"Chat {target_chat_id}: Error during scheduled processing: {e}", exc_info=True)
            try:
                 await context.bot.send_message(target_chat_id, "😭 Apologies, an internal error occurred while processing your request.")
            except Exception as send_err:
                 logger.error(f"Chat {target_chat_id}: Failed to send processing error message: {send_err}")
        finally:
            current_state = chat_states.get(target_chat_id)
            if current_state and current_state.debounce_task and current_state.debounce_task.done():
                 current_state.debounce_task = None

    logger.debug(f"Chat {chat_id}: Scheduling processing in {debounce_delay}s.")
    new_task = asyncio.create_task(_schedule_processing(debounce_delay, chat_id))
    chat_state.debounce_task = new_task

async def _process_accumulated_messages(chat_id: int, context: CallbackContext):
    """Processes all accumulated messages for a chat after debounce, combining consecutive text."""
    chat_state = _get_chat_state(chat_id)

    if not chat_state.pending_messages:
        logger.warning(f"Chat {chat_id}: Processing called with no pending messages. Skipping.")
        return

    # --- Prepare for API Call ---
    messages_to_process = list(chat_state.pending_messages)
    chat_state.pending_messages.clear()
    logger.info(f"Chat {chat_id}: Processing {len(messages_to_process)} accumulated message parts.")
    combined_content_parts = []
    for part in messages_to_process:
        if part.get("type") == "input_text":
            if combined_content_parts and combined_content_parts[-1].get("type") == "input_text":
                combined_content_parts[-1]["text"] += "\n" + part.get("text", "")
                logger.debug(f"Chat {chat_id}: Appended text to previous part.")
            else:
                combined_content_parts.append(part)
                logger.debug(f"Chat {chat_id}: Added new text part.")
        else:
            combined_content_parts.append(part)
            logger.debug(f"Chat {chat_id}: Added non-text part ({part.get('type')}).")
    if not combined_content_parts:
         logger.warning(f"Chat {chat_id}: No processable content after combining. Skipping API call.")
         return
    api_input_payload = [{"role": "user", "content": combined_content_parts}]

    current_mode = chat_state.mode_config
    previous_response_id = chat_state.last_response_id
    mode_param_log = f"Effort={current_mode.reasoning_effort}" if current_mode.reasoning_effort is not None else f"Temp={current_mode.temperature}"
    logger.info(f"Processing for Chat {chat_id}: Mode='{current_mode.name}', Model='{current_mode.model_id}', {mode_param_log}, PrevID='{previous_response_id}'")
    
    # --- Call the Assistant ---
    reply_text = None
    response_id = None
    assistant_task = None
    typing_task = None
    stop_typing = asyncio.Event()

    try:
        # --- Start Typing Indicator ---
        await asyncio.sleep(0.5)
        async def keep_typing():
            while not stop_typing.is_set():
                try:
                    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
                    await asyncio.sleep(4)
                except asyncio.CancelledError:
                    logger.debug(f"Chat {chat_id}: Keep typing task cancelled.")
                    break
                except Exception as e:
                    logger.warning(f"Chat {chat_id}: Error sending chat action (might be expected): {e}")
                    await asyncio.sleep(10)

        typing_task = asyncio.create_task(keep_typing())

        # --- Call the Assistant ---
        assistant_task = asyncio.to_thread(
            query_assistant,
            api_input_payload,
            model_id=current_mode.model_id,
            system_instructions=current_mode.instructions,
            previous_response_id=previous_response_id,
            temperature=current_mode.temperature,
            reasoning_effort=current_mode.reasoning_effort
        )
        reply_text, response_id = await assistant_task

        # --- Log response ---
        if reply_text is not None: logger.info(f"Assistant raw response for chat_id {chat_id} (Mode: {current_mode.name}, Model: {current_mode.model_id}): {reply_text}")
        elif response_id: logger.info(f"Assistant call succeeded for chat_id {chat_id} (Mode: {current_mode.name}, Model: {current_mode.model_id}) but returned empty text.")

        # --- Update State ---
        if response_id:
            chat_state.last_response_id = response_id
            logger.debug(f"Chat {chat_id}: Stored new response_id: {response_id}")
        else:
            if chat_state.last_response_id:
                 logger.info(f"Chat {chat_id}: Clearing previous response_id due to error or missing ID in current response.")
                 chat_state.last_response_id = None

        # --- Format and Prepare Reply ---
        if reply_text is None or reply_text == "":
             logger.warning(f"Assistant task finished but reply_text is empty/None for chat {chat_id}. No message sent.")
             stop_typing.set()
             return

        original_reply_text_for_logging = reply_text
        if reply_text.startswith("Error:"):
            reply_text += (f"\n\nCurrent Mode: **{current_mode.name}**\nTap a button below to start a new conversation or switch mode.")

        converted_reply = reply_text
        try:
            if not original_reply_text_for_logging.startswith("Error:"):
                 converted_reply = telegramify_markdown.markdownify(reply_text)
                 logger.debug(f"Markdown conversion successful for chat_id {chat_id}.")
            else:
                 logger.debug(f"Skipping markdown conversion for error message chat_id {chat_id}.")
        except Exception as e:
            logger.warning(f"Markdown conversion failed for chat_id {chat_id}: {e}. Sending as plain text.")
            converted_reply = reply_text # Fallback

        logger.info(f"Formatted response prepared for chat_id {chat_id}. Length: {len(converted_reply)}")

        # --- Send Reply Chunks ---
        chunks = split_text(converted_reply, 4050)
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            keyboard = None
            if idx == total_chunks - 1:
                keyboard = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton(text=f"{MODE_CASUAL.emoji} New {MODE_CASUAL.name}", callback_data=MODE_CASUAL.callback_data),
                        InlineKeyboardButton(text=f"{MODE_STANDARD.emoji} New {MODE_STANDARD.name}", callback_data=MODE_STANDARD.callback_data)
                    ],
                    [
                        InlineKeyboardButton(text=f"{MODE_SHARP.emoji} New {MODE_SHARP.name}", callback_data=MODE_SHARP.callback_data),
                        InlineKeyboardButton(text=f"{MODE_REASONING.emoji} New {MODE_REASONING.name}", callback_data=MODE_REASONING.callback_data)
                    ]
                ])

            parse_mode_to_use = None
            if converted_reply != reply_text: parse_mode_to_use = "MarkdownV2"

            # Send the chunk
            await context.bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode=parse_mode_to_use,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
            logger.debug(f"Successfully sent chunk {idx+1}/{total_chunks} to chat_id {chat_id} (ParseMode: {parse_mode_to_use})")

        # --- Stop Typing AFTER sending all chunks ---

    except telegram_error.BadRequest as e:
         if parse_mode_to_use == "MarkdownV2" and "can't parse entities" in str(e):
             logger.error(f"MarkdownV2 parse error sending chunk for chat {chat_id}: {e}. Retrying as plain text.")
             try:
                 await context.bot.send_message(chat_id=chat_id, text="😭 Sorry, there was a formatting problem sending my response.")
             except Exception as retry_e: logger.error(f"Error sending plain text retry chunk for chat {chat_id}: {retry_e}")
         else:
             logger.error(f"BadRequest error sending chunk for chat {chat_id} (ParseMode: {parse_mode_to_use}): {e}")
             try: await context.bot.send_message(chat_id=chat_id, text="😭 Sorry, there was a problem sending my response.")
             except Exception as final_err: logger.error(f"Failed even to send the final error message for chat {chat_id}: {final_err}")
         stop_typing.set()

    except Exception as e:
        logger.error(f"Error during message processing/sending for chat {chat_id}: {e}", exc_info=True)
        try:
            if not (reply_text and reply_text.startswith("Error:")) :
                 await context.bot.send_message(chat_id=chat_id, text="😭 Sorry, an unexpected error occurred while preparing your response.")
        except Exception as final_err:
             logger.error(f"Failed even to send the final error message for chat {chat_id}: {final_err}")
        stop_typing.set()

    finally:
        # 1. Signal the keep_typing loop to stop.
        stop_typing.set()

        # 2. Clean up the typing task.
        if typing_task and not typing_task.done():
            typing_task.cancel()
            try:
                await asyncio.wait_for(typing_task, timeout=0.1)
            except asyncio.TimeoutError:
                logger.debug(f"Chat {chat_id}: Typing task cleanup timed out (likely ok).")
            except asyncio.CancelledError:
                 logger.debug(f"Chat {chat_id}: Typing task already cancelled.")
            except Exception as e:
                logger.error(f"Chat {chat_id}: Error during typing task cleanup: {e}")
        logger.debug(f"Chat {chat_id}: Typing indicator stopped and cleaned up.")
        
# --- CALLBACK ---
async def set_mode_callback(update: Update, context: CallbackContext):
    """Handles all mode-setting button presses."""
    query = update.callback_query
    if not query or not query.data or not query.message:
        logger.warning("Received callback query without data or message.")
        return

    callback_data = query.data
    chat_id = query.message.chat.id

    selected_mode_config = MODES_BY_CALLBACK.get(callback_data)

    if not selected_mode_config:
        logger.error(f"Chat {chat_id}: Received unknown callback data: {callback_data}")
        try:
             await query.edit_message_text(text="⚠️ Error: Unknown mode selected.", reply_markup=None)
        except Exception as e:
             logger.error(f"Chat {chat_id}: Failed to edit message on unknown callback: {e}")
        return

    chat_state = _get_chat_state(chat_id)

    # --- Update chat state ---
    chat_state.mode_config = selected_mode_config
    chat_state.last_response_id = None
    chat_state.pending_messages.clear()

    # Cancel any pending debounce task for this chat when switching modes
    if chat_state.debounce_task and not chat_state.debounce_task.done():
        try:
            chat_state.debounce_task.cancel()
            chat_state.debounce_task = None
            logger.debug(f"Chat {chat_id}: Debounce task cancelled due to mode switch.")
        except Exception as e:
            logger.error(f"Chat {chat_id}: Error cancelling debounce task during mode switch: {e}", exc_info=False)

    mode_param_log = f"Temp={selected_mode_config.temperature}" if selected_mode_config.temperature is not None else f"Effort={selected_mode_config.reasoning_effort}"
    logger.info(f"Chat {chat_id}: Switched to {selected_mode_config.name} mode (Model: {selected_mode_config.model_id}, {mode_param_log}).")

    await query.answer(f"{selected_mode_config.emoji} {selected_mode_config.name} conversation started.")

    message_text = f"{selected_mode_config.emoji} **{selected_mode_config.name}** conversation started."
    if selected_mode_config.name == "Casual": message_text += " What's up?\n\nUse me for quick searches, small inquiries, and creative writing tasks in a casual tone."
    elif selected_mode_config.name == "Standard": message_text += " How can I help you?\n\nUse me for pooling ideas, most inquiries, and most writing tasks in a professional tone."
    elif selected_mode_config.name == "Sharp": message_text += " What do you need?\n\nUse me for brainstorming, technical inquiries, and writing tasks that require more precision."
    elif selected_mode_config.name == "Reasoning": message_text += " Ready. Beep boop.\n\nUse me for reasoning, step-by-step thinking, and complex logic."

    try:
        safe_markdown_text = telegramify_markdown.markdownify(message_text)

        await context.bot.send_message(
            chat_id=chat_id,
            text=safe_markdown_text,
            parse_mode="MarkdownV2"
        )

    except telegram_error.BadRequest as e:
         if "can't parse entities" in str(e):
              logger.error(f"Chat {chat_id}: Markdown parse error in mode confirmation: {e}. Sending plain.")
              try:
                   await context.bot.send_message(chat_id=chat_id, text=message_text)
                   await query.edit_message_reply_markup(reply_markup=None)
              except Exception as send_err:
                   logger.error(f"Chat {chat_id}: Failed to send plain text confirmation or edit markup: {send_err}")
         else:
              logger.error(f"Chat {chat_id}: BadRequest sending mode confirmation: {e}")
    except Exception as e:
         logger.error(f"Chat {chat_id}: Unexpected error sending mode confirmation: {e}", exc_info=True)


# --- BOT STARTUP ---
def start_bot():
    """Starts the Telegram bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CallbackQueryHandler(set_mode_callback, pattern=f"^{MODE_CASUAL.callback_data}$"))
    application.add_handler(CallbackQueryHandler(set_mode_callback, pattern=f"^{MODE_STANDARD.callback_data}$"))
    application.add_handler(CallbackQueryHandler(set_mode_callback, pattern=f"^{MODE_SHARP.callback_data}$"))
    application.add_handler(CallbackQueryHandler(set_mode_callback, pattern=f"^{MODE_REASONING.callback_data}$"))
    application.add_handler(MessageHandler(filters.TEXT | filters.PHOTO | filters.CAPTION, handle_message))

    default_param_log = f"Temp={DEFAULT_MODE.temperature}" if DEFAULT_MODE.temperature is not None else f"Effort={DEFAULT_MODE.reasoning_effort}"
    logger.info(f"Bot is running... Default mode: {DEFAULT_MODE.name} (Model: {DEFAULT_MODE.model_id}, {default_param_log})")
    application.run_polling()

if __name__ == '__main__':
    start_bot()
