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

# --- MODE DEFINITIONS ---
# Replace these with actual values as needed.
# Mode 1: Casual
MODE_NAME_CASUAL = "Casual"
MODEL_ID_CASUAL = "model-1"
INSTRUCTIONS_CASUAL = """instructions 1"""
TEMPERATURE_CASUAL = 0.8
CALLBACK_CASUAL = "set_mode_casual"
EMOJI_CASUAL = "💬"

# Mode 2: Standard
MODE_NAME_STANDARD = "Standard"
MODEL_ID_STANDARD = "model-2"
INSTRUCTIONS_STANDARD = """instructions 2"""
TEMPERATURE_STANDARD = 0.6
CALLBACK_STANDARD = "set_mode_standard"
EMOJI_STANDARD = "💭"

# Mode 3: Analytical
MODE_NAME_ANALYTICAL = "Analytical"
MODEL_ID_ANALYTICAL = "model-3"
INSTRUCTIONS_ANALYTICAL = """instructions 3"""
TEMPERATURE_ANALYTICAL = 0.4
CALLBACK_ANALYTICAL = "set_mode_analytical"
EMOJI_ANALYTICAL = "🧠"

# --- STATE MANAGEMENT DICTIONARIES ---
system_message_preferences = {}
model_preferences = {}
conversation_response_ids = {}
mode_name_preferences = {}
temperature_preferences = {}

# --- DEFAULT SETTINGS ---
DEFAULT_MODE_NAME = MODE_NAME_STANDARD
DEFAULT_MODEL = MODEL_ID_STANDARD
DEFAULT_SYSTEM_MESSAGE = INSTRUCTIONS_STANDARD
DEFAULT_TEMPERATURE = TEMPERATURE_STANDARD


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


def query_assistant(user_content, model_id: str, system_instructions: str = None, previous_response_id: str = None, temperature: float = 1.0):
    """Queries the OpenAI Responses API with conversation state, model selection, temperature, and truncation."""
    input_payload = []
    if isinstance(user_content, str):
        input_payload.append({"role": "user", "content": [{"type": "input_text", "text": user_content}]})
    elif isinstance(user_content, list):
        input_payload.append({"role": "user", "content": user_content})
    else:
        logger.warning(f"Unexpected user_content type: {type(user_content)}. Converting to string.")
        input_payload.append({"role": "user", "content": [{"type": "input_text", "text": str(user_content)}]})

    api_payload = {
        "model": model_id,
        "input": input_payload,
        "tools": [{"type": "web_search_preview"}],
        "previous_response_id": previous_response_id,
        "truncation": "auto",
        "instructions": system_instructions if system_instructions else None,
        "temperature": temperature,
    }

    api_payload = {k: v for k, v in api_payload.items() if v is not None}
    logger.debug(f"Sending payload to OpenAI Responses API: {api_payload}")

    try:
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
             return f"Error: OpenAI response issue (Status: {response.status}).", None
        return reply_text, response_id_to_store
    except Exception as e:
        logger.error(f"Error calling OpenAI Responses API with model {model_id}: {e}", exc_info=True)
        return "Error: Could not get response from OpenAI.", None


# --- TELEGRAM ---
async def handle_message(update: Update, context: CallbackContext):
    """Handles incoming messages."""
    global system_message_preferences, model_preferences, conversation_response_ids, mode_name_preferences, temperature_preferences

    if not update.message: return
    chat_id = update.message.chat.id
    message_id = update.message.message_id

    # --- Retrieve state for this chat, using defaults if not set ---
    selected_mode_name = mode_name_preferences.get(chat_id, DEFAULT_MODE_NAME)
    selected_system_instructions = system_message_preferences.get(chat_id, DEFAULT_SYSTEM_MESSAGE)
    selected_model_id = model_preferences.get(chat_id, DEFAULT_MODEL)
    selected_temperature = temperature_preferences.get(chat_id, DEFAULT_TEMPERATURE)
    previous_response_id = conversation_response_ids.get(chat_id)

    logger.info(f"Chat {chat_id}: Mode='{selected_mode_name}', Model='{selected_model_id}', Temp={selected_temperature}, PrevID='{previous_response_id}'")

    user_content = None

    # --- Handle Image Input ---
    if update.message.photo:
        photo = update.message.photo[-1]
        try:
            photo_file = await photo.get_file()
            image_url = photo_file.file_path
            content_items = []
            if update.message.caption:
                content_items.append({"type": "input_text", "text": update.message.caption})
            content_items.append({"type": "input_image", "image_url": image_url})
            user_content = content_items
            logger.info(f"Received photo from chat_id: {chat_id} (message_id: {message_id}) with caption: {update.message.caption}")
        except Exception as e:
            logger.error(f"Error getting photo file/URL for chat {chat_id}: {e}")
            await context.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't process the image file.")
            return

    # --- Handle Text Input ---
    elif update.message.text:
        user_content = update.message.text
        logger.info(f"Received text message from chat_id: {chat_id} (message_id: {message_id}): {user_content}")
    else:
        logger.info(f"Ignoring non-text/photo message from chat_id: {chat_id}")
        return

    # --- Call the Assistant ---
    assistant_task = asyncio.create_task(
        asyncio.to_thread(
            query_assistant,
            user_content,
            model_id=selected_model_id,
            system_instructions=selected_system_instructions,
            previous_response_id=previous_response_id,
            temperature=selected_temperature
        )
    )

    # --- Typing Indicator ---
    stop_typing = asyncio.Event()
    typing_task = None
    await asyncio.sleep(0.5)
    if not assistant_task.done():
        async def keep_typing():
            while not stop_typing.is_set() and not assistant_task.done():
                try: await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
                except Exception as e: logger.warning(f"Error sending chat action (might be expected) for chat {chat_id}: {e}")
                await asyncio.sleep(4)
        typing_task = asyncio.create_task(keep_typing())

    # --- Await Response and Handle Results ---
    reply_text = "Error: An unexpected issue occurred."
    response_id = None
    try:
        reply_text, response_id = await assistant_task
        if reply_text is not None: logger.info(f"Assistant raw response for chat_id {chat_id} (Mode: {selected_mode_name}, Model: {selected_model_id}): {reply_text}")
        elif response_id: logger.info(f"Assistant call succeeded for chat_id {chat_id} (Mode: {selected_mode_name}, Model: {selected_model_id}) but returned empty text.")
    except Exception as e:
        logger.error(f"Error awaiting assistant task for chat {chat_id}: {e}", exc_info=True)
        reply_text = f"Error: An unexpected error occurred while processing your request."
        response_id = None
    finally:
        if typing_task:
            stop_typing.set()
            try: await asyncio.wait_for(typing_task, timeout=0.1)
            except asyncio.TimeoutError: pass
            except Exception as e: logger.error(f"Error cleaning up typing task for chat {chat_id}: {e}")

    # --- Update Conversation State ---
    if response_id:
        conversation_response_ids[chat_id] = response_id
        logger.debug(f"Stored new response_id for chat {chat_id}: {response_id}")
    else:
        if chat_id in conversation_response_ids:
            del conversation_response_ids[chat_id]
            logger.info(f"Cleared response_id for chat {chat_id} due to error or missing ID.")

    # --- Format and Send Reply ---
    if reply_text is None or reply_text == "":
         logger.warning(f"Assistant task finished but reply_text is empty/None for chat {chat_id}. No message sent.")
         return

    original_reply_text_for_logging = reply_text
    if reply_text.startswith("Error:"):
        reply_text += (f"\n\nCurrent Mode: **{selected_mode_name}**\nTap a button below to start a new conversation or switch mode.")

    converted_reply = reply_text
    try:
        if not original_reply_text_for_logging.startswith("Error:"):
             converted_reply = telegramify_markdown.markdownify(reply_text)
             logger.debug(f"Markdown conversion successful for chat_id {chat_id}.")
        else:
             logger.debug(f"Skipping markdown conversion for error message chat_id {chat_id}.")
    except Exception as e:
        logger.warning(f"Markdown conversion failed for chat_id {chat_id}: {e}. Sending as plain text.")
        converted_reply = reply_text

    logger.info(f"Formatted response prepared for chat_id {chat_id}. Length: {len(converted_reply)}")

    chunks = split_text(converted_reply, 4050)
    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks):
        keyboard = None
        if idx == total_chunks - 1:
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton(text=f"{EMOJI_CASUAL} New {MODE_NAME_CASUAL}", callback_data=CALLBACK_CASUAL),
                    InlineKeyboardButton(text=f"{EMOJI_STANDARD} New {MODE_NAME_STANDARD}", callback_data=CALLBACK_STANDARD),
                    InlineKeyboardButton(text=f"{EMOJI_ANALYTICAL} New {MODE_NAME_ANALYTICAL}", callback_data=CALLBACK_ANALYTICAL)
                ]
            ])

        parse_mode_to_use = None
        if converted_reply != reply_text:
            parse_mode_to_use = "MarkdownV2"

        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode=parse_mode_to_use,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
            logger.debug(f"Successfully sent chunk {idx+1}/{total_chunks} to chat_id {chat_id} (ParseMode: {parse_mode_to_use})")

        except telegram_error.BadRequest as e:
             if parse_mode_to_use == "MarkdownV2" and "can't parse entities" in str(e):
                 logger.error(f"MarkdownV2 parse error sending chunk {idx+1} for chat {chat_id}: {e}. Retrying as plain text.")
                 try:
                     original_chunks = split_text(original_reply_text_for_logging, 4050)
                     plain_chunk = original_chunks[idx] if idx < len(original_chunks) else chunk
                     await context.bot.send_message(
                         chat_id=chat_id,
                         text=plain_chunk,
                         reply_markup=keyboard,
                         disable_web_page_preview=True
                     )
                     logger.info(f"Successfully sent chunk {idx+1}/{total_chunks} as plain text retry for chat {chat_id}.")
                 except Exception as retry_e:
                     logger.error(f"Error sending plain text retry chunk {idx+1} for chat {chat_id}: {retry_e}")
             else:
                 logger.error(f"BadRequest error sending chunk {idx+1} for chat {chat_id} (ParseMode: {parse_mode_to_use}): {e}")
                 break
        except Exception as e:
            logger.error(f"General error sending chunk {idx+1} for chat {chat_id}: {e}", exc_info=True)
            if idx == 0:
                 try: await context.bot.send_message(chat_id=chat_id, text="😭 Sorry, there was a problem sending my response.")
                 except Exception as final_err: logger.error(f"Failed even to send the final error message for chat {chat_id}: {final_err}")
            break


# --- CALLBACK HANDLERS FOR MODE SWITCHING ---
async def set_mode_casual(update: Update, context: CallbackContext):
    """Switches to the Casual mode."""
    query = update.callback_query
    await query.answer(f"{EMOJI_CASUAL} Switched to {MODE_NAME_CASUAL} mode.")
    chat_id = query.message.chat.id
    mode_name_preferences[chat_id] = MODE_NAME_CASUAL
    system_message_preferences[chat_id] = INSTRUCTIONS_CASUAL
    model_preferences[chat_id] = MODEL_ID_CASUAL
    temperature_preferences[chat_id] = TEMPERATURE_CASUAL
    conversation_response_ids[chat_id] = None
    logger.info(f"Chat {chat_id}: Switched to {MODE_NAME_CASUAL} mode (Model: {MODEL_ID_CASUAL}, Temp: {TEMPERATURE_CASUAL}).")

    message_text = f"{EMOJI_CASUAL} **{MODE_NAME_CASUAL}** mode activated."
    safe_markdown_text = telegramify_markdown.markdownify(message_text)

    await context.bot.send_message(
        chat_id=chat_id,
        text=safe_markdown_text,
        parse_mode="MarkdownV2"
    )

async def set_mode_standard(update: Update, context: CallbackContext):
    """Switches to the Standard mode."""
    query = update.callback_query
    await query.answer(f"{EMOJI_STANDARD} Switched to {MODE_NAME_STANDARD} mode. What's up?")
    chat_id = query.message.chat.id
    mode_name_preferences[chat_id] = MODE_NAME_STANDARD
    system_message_preferences[chat_id] = INSTRUCTIONS_STANDARD
    model_preferences[chat_id] = MODEL_ID_STANDARD
    temperature_preferences[chat_id] = TEMPERATURE_STANDARD
    conversation_response_ids[chat_id] = None
    logger.info(f"Chat {chat_id}: Switched to {MODE_NAME_STANDARD} mode (Model: {MODEL_ID_STANDARD}, Temp: {TEMPERATURE_STANDARD}).")

    message_text = f"{EMOJI_STANDARD} **{MODE_NAME_STANDARD}** mode activated. Need help with something?"
    safe_markdown_text = telegramify_markdown.markdownify(message_text)

    await context.bot.send_message(
        chat_id=chat_id,
        text=safe_markdown_text,
        parse_mode="MarkdownV2"
    )

async def set_mode_analytical(update: Update, context: CallbackContext):
    """Switches to the Analytical mode."""
    query = update.callback_query
    await query.answer(f"{EMOJI_ANALYTICAL} Switched to {MODE_NAME_ANALYTICAL} mode.")
    chat_id = query.message.chat.id
    mode_name_preferences[chat_id] = MODE_NAME_ANALYTICAL
    system_message_preferences[chat_id] = INSTRUCTIONS_ANALYTICAL
    model_preferences[chat_id] = MODEL_ID_ANALYTICAL
    temperature_preferences[chat_id] = TEMPERATURE_ANALYTICAL
    conversation_response_ids[chat_id] = None
    logger.info(f"Chat {chat_id}: Switched to {MODE_NAME_ANALYTICAL} mode (Model: {MODEL_ID_ANALYTICAL}, Temp: {TEMPERATURE_ANALYTICAL}).")

    message_text = f"{EMOJI_ANALYTICAL} **{MODE_NAME_ANALYTICAL}** mode activated. What do you need?"
    safe_markdown_text = telegramify_markdown.markdownify(message_text)

    await context.bot.send_message(
        chat_id=chat_id,
        text=safe_markdown_text,
        parse_mode="MarkdownV2"
    )

# --- BOT STARTUP ---
def start_bot():
    """Starts the Telegram bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CallbackQueryHandler(set_mode_casual, pattern=f"^{CALLBACK_CASUAL}$"))
    application.add_handler(CallbackQueryHandler(set_mode_standard, pattern=f"^{CALLBACK_STANDARD}$"))
    application.add_handler(CallbackQueryHandler(set_mode_analytical, pattern=f"^{CALLBACK_ANALYTICAL}$"))
    application.add_handler(MessageHandler(filters.ALL & (~filters.COMMAND), handle_message))
    logger.info(f"Bot is running... Default mode: {DEFAULT_MODE_NAME} (Model: {DEFAULT_MODEL}, Temp: {DEFAULT_TEMPERATURE})")
    application.run_polling()

if __name__ == '__main__':
    start_bot()
