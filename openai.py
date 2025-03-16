import logging
import requests
import time
import asyncio
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ChatAction
from telegram.ext import Application, MessageHandler, CallbackContext, filters, CommandHandler, CallbackQueryHandler
import telegramify_markdown
from openai import OpenAI

# --- CONFIGURATION ---
TELEGRAM_TOKEN = 'BOT_TOKEN'
OPENAI_API_KEY = 'OPENAI_KEY'
client = OpenAI(api_key=OPENAI_API_KEY)

# --- LOGGER SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Bot state
system_message_preferences = {}
conversation_response_ids = {}

# --- SYSTEM MESSAGES ---
SYSTEM_MESSAGE_4O_MINI = """instructions here"""

SYSTEM_MESSAGE_OTHER = """other instructions here"""


def split_text(text, max_length=4050):
    """Splits text into chunks."""
    if len(text) <= max_length:
        return [text]
    lines = text.split("\n")
    parts = []
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > max_length:
            parts.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        parts.append(current)
    final_parts = []
    for part in parts:
        if len(part) > max_length:
            for i in range(0, len(part), max_length):
                final_parts.append(part[i:i+max_length])
        else:
            final_parts.append(part)
    return final_parts

def query_assistant(user_content, chat_id, system_message: str = None, previous_response_id: str = None):
    """Queries the OpenAI Responses API with conversation state and truncation."""

    messages = []
    if system_message:
        messages.append({"role": "developer", "content": system_message})

    input_content = []
    if isinstance(user_content, str):
        input_content.append({"type": "input_text", "text": user_content})
    elif isinstance(user_content, list):
        input_content.extend(user_content)
    else:
        input_content.append({"type": "input_text", "text": str(user_content)})

    # Add user message to the messages list
    messages.append({
        "role": "user",
        "content": input_content
    })

    payload_input = {
        "model": "gpt-4o",
        "input": messages,
        "tools": [{"type": "web_search_preview"}],
        "previous_response_id": previous_response_id,
        "truncation": "auto"
    }

    try:
        response = client.responses.create(**payload_input)
        response_id_to_store = response.id

        reply_text = response.output_text if hasattr(response, 'output_text') and response.output_text else ""
        if not reply_text:
            for output_item in response.output:
                if output_item.type == "message":
                    for content_item in output_item.content:
                        if content_item.type == "output_text":
                            reply_text += content_item.text

        return reply_text, response_id_to_store

    except Exception as e:
        logger.error(f"Error calling Responses API: {e}")
        return "Error: Could not get response from OpenAI Responses API.", None


async def handle_message(update: Update, context: CallbackContext):
    global system_message_preferences, conversation_response_ids

    chat_id = update.message.chat.id
    message_id = update.message.message_id

    system_message = system_message_preferences.get(chat_id) # Get system message preference
    previous_response_id = conversation_response_ids.get(chat_id)

    user_content = None
    if update.message.photo:
        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        image_url = photo_file.file_path
        content_items = []
        if update.message.caption:
            content_items.append({"type": "input_text", "text": update.message.caption})
        content_items.append({"type": "input_image", "image_url": image_url})
        user_content = content_items
    else:
        user_content = update.message.text

    logger.info(f"Received message from chat_id: {chat_id} (message_id: {message_id}): {user_content}")

    # Determine system message, default to 4o_mini if no preference set
    effective_system_message = SYSTEM_MESSAGE_4O_MINI if system_message is None else system_message

    assistant_task = asyncio.create_task(
        asyncio.to_thread(query_assistant, user_content, chat_id, effective_system_message, previous_response_id)
    )
    await asyncio.sleep(1)

    stop_typing = asyncio.Event()
    typing_task = None
    if not assistant_task.done():
        async def keep_typing():
            while not stop_typing.is_set() and not assistant_task.done():
                try:
                    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
                except Exception as e:
                    logger.error(f"Error sending chat action: {e}")
                await asyncio.sleep(4)
        typing_task = asyncio.create_task(keep_typing())

    try:
        reply_text, response_id = await assistant_task
    except Exception as e:
        reply_text = f"Error: An unexpected error occurred: {str(e)}"
        response_id = None
    finally:
        if typing_task:
            stop_typing.set()
            await typing_task

    if response_id:
        conversation_response_ids[chat_id] = response_id

    if reply_text.startswith("Error:"):
        reply_text += ("\n\nTap a button below to start a new conversation.")

    try:
        converted_reply = telegramify_markdown.markdownify(reply_text)
    except:
        converted_reply = reply_text

    logger.info(f"Formatted response from LLM: {converted_reply}")

    chunks = split_text(converted_reply, 4050)
    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks):
        if idx == total_chunks - 1:
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton(text="✨ New 4o-mini", callback_data="new_4o_mini"),
                    InlineKeyboardButton(text="✨ New other Writer", callback_data="new_other")
                ]
            ])
        else:
            keyboard = None

        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode="MarkdownV2",
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
        except Exception as e:
            logger.error(f"Error sending chunk {idx+1}: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                text="😭 Sorry, there was a problem sending my response.",
                reply_to_message_id=message_id
            )


async def new_conversation_4o_mini(update: Update, context: CallbackContext):
    """Starts a new conversation with the 4O-MINI system message."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat.id
    system_message_preferences[chat_id] = SYSTEM_MESSAGE_4O_MINI # Store system message preference
    conversation_response_ids[chat_id] = None
    await context.bot.send_message(chat_id=chat_id, text="✨ New 4O-MINI conversation started. How can I help?")

async def new_conversation_other(update: Update, context: CallbackContext):
    """Starts a new conversation with the OTHER system message."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat.id
    system_message_preferences[chat_id] = SYSTEM_MESSAGE_OTHER # Store system message preference
    conversation_response_ids[chat_id] = None
    await context.bot.send_message(chat_id=chat_id, text="✨ New OTHER conversation started. What's up?")


def start_bot():
    """Starts the Telegram bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CallbackQueryHandler(new_conversation_4o_mini, pattern="^new_4o_mini$"))
    application.add_handler(CallbackQueryHandler(new_conversation_other, pattern="^new_other$"))
    application.add_handler(MessageHandler(filters.ALL & (~filters.COMMAND), handle_message))

    logger.info("Bot is running... Waiting for messages.")
    application.run_polling()

if __name__ == '__main__':
    start_bot()
