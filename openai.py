import logging
import asyncio
import base64
import re
import time
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, error as telegram_error
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, MessageHandler, CallbackContext, filters, CommandHandler, CallbackQueryHandler
import telegramify_markdown
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
    supports_web_search: bool = True 
    supports_code_interpreter: bool = False 

@dataclass
class ChatState:
    """Holds the state for a single chat conversation."""
    mode_config: ModeConfig
    last_response_id: Optional[str] = None
    debounce_task: Optional[asyncio.Task] = None
    pending_messages: List[Dict[str, Any]] = field(default_factory=list)
    container_id: Optional[str] = None
    pending_file_ids: List[str] = field(default_factory=list)
    
# --- MODE DEFINITIONS ---
# Mode 1: Casual
MODE_CASUAL = ModeConfig(
    name="Casual",
    model_id="gpt-4o-mini",
    instructions="""instructions""",
    temperature=0.8,
    emoji="💬",
    callback_data="set_mode_casual",
    supports_web_search=True,
    supports_code_interpreter=True  
)

# Mode 2: Standard
MODE_STANDARD = ModeConfig(
    name="Standard",
    model_id="gpt-4.1-mini",
    instructions="""instructions""",
    temperature=0.6,
    emoji="💭",
    callback_data="set_mode_standard",
    supports_web_search=True,
    supports_code_interpreter=True  
)

# Mode 3: Sharp
MODE_SHARP = ModeConfig(
    name="Sharp",
    model_id="gpt-4.1",
    instructions="""instructions""",
    temperature=0.4,
    emoji="💡",
    callback_data="set_mode_sharp",
    supports_web_search=True,
    supports_code_interpreter=True  
)
# Mode 4: Reasoning
MODE_REASONING = ModeConfig(
    name="Reasoning",
    model_id="o4-mini",
    instructions="""instructions""",
    reasoning_effort="high",
    temperature=None,
    emoji="🤔",
    callback_data="set_mode_reasoning",
    supports_web_search=True,
    supports_code_interpreter=True  
)

# Add the new mode to the lookup dictionary
MODES_BY_CALLBACK: Dict[str, ModeConfig] = {
    MODE_CASUAL.callback_data: MODE_CASUAL,
    MODE_STANDARD.callback_data: MODE_STANDARD,
    MODE_SHARP.callback_data: MODE_SHARP,
    MODE_REASONING.callback_data: MODE_REASONING,
}

# --- DEFAULT MODE ---
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
                    tools: List[Dict[str, Any]],
                    system_instructions: str = None,
                    previous_response_id: str = None,
                    temperature: Optional[float] = None,
                    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None):
    """
    Queries the OpenAI Responses API, handling tools, temperature, and reasoning effort.
    """
    # --- Prepare the API request payload ---
    api_payload = {
        "model": model_id,
        "input": input_messages,
        "previous_response_id": previous_response_id,
        "truncation": "auto",
        "instructions": system_instructions,
        "include": [
            "code_interpreter_call.outputs",
            "message.input_image.image_url"
        ],
    }

    if tools:
        api_payload["tools"] = tools
    if reasoning_effort is not None:
        api_payload["reasoning"] = {"effort": reasoning_effort}
    elif temperature is not None:
        api_payload["temperature"] = temperature

    # Remove None values from payload
    api_payload = {k: v for k, v in api_payload.items() if v is not None}

    logger.debug(f"Sending payload to OpenAI Responses API: {api_payload}")

    try:
        response = client.responses.create(**api_payload)
        logger.debug(f"Received response object from OpenAI: {response}")
        return response
    except Exception as e:
        logger.error(f"Error calling OpenAI Responses API with model {model_id}: {e}", exc_info=True)
        return e

# --- TELEGRAM HANDLERS ---

def _get_chat_state(chat_id: int) -> ChatState:
    """Gets or creates the ChatState for a given chat_id."""
    if chat_id not in chat_states:
        logger.info(f"Creating new chat state for chat_id {chat_id} with default mode '{DEFAULT_MODE.name}'.")
        chat_states[chat_id] = ChatState(mode_config=DEFAULT_MODE)
    return chat_states[chat_id]

async def handle_message(update: Update, context: CallbackContext):
    """Handles incoming messages, including files, accumulates during debounce, schedules processing."""
    if not update.message or not update.message.chat or not update.message.from_user:
        logger.debug("Ignoring update without message, chat, or user.")
        return

    chat_id = update.message.chat.id
    debounce_delay = 3.0 # 3 seconds

    # Get or create the state for this chat
    chat_state = _get_chat_state(chat_id)

    # --- Extract message content ---
    message_content_parts = []
    file_uploaded = False

    # --- Handle Document Uploads ---
    if update.message.document:
        # Check if the current mode supports Code Interpreter
        if not chat_state.mode_config.supports_code_interpreter:
            await context.bot.send_message(chat_id=chat_id, text="⚠️ File uploads are only supported in modes with Code Interpreter enabled.")
            return

        document = update.message.document
        logger.info(f"Chat {chat_id}: Received document '{document.file_name}' ({document.mime_type}).")
        await context.bot.send_message(chat_id=chat_id, text=f"⏳ Uploading '{document.file_name}' to be used by Code Interpreter...")
        try:
            # Download file content from Telegram into memory
            tg_file = await document.get_file()
            file_bytes = await tg_file.download_as_bytearray()

            # Upload the file to OpenAI
            # The 'purpose' must be 'assistants' for Code Interpreter
            openai_file = client.files.create(
                file=(document.file_name, bytes(file_bytes)),
                purpose="assistants"
            )
            logger.info(f"Chat {chat_id}: File uploaded to OpenAI with ID: {openai_file.id}")

            # Add the file ID to the pending list for this chat
            chat_state.pending_file_ids.append(openai_file.id)
            file_uploaded = True

            # Add the caption as a text part if it exists
            if update.message.caption:
                message_content_parts.append({"type": "input_text", "text": update.message.caption})

        except Exception as e:
            logger.error(f"Chat {chat_id}: Failed to upload file to OpenAI: {e}", exc_info=True)
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Error uploading '{document.file_name}'. Please try again.")
            return

    if update.message.text:
        message_content_parts.append({"type": "input_text", "text": update.message.text})
        logger.info(f"Chat {chat_id}: Received text: '{update.message.text[:50]}...'")

    if update.message.photo:
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
    if not message_content_parts and not file_uploaded:
        logger.info(f"Chat {chat_id}: Ignoring message with no processable content.")
        return

    if message_content_parts:
        chat_state.pending_messages.extend(message_content_parts)
        logger.debug(f"Chat {chat_id}: Added content to pending. Total pending: {len(chat_state.pending_messages)}")

    # --- Debounce ---
    if chat_state.debounce_task and not chat_state.debounce_task.done():
        try:
            chat_state.debounce_task.cancel()
            logger.debug(f"Chat {chat_id}: Previous debounce task cancelled.")
        except Exception as e:
            logger.error(f"Chat {chat_id}: Error cancelling debounce task: {e}", exc_info=False)

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
    """Builds tools, processes messages, and handles direct image outputs from Code Interpreter."""
    chat_state = _get_chat_state(chat_id)

    if not chat_state.pending_messages and not chat_state.pending_file_ids:
        logger.warning(f"Chat {chat_id}: Processing called with no pending content. Skipping.")
        return
    current_mode = chat_state.mode_config
    tools_to_use = []
    if current_mode.supports_web_search: tools_to_use.append({"type": "web_search_preview"})
    if current_mode.supports_code_interpreter:
        code_interpreter_tool = {"type": "code_interpreter", "container": {"type": "auto"}}
        if chat_state.pending_file_ids:
            code_interpreter_tool["container"]["file_ids"] = list(chat_state.pending_file_ids)
        tools_to_use.append(code_interpreter_tool)
    messages_to_process = list(chat_state.pending_messages)
    chat_state.pending_messages.clear()
    chat_state.pending_file_ids.clear()
    combined_content_parts = []
    for part in messages_to_process:
        if part.get("type") == "input_text":
            if combined_content_parts and combined_content_parts[-1].get("type") == "input_text":
                combined_content_parts[-1]["text"] += "\n" + part.get("text", "")
            else:
                combined_content_parts.append(part)
        else:
            combined_content_parts.append(part)
    if not combined_content_parts and tools_to_use:
         combined_content_parts.append({"type": "input_text", "text": "Process the attached file(s)."})
    api_input_payload = [{"role": "user", "content": combined_content_parts}]
    previous_response_id = chat_state.last_response_id
    mode_param_log = f"Effort={current_mode.reasoning_effort}" if current_mode.reasoning_effort is not None else f"Temp={current_mode.temperature}"
    logger.info(f"Processing for Chat {chat_id}: Mode='{current_mode.name}', Model='{current_mode.model_id}', {mode_param_log}, PrevID='{previous_response_id}'")
    typing_task = None
    stop_typing = asyncio.Event()

    try:
        await asyncio.sleep(0.5)
        async def keep_typing():
            while not stop_typing.is_set():
                try:
                    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
                    await asyncio.sleep(4)
                except asyncio.CancelledError: break
                except Exception as e:
                    logger.warning(f"Chat {chat_id}: Error sending chat action: {e}")
                    await asyncio.sleep(10)
        typing_task = asyncio.create_task(keep_typing())

        response = await asyncio.to_thread(
            query_assistant,
            api_input_payload,
            model_id=current_mode.model_id,
            tools=tools_to_use,
            system_instructions=current_mode.instructions,
            previous_response_id=previous_response_id,
            temperature=current_mode.temperature,
            reasoning_effort=current_mode.reasoning_effort
        )
        if isinstance(response, Exception):
            raise response

        # --- Process the successful response object ---
        chat_state.last_response_id = response.id
        logger.debug(f"Chat {chat_id}: Stored new response_id: {response.id}")
        reply_text = ""
        has_text_to_send = False

        for output_item in response.output:
            if output_item.type == 'code_interpreter_call':
                if output_item.container_id and chat_state.container_id != output_item.container_id:
                    logger.info(f"Chat {chat_id}: Associated with container_id: {output_item.container_id}")
                    chat_state.container_id = output_item.container_id
                if output_item.outputs:
                    for tool_output in output_item.outputs:
                        if tool_output.get('type') == 'image':
                            logger.info(f"Chat {chat_id}: Found direct image output.")
                            image_url = tool_output.get('url')
                            if image_url:
                                asyncio.create_task(
                                    _handle_base64_image(image_url, chat_id, context)
                                )

            elif output_item.type == 'message' and output_item.role == 'assistant':
                for content_part in output_item.content:
                    if content_part.type == 'output_text' and content_part.text:
                         reply_text += content_part.text
                         has_text_to_send = True

        logger.info(f"Assistant raw response for chat_id {chat_id}: {reply_text}")

        if has_text_to_send:
            original_reply_text_for_logging = reply_text
            converted_reply = telegramify_markdown.markdownify(reply_text)
            logger.info(f"Formatted response prepared for chat_id {chat_id}. Length: {len(converted_reply)}")
            chunks = split_text(converted_reply, 4050)
            for idx, chunk in enumerate(chunks):
                keyboard = None
                if idx == len(chunks) - 1:
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
                await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode="MarkdownV2", reply_markup=keyboard, disable_web_page_preview=True)
        else:
            logger.info(f"Chat {chat_id}: No text content in assistant's final message to send.")

    except Exception as e:
        logger.error(f"Error during message processing for chat {chat_id}: {e}", exc_info=True)
        try:
            await context.bot.send_message(chat_id=chat_id, text="😭 Sorry, an unexpected error occurred while preparing your response.")
        except Exception as final_err:
             logger.error(f"Failed even to send the final error message for chat {chat_id}: {final_err}")
        chat_state.last_response_id = None

    finally:
        stop_typing.set()
        if typing_task and not typing_task.done():
            typing_task.cancel()
            try: await asyncio.wait_for(typing_task, timeout=0.1)
            except (asyncio.TimeoutError, asyncio.CancelledError): pass
            except Exception as e: logger.error(f"Chat {chat_id}: Error during typing task cleanup: {e}")
        logger.debug(f"Chat {chat_id}: Typing indicator stopped and cleaned up.")

async def _handle_base64_image(data_url: str, chat_id: int, context: CallbackContext) -> Optional[str]:
    """Decodes a Base64 data URL, sends it as a photo, and returns the filename on success."""
    try:
        match = re.match(r'data:image/(?P<format>\w+);base64,(?P<data>.+)', data_url)
        if not match:
            logger.error(f"Chat {chat_id}: Could not parse Base64 data URL.")
            return None

        image_data = match.group('data')
        image_format = match.group('format')
        image_bytes = base64.b64decode(image_data)
        filename = f"generated_graph_{int(time.time())}.{image_format}"

        logger.info(f"Chat {chat_id}: Sending decoded {image_format} image to user as '{filename}'.")
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=image_bytes,
            filename=filename,
            caption="Here is the generated graph."
        )
        return filename
    except Exception as e:
        logger.error(f"Chat {chat_id}: Failed to decode or send Base64 image: {e}", exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text="⚠️ I created a graph but failed to send the image.")
        return None

        
# --- CALLBACK ---
async def set_mode_callback(update: Update, context: CallbackContext):
    """Handles all mode-setting button presses."""
    query = update.callback_query
    if not query or not query.data or not query.message:
        logger.warning("Received callback query without data or message.")
        return

    callback_data = query.data
    chat_id = query.message.chat.id

    # Find the selected mode config based on callback data
    selected_mode_config = MODES_BY_CALLBACK.get(callback_data)

    if not selected_mode_config:
        logger.error(f"Chat {chat_id}: Received unknown callback data: {callback_data}")
        try:
             await query.edit_message_text(text="⚠️ Error: Unknown mode selected.", reply_markup=None)
        except Exception as e:
             logger.error(f"Chat {chat_id}: Failed to edit message on unknown callback: {e}")
        return

    # Get or create chat state
    chat_state = _get_chat_state(chat_id)

    # --- Update chat state ---
    chat_state.mode_config = selected_mode_config
    chat_state.last_response_id = None # Reset conversation history on mode switch
    chat_state.pending_messages.clear() # Clear any pending messages on mode switch

    # Cancel any pending debounce task for this chat when switching modes
    if chat_state.debounce_task and not chat_state.debounce_task.done():
        try:
            chat_state.debounce_task.cancel()
            chat_state.debounce_task = None # Clear reference
            logger.debug(f"Chat {chat_id}: Debounce task cancelled due to mode switch.")
        except Exception as e:
            logger.error(f"Chat {chat_id}: Error cancelling debounce task during mode switch: {e}", exc_info=False)

    mode_param_log = f"Temp={selected_mode_config.temperature}" if selected_mode_config.temperature is not None else f"Effort={selected_mode_config.reasoning_effort}"
    logger.info(f"Chat {chat_id}: Switched to {selected_mode_config.name} mode (Model: {selected_mode_config.model_id}, {mode_param_log}).")

    await query.answer(f"{selected_mode_config.emoji} {selected_mode_config.name} conversation started.")

    # --- Send a NEW confirmation message ---
    message_text = f"{selected_mode_config.emoji} **{selected_mode_config.name}** conversation started."
    # Add a short description based on mode
    if selected_mode_config.name == "Casual": message_text += " Hi.\n\nUse me for quick searches, small inquiries, and creative writing tasks in a casual tone.\nI can also see files like images, PDFs, and documents."
    elif selected_mode_config.name == "Standard": message_text += " How can I help you?\n\nUse me for pooling ideas, most inquiries, and most writing tasks in a professional tone.\nI can also see files like images, PDFs, and documents."
    elif selected_mode_config.name == "Sharp": message_text += " What do you need?\n\nUse me for brainstorming, technical inquiries, and writing tasks that require more precision.\nI can also see files like images, PDFs, and documents."
    elif selected_mode_config.name == "Reasoning": message_text += " Ready. Beep boop.\n\nUse me for reasoning, step-by-step thinking, and complex logic.\nI can also see files like images, PDFs, and documents.\nNo web search."

    try:
        safe_markdown_text = telegramify_markdown.markdownify(message_text)

        # Always send a new message for confirmation
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
    application.add_handler(MessageHandler(filters.TEXT | filters.PHOTO | filters.CAPTION | filters.Document.ALL, handle_message))

    default_param_log = f"Effort={DEFAULT_MODE.reasoning_effort}" if DEFAULT_MODE.reasoning_effort is not None else f"Temp={DEFAULT_MODE.temperature}"
    logger.info(f"Bot is running... Default mode: {DEFAULT_MODE.name} (Model: {DEFAULT_MODE.model_id}, {default_param_log})")
    application.run_polling()

if __name__ == '__main__':
    start_bot()
