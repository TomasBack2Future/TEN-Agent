#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import asyncio
import traceback
import random

from openai import AsyncClient
from base64 import b64encode
from io import BytesIO
from PIL import Image
from datetime import datetime
from typing import List

from ten import (
    AudioFrame,
    VideoFrame,
    AsyncExtension,
    AsyncTenEnv,
    Cmd,
    StatusCode,
    CmdResult,
    Data,
)

CMD_IN_FLUSH = "flush"
CMD_IN_ON_USER_JOINED = "on_user_joined"
CMD_IN_ON_USER_LEFT = "on_user_left"
CMD_OUT_FLUSH = "flush"
DATA_IN_TEXT_DATA_PROPERTY_TEXT = "text"
DATA_IN_TEXT_DATA_PROPERTY_IS_FINAL = "is_final"
DATA_OUT_TEXT_DATA_PROPERTY_TEXT = "text"
DATA_OUT_TEXT_DATA_PROPERTY_TEXT_END_OF_SEGMENT = "end_of_segment"

PROPERTY_BASE_URL = "base_url"  # Optional
PROPERTY_API_KEY = "api_key"  # Required
PROPERTY_MODEL = "model"  # Optional
PROPERTY_PROMPT = "prompt"  # Optional
PROPERTY_FREQUENCY_PENALTY = "frequency_penalty"  # Optional
PROPERTY_PRESENCE_PENALTY = "presence_penalty"  # Optional
PROPERTY_TEMPERATURE = "temperature"  # Optional
PROPERTY_TOP_P = "top_p"  # Optional
PROPERTY_MAX_TOKENS = "max_tokens"  # Optional
PROPERTY_GREETING = "greeting"  # Optional
PROPERTY_ENABLE_TOOLS = "enable_tools"  # Optional
PROPERTY_PROXY_URL = "proxy_url"  # Optional
PROPERTY_MAX_MEMORY_LENGTH = "max_memory_length"  # Optional
PROPERTY_CHECKING_VISION_TEXT_ITEMS = "checking_vision_text_items"  # Optional

TASK_TYPE_CHAT_COMPLETION = "chat_completion"
TASK_TYPE_CHAT_COMPLETION_WITH_VISION = "chat_completion_with_vision"


def is_punctuation(char):
    if char in [",", "，", ".", "。", "?", "？", "!", "！"]:
        return True
    return False

def parse_sentences(sentence_fragment, content):
    sentences = []
    current_sentence = sentence_fragment
    for char in content:
        current_sentence += char
        if is_punctuation(char):
            stripped_sentence = current_sentence
            if any(c.isalnum() for c in stripped_sentence):
                sentences.append(stripped_sentence)
            current_sentence = ""

    remain = current_sentence
    return sentences, remain

def rgb2base64jpeg(rgb_data, width, height):
    # Convert the RGB image to a PIL Image
    pil_image = Image.frombytes("RGBA", (width, height), bytes(rgb_data))
    pil_image = pil_image.convert("RGB")

    # Resize the image while maintaining its aspect ratio
    pil_image = resize_image_keep_aspect(pil_image, 320)

    # Save the image to a BytesIO object in JPEG format
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    # pil_image.save("test.jpg", format="JPEG")

    # Get the byte data of the JPEG image
    jpeg_image_data = buffered.getvalue()

    # Convert the JPEG byte data to a Base64 encoded string
    base64_encoded_image = b64encode(jpeg_image_data).decode("utf-8")

    # Create the data URL
    mime_type = "image/jpeg"
    base64_url = f"data:{mime_type};base64,{base64_encoded_image}"
    return base64_url


def resize_image_keep_aspect(image, max_size=512):
    """
    Resize an image while maintaining its aspect ratio, ensuring the larger dimension is max_size.
    If both dimensions are smaller than max_size, the image is not resized.

    :param image: A PIL Image object
    :param max_size: The maximum size for the larger dimension (width or height)
    :return: A PIL Image object (resized or original)
    """
    # Get current width and height
    width, height = image.size

    # If both dimensions are already smaller than max_size, return the original image
    if width <= max_size and height <= max_size:
        return image

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Determine the new dimensions
    if width > height:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(max_size * aspect_ratio)

    # Resize the image with the new dimensions
    resized_image = image.resize((new_width, new_height))

    return resized_image

class VisionExtension(AsyncExtension):
    ten_env: AsyncTenEnv = None
    loop: asyncio.AbstractEventLoop = None
    image_data = None
    image_width = 0
    image_height = 0
    history = []
    max_memory_length = 10
    loop = None
    stopped: bool = False
    sentence_fragment = ""
    outdate_ts = datetime.now()
    
    base_url="https://api.openai.com/v1"
    api_key=""
    model="gpt-4"  # Adjust this to match the equivalent of `openai.GPT4o` in the Python library
    prompt="You are a voice assistant who talks in a conversational way and can chat with me like my friends. I will speak to you in English or Chinese, and you will answer in the corrected and improved version of my text with the language I use. Don’t talk like a robot, instead I would like you to talk like a real human with emotions. I will use your answer for text-to-speech, so don’t return me any meaningless characters. I want you to be helpful, when I’m asking you for advice, give me precise, practical and useful advice instead of being vague. When giving me a list of options, express the options in a narrative way instead of bullet points.",
    frequency_penalty=0.9
    presence_penalty=0.9
    top_p=1.0
    temperature=0.1
    max_tokens=512
    seed=random.randint(0, 10000)
    proxy_url=""

    # Create the queue for message processing
    queue = asyncio.Queue()

    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log_debug("on_init")
        ten_env.on_init_done()

    async def on_start(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log_debug("on_start")

        self.loop = asyncio.get_event_loop()

        # Prepare configuration
        try:
            self.api_key = ten_env.get_property_string(PROPERTY_API_KEY)
        except Exception as e:
            ten_env.log_error(f"Error to get {PROPERTY_API_KEY}: {e}")
            return

        try:
            self.model = ten_env.get_property_string(PROPERTY_MODEL)
        except:
            ten_env.log_debug(f"Error to get {PROPERTY_MODEL}")

        try:
            self.prompt = ten_env.get_property_string(PROPERTY_PROMPT)
        except:
            ten_env.log_debug(f"Error to get {PROPERTY_PROMPT}")

        try:
            self.frequency_penalty = ten_env.get_property_float(PROPERTY_FREQUENCY_PENALTY)
        except:
            ten_env.log_debug(f"Error to get {PROPERTY_FREQUENCY_PENALTY}")

        try:
            self.presence_penalty = ten_env.get_property_float(PROPERTY_PRESENCE_PENALTY)
        except:
            ten_env.log_debug(f"Error to get {PROPERTY_PRESENCE_PENALTY}")

        try:
            self.temperature = ten_env.get_property_float(PROPERTY_TEMPERATURE)
        except:
            ten_env.log_debug(f"Error to get {PROPERTY_TEMPERATURE}")

        try:
            self.top_p = ten_env.get_property_float(PROPERTY_TOP_P)
        except:
            ten_env.log_debug(f"Error to get {PROPERTY_TOP_P}")

        try:
            self.max_tokens = ten_env.get_property_int(PROPERTY_MAX_TOKENS)
        except:
            ten_env.log_debug(f"Error to get {PROPERTY_MAX_TOKENS}")

        try:
            self.greeting = ten_env.get_property_string(PROPERTY_GREETING)
        except:
            ten_env.log_debug(f"Error to get {PROPERTY_GREETING}")
        
        try:
            self.max_memory_length = ten_env.get_property_int(PROPERTY_MAX_MEMORY_LENGTH)
        except:
            ten_env.log_debug(f"Error to get {PROPERTY_MAX_MEMORY_LENGTH}")

        self.ten_env = ten_env
        self.users_count = 0
        self.client = AsyncClient(api_key=self.api_key)
        self.loop.create_task(self._consume())

        ten_env.on_start_done()

    async def on_stop(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log_debug("on_stop")

        self.stopped = True
        await self.queue.put(None)

        ten_env.on_stop_done()

    async def on_deinit(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log_debug("on_deinit")
        ten_env.on_deinit_done()

    async def on_cmd(self, ten_env: AsyncTenEnv, cmd: Cmd) -> None:
        cmd_name = cmd.get_name()
        ten_env.log_debug("on_cmd name {}".format(cmd_name))

        status_code, detail = StatusCode.OK, "success"
        if cmd_name == CMD_IN_FLUSH:
            await self._flush(ten_env)
            await ten_env.send_cmd(Cmd.create(CMD_OUT_FLUSH))
            ten_env.log_info("on_cmd sent flush")
        elif cmd_name == CMD_IN_ON_USER_JOINED:
            self.users_count += 1
            # Send greeting when first user joined
            if self.greeting and self.users_count == 1:
                await self._send_data(self.greeting,  True)
                ten_env.log_debug(f"Greeting [{self.greeting}] sent")
        elif cmd_name == CMD_IN_ON_USER_LEFT:
            self.users_count -= 1
        else:
            ten_env.log_debug(f"on_cmd unknown cmd: {cmd_name}")
            status_code, detail = StatusCode.ERROR, "unknown cmd"

        cmd_result = CmdResult.create(status_code)
        cmd_result.set_property_string("detail", detail)
        ten_env.return_result(cmd_result, cmd)

    async def on_data(self, ten_env: AsyncTenEnv, data: Data) -> None:
        data_name = data.get_name()
        ten_env.log_debug("on_data name {}".format(data_name))

        is_final = False
        input_text = ""
        try:
            is_final = data.get_property_bool(DATA_IN_TEXT_DATA_PROPERTY_IS_FINAL)
        except Exception as err:
            ten_env.log_info(f"GetProperty optional {DATA_IN_TEXT_DATA_PROPERTY_IS_FINAL} failed, err: {err}")

        try:
            input_text = data.get_property_string(DATA_IN_TEXT_DATA_PROPERTY_TEXT)
        except Exception as err:
            ten_env.log_info(f"GetProperty optional {DATA_IN_TEXT_DATA_PROPERTY_TEXT} failed, err: {err}")

        if not is_final:
            ten_env.log_info("ignore non-final input")
            return
        if not input_text:
            ten_env.log_info("ignore empty text")
            return

        ten_env.log_info(f"OnData input text: [{input_text}]")

        ts = datetime.now()
        await self.queue.put((input_text, ts))

    async def on_audio_frame(self, ten_env: AsyncTenEnv, audio_frame: AudioFrame) -> None:
        pass

    async def on_video_frame(self, ten_env: AsyncTenEnv, video_frame: VideoFrame) -> None:
        video_frame_name = video_frame.get_name()
        ten_env.log_debug("on_video_frame name {}".format(video_frame_name))

        self.image_data = video_frame.get_buf()
        self.image_width = video_frame.get_width()
        self.image_height = video_frame.get_height()

    async def _flush(self):
        """Flushes the self.queue and cancels the current task."""
        # Flush the queue by consuming all items
        self.ten_env.log_info("flush")

        while not self.queue.empty():
            self.queue.get_nowait()

        self.outdate_ts = datetime.now()
        if self.current_task:
            self.current_task.cancel()

    async def _add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_memory_length:
            self.history = self.history[1:]
    
    async def _get_messages(self) -> List[dict]:
        messages = []
        if self.prompt:
            messages.append({"role": "system", "content": self.prompt})
        messages.extend(self.history)
        return messages

    async def _chat_completion(self, input: str, ts: datetime):
        """Run the chatflow asynchronously."""
        try:
            total_output = ""
            messages = await self._get_messages()
            contents = [
                {"type": "text", "text": input}
            ]
            if self.image_data:
                contents.append({"type": "image_url", "image_url": {"url": rgb2base64jpeg(self.image_data, self.image_width, self.image_height)}})
            messages.append({"role": "user", "content": contents})
            await self._add_to_history("user", input)
            # Make an async API call to get chat completions
            params = {
                "model": self.model,
                "messages":messages,
                "temperature": self.temperature,
                "max_tokens":self.max_tokens,
                "top_p":self.top_p,
                "frequency_penalty":self.frequency_penalty,
                "presence_penalty":self.presence_penalty,
                "stream":True
            }
            self.ten_env.log_info(f"before chat completion {params}")
            response = await self.client.chat.completions.create(**params)

            async for message in response:
                if message.choices[0].delta.content is not None:
                    sentences, self.sentence_fragment = parse_sentences(self.sentence_fragment, message.choices[0].delta.content)
                    for s in sentences:
                        await self._send_data(s, True)
                        total_output += s
                # TODO add error handling
        except Exception as e:
            self.ten_env.log_error(
                f"Error in chat_completion: {traceback.format_exc()} for input text: {input} {e}")
        finally:
            if total_output:
                await self._add_to_history("assistant", total_output)

    def _append_memory(self, message: str):
        if len(self.memory) > self.max_memory_length:
            self.memory.pop(0)
        self.memory.append(message)

    async def _send_data(self, sentence: str, end_of_segment: bool):
        try:
            output_data = Data.create("text_data")
            output_data.set_property_string(
                DATA_OUT_TEXT_DATA_PROPERTY_TEXT, sentence)
            output_data.set_property_bool(
                DATA_OUT_TEXT_DATA_PROPERTY_TEXT_END_OF_SEGMENT, end_of_segment
            )
            self.ten_env.send_data(output_data)
            self.ten_env.log_debug(
                f"{'end of segment ' if end_of_segment else ''}sent sentence [{sentence}]"
            )
        except Exception as err:
            self.ten_env.log_debug(
                f"send sentence [{sentence}] failed, err: {err}"
            )

    async def _consume(self) -> None:
        self.ten_env.log_info("start async loop")
        while not self.stopped:
            try:
                value = await self.queue.get()
                if value is None:
                    self.ten_env.log_info("async loop exit")
                    break
                input, ts = value
                if self._need_interrrupt(ts):
                    continue

                self.current_task = self.loop.create_task(self._chat_completion(input, ts))
                try:
                    await self.current_task
                except asyncio.CancelledError:
                    self.ten_env.log_debug(f"Task cancelled: {input}")
            except Exception as e:
                self.ten_env.log_error(f"Failed to handle {e}")
            finally:
                self.current_task = None

    def _need_interrrupt(self, ts: datetime) -> bool:
        return self.outdate_ts > ts