#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import websockets
import asyncio
import json
import base64
import uuid

from typing import Any
import traceback

from ten import (
    AudioFrame,
    VideoFrame,
    AsyncExtension,
    AudioFrameDataFmt,
    AsyncTenEnv,
    Cmd,
    StatusCode,
    CmdResult,
    Data,
)

PROPERTY_TOKEN = "token"  # Required
PROPERTY_BASE_URI = "base_uri"  # Optional

EVENT_SESSION_UPDATE = "session.update"
EVENT_SESSION_UPDATED = "session.updated"

EVENT_CONVERSATION_ITEM_CREATE = "conversation.item.create"
EVENT_CONVERSATION_ITEM_DELETE = "conversation.item.delete"
EVENT_INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
EVENT_INPUT_ADUIO_BUFFER_COMMIT = "input_audio_buffer.commit"
EVENT_INPUT_ADUIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
EVENT_INPUT_ADUIO_BUFFER_CLEAR = "input_audio_buffer.clear"
EVENT_INPUT_ADUIO_BUFFER_CLEARED = "input_audio_buffer.cleared"

EVENT_RESPONSE_CREATE = "response.create"
EVENT_RESPONSE_CREATED = "response.created"
EVENT_RESPONSE_DONE = "response.done"

EVENT_RESPONSE_AUDIO_DELTA = "response.audio.delta"
EVENT_RESPONSE_AUDIO_DONE = "response.audio.done"

EVENT_RESPONSE_AUDIO_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
EVENT_RESPONSE_AUDIO_TRANSCRIPT_DONE = "response.audio_transcript.done"

EVENT_RESPONSE_OUTPUT_ITEM_DELTA = "response.output_item.delta"

def generate_event_id() -> str:
    return str(uuid.uuid4())

class MMRealtimeExtension(AsyncExtension):
    base_uri = "wss://api.minimax.chat/ws/v1/realtime"
    token = ""
    prompt = "You are a voice assistant who talks in a conversational way and can chat with me like my friends. I will speak to you in English or Chinese, and you will answer in the corrected and improved version of my text with the language I use. Don’t talk like a robot, instead I would like you to talk like a real human with emotions. I will use your answer for text-to-speech, so don’t return me any meaningless characters. I want you to be helpful, when I’m asking you for advice, give me precise, practical and useful advice instead of being vague. When giving me a list of options, express the options in a narrative way instead of bullet points."
    current_response = ""
    ignore_response = ""

    websocket: websockets.WebSocketClientProtocol = None
    ten_env: AsyncTenEnv = None
    stopped = False
    sample_rate: int = 24000
    remote_stream_id: int = 0

    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log_debug("on_init")
        ten_env.on_init_done()

    async def on_start(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log_debug("on_start")

        self.loop = asyncio.get_event_loop()

        try:
            token = ten_env.get_property_string(PROPERTY_TOKEN)
            self.token = token
        except Exception as err:
            ten_env.log_warn(
                f"GetProperty required {PROPERTY_TOKEN} failed, err: {err}")
            return

        try:
            base_uri = ten_env.get_property_string(PROPERTY_BASE_URI)
            if base_uri:
                self.base_uri = base_uri
        except Exception as err:
            ten_env.log_warn(
                f"GetProperty required {PROPERTY_TOKEN} failed, err: {err}")

        self.ten_env = ten_env
        ten_env.on_start_done()

    async def on_stop(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log_debug("on_stop")


        self.stopped = True
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        ten_env.on_stop_done()

    async def on_deinit(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log_debug("on_deinit")
        ten_env.on_deinit_done()

    async def on_cmd(self, ten_env: AsyncTenEnv, cmd: Cmd) -> None:
        cmd_name = cmd.get_name()
        ten_env.log_debug("on_cmd name {}".format(cmd_name))

        if cmd_name == "flush":
            await self._flush()
            await ten_env.send_cmd(Cmd.create("flush"))
            ten_env.log_debug("flush done")

        cmd_result = CmdResult.create(StatusCode.OK)
        ten_env.return_result(cmd_result, cmd)

    async def on_data(self, ten_env: AsyncTenEnv, data: Data) -> None:
        pass

    async def on_audio_frame(
        self, ten_env: AsyncTenEnv, audio_frame: AudioFrame
    ) -> None:
        audio_frame_name = audio_frame.get_name()
        ten_env.log_info(f"on_audio_frame name {audio_frame_name}")

        try:
            stream_id = audio_frame.get_property_int("stream_id")

            if self.remote_stream_id == 0:
                self.remote_stream_id = stream_id
            
            if self.websocket is None:
                await self._start_conn()
                ten_env.log_info(f"Start session for {stream_id}")

            frame_buf = audio_frame.get_buf()
            await self._send_audio(frame_buf)
        except Exception as e:
            ten_env.log_error(f"on audio frame failed {e} {traceback.format_exc()}")

    async def on_video_frame(
        self, ten_env: AsyncTenEnv, video_frame: VideoFrame
    ) -> None:
        pass

    async def _start_conn(self):
        self.websocket = await websockets.connect(
            self.base_uri,
            extra_headers={"Authorization": f"Bearer {self.token}"}
        )
        
        await self._send({
            "event_id": generate_event_id(),
            "type": EVENT_SESSION_UPDATE,
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self.prompt,
                "voice": "female-shaonv",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "asr-01"
                },
                "temperature": 0.8
            }
        })
        self.loop.create_task(self._recv_loop())
        

    async def _send_audio(self, frame_buf: bytes):
        await self._send({
            "event_id": generate_event_id(),
            "type": EVENT_INPUT_AUDIO_BUFFER_APPEND,
            "audio": base64.b64encode(frame_buf).decode("ISO-8859-1"),
        })
        await self._send({
            "event_id": generate_event_id(),
            "type": EVENT_INPUT_ADUIO_BUFFER_COMMIT
        })
        await self._send({
            "event_id": generate_event_id(),
            "type": EVENT_RESPONSE_CREATE
        })

    async def _recv_loop(self):
        while not self.stopped:
            try:
                data = await self.websocket.recv()
                message_data = json.loads(data)
                self.ten_env.log_info(f"incoming message {message_data}")
                message_type = message_data.get("type")
                if message_type == EVENT_RESPONSE_AUDIO_DELTA:
                    if self.ignore_response == message_data.get("response_id"):
                        continue
                    delta = message_data.get("delta", "")
                    await self._on_audio_recv(base64.b64decode(delta))
                elif message_type == EVENT_SESSION_UPDATED:
                    self.ten_env.log_info(f"Session updated {message_data}")
                elif message_type == EVENT_RESPONSE_DONE:
                    self.current_response = ""
                    self.ignore_response = ""
                elif message_type == EVENT_RESPONSE_CREATED:
                    self.current_response = message_data.get("response_id")
                    self.ignore_response = ""
            except websockets.exceptions.ConnectionClosed:
                self.ten_env.log_info("Connection closed")
                break

        self.websocket = None

    async def _on_audio_recv(self, audio_data):
        f = AudioFrame.create("pcm_frame")
        f.set_sample_rate(self.sample_rate)
        f.set_bytes_per_sample(2)
        f.set_number_of_channels(1)
        f.set_data_fmt(AudioFrameDataFmt.INTERLEAVE)
        f.set_samples_per_channel(len(audio_data) // 2)
        f.alloc_buf(len(audio_data))
        buff = f.lock_buf()
        buff[:] = audio_data
        f.unlock_buf(buff)
        self.ten_env.send_audio_frame(f)

    async def _flush(self):
        self.ignore_response = self.current_response
        await self._send({
            "event_id": generate_event_id(),
            "type": EVENT_INPUT_ADUIO_BUFFER_CLEAR,
        })
    
    async def _send_text(self, text:str, role:str, end_of_segment:bool):
        stream_id = self.remote_stream_id if role == "user" else 0

        d = Data.create("text_data")
        d.set_property_string("text", text)
        d.set_property_bool("end_of_segment", end_of_segment)
        d.set_property_string("role", role)
        d.set_property_int("stream_id", stream_id)
        self.ten_env.log_info(
            f"send transcript text [{text}] {stream_id} end_of_segment {end_of_segment} role {role}"
        )
        self.ten_env.send_data(d)

    async def _send(self, event: Any):
        if self.websocket:
            if event.get("type") != EVENT_INPUT_AUDIO_BUFFER_APPEND:
                self.ten_env.log_info(f"outgoing message {event}")
            await self.websocket.send(json.dumps(event))