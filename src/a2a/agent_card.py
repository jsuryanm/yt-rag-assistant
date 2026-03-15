from __future__ import annotations
import logging 
import re 
from src.logger.custom_logger import logger 
from typing import Any

from a2a.server.agent_execution import AgentExecutor,RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import (Part,
                       TaskState,
                       TextPart,
                       UnsupportedOperationError)

from a2a.utils import new_agent_text_message
from src.agents.orchestrator import OrchestratorAgent

"""
he A2A AgentExecutor for the YouTube RAG Assistant.
 
Responsibilities:
  1. Parse the incoming A2A Message into (video_url, question, skill_id)
  2. Emit TaskState transitions via TaskUpdater so the caller can track progress
  3. Dispatch to the LangGraph OrchestratorAgent
  4. Return the final answer as a TextPart artifact
 
Message format understood (text content of the first TextPart):
┌────────────────────────────────────────────────────────────────────┐
  │ Simple summarise (no question needed):                             │
  │   https://www.youtube.com/watch?v=<id>                            │
  │                                                                    │
  │ QA / search (url + question on same line or pipe-separated):       │
  │   url: <youtube_url> | question: <your question>                   │
  │   OR just:  <youtube_url>  <question text>                         │
  └────────────────────────────────────────────────────────────────────┘
 
TaskState lifecycle emitted:
  submitted → working → completed  (happy path)
  submitted → working → failed     (error path)
"""

from __future__ import annotations 

import re 
import logging 
from typing import Any 

from a2a.server.agent_execution import AgentExecutor,RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import Part,TaskState,TextPart,UnsupportedOperationError
from a2a.utils import new_agent_text_message

from src.agents.orchestrator import OrchestratorAgent
from src.logger.custom_logger import logger 

# Message parsing 
# Matches url: <url>| question: <text>
_PIPE_PATTERN = re.compile(r"url\s*:\s*(?P<url>https?://\S+)\s*\|\s*question\s*:\s*(?P<question>.+)",
                           re.IGNORECASE)

# Matches a bare YouTube URL at the start of the message
_URL_PATTERN =  re.compile(r"(?P<url>https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w\-]{11}\S*)")


def _parse_message():
    pass 