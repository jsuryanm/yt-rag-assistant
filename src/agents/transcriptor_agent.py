import re 
from typing import Optional 
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled,NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.state import AgentState
from src.core.config import settings 
from src.logger.custom_logger import logger
from src.exceptions.custom_exception import YtException
import sys

class TranscriptAgent:
    """
    1. Extracts video id from URL
    2. Fetch transcript
    3. Convert transcript obj to clean text
    4. Chunking text for vector store indexing 
    """

    _VIDEO_ID_PATTERNS = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,
                                                        chunk_overlap=settings.chunk_overlap,
                                                        separators=["\n\n","\n"," ",""])
    
    def extract_video_id(self,url: str) -> Optional[str]:
        """Extract 11 char video id from youtube url"""
        for pattern in self._VIDEO_ID_PATTERNS:
            match = re.search(pattern,url)

            if match: 
                return match.group(1)
        return None 
    
    def fetch_transcript(self,video_id: str) -> Optional[str]:
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id) 
            # list all transcripts which are available for a given video you can call 
            manual_transcript = None 
            auto_transcript = None
            
            for t in transcript_list:
                if t.language_code == "en":
                    manual_transcript = t.fetch()
                    break 

                elif auto_transcript is None:
                    auto_transcript = t.fetch()
        
            raw = manual_transcript or auto_transcript
            if raw is None:
                logger.warning(f"No English transcript for video_id:{video_id}")
                return None 
            return self._transcript_to_text(raw)
        
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.error(f"Transcript unavailable for {video_id}: {e}")
            return None
        
        except Exception as e:
            raise YtException(e,sys)
    
    def chunk_text(self,text: str) -> list[str]:
        """Split processed transcript into overlapping chunks for RAG"""
        return self._splitter.split_text(text)
    
    def run(self,state: AgentState) -> dict:
        """
        LangGraph node function
        Called by orchestrator graph with AgentState
        Returns only the keys this node updates 
        """ 
        logger.info("Transcriptor agent: Starting")

        video_url = state.get("video_url","")
        video_id = self.extract_video_id(video_url)

        if not video_id:
            return {"error":f"Could not extract video id from URL: {video_url}",
                    "agent_trace":["TranscriptorAgent: invalid URL"]}
        
        logger.info(f"TranscriptorAgent: fetching transcript for video_id: {video_id}")
        transcript_text = self.fetch_transcript(video_id)

        if not transcript_text:
            return {"error":"No english transcript found for this video",
                    "agent_trace":["TranscriptorAgent: no transcript available"]}
        
        chunks = self.chunk_text(transcript_text)
        logger.info(f"TranscriptorAgent: got {len(transcript_text)} chars, {len(chunks)} chunks")
        return {"processed_transcript":transcript_text,
                "chunks":chunks,
                "agent_trace":[f"TranscriptAgent: fetched transcript ({len(transcript_text)} chars, {len(chunks)} chunks)"]}
    
    def _transcript_to_text(self, transcript_data) -> str:
        """
        Convert list of transcript snippet objects to a single clean string.
        Each snippet has .text and .start attributes.
        We keep text only (no timestamps) — cleaner for LLM processing.
        """
        parts = []
        for snippet in transcript_data:
            try:
                text = snippet.text.strip()
            except AttributeError:
                text = snippet.get("text", "").strip()
            if text:
                parts.append(text)

        # Join with space — transcript is naturally continuous speech
        return " ".join(parts)