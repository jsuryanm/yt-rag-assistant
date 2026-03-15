from src.agents.transcriptor_agent import TranscriptAgent
from src.logger.custom_logger import logger 

if __name__ == "__main__":
    logger.info("Test for running the transcriptor agent")
    agent = TranscriptAgent()
    state = {"video_url": "https://www.youtube.com/watch?v=BV0YUeam4y8"}

    result = agent.run(state)

    print("\nRESULT:")
    print(result.keys())

    print("\nTranscript preview:")
    print(result.get("processed_transcript","")[:500])

    print("\nChunks count:")
    print(len(result.get("chunks",[])))