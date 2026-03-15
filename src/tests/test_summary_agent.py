from src.agents.transcriptor_agent import TranscriptAgent
from src.agents.summary_agent import SummaryAgent
from src.logger.custom_logger import logger 

if __name__ == "__main__":
    logger.info("Test pipeline: Transcript -> Summary")

    transcriptor_agent = TranscriptAgent()
    summary_agent = SummaryAgent()
    url = "https://www.youtube.com/watch?v=BV0YUeam4y8"
    
    state = {"video_url":url,
             "agent_trace":[]} 

    transcript_result = transcriptor_agent.run(state)
    state.update(transcript_result)

    if "error" in transcript_result:
        print("Transcript error:",transcript_result['error'])
        exit()

    print("\nTranscript preview:")
    print(state.get("processed_transcript","")[:500])

    print("\nChunks count:")
    print(len(state.get("chunks",[])))

    summary_result = summary_agent.run(state)
    state.update(summary_result)

    if "error" in summary_result:
        print("Summary error:",summary_result['error'])
        exit()

    print("\nSummary:")
    print(state.get("summary"))

    print("\nAgent Trace:")
    print(state.get("agent_trace"))