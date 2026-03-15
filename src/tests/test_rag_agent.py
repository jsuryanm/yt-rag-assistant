from src.agents.transcriptor_agent import TranscriptAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.rag_agent import AgenticRAGAgent
from src.logger.custom_logger import logger 

if __name__ == "__main__":
    logger.info("Test pipeline: Transcript -> RAG QA")
    transcript_agent = TranscriptAgent()
    rag_agent = AgenticRAGAgent()

    url = "https://www.youtube.com/watch?v=BV0YUeam4y8"
    
    state = {"video_url":url,
             "user_question":"What is the main topic of the video?",
             "rewrite_count":0,
             "agent_trace":[]}
    
    result = transcript_agent.run(state)
    state.update(result)

    if state.get("error"):
        print(state['error'])
        exit()

    result = rag_agent.build_index(state)
    state.update(result)
    
    result = rag_agent.retrieve(state)
    state.update(result)

    result = rag_agent.grade_docs(state)
    state.update(result)

    if not state.get("is_relevant"):
        result = rag_agent.rewrite_query(state)
        state.update(result)

        result = rag_agent.retrieve(state)
        state.update(result)

    result = rag_agent.generate(state)
    state.update(result)

    print("\nANSWER:")
    print(state.get("answer"))

    print("\nAGENT TRACE:")
    for step in state.get("agent_trace", []):
        print(step)