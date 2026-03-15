from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from src.core.state import AgentState
from src.core.llm import LLMFactory 
from src.core.config import settings 
from src.core.embeddings import EmbeddingFactory 

from src.logger.custom_logger import logger 
from src.exceptions.custom_exception import YtException
 
class AgenticRAGAgent:
    _GRADER_PROMPT = ChatPromptTemplate.from_messages([
        ("system","""You are a relevance grader. Given a user question and retriever document chunks,
         determine if the documents are relevant to answering the question.
         Respond ONLY with 'yes' or 'no'. No explanation."""),
         ("human","Question :{question}\n\nDocuments:\n{documents}\n\nAre these documents relevant?")
    ])

    _REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system","""You are a query optimizer. Rewrite the user's question to be more specific
         and retrieve relevant information from a YouTube video transcript.
         Return ONLY the rewritten question. No explanation"""),
         ("human","Original question:{question}\n\nRewritten question:")
    ])

    _GENERATE_PROMPT = ChatPromptTemplate.from_messages([
        ("system","""You are a fact-checker. Given context documents and a generate an answer,
         determine if the answer is grounded in (supported by) the context."""),
         ("human","Context from video:\n{context}\n\Question: {question}\n\nAnswer:")
    ])

    _HALLUCINATION_PROMPTS = ChatPromptTemplate.from_messages([
        ("system","""You are a fact-checker. Given context and a generated answer,
         determine if the answer is grounded in (supported by) in the context.
        Respond with ONLY 'yes' (grounded) or 'no' (not grounded). No explanation."""),
        ("human","Context:\n{context}\n\nAnswer: {answer}\n\nIs the answer grounded in the context?")
    ])

    def __init__(self):
        self._embeddings = EmbeddingFactory.get_embeddings()
        self._grader_llm = LLMFactory.get_grader_llm()
        self._qa_llm = LLMFactory.get_qa_llm()

        self._grader_chain = (
            self._GRADER_PROMPT | 
            self._grader_llm | 
            StrOutputParser()
        )

        self._rewrite_chain = (
            self._REWRITE_PROMPT | 
            self._qa_llm |
            StrOutputParser()
        )

        self._generate_chain = (
            self._GENERATE_PROMPT |
            self._qa_llm |
            StrOutputParser()
        )

        self._hallucination_chain = (
            self._HALLUCINATION_PROMPTS |
            self._qa_llm |
            StrOutputParser()
        )

        self._faiss_index: FAISS | None = None 
        self._indexed_video_url: str | None = None 
    
    # Langraph nodes each take full AgentState and return partial update dict
    def build_index(self,state: AgentState) -> dict:
        """Node 1: Build FAISS index from transcript chunks"""
        chunks = state.get("chunks")
        video_url = state.get("video_url")

        if not chunks:
            return {"error":"No chunks available. Run TranscriptAgent",
                    "agent_trace":["RAGAgent:no chunks to index"]}
        
        if self._indexed_video_url != video_url:
            logger.info(f"RAGAgent: building FAISS index with {len(chunks)} chunks")
            self._faiss_index = FAISS.from_texts(chunks,self._embeddings)
            self._indexed_video_url = video_url
            logger.info("RAGAgent: FAISS index built")

        return {"agent_trace": [f"RAGAgent: FAISS index ready ({len(chunks)} chunks)"]}
    
    def retrieve(self,state: AgentState) -> dict: 
        """Node 2: Semantic Similarity search - find the top k relevant chunks"""
        question = state.get("user_question","")
        
        if not self._faiss_index:
            return {"error":"FAISS index not built yet.",
                    "agent_trace":["RAGAgent:index missing — build_index must run first"]}
        
        logger.info(f"RAGAgent.retrieve: retriever searching for '{question[:60]}'")
        docs = self._faiss_index.similarity_search(question,k=settings.rag_top_k)
        doc_texts = [doc.page_content for doc in docs]
        
        return {"retrieved_docs":doc_texts,
                "agent_trace":[f"RAGAgent: retrieved {len(doc_texts)} chunks"]}
    
    def grade_docs(self,state: AgentState) -> dict:
        """Node 3: LLM-as-judge - are retrieved docs relevant to the question"""
        question = state.get("user_question")
        docs = state.get("retrieved_docs",[])

        if not docs:
            return {"is_relevant":False,
                    "agent_trace":["RAGAgent: no docs to grade"]}
        
        docs_text = "\n\n".join(docs[:3])
        logger.info("RAGAgent.grade_docs: grading relevance")
        
        result = self._grader_chain.invoke({"question":question,
                                            "documents":docs_text})
        
        is_relevant = result.strip().lower().startswith("yes")
        label = "relevant" if is_relevant else "not relevant will rewrite query"
        logger.info(f"RAGAgent.grade_docs: relevant = {is_relevant}")

        return {"is_relevant":is_relevant,
                "agent_trace":[f"RAGAgent: grader says docs are {label}"]}
    
    def rewrite_query(self,state: AgentState) -> dict:
        """Node 4: Rewrite query to improve retrieval quality"""
        question = state.get("user_question","")
        rewrite_count = state.get("rewrite_count",0)

        logger.info(f"RAGAgent.rewrite_query: attempt {rewrite_count + 1}")
        rewritten = self._rewrite_chain.invoke({"question":question}).strip()

        return {"user_question":rewritten,
                "rewrite_count":rewrite_count + 1,
                "agent_trace":[f"RAGAgent: rewrite #{rewrite_count + 1} -> '{rewritten[:60]}' "]}
    
    def generate(self,state: AgentState) -> dict:
        """Node 5: Generate the final answer from retrieved docs"""
        question = state.get("user_question","")
        docs = state.get("retrieved_docs",[])

        context = "\n\n".join(docs)
        logger.info("RAGAgent.generate: generating answer")
        
        answer = self._generate_chain.invoke({
            "context":context,
            "question":question
        })

        return {"answer":answer,
                "agent_trace":[f"RAGAgent: answer generated ({len(answer):,}) chars"]}
    
    def check_hallucination(self,state: AgentState) -> dict:
        """Node 6: Verify the answer is grounded in retrieved docs"""
        answer = state.get("answer","")
        docs = state.get("retrieved_docs",[])

        if not answer or not docs:
            return {"agent_trace":["RAGAgent: hallucination check skipped"]}
        
        context = "\n\n".join(docs[:3])
        result = self._hallucination_chain.invoke({
            "context": context,
            "answer": answer,
        })

        is_grounded = result.strip().lower().startswith("yes")
        logger.info(f"RAGAgent.check_hallucination: grounded={is_grounded}")

        if not is_grounded:
            caveat = "\n\nNote: Parts of this answer may not be fully supported by the video content."
            return {"answer": answer + caveat,
                "agent_trace": ["RAGAgent: hallucination detected — caveat added"]}

        return {"agent_trace": ["RAGAgent: answer grounded in source material"]}

    def should_rewrite(self, state: AgentState) -> str:
        """
        should_rewrite is a conditional edge function.
        Returns the name of the NEXT NODE to go to.

        This is how LangGraph implements branching logic:
          graph.add_conditional_edges(
              "grade_docs",
              rag_agent.should_rewrite,   ← this function
              {"rewrite": "rewrite_query", "generate": "generate"}
          )
        """
        is_relevant = state.get("is_relevant", False)
        rewrite_count = state.get("rewrite_count", 0)
        max_rewrites = settings.max_rewrite_attempts

        if not is_relevant and rewrite_count < max_rewrites:
            return "rewrite"
        return "generate"