import asyncio 
import json 
import argparse 
import sys 
from pathlib import Path 
from collections import Counter
from datetime import datetime 

from langchain_core.documents import Document 
from openai import OpenAI

from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.transforms import default_transforms

from src.agents.transcriptor_agent import TranscriptAgent
from src.core.config import settings 
from src.logger.custom_logger import logger 
from src.core.llm import LLMFactory
from src.core.embeddings import EmbeddingFactory
from src.exceptions.custom_exception import YtException

"""
Generates a RAGAS EvaluationDataset from a YouTube video transcript.
 
Strategy
--------
1. Fetch transcript via your existing TranscriptAgent
2. Wrap chunks as LangChain Documents  (same granularity as your retriever)
3. Run RAGAS TestsetGenerator — internally it:
     a. Extracts keyphrases / concepts from chunks  (knowledge graph nodes)
     b. Generates questions of 3 types:
          simple        → single-chunk factual   "What is X?"
          reasoning     → multi-hop across chunks "Why does X cause Y?"
          multi_context → comparison / synthesis  "How does X differ from Y?"
     c. Generates a reference answer grounded in the source chunks
4. Save to JSON — pin this file so baselines stay comparable across runs
 
Usage
-----
    python -m src.evaluation.synthetic_testset \
        --url https://www.youtube.com/watch?v=VIDEO_ID \
        --size 20 \
        --out eval_results/testset_v1.json
 
Output schema (per sample)
--------------------------
    {
      "user_input":          str,        # generated question
      "reference":           str,        # ground-truth answer (LLM-written from chunks)
      "reference_contexts":  list[str],  # chunks RAGAS used to write the reference answer
      "evolution_type":      str,        # "simple" | "reasoning" | "multi_context"
      "retrieved_contexts":  list[str],  # placeholder — filled at eval time by your pipeline
      "response":            str,        # placeholder — filled at eval time by your pipeline
    }
"""

def select_representative_chunks(
    chunks: list[str],
    max_chunks: int = 50
):

    if len(chunks) <= max_chunks:
        return chunks

    # prioritize longer chunks (more information)
    chunks = sorted(
        chunks,
        key=lambda x: len(x),
        reverse=True
    )

    return chunks[:max_chunks]

def _get_generator_llm_and_embeddings():
    # create judge llm and embeddings for TestsetGenerator
    try:
        if settings.openai_api_key:
            client = OpenAI(api_key=settings.openai_api_key)
            logger.info(f"Creating RAGAS synthetic test set using OpenAI LLM : {settings.openai_judge_model} and OpenAI Embeddings: {settings.openai_embedding_model}")
            
            ragas_llm = llm_factory(model=settings.openai_judge_model,
                                    client=client)
            
            ragas_embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model,
                                                client=client)
            
            return ragas_llm,ragas_embeddings
    
    except Exception as e:
        logger.error("Failed to create RAGAS generator")
        YtException(str(e),sys)
    

def chunks_to_documents(chunks: list[str],
                        video_url: str) -> list[Document]:
    """Wrap transcript chunks as LangChain Documents"""

    return [Document(page_content=chunk,
                     metadata={"source":video_url,"chunk_index":i}) for i,chunk in enumerate(chunks)]

async def generate_testset(video_url: str,
                           testset_size: int=10,
                           output_path: str | None = None) -> list[dict]:
    """
    Full generation pipeline. Returns list of sample dicts.
 
    LLM call volume:
        approx 3 × testset_size calls internally (concept extraction + Q gen + A gen).
        For testset_size=20 → ~60 calls. Start with --size 10 on first run.
    """
    agent = TranscriptAgent()
    video_id = agent.extract_video_id(video_url)
    if not video_id:
        raise ValueError(f"SyntheticTestset: Could not extract video ID from URL: {video_id}")
    
    logger.info(f"SyntheticTestset: Fetching transcript for video_id: {video_id}")
    transcript = agent.fetch_transcript(video_id)
    if not transcript:
        raise ValueError(f"SyntheticTestset: No transcript found for video_id:{video_id}")
    
    chunks = agent.chunk_text(transcript)
    logger.info(f"SyntheticTestset: {len(transcript):,} chars,  {len(chunks)} chunks")

    chunks = select_representative_chunks(chunks,max_chunks=80)
    logger.info(f"Selected chunks: {len(chunks)}")

    docs = chunks_to_documents(chunks,video_url)
    
    generator_llm,generator_embeddings = _get_generator_llm_and_embeddings()

    transforms = default_transforms(documents=docs,
                                    llm=generator_llm,
                                    embedding_model=generator_embeddings)
    # default_transform() performs data normalization and schema alignment 
    generator = TestsetGenerator(llm=generator_llm,
                                 embedding_model=generator_embeddings)
    
    logger.info(f"SyntheticTestset: generating {testset_size} Q&A pairs")

    dataset = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: generator.generate_with_langchain_docs(documents=docs,
                                                       testset_size=testset_size,
                                                       transforms=transforms) 
    )

    # convert to plain dicts 
    samples = _dataset_to_samples(dataset)
    logger.info(f"SyntheticTestset: generated {len(samples)} samples")
    _log_type_distribution(samples)
 
    # 5. Persist
    if output_path:
        _save(samples, video_url, output_path)
 
    return samples

def _dataset_to_samples(dataset)->list[dict]:
    """Convert RAGAS dataset to list of dicts"""
    df = dataset.to_pandas()
    samples = []

    for _,row in df.iterrows():
        ref_ctx = row.get("reference_contexts",[])
        
        if not isinstance(ref_ctx,list):
            ref_ctx = list(ref_ctx) if ref_ctx else []
        
        samples.append({"user_input": str(row["user_input"]),
                        "reference": str(row["reference"]),
                        "reference_contexts": [str(c) for c in ref_ctx],
                        "evolution_type": str(row.get("synthesizer_name", "unknown")),
                        "retrieved_contexts": [],
                        "response": "",})
    return samples

def _log_type_distribution(samples: list[dict]) -> None:
    dist = Counter(s["evolution_type"] for s in samples)
    logger.info("SyntheticTestset: question type distribution:")
    for qtype, count in sorted(dist.items()):
        logger.info(f"  {qtype}: {count}")
 
 
def _save(samples: list[dict], video_url: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    path.write_text(json.dumps({"generated_at": datetime.now().isoformat(),
                                "video_url": video_url,
                                "num_samples": len(samples),
                                "samples": samples,},
                                indent=2,))
    
    logger.info(f"SyntheticTestset: saved to {path}")
 
 
def load_testset(path: str) -> list[dict]:
    """Load a previously generated testset JSON. Returns list of sample dicts."""
    payload = json.loads(Path(path).read_text())
    logger.info(
        f"SyntheticTestset: loaded {payload['num_samples']} samples "
        f"from {path} (generated {payload.get('generated_at', 'unknown')})"
    )
    return payload["samples"]
 
 
# CLI 
 
async def _main():
    parser = argparse.ArgumentParser(
        description="Generate a RAGAS evaluation testset from a YouTube video"
    )
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--size", type=int, default=20, help="Number of Q&A pairs")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()
 
    out = args.out or f"eval_results/testset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    samples = await generate_testset(video_url=args.url, testset_size=args.size, output_path=out)
 
    print(f"\nGenerated {len(samples)} samples → {out}\n")
    for s in samples[:3]:
        print(f"  [{s['evolution_type']}]  Q: {s['user_input']}")
        print(f"  A: {s['reference'][:150]}...\n")
 
 
if __name__ == "__main__":
    asyncio.run(_main())
