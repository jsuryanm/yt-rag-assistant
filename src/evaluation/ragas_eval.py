"""Usage
-----
    # Generate testset and evaluate in one shot
    python -m src.evaluation.ragas_eval \
        --url https://www.youtube.com/watch?v=VIDEO_ID \
        --generate --size 20 \
        --run-name "baseline_chunk500_topk3"
 
    # Evaluate against a pinned testset (reproducible baseline)
    python -m src.evaluation.ragas_eval \
        --url https://www.youtube.com/watch?v=VIDEO_ID \
        --testset eval_results/testset_v1.json \
        --run-name "exp01_chunk300_topk5" \
        --note "reduced chunk_size, increased top_k"
"""


import asyncio 
import json 
import argparse
import sys
from pathlib import Path 
from datetime import datetime

from openai import OpenAI

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset,SingleTurnSample
from ragas.metrics.collections import (Faithfulness,
                                       AnswerRelevancy,
                                       ContextPrecision,
                                       ContextRecall)


from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

from src.agents.orchestrator import OrchestratorAgent
from src.evaluation.synthetic_testset import generate_testset,load_testset

from src.logger.custom_logger import logger 
from src.core.config import settings 
from src.exceptions.custom_exception import YtException

def _get_judge_llm_and_embeddings():
    if settings.openai_api_key:
        try:
            logger.info(f"Using RAGASEval: judge llm = {settings.openai_judge_model}")
            logger.info(f"embeddings = {settings.openai_embedding_model}")

            openai_client = OpenAI(api_key=settings.openai_api_key)
            ragas_llm = llm_factory(model=settings.openai_judge_model,
                                        client=openai_client)
                
            ragas_embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model,
                                                    client=openai_client)
                
            return ragas_llm,ragas_embeddings

        
        except Exception as e:
            logger.warning(f"RAGASEval: OpenAI import failed ({e})")
            raise YtException(str(e),sys)
        
async def collect_pipeline_outputs(samples: list[dict],
                                   video_url: str) -> list[dict]:
    """
    Run each question through OrchestratorAgent.arun() and capture:
      - response            → state["answer"]
      - retrieved_contexts  → state["retrieved_docs"]
    """
    orchestrator = OrchestratorAgent()
    results = []

    for i,sample in enumerate(samples):
        question = sample["user_input"]
        logger.info(f"RAGASEval [{i + 1}/{len(samples)}]: '{question[:70]}'")

        try:
            state = await orchestrator.arun(video_url=video_url,
                                            question=question)
            
            retrieved = state.get("retrieved_docs") or []

            if retrieved and not isinstance(retrieved[0],str):
                retrieved = [str(d) for d in retrieved]
            
            results.append({**sample,
                            "response":state.get("answer") or "",
                            "retrieved_context":retrieved})

        except Exception as e:
            logger.error(f"RAGASEval: sample {i + 1} failed — {str(e)}")
            results.append({**sample, "response": "", "retrieved_contexts": []}) 
            raise YtException(str(e),sys)
    
    successful = sum(1 for r in results if r.get("response"))
    logger.info(f"RAGASEval: pipeline complete — "
                f"{successful}/{len(samples)} samples answered successfully")
    
    return results

def run_ragas_scoring(collected: list[dict]) -> dict:
    """Returns dictionary of ragas metrics"""

    valid = [s for s in collected if s.get("response") or s.get("retrieved_contexts")]
    num_failed = len(collected) - len(valid)

    if not valid:
        logger.error("RAGASEval: no valid samples to score - check the pipeline errors above")
        return {}
    
    logger.info(f"RAGASEval: scoring {len(valid)} samples"
                f"({num_failed} excluded due to pipeline features)")
    
    ragas_samples = [SingleTurnSample(user_input=s["user_input"],
                                      response=s["response"],
                                      retrieved_contexts=s["retrieved_contexts"] or [""],
                                      reference=s["reference"])
                                      for s in valid]
    
    dataset = EvaluationDataset(samples=ragas_samples)

    ragas_llm,ragas_embeddings = _get_judge_llm_and_embeddings()

    logger.info("RAGASEval: performing evaluation")

    result = evaluate(dataset=dataset,
                      metrics=[Faithfulness(llm=ragas_llm),
                               AnswerRelevancy(llm=ragas_llm),
                               ContextPrecision(llm=ragas_llm),
                               ContextRecall(llm=ragas_llm)],
                      llm=ragas_llm,
                      embeddings=ragas_embeddings,
                      allow_nest_asyncio=True)
    
    df = result.to_pandas()
    score_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores = {
        col: round(float(df[col].mean()), 4)
        for col in score_cols
        if col in df.columns
    }
    scores["num_samples"] = len(valid)
    scores["num_failed"] = num_failed
 
    return scores

def save_results_locally(scores: dict,
                         collected: list[dict],
                         video_url: str,
                         testset_source: str) -> str:
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"scores_{timestamp}.json"
    out_path.write_text(json.dumps({
        "timestamp":timestamp,
        "video_url":video_url,
        "testset_source":testset_source,
        "scores":scores,
        "per_sample":collected
    },indent=2))

    logger.info(f"RAGASEval local results saved to the {out_path}")
    return str(out_path)

async def _load_or_generate(video_url: str,
                            testset_path: str | None,
                            generate: bool,
                            size: int) -> tuple[list[dict],str]:
    
    if testset_path and Path(testset_path).exists():
        return load_testset(testset_path),testset_path
    
    if generate:
        out_path = f"eval_results/testset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        samples = await generate_testset(video_url=video_url,
                                         testset_size=size,
                                         output_path=out_path)
        return samples, out_path
 
    raise ValueError(
        "No testset found. "
        "Pass --testset path/to/testset.json  OR  use --generate to create one."
    )

def _print_results(scores: dict, local_path: str, run_url: str | None = None) -> None:
    width = 62
    print("\n" + "=" * width)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * width)
    for key, label in [
        ("faithfulness",      "Faithfulness      grounded in context"),
        ("answer_relevancy",  "Answer relevancy  answers the question"),
        ("context_precision", "Context precision useful chunks ranked first"),
        ("context_recall",    "Context recall    all needed info retrieved"),
    ]:
        if key in scores:
            filled = int(scores[key] * 20)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"  {scores[key]:.4f}  {bar}  {label}")
    print("-" * width)
    print(f"  Samples : {scores.get('num_samples', '?')}  |  Failed: {scores.get('num_failed', 0)}")
    print(f"  Local   : {local_path}")
    if run_url:
        print(f"  DagsHub : {run_url}")
    print("=" * width + "\n")
 
 
def _print_sample_preview(collected: list[dict], n: int = 3) -> None:
    print(f"Sample preview (first {n}):\n")
    for s in collected[:n]:
        print(f"  [{s.get('evolution_type', '?')}]")
        print(f"  Q:        {s['user_input'][:100]}")
        print(f"  Response: {s.get('response', '')[:120]}...")
        print(f"  Contexts: {len(s.get('retrieved_contexts') or [])} retrieved")
        print()
 
 
# ── Public async API ──────────────────────────────────────────────────────────
 
async def run_evaluation(
    video_url: str,
    testset_path: str | None = None,
    generate: bool = False,
    size: int = 20,
    run_name: str | None = None,
    extra_params: dict | None = None,
    preview: bool = True,
) -> dict:
    """Full pipeline: load/generate → run pipeline → score → track → display."""
    from src.evaluation.mlflow_tracker import EvalTracker
 
    samples, testset_source = await _load_or_generate(
        video_url=video_url,
        testset_path=testset_path,
        generate=generate,
        size=size,
    )
 
    collected = await collect_pipeline_outputs(samples=samples, video_url=video_url)
    scores = run_ragas_scoring(collected)
 
    if not scores:
        logger.error("RAGASEval: scoring returned empty — aborting")
        return {}
 
    local_path = save_results_locally(scores, collected, video_url, testset_source)
 
    tracker = EvalTracker()
    run_url = None
    _run_name = run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M')}"
 
    with tracker.start_run(_run_name):
        tracker.log_all(
            scores=scores,
            samples=samples,
            collected=collected,
            video_url=video_url,
            testset_source=testset_source,
            extra_params=extra_params,
        )
        run_url = tracker.active_run_url()
 
    _print_results(scores, local_path, run_url)
    if preview:
        _print_sample_preview(collected)
 
    return scores
 
 
# ── CLI ───────────────────────────────────────────────────────────────────────
 
async def _main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation with MLflow / DagsHub tracking"
    )
    parser.add_argument("--url",        required=True,       help="YouTube video URL")
    parser.add_argument("--testset",    default=None,        help="Path to existing testset JSON")
    parser.add_argument("--generate",   action="store_true", help="Generate new testset")
    parser.add_argument("--size",       type=int, default=20, help="Q&A pairs to generate")
    parser.add_argument("--run-name",   default=None,        help="MLflow run name")
    parser.add_argument("--note",       default=None,        help="Freetext note as extra param")
    parser.add_argument("--no-preview", action="store_true", help="Skip sample preview")
    args = parser.parse_args()
 
    await run_evaluation(
        video_url=args.url,
        testset_path=args.testset,
        generate=args.generate,
        size=args.size,
        run_name=args.run_name,
        extra_params={"note": args.note} if args.note else None,
        preview=not args.no_preview,
    )
 
 
if __name__ == "__main__":
    asyncio.run(_main())