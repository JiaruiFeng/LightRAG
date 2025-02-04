# import os, sys
# sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_complete, gpt_4o_mini_complete
import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="query graphrag. ",
    )

    parser.add_argument(
        "--question_dir",
        default="../eval_result/questions.txt",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../eval_result",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./ad_paper_0122",
    )
    parser.add_argument(
        "--search_mode",
        type=str,
        choices=("local", "global", "hybrid", "naive"),
        default="hybrid",
    )

    args = parser.parse_args()

    if args.question_dir:
        with open(args.question_dir, "r") as f:
            questions = [question.rstrip("\n") for question in f]
    else:
        questions = ["How do amyloid beta circadian patterns change with respect to age and amyloid pathology"]

    rag = LightRAG(
        working_dir=args.root_dir,
        llm_model_func=gpt_4o_complete,  # Use gpt_4o_mini_complete LLM model
        # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
    )

    results = []
    for question in questions:
        response = rag.query(question, param=QueryParam(mode=args.search_mode, print_reference=False))
        results.append({"response": response, "query": question})
    with open(f"{args.output_dir}/lightrag_{args.search_mode}_new_prompt_result.json", "w") as f:

        json.dump(results, f, indent=2)


