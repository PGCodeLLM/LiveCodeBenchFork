import os
import json

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import create_generic_openai_model, LanguageModelStore
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.evaluation import extract_instance_results, extract_test_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
)


def main():
    args = get_args()

    if args.use_generic_openai_server:
        model = create_generic_openai_model(args.model)
    else:
        model = LanguageModelStore[args.model]
    benchmark, format_prompt = build_prompt_benchmark(args)
    if args.debug:
        benchmark = benchmark[:15]
        print(f"Running with {len(benchmark)} instances in debug mode")

    output_path = get_output_path(model.model_repr, args)
    eval_file = output_path.replace(".json", "_eval.json")
    eval_all_file = output_path.replace(".json", "_eval_all.json")

    if args.continue_existing or args.continue_existing_with_eval:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                old_save_results = json.load(f)
        elif os.path.exists(eval_all_file):
            with open(eval_all_file, "r") as f:
                old_save_results = json.load(f)
        else:
            print(
                f"File {output_path} does not exist in --continue_existing, starting from scratch"
            )
            old_save_results = []

        old_save_results = [
            instance
            for instance in old_save_results
            if instance["output_list"] and [x for x in instance["output_list"] if x]
        ]
        old_save_results_question_ids = [
            instance["question_id"] for instance in old_save_results
        ]
        remaining_benchmark = [
            instance
            for instance in benchmark
            if instance.question_id not in old_save_results_question_ids
        ]
        print(
            f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
        )
    else:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        runner = build_runner(args, model)
        results: list[list[str]] = runner.run_main(remaining_benchmark, format_prompt)
    else:
        results = []

    # If model outputs are empty (probably due to failed inference requests),
    # add args.n empty strings to ensure failed tasks are still evaluated
    # results = [r if r else [""] * args.n for r in results]
    combined_results = combine_results(
        args.scenario, results, model, args.cot_code_execution
    )

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            remaining_benchmark, combined_results
        )
    ]

    if args.continue_existing or args.continue_existing_with_eval:
        save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    # for i in range(len(combined_results)):
    #     for j in range(len(combined_results[i][1])):
    #         if "def solve()" in combined_results[i][1][j]:
    #             from lcb_runner.utils.extraction_utils import extract_code, LMStyle

    #             combined_results[i][1][j] = extract_code(
    #                 combined_results[i][0][j], LMStyle.Gemini
    #             )
    #             if "\nsolve()" not in combined_results[i][1][j]:
    #                 combined_results[i][1][j] += "\n\nsolve()"

    #                 # combined_results[i][1][j] += "\n\nsolve()"
    #                 print(combined_results[i][1][j])

    if args.evaluate:
        if args.continue_existing_with_eval and os.path.exists(eval_all_file):
            with open(eval_all_file) as fp:
                old_eval_all_results = json.load(fp)

            if os.path.exists(eval_file):
                with open(eval_file) as fp:
                    old_eval_results = json.load(fp)
            else:
                old_eval_results = None

            old_eval_results_question_ids = [
                instance["question_id"] for instance in old_eval_all_results
            ]
            remaining_indices = [
                idx
                for idx in range(len(benchmark))
                if benchmark[idx].question_id not in old_eval_results_question_ids
            ]
            benchmark = [benchmark[idx] for idx in remaining_indices]
            combined_results = [combined_results[idx] for idx in remaining_indices]

            old_eval_size = len(old_eval_results_question_ids)
            new_eval_size = len(benchmark)

            if new_eval_size == 0:
                return

            print(f"Found {old_eval_size}, running evals for {new_eval_size} problems")

            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])

            if old_eval_results:
                for key in metrics[0]:
                    if key in old_eval_results[0]:
                        if key != "detail":
                            metrics[0][key] = (
                                old_eval_size * old_eval_results[0][key]
                                + new_eval_size * metrics[0][key]
                            )
                            metrics[0][key] /= old_eval_size + new_eval_size

                for key in metrics[0]["detail"]:
                    if key in old_eval_results[0]["detail"]:
                        metrics[0]["detail"][key] = {
                            **metrics[0]["detail"][key],
                            **old_eval_results[0]["detail"][key],
                        }
                metrics[1] = {**metrics[1], **old_eval_results[1]}
            else:
                print("Old eval file not present, cannot update eval file")
                metrics = {}

        else:
            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])
            old_eval_all_results = []
            old_eval_results = []

        if args.scenario == Scenario.codegeneration:
            if metrics:
                metadatas = metrics[2]
                test_results = extract_test_results(metrics[1])
            else:
                metadatas = [[] for _ in benchmark]
                test_results = [[] for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list, metadata=meta, test_results_list=test_result
                )
                for instance, (outputs_list, extracted_list), graded_list, meta, test_result in zip(
                    benchmark, combined_results, graded, metadatas, test_results
                )
            ]
            if metrics and old_eval_results:
                old_eval_results
                metrics[2] = old_eval_results[2] + metrics[2]
        elif args.scenario == Scenario.selfrepair:
            metadatas = metrics[2]
            with open(
                f"output/{model.model_repr}/{Scenario.codegeneration}_{args.codegen_n}_{args.temperature}_eval_all.json"
            ) as f:
                code_gen_evals = json.load(f)
            original_code_lists = [
                code_gen_eval["code_list"] for code_gen_eval in code_gen_evals
            ]

            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list,
                    extracted_list,
                    graded_list,
                    metadata=meta,
                    original_code_list=original_code_list,
                )
                for instance, (
                    outputs_list,
                    extracted_list,
                ), graded_list, meta, original_code_list in zip(
                    benchmark, combined_results, graded, metadatas, original_code_lists
                )
            ]

        else:
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list
                )
                for instance, (outputs_list, extracted_list), graded_list in zip(
                    benchmark, combined_results, graded
                )
            ]

        save_eval_results = old_eval_all_results + save_eval_results

        with open(eval_file, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(eval_all_file, "w") as f:
            json.dump(save_eval_results, f, indent=4)

        # Generate detailed results file with separated reasoning and output
        detailed_results_file = eval_all_file.replace("_eval_all.json", "_detailed_results.json")
        detailed_results = generate_detailed_results(save_eval_results, output_path)
        with open(detailed_results_file, "w") as f:
            json.dump(detailed_results, f, indent=2)
        print(f"Generated detailed results file: {detailed_results_file}")


def generate_detailed_results(eval_results, output_path):
    """
    Generate detailed results with separated reasoning and output for each sample.

    Transforms task-level eval_all.json entries (with arrays) into sample-level entries.

    Args:
        eval_results: List of evaluation results from eval_all.json (task-level)
        output_path: Path to the output file (used to extract exp_id)

    Returns:
        List of detailed result entries (sample-level) with fields:
        - exp_id, task_id, kth_sample, reasoning, output
        - question_title, difficulty, platform, contest_id, contest_date
        - passed (bool), test_results (array), metadata (dict)
    """
    from pathlib import Path

    output_dir = Path(output_path).parent.name
    exp_id = output_dir.split('--', 1)[1] if '--' in output_dir else output_dir

    # Load reasoning content if available
    reasoning_map = {}
    reasoning_file = Path(output_path).parent / "reasoning_result.jsonl"
    if reasoning_file.exists():
        try:
            malformed_lines = 0
            with open(reasoning_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        for response in data.get('responses', []):
                            content = response.get('content', '')
                            reasoning_content = response.get('reasoning_content', '')
                            # Only require content to be non-empty (reasoning_content already filtered when saving)
                            if content:
                                reasoning_map[content] = reasoning_content
                    except json.JSONDecodeError as e:
                        malformed_lines += 1
                        if malformed_lines == 1:  # Only log first error to avoid spam
                            print(f"Warning: Malformed JSON on line {line_num} in reasoning file: {e}")
                        continue

            if malformed_lines > 0:
                print(f"Warning: Skipped {malformed_lines} malformed lines in reasoning file")
            print(f"Loaded {len(reasoning_map)} reasoning entries from {reasoning_file}")
        except Exception as e:
            print(f"Warning: Failed to load reasoning file: {e}")

    detailed_results = []

    for entry in eval_results:
        task_id = entry.get('question_id', 'unknown')
        output_list = entry.get('output_list', [])
        test_results_list = entry.get('test_results_list', [])
        metadata_list = entry.get('metadata', [])
        graded_list = entry.get('graded_list', [])

        for kth_sample in range(len(output_list)):
            output = output_list[kth_sample] if kth_sample < len(output_list) else ""
            test_results = test_results_list[kth_sample] if kth_sample < len(test_results_list) else []
            metadata = metadata_list[kth_sample] if kth_sample < len(metadata_list) else {}
            passed = graded_list[kth_sample] if kth_sample < len(graded_list) else False

            # Look up reasoning content
            reasoning_content = reasoning_map.get(output, "")

            detailed_results.append({
                'exp_id': exp_id,
                'task_id': str(task_id),
                'kth_sample': kth_sample,
                'output': output,
                'reasoning_content': reasoning_content,
                'question_title': entry.get('question_title', ''),
                'difficulty': entry.get('difficulty', ''),
                'platform': entry.get('platform', ''),
                'contest_id': entry.get('contest_id', ''),
                'contest_date': entry.get('contest_date', ''),
                'passed': passed,
                'test_results': test_results,
                'metadata': metadata,
            })

    return detailed_results


if __name__ == "__main__":
    main()
