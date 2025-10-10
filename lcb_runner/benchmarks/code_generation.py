import os
import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from datasets import load_dataset, Features, Value


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str], reasoning_list: list[str], output_list_extracted: list[str]) -> dict:
        result = {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }
        if reasoning_list is not None:
            result["reasoning_list"] = reasoning_list
        if output_list_extracted is not None:
            result["output_list_extracted"] = output_list_extracted
        return result

    def get_num_test_cases(self) -> int:
        """Get total number of test cases (public + private) for this problem."""
        return len(self.public_test_cases) + len(self.private_test_cases)

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        reasoning_list: list[str],
        output_list_extracted: list[str],
        test_results_list: list[list] = None,
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list, reasoning_list, output_list_extracted)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        output["num_test_cases"] = self.get_num_test_cases()
        if test_results_list is not None:
            output["test_results_list"] = test_results_list
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def load_code_generation_dataset(release_version="release_v1", start_date=None, end_date=None) -> list[CodeGenerationProblem]:

    # Data folder path
    DATA_FOLDER = os.path.abspath(os.path.join(__file__, "..", "..", "..", "data"))
    
    data_files = {
        "release_v1": os.path.join(DATA_FOLDER, "test.jsonl"),
        "release_v2": os.path.join(DATA_FOLDER, "test2.jsonl"),
        "release_v3": os.path.join(DATA_FOLDER, "test3.jsonl"),
        "release_v4": os.path.join(DATA_FOLDER, "test4.jsonl"),
        "release_v5": os.path.join(DATA_FOLDER, "test5.jsonl"),
        "release_v6": os.path.join(DATA_FOLDER, "test6.jsonl"),
        "release_latest": os.path.join(DATA_FOLDER, "test6.jsonl"),
    }

    # As defined in https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/main/code_generation_lite.py
    features=Features({
        "question_title": Value("string"),
        "question_content": Value("string"),
        "platform": Value("string"),
        "question_id": Value("string"),
        "contest_id": Value("string"),
        "contest_date": Value("string"),
        "starter_code": Value("string"),
        "difficulty": Value("string"),
        "public_test_cases": Value("string"),
        "private_test_cases": Value("string"),
        "metadata": Value("string"),
    })
    
    dataset = load_dataset("json", data_files=data_files[release_version], split="train", features=features)
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


if __name__ == "__main__":
    dataset = load_code_generation_dataset()
