import re
from lcb_runner.lm_styles import LMStyle


def extract_from_output(model_output: str) -> tuple[str, str]:
    """
    Extract both reasoning and output content from model output in a single pass.
    Checks for reasoning in this order:
    1. [REASONING] tags (from OpenAI/vLLM reasoning_content field)
    2. <think> tags (fallback for some models if vLLM reasoning parser not enabled)
    3. [unused##] tags (pangu)

    NOTE: For some models we might need add parser to properly extract reasoning into reasoning_content field:
        vllm serve model_name --enable-reasoning --reasoning-parser <parser_name>

    Args:
        model_output: The full output text from the model

    Returns:
        tuple[str, str]: (reasoning_content, output_content)
    """
    if not model_output:
        return "", ""

    if model_output.startswith('[REASONING]'):
        match = re.search(r'\[REASONING\](.*?)\[/REASONING\](.*)', model_output, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()

    if '<think>' in model_output:
        match = re.search(r'<think>(.*?)</think>(.*)', model_output, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()

    if '[unused' in model_output:
        parts = re.split(r'\[unused\d+\]', model_output)
        if len(parts) > 1:
            reasoning = parts[0].strip() if parts[0] else (parts[1].strip() if len(parts) > 1 else "")
            reasoning = re.sub(r'\[unused\d+\]\s*', '', reasoning)
            output = parts[-1].strip()
            return reasoning, output

    return "", model_output


def extract_code(model_output: str, lmstyle: LMStyle):
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
        if len(indexlines) < 2:
            indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    elif lmstyle == LMStyle.GenericBase:
        return model_output.strip()
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if len(indexlines) < 2:
            return ""
        # return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
        return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])


def extract_test_output_code(model_output: str, lmstyle: LMStyle = None):
    outputlines = model_output.split("\n")
    # find the last line startwith assert...
    indexlines = [i for i, line in enumerate(outputlines) if line.startswith("assert")]
    if indexlines:
        return outputlines[indexlines[-1]]
    if lmstyle and lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        # first try to extract ```python if not then try ```
        indexlines = [
            i
            for i, line in enumerate(outputlines)
            if "```python" in line or "```Python" in line
        ]
        if indexlines:
            start_index = indexlines[0]
        else:
            start_index = None
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if start_index is not None:
            indexlines = [i for i in indexlines if i > start_index]
            indexlines = [start_index] + indexlines

    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def extract_execution_code(model_output: str, lmstyle: LMStyle, cot: bool = False):
    if cot:
        if "[ANSWER]" in model_output:
            model_output = model_output.split("[ANSWER]")[1].strip()
    if "==" in model_output:
        model_output = model_output.split("==")[1].strip()
    if "[/ANSWER]" in model_output:
        model_output = model_output.split("[/ANSWER]")[0].strip()
    else:
        model_output = model_output.split("\n")[0].strip()
    return model_output.strip()
