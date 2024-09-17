from typing import (
    List,
    Literal,
    TypedDict,
)
import json
from transformers import (
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from peft import (
    PeftModel,
)

Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def format_tokens(dialogs, tokenizer):
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]"
        to your inputs. Here, we are adding it manually.
        """
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} "
                    f"{E_INST} {(answer['content']).strip()} ",
                )
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens


def read_dialogs_from_file(file_path):
    with open(file_path, 'r') as file:
        dialogs = json.load(file)
    return dialogs


def load_model(model_name, quantization=False):
    access_token = open('access_token').read()

    if quantization:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=access_token,
            load_in_8bit=True,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            use_auth_token=access_token,
        )

    return model


def load_trained_model(model_name, epoch, db_size=None):
    access_token = open('access_token').read()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=access_token,
        load_in_8bit=True,
    )
    if db_size == '10000':
        bas_model_dir = (
            "/scratch/pbhanda2/projects/travel_survey_llm/model_output_large/"
        )
        if epoch == 1:
            model_dir = "checkpoint-4963"
        elif epoch == 3:
            model_dir = "checkpoint-14889"
        else:
            raise ValueError("Invalid epoch number")
    elif db_size == '10000-exclude-inf-cities':
        bas_model_dir = (
            "/scratch/pbhanda2/projects/travel_survey_llm"
            "/model_output_large_exclude_inf_cities/"
        )
        if epoch == 1:
            model_dir = "checkpoint-4958"
        elif epoch == 3:
            model_dir = "checkpoint-14874"
        else:
            raise ValueError("Invalid epoch number")
    else:
        bas_model_dir = (
            "/scratch/pbhanda2/projects/travel_survey_llm/model_output/"
        )
        if epoch == 1:
            model_dir = "checkpoint-497"
        elif epoch == 5:
            model_dir = "checkpoint-2485"
        elif epoch == 10:
            model_dir = "checkpoint-4970"
        elif epoch == 20:
            model_dir = "checkpoint-9940"
        else:
            raise ValueError("Invalid epoch number")

    final_model_dir = bas_model_dir + model_dir
    # config = PeftConfig.from_pretrained(final_model_dir)
    peft_model = PeftModel.from_pretrained(
        model,
        final_model_dir
    )
    return peft_model


def create_update_prompt(prompt, question, last_answer):
    if last_answer is not None:
        ans = {
            "role": "assistant",
            "content": last_answer
        }
        prompt[0].append(ans)
    que = {
        "role": "user",
        "content": question['question'],
    }

    if question['type'] == 'multiple_choice':
        que['content'] += (
            "\n Please write the number that corresponds to your choice "
            "from the following options: \n"
        )
        for option in question['options']:
            que['content'] += f"({option['id']})- {option['text']}\n"
    elif question['type'] == 'free_text':
        que['content'] += "\n"
    prompt[0].append(que)

    return prompt
