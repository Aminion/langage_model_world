from typing import Optional
import fire
from llama_models.llama3.api.datatypes import RawMessage, StopReason
from llama_models.llama3.reference_impl.generation import Llama


def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    model_parallel_size: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )
    sys_phrase = "Format answer as JSON expression. Imagine the world (note there is no civilization) based on following keywords: worm grab spoon beef sand search dune bulb dry apocalyptic violin orange; do not use those keywords further in answers, they only for initial imaginary"

    dialogs = [
        [
            RawMessage(role="system", content=sys_phrase),
            RawMessage(
                role="user",
                content=(
                    
                    "Provide 5 JSON objects like: {biom_id: BIOM_ID, features: ['FEATURE', ...], paths : [{other_biom_id: OTHER_BIOM_ID, path : PATH}, ...]}."
                    "Where: BIOM_ID is unique integer identifier of biom;"
                    "FEATURES is one or many words from list [trees, flowers, dunes, cliff, hill, stones] that fits to current biom;"
                    "OTHER_BIOM_ID is one of generated BIOM_ID;"
                    "PATH is one of worlds from list [by the river, down the hill, across the lake, across the river, down the valley] describing the way to current biom from other; "
                ),
            ),
        ],
    ]

    for dialog in dialogs:
        result = generator.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for msg in dialog:
            print(f"{msg.role.capitalize()}: {msg.content}\n")

        out_message = result.generation
        print(f"> {out_message.role.capitalize()}: {out_message.content}")
        print("\n==================================\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
