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
    sys_phrase = "Format answer as JSON expression. Imagine the world (note there is no civilization) based on following keywords: worm grab spoon beef sand search dune bulb dry apocalyptic violin orange"

    dialogs = [
        [
           RawMessage(role="system", content= sys_phrase), 
           RawMessage(role="user", content=("replace VALUE in { min : VALUE , max : VALUE } with temperature limits of the world")),
        ],
        [
           RawMessage(role="system", content= sys_phrase), 
           RawMessage(role="user", content=("replace VALUE in { north : VALUE, west : VALUE, south : VALUE, west : VALUE } with numbers describing wind rose in meters per second")),
        ],
        [
           RawMessage(role="system", content= sys_phrase), 
           RawMessage(role="user", content=("replace VALUE in { soil_types : VALUE } with list of {type: TOPSOIL , percentage : PERCENTAGE} where TOPSOIL is topsoil type (sand, clay, silt, peat, chalk, loam) and PERCENTAGE is percentage of soil type in the world. Do not repeat topsoil types.")),
        ],
                [
           RawMessage(role="system", content= sys_phrase), 
           RawMessage(role="user", content=("replace VALUE in { ores_and_minerals : VALUE } with list of 10 elements of {name: NAME , percentage : PERCENTAGE} where NAME is ether mineral or ore; PERCENTAGE is percentage of it in the soil.")),
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

