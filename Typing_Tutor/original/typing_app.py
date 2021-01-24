import time
import difflib
import logging
import textwrap
import SessionState
import streamlit as st

from random import choice
from streamlit_ace import st_ace
from tokenizers import AddedToken
from transformers import AutoTokenizer, AutoModelWithLMHead


CONTEXTS = [
    "def fib",
    "def fact",
    "def sum_of_int",
    "def sum_of_fact",
    "def sum_of_square_error",
    "def get_val",
    "def convert_to_num",
    "def convolute",
    "def dict_sort",
]


@st.cache(
    hash_funcs={
        st.delta_generator.DeltaGenerator: lambda x: None,
        AddedToken: lambda x: None,
        "_regex.Pattern": lambda x: None,
    },
    allow_output_mutation=True,
)
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "congcongwang/distilgpt2_fine_tuned_coder"
    )
    model = AutoModelWithLMHead.from_pretrained(
        "congcongwang/distilgpt2_fine_tuned_coder"
    )
    model.eval()

    return tokenizer, model


class TypingTutor:
    def __init__(self):

        st.set_page_config(page_title="Typing Tutor", layout="wide")

        self.tokenizer, self.model = _load_model()

        self.session_state = SessionState.get(
            name="typingSession",
            start_time=0,
            end_time=0,
            num_chars=0,
            text="",
            content="",
        )

        st.markdown(
            "<h1 style='text-align: center; color: black;'>Typing Tutor</h1>",
            unsafe_allow_html=True,
        )

        self.col1, self.col2 = st.beta_columns(2)
        placeholder = st.empty()

        with self.col1:
            self.start_button = st.button("Start!", key="start_button")
            st.subheader("Text to write")

            with placeholder.beta_container():
                st.subheader("Steps to check your Typing speed")
                st.write(
                    "1. When you are ready, click on the start button which will generate code for you to write on the left hand side. A point to note that the timer starts as soon as you click on the start button"
                )
                st.write(
                    "2. Start writing the same code on the code window given on the right hand side. When you're done - press 'CTRL + ENTER' to save your code. **Remember to do this as this ensures that the code you have written is ready for submission**"
                )
                st.write(
                    "3. Lastly, click on Check Speed button to check you writing accuracy and the writing speed. Good luck!"
                )

        with self.col2:
            self.eval_button = st.button("Check Speed", key="eval_button")
            st.subheader("Text Input")
            st.write("")

            self.session_state.content = st_ace(
                placeholder="Start typing here ...",
                language="python",
                theme="solarized_light",
                keybinding="sublime",
                font_size=20,
                tab_size=4,
                show_gutter=True,
                show_print_margin=True,
                wrap=True,
                readonly=False,
                auto_update=False,
                key="ace-editor",
            )

    def _code_gen(self, context, realtime=True):
        if realtime:
            input_ids = self.tokenizer.encode(
                "<python> " + context, return_tensors="pt"
            )
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=256,
                temperature=0.7,
                num_return_sequences=1,
            )

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return text
        else:
            with open("example_code.py", "r") as f:
                text = "".join(f.readlines())

            return text

    def on_start_click(self):
        with self.col1:
            context = choice(CONTEXTS)
            self.session_state.text = self._code_gen(context, realtime=True)

            self.session_state.num_chars = len(self.session_state.text)
            st.code(textwrap.dedent(self.session_state.text))

        self.session_state.start_time = time.time()

        logging.info(f"On start click, start time is {self.session_state.start_time}")
        logging.info(
            f"On start click, num_chars to type are {self.session_state.num_chars}"
        )

    def on_eval_click(self):
        self.session_state.end_time = time.time() - self.session_state.start_time

        logging.info(f"On eval click, current time is {time.time()}")
        logging.info(f"On eval click, start time is {self.session_state.start_time}")
        logging.info(f"On eval click, end time is {self.session_state.end_time}")

        speed = ((self.session_state.num_chars / self.session_state.end_time) / 5) * 60
        accuracy = difflib.SequenceMatcher(
            None, self.session_state.text, self.session_state.content
        ).ratio()
        with self.col1:
            st.write("Time to write:", round(speed), "WPM")
            st.write("Accuracy:", round(accuracy * 100, 2), "%")

        logging.info(f"On eval click, speed is {speed}")


if __name__ == "__main__":
    logging.info(f"Starting init")
    tt = TypingTutor()
    logging.info(f"Done with init")

    if tt.start_button:
        logging.info(f"Start button clicked at {time.time()}")
        tt.on_start_click()
        logging.info(f"Done with Start button")

    if tt.eval_button:
        logging.info(f"Eval button clicked at {time.time()}")
        tt.on_eval_click()
        logging.info(f"Done with Eval button")
