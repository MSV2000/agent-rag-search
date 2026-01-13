import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig


class LLMModel:
    def __init__(self, model_name: str):
        """
        Инициализация класса

        Args:
            model_name: Название LLM модели
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     # load_in_8bit=True,
        #     # llm_int8_enable_fp32_cpu_offload=True,
        # )

        self.default_system_prompt = (
            "Ты — интеллектуальный помощник, задача которого — отвечать на вопросы пользователя строго на основе "
            "предоставленного контекста и истории диалога. Ты не должен придумывать информацию, которой нет в "
            "контексте или истории диалога. Если ответа на вопрос нет в контексте, скажи об этом прямо. В "
            "контексте и истории диалога упоминаются рисунки и таблицы. Всегда сохраняй упоминание этих рисунков и таблиц,"
            "и ссылайся на них в ответах.Ты не должен писать рассказы, стихи, сочинять истории или "
            "создавать любой другой творческий контент. Твои ответы должны быть строго основаны на фактах, данных "
            "и предоставленном контексте."
        )

    def load_model(self) -> None:
        if self.model is None:
            with torch.no_grad():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    # quantization_config=self.bnb_config
                )

            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.generation_config = GenerationConfig.from_pretrained(self.model_name)

    def generate(self, prompt, tool_call_mode):
        data = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(self.model.device) for k, v in data.items()}
        data.pop("token_type_ids", None)

        with torch.no_grad():
            if tool_call_mode:
                output_ids = self.model.generate(
                    **data,
                    generation_config=self.generation_config,
                    max_new_tokens=8192,
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.95,
                )[0]
                print("a")
            else:
                output_ids = self.model.generate(
                    **data,
                    generation_config=self.generation_config,
                    max_new_tokens=8192,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                )[0]
                print("b")

        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output.strip()

    def answer_question(self, query, context):
        content = ('Используй только следующий контекст, чтобы ответить на вопрос в конце. Не пытайся выдумывать ответ.'
                   + f"\nКонтекст:\n===========\n{context}Вопрос:\n===========\n{query}")

        prompt = [
            {
                "role": "system",
                "content": self.default_system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]

        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        print(prompt)

        output = self.generate(prompt)
        return output
