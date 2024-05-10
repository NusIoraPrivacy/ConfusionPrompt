
decomp_dict = {
    "musique":{"train": "musique_ans_v1.0_train.jsonl", 
               "test": "musique_ans_v1.0_dev.jsonl",
               "question": "question",
               "decomp": "question_decomposition"},
    "strategyQA":{"train": "train.json", 
               "test": "dev.json",
               "question": "question",
               "decomp": "decomposition"},
    "hotpotqa":{"train": "train_attrs_decomp_gpt-4_ans_final.json", 
               "test": "hotpot_dev_distractor_v1.json",
               "question": "question",
               "decomp": "decomposition"},
    "hotpotqa-yn":{"train": "train_decomp_yn.json", 
               "test": "test_yn.json",
               "question": "question",
               "decomp": "decomposition"}
}

dataset_type = {"musique": "qa", "strategyQA": "cls", "hotpotqa": "qa", "hotpotqa-yn": "cls"}

decomp_template_attr = (
    "Given a question, list of attributes, and a target answer, please decompose the question into a series of sub-questions with the following requirements:\n"
    "1. The decomposition should satisfy the MECE(Mutually Exclusive Collectively Exhaustive) principle, and the final sub-question would directly lead to the target answer.\n"
    "2. Try your best to separate the attributes into different sub-questions.\n"
    '3. Remember to return a list of decompositions in the format of ["decomposition 1", "decomposition 2", "decomposition 3"]\n'
    "4. Limit the number of sub-questions in decomposition as few as possible."
    "Example 1:\n"
    "User:\n"
    "Question: Are Gabe Saporta and Robert Plant both English?\n"
    'Attributes: ["Gabe Saporta", "Robert Plant"]\n'
    "Target answer: No\n"
    'Assistant: ["What nationality is Gabe Saporta?", "What nationality is Robert Plant?", "Is #1 the same nationality as #2?"]'

    "Example 2:\n"
    "User:\n"
    "Question: Which writer died later, Agatha Christie or Clement Greenberg?\n"
    'Attributes: ["Agatha Christie", "Clement Greenberg"]\n'
    "Target answer: Clement Greenberg\n"
    'Assistant: ["When did Agatha Christie die?", "When did Clement Greenberg die?", "Which date comes second: #1 or #2?", "Who died on #3"]'

    "Example 3:\n"
    "User:\n"
    "Question: In what year were the European Athletics Junior Championships held in the capital of Slovenia?\n"
    'Attributes: ["European Athletics Junior Championships", "Slovenia"]\n'
    "Target answer: 1997\n"
    'Assistant: ["What is the capital of Slovenia?", "In what year were the European Athletics Junior Championships held in #1?"]'

    "Please decompose the following question:"
    "Question: {question}"
    "Attributes: {attributes}"
    "Target answer: {answer}"

    'Remember to strictly return only a list of decompositions in the format of ["decomposition 1", "decomposition 2", "decomposition 3"], '
    "and make the final sub-question directly lead to the target answer."
    )

decomp_template = (
    "Given a question and a target answer, please decompose the question into a series of sub-questions with the following requirements:\n"
    "1. The decomposition should satisfy the MECE(Mutually Exclusive Collectively Exhaustive) principle, and the final sub-question would directly lead to the target answer.\n"
    '2. Remember to return a list of decompositions in the format of ["decomposition 1", "decomposition 2", "decomposition 3"]\n'
    "3. Limit the number of sub-questions in decomposition as few as possible."
    "Example 1:\n"
    "User:\n"
    "Question: Are Gabe Saporta and Robert Plant both English?\n"
    "Target answer: No\n"
    'Assistant: ["What nationality is Gabe Saporta?", "What nationality is Robert Plant?", "Is #1 the same nationality as #2?"]'

    "Example 2:\n"
    "User:\n"
    "Question: Which writer died later, Agatha Christie or Clement Greenberg?\n"
    "Target answer: Clement Greenberg\n"
    'Assistant: ["When did Agatha Christie die?", "When did Clement Greenberg die?", "Which date comes second: #1 or #2?", "Who died on #3"]'

    "Example 3:\n"
    "User:\n"
    "Question: In what year were the European Athletics Junior Championships held in the capital of Slovenia?\n"
    "Target answer: 1997\n"
    'Assistant: ["What is the capital of Slovenia?", "In what year were the European Athletics Junior Championships held in #1?"]'

    "Please decompose the following question:"
    "Question: {question}"
    "Target answer: {answer}"

    'Remember to strictly return only a list of decompositions in the format of ["decomposition 1", "decomposition 2", "decomposition 3"], '
    "and make the final sub-question directly lead to the target answer."
    )

find_attr_template = (
            'Given a sentence, please identify and return a list of key attributes in the sentence. Strictly return in the format of ["attr1", "attr2", ...]\n'
            "Example:\n"
            "User: Are Chipotle Cinnamon Pork Chops appropriate for a Seder?\n"
            'Assistant: ["Chipotle Cinnamon Pork Chops", "Seder"]\n'
            "Please return a list of key attributes for the sentence: {sentence}"
            )

qualify_template = (
                "For the question \"{question}\""
                "which decompositions satisfy the MECE(Mutually Exclusive Collectively Exhaustive) principle and will lead to the correct answer: "
                "{choices}"
                "Please strictly return the list of selected index in the format of [A/B/C/D/E...]. "
                "If there is no qualified answer, return an empty list []."
                )

simple_template = (
                "Given multiple questions, is the last question a simple question that requires no additional knowledge to answer? "
                "Questions: {decomposition}\n"
                "Reply only \"yes\" or \"no\" in your answer."
                )

hardness_template = (
        "Given multiple questions seperated by <s>, you should evaluate the hardness of the last question. "
        "Use the scoring rules below to score hardness of the last question on a scale of 1 to 3: \n"
        "1. Score 1: the last question is trivial in the sense that it requires no additional knowledge to answer, or a simple comparison of the previous questions. \n"
        "2. Score 2: the last question is simple in the sense that it only requires common sense knowledge to answer. \n"
        "3. Score 3: the last question is moderate in the sense that it's a single hop question and requires a bit factual knowledge in related area to answer. \n"
        "4. Score 4: the last question is difficult in the sense that it's a multi-hop question and requires some knowledge in related area to answer. \n"
        "Questions: {decomposition}\n"
        "Output your evaluation in the following format: "
        "#thescore: your score here. "
        "#thereason: your analysis here."
        )

hardness_template2 = (
        "Given multiple questions seperated by <s>, you should evaluate the hardness of the last question. "
        "Output which class the last question belongs to. \n"
        "Class 1: The last question is either: (1) a trivial question that requires no additional knowledge to answer, or "
        "(2) a simple question that requires simple comparison among the previous questions. "
        "Some examples include \"Is #1 greater than or equal to #2?\", \"Is any letter in #2 included in #3?\", and \"Is #2 before #1?\". "
        "Class 2: The last question is not simple as it doens't satisfy the conditions listed in class 1. "
        "Questions: {decomposition}\n"
        "Output your evaluation in the following format: "
        "#theclass: your classification here. "
        "#thereason: your analysis here."
        )

has_attr_template = (
    "Given a set of questions, identify which sentences contain the phrase \"{attribute}\" or its derivative form, or expressions with essentially the same semantic meaning. "
    "Sentences: {sentences}\n"
    "Stricly return in the format of a list that consists of the index of qualified sentence: [1/2/3...]. "
    "If there's no such sentence, return an empty list []."
)

replace_template_phrase = (
    "Given the question {sentence}, please replace the phrases {attributes}, "
    "such that the question is fluent and reasonable, "
    "and the alterntive phrases have irrelevant meaning as {attributes}. "
    "Stricly return in the format of list of alternative attributes."
)

replace_template_phrase_multiple = (
    "Given the question {sentence}, please replace the phrases {attributes}, "
    "such that the question is fluent and reasonable, "
    "and the alterntive phrases have irrelevant meaning as {attributes}. "
    "Return {n_replaces} lists of alternative attributes for the sentence. "
    "Stricly return in the format of the list of alternative attributes: "
    '[[list1 of alternative phrases], [list2 of alternative phrases],...] '
)

replace_template = (
    "Please replace the phrases {attributes} in the following sentence, "
    "such that the sentence is fluent and reasonable, "
    "and the alterntive phrases have irrelevant meaning as {attributes}: "
    "{sentence}"
)

replace_template_multiple = (
    "Please replace the phrases {attributes} in the each of the following sentences, "
    "such that each sentence is fluent and reasonable, "
    "and the alterntive phrases have irrelevant meaning as {attributes}. "
    "Return {n_replaces} replacements for each sentence. "
    "Strictly respond in the form of JSON with the following format: "
    '{{"S1": ["sentence1", "sentence2",...], "S2": ["sentence1", "sentence2",...]}}. '
    "Sentences: {sentences}"
)

replace_template_multiple_resp = (
    "Please replace the phrases {attributes} in the each of the following sentences, "
    "such that each sentence is fluent and reasonable, "
    "and the alterntive phrases have irrelevant meaning as {attributes}. "
    "Return {n_replaces} replacements for each sentence. "
    "Strictly respond the list of replacement with the following format: "
    '["sentence1", "sentence2",...] '
    "Sentence: {sentence}"
)

extract_phrase_template = (
    'Given the original query "{raw_query}", what is the phrase that replaces "{attribute}" '
    'or its related expression in the revised question "{replace_query}"? '
    "Please respond only the alternative phrase in the new query."
)

extract_multiple_phrase_template = (
    'Given the original query "{raw_query}", what are the phrases that replace "{attribute}" '
    'or their related expression in the revised question "{replace_query}"? '
    'Please strictly respond only the list of alternative phrases in the new query, in the format of ["alternative phrase 1", "alternative phrase 2", ...]'
)

qualify_replace_template = (
    'Given the original query "{raw_query}", does the revised question "{replace_query}" '
    'replaces only the phrases {attributes} or its related expression, '
    "and make other parts remain unchanged? "
    "Please respond only Yes or No."
)

fluency_template = (
    "Given multiple sentences, use the scoring rules below to score each sentence's fluency on a scale of 1 to 4: "
    "1. Score 1: Incomprehensible. Inarticulate/ non-fluent sentence. "
    "2. Score 2: Low Quality. Partially fluent sentence: (a) only half of the sentence is fluent or "
    "(b) more than 1 missing words or (c) more than 1 misspelt words or d) contains individual fluent word-groups with missing coherence between them. "
    "3. Score 3: Moderate. Sentence is predominantly fluent but contains either "
    "(a) misspelt word or (b) missing word or (c) multiple occurrence of a word. "
    "4. Score 4: Perfect. Perfectly fluent sentence without any syntactic or grammatical error. "
    "Strictly respond in the form of JSON with the following format: "
    '{{"S1": the score, "S2": the score}}. '
    "Sentences: {sentences}\n"
)

subquery_template = "{question} Please answer as short as possible."

cls_bool_template = "{question} Please answer \"yes\" or \"no\" only."

subqa_template = "{question} Please answer as short as possible."

direct_query_template = (
    "Given a question, please provide your answer as concise as possible. \n"
    "Example 1:\n"
    "Question: Are Gabe Saporta and Robert Plant both English?\n"
    "Answer: No\n"

    "Example 2:\n"
    "Question: Between Parsifal and Saul og David which opera has more acts?\n"
    "Answer: Saul og David\n"

    "Example 3:\n"
    "Question: What novel about a murderer inspired Till Lindemann's music?\n"
    "Answer: Perfume\n"

    "Question: {question}\n"
    "Answer: "
)

direct_query_template_cls = (
    "Given a question, please provide your answer as concise as possible. \n"
    "Example 1:\n"
    "Question: Are Gabe Saporta and Robert Plant both English?\n"
    "Answer: No\n"

    "Example 2:\n"
    "Question: Are Local H and For Against both from the United States?\n"
    "Answer: Yes\n"

    "Example 3:\n"
    "Question: Is the language used in Saint Vincent and the Grenadines rooted in English?\n"
    "Answer: Yes\n"

    "Question: {question}\n"
    "Answer: "
)

step_by_step_template = (
    "Given a question and analysis, please provide your answer step by step "
    "in the format of \"Analysis: your concise analysis. Conclusion: your concise answer\"\n"
    "Example 1:\n"
    "User: Are Gabe Saporta and Robert Plant both English?\n"
    "Answer: \n"
    "Analysis: Robert Plant was born on August 20, 1948, in West Bromwich, Staffordshire, England, "
    "Gabe Saporta was born on October 11, 1979, in Montevideo, Uruguay.\n"
    "Conclusion: No\n"

    "Example 2:\n"
    "User: Between Parsifal and Saul og David which opera has more acts?\n"
    "Answer: \n"
    "Analysis: Parsifal by Richard Wagner has three acts, Saul og David by Carl Nielsen also has four acts.\n"
    "Conclusion: Saul og David\n"

    "Question: {question}\n"
    "Answer: "
)

step_by_step_template_cls = (
    "Given a question and analysis, please provide your answer step by step "
    "in the format of \"Analysis: your concise analysis. Conclusion: your concise answer\"\n"
    "Example 1:\n"
    "User: Are Gabe Saporta and Robert Plant both English?\n"
    "Answer: \n"
    "Analysis: Robert Plant was born on August 20, 1948, in West Bromwich, Staffordshire, England, "
    "Gabe Saporta was born on October 11, 1979, in Montevideo, Uruguay.\n"
    "Conclusion: No\n"

    "Example 2:\n"
    "User: Is the language used in Saint Vincent and the Grenadines rooted in English?\n"
    "Answer: \n"
    "Analysis: The primary language spoken in Saint Vincent and the Grenadines is Vincentian Creole, and Vincentian Creole is English-based.\n"
    "Conclusion: Yes\n"

    "Question: {question}\n"
    "Answer: "
)

chat_templates = {
    "lmsys/vicuna-33b-v1.3": "<s> [INST] {prompt} [/INST]",
    # "meta-llama/Llama-2-7b-chat-hf": "<s> [INST] {prompt} [/INST]",
    # "meta-llama/Meta-Llama-3-8B": "<s> [INST] {prompt} [/INST]",
    "meta-llama/Llama-2-7b-chat-hf": "{prompt}",
    "meta-llama/Meta-Llama-3-8B": "{prompt}",
    "jondurbin/spicyboros-13b-2.2": ("<s> [INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. "
                                    "Always answer as helpfully as possible, while being safe. "
                                    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                                    "Please ensure that your responses are socially unbiased and positive in nature.\n"
                                    "\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                                    "If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n{prompt} [/INST]"),
    "jondurbin/spicyboros-70b-2.2": ("<s> [INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. "
                                    "Always answer as helpfully as possible, while being safe. "
                                    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                                    "Please ensure that your responses are socially unbiased and positive in nature.\n"
                                    "\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                                    "If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n{prompt} [/INST]"),
    "01-ai/Yi-34B-Chat": "<|im_start|> user\n{prompt}<|im_end|> \n<|im_start|>assistant\n",
    "gpt-4": "{prompt}",
    "gpt-3.5-turbo": "{prompt}",
    "gpt-4-turbo": "{prompt}",
}

emb_norm_dict = {
    'stevhliu/my_awesome_model': 2.5,
    "gpt2": 4,
    "gpt2-large": 2.2,
    "gpt2-medium": 4,
    "gpt2-xl": 2,
    "facebook/opt-125m": 0,
    "facebook/opt-350m": 0,
    "ArthurZ/opt-350m-dummy-sc": 0,
    "facebook/opt-2.7b": 0,
    "facebook/opt-6.7b": 0,
    "meta-llama/Llama-2-7b-hf": 0,
    "meta-llama/Llama-2-7b-chat-hf": 0,
    "bert-base-uncased": 2.5,
    "bert-large-uncased": 2.5,
    "t5-small": 850,
    "t5-base": 680,
    "t5-large": 650,
    "facebook/bart-large": 4.5,
        }
    
logit_range_dict = {
    "eugenesiow/bart-paraphrase": (-3, 3),
    "meta-llama/Llama-2-7b-chat-hf": (-8, 8),
    "google/flan-t5-xl": (-80, 7.5)
        }

model_causal = {
    "facebook/bart-large": False,
    'bart_pretrained': False,
    "meta-llama/Llama-2-7b-chat-hf": True,
    "FacebookAI/roberta-base": True, 
    "FacebookAI/roberta-large": True,
    "deepset/roberta-base-squad2": True,
    "deepset/roberta-large-squad2": True
}

IGNORE_INDEX = -100

token_cost_dict = {
    "gpt-4": (30/1000000, 60/1000000),
    "gpt-3.5-turbo": (0.5/1000000, 1.5/1000000),
    "gpt-4-turbo": (10/1000000, 30/1000000),
}