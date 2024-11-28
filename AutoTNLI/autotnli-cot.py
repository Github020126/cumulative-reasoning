# AutoTNLI with Cumulative Reasoning + CoT
"""AutoTNLI是一个用于自然语言推理（特别是表格推理）的数据增强框架。它旨在通过生成可转移的假设
模板和创建基于人类编写的逻辑约束的理性反事实表格，来增强训练数据。这个框架特别适用于在监督有
限的情况下，提供更高质量和更复杂的训练示例，以改善半结构化表格推理任务的性能。"""
# @Jingqin Yang

#"import guidance" 就是关于如何导入的指导或建议。这通常出现在文档或教程中，告诉用户如何正确地使用 "import" 语句来引入他们需要的代码库。
import guidance
import torch
import ast
import datasets
import numpy as np
import argparse

#使用 Python 的 argparse 模块定义一个命令行参数解析器的函数。argparse 是 Python 标准库的一部分，用于编写用户友好的命令行接口。
def get_parser():
    #创建了一个对象
    parser = argparse.ArgumentParser(description="Cumulative Reasoning")
    #添加参数名称、类型、默认值、在帮助文档里的描述
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
    parser.add_argument('--max_tokens', type=int, default=50, help='max tokens')
    parser.add_argument('--save_suffix', type=str, default='example-suffix', help='save suffix')
    parser.add_argument('--sc_cnt', type=int, choices=range(1, 30), default=1, help='number of sc cnt')
    parser.add_argument('--model', type=str, default='/data/model/llama-13b', help='model to use')
    parser.add_argument('--dataset', type=str, default='/data/datasets/AutoTNLI', help='dataset to use')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    return parser

#这一行代码调用了一个名为get_parser的函数，这个函数的作用是创建并返回一个ArgumentParser对象，这个对象用于解析命令行参数。parser是这个对象的变量名。
parser = get_parser()
#解析后的参数被存在这里
args = parser.parse_args()

#将一个变量guidance.llm赋值为接下来的表达式的结果。
# token_healing可能是用于修复或优化某些特定输入类型的表现
guidance.llm = guidance.llms.transformers.LLaMA(args.model, device_map="auto", token_healing=True,
                                                torch_dtype=torch.bfloat16, caching=False)

import json
import time
import numpy
#tqdm是一个快速，可扩展的Python进度条库，可以在长循环中添加一个进度条，用户只需要封装任意的迭代器tqdm(iterator)。
from tqdm import tqdm

#这个例子包含前提、命题、结论、判断
examples = [
    #前提、命题、entail是必然的意思
    {
        'premises': 'Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.',
        'propositions': 'Miroslav Venhoda, who published a book in 1946 called Method of Studying Gregorian Chant, is a musician as he is a choral conductor.',
        'conclusion': 'A Czech person wrote a book in 1946.',
        'judgement': 'entail'},
    {
        'premises': 'All eels are fish. No fish are plants. A thing is either a plant or animal. Nothing that breathes is paper. All animals breathe. If a sea eel is either an eel or a plant, then a sea eel is an eel or an animal.',
        'propositions': 'No eels are plants. All eels are animals.',
        'conclusion': 'Sea eel is an eel.',
        'judgement': 'contradict'},
    {
        'premises': 'Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.',
        'propositions': 'Miroslav Venhoda specialized in the performance of Renaissance and Baroque music.',
        'conclusion': 'No choral conductor specialized in the performance of Renaissance.',
        'judgement': 'contradict'},
]

#这些例子包含前提、命题、结论、解释
gen_proposition_examples = [
    {'premises': 'All eels are fish. No fish are plants. ',
     'proposition': 'No eels are plants.',
     'conclusion': 'Sea eel is an eel.',
     'explanation': 'This expression is deduced from the two premises as follows: if x is an eel, then it is a fish (from Premise 1), and if it is a fish, then it is not a plant (from Premise 2). Thus, if x is an eel, then it is not a plant.'},
    {'premises': 'All eels are fish. A thing is either a plant or animal.',
     'proposition': 'All eels are animals.',
     'conclusion': 'Sea eel is an eel.',
     'explanation': 'This statement follows from the premises as follows: If x is an eel, then it is a fish (from Premise 1). If x is a thing (which includes being a fish, hence an eel), then it is either a plant or an animal (from Premise 2). Since it cannot be a plant (because it is a fish and no fish is a plant), it must be an animal. Thus, if x is an eel, it is an animal.'},
    {'premises': 'A thing is either a plant or animal. All animals breathe.',
     'proposition': 'All things that breathe are animals.',
     'conclusion': 'Sea eel is an eel.',
     'explanation': 'This statement is deduced from the premises as follows: If x is a thing, then it is either a plant or an animal (from Premise 1), and if x is an animal, then it breathes (from Premise 2). Therefore, if a thing breathes, it must be an animal, because it can not be a plant that breathes based on these premises.'},
    {
        'premises': 'All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. ',
        'proposition': 'All people who joke about being addicted to caffeine are not dependent on caffeine.',
        'conclusion': 'Rina is either a person who regularly drinks coffee or a person who is unaware that caffeine is a drug.',
        'explanation': 'Since all people who regularly drink coffee are dependent on caffeine, those who just joke about being addicted (and don\'t regularly drink coffee) are not dependent on caffeine.'},
    {
        'premises': 'Any choral conductor is a musician. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.',
        'proposition': 'Miroslav Venhoda, who published a book in 1946 called Method of Studying Gregorian Chant, is a musician as he is a choral conductor.',
        'conclusion': 'A Czech person wrote a book in 1946',
        'explanation': 'This follows from the universal rule that any choral conductor is a musician (Premise 1), so since Miroslav Venhoda is a choral conductor who published a book in 1946 called Method of Studying Gregorian Chant (Premise 2), he is therefore a musician.'
    }
]

#验证推理的例子
validate_deduction_examples = [
    {'premises': 'All eels are fish. No fish are plants.',
     'proposition': 'No eels are plants.',
     'validation': 'True'},
    {'premises': 'All eels are fish. A thing is either a plant or animal.',
     'proposition': 'All eels are animals.',
     'validation': 'True'},
    {'premises': 'Nothing that breathes is paper. All animals breathe.',
     'proposition': 'All animals are paper.',
     'validation': 'False'},
    {'premises': 'A thing is either a plant or animal. All animals breathe.',
     'proposition': 'All things that breathe are animals.',
     'validation': 'True'},
    {
        'premises': 'All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine.',
        'proposition': 'All people who joke about being addicted to caffeine are dependent on caffeine.',
        'validation': 'False'},
    {
        'premises': 'Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician.',
        'proposition': 'Miroslav Venhoda, being a Czech choral conductor specializing in Renaissance and Baroque music, is also a musician.',
        'validation': 'True'},
    {'premises': 'Any choral conductor is a musician. Some musicians love music.',
     'proposition': 'All choral conductor love music.',
     'validation': 'False'},
    {
        'premises': 'Any choral conductor is a musician. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.',
        'proposition': 'Miroslav Venhoda, who published a book in 1946 called Method of Studying Gregorian Chant, is a musician as he is a choral conductor.',
        'validation': 'True'}
]

#有用推理的例子
useful_deduction_examples = [
    {
        'premises': 'Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.',
        'proposition': 'Miroslav Venhoda, who published a book in 1946 called Method of Studying Gregorian Chant, is a musician as he is a choral conductor.',
        'conclusion': 'A Czech person wrote a book in 1946.',
        'usefulness': 'Useful'},
    {
        'premises': 'All eels are fish. No fish are plants. A thing is either a plant or animal. Nothing that breathes is paper. All animals breathe. If a sea eel is either an eel or a plant, then a sea eel is an eel or an animal.',
        'proposition': 'No animals are paper.',
        'conclusion': 'Sea eel is an eel.',
        'usefulness': 'Unuseful'}
]

#重复推理的例子
duplicated_deduction_examples = [
    {
        'premises': 'Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.',
        'proposition': 'Any choral conductor is a musician.',
        'conclusion': 'A Czech person wrote a book in 1946.',
        'duplicated': 'True'},
    {
        'premises': 'All eels are fish. No fish are plants. A thing is either a plant or animal. Nothing that breathes is paper. All animals breathe. If a sea eel is either an eel or a plant, then a sea eel is an eel or an animal.',
        'proposition': 'No animals are paper.',
        'duplicated': 'False'
    }
]

#来源推理的例子
sourced_deduction_examples = [
    {'premises': 'All eels are fish. No fish are plants.',
     'proposition': 'No eels are plants.',
     'sourced': 'True'},
    {
        'premises': 'Nothing that breathes is paper. All animals breathe.',
        'proposition': 'All animals need food.',
        'sourced': 'False'}
]

#定义用于判断命题各种性质的选项集
# we can pre-define valid option sets
valid_judgement = ["entail", "contradict"]

# we can pre-define valid option sets
valid_validation = ["True", "False"]

# we can pre-define valid option sets
valid_usefulness = ["Useful", "Unuseful"]

# we can pre-define valid option sets
valid_duplicated = ["True", "False"]

# we can pre-define valid option sets
valid_sourced = ["True", "False"]

#生成性测试的prompt
gen_proposition = guidance(
    '''
    ### Instruction:
    Suppose you are one of the greatest AI scientists, logicians and mathematicians. Let us think step by step. Please deduce a "Proposition" from two given "Premises". 
    Please make sure that the "Proposition" is logically correct. 
    Please make sure that the "Proposition" is not a duplicate of the "Premises".
    Please remember that your "Proposition" should be useful to determine whether the "Premises" entail or contradict the "Hypothesis". 
    ----
    {{~! display the few-shot examples ~}}
    {{~#each examples}}
    ### Input:
    "Premises": "{{this.premises}}"
    We want to deduce more propositions to determine whether the "Premises" entail or contradict the following "Hypothesis":
    "Hypothesis": "{{this.conclusion}}"

    ### Response:
    "Proposition": "{{this.proposition}}"
    ---
    {{~/each}}

    {{~! place the real question at the end }}
    ### Input:
    "Premises": "{{premises}}"
    We want to deduce more propositions to determine whether the "Premises" entail or contradict the following "Hypothesis":
    "Hypothesis": "{{hypothesis}}"

    ### Response:
    "Proposition {{prop_id}}": "{{gen "proposition" temperature=0.7 max_tokens=50 stop='\"\\n'}}"
    ''')

# Define the guidance program
validate_deduction = guidance(
    '''
    ### Instruction:
    Suppose you are one of the greatest AI scientists, logicians and mathematicians. Let us think step by step. Please determine whether the deduction of given "Premises" to a "Proposition" is True or False.

    {{~! display the few-shot examples ~}}
    {{~#each examples}}
    ### Input:
    "Premises": "{{this.premises}}"
    "Proposition": "{{this.proposition}}"

    ### Response:
    "Judgement": "Now we know that this deduction is {{this.validation}}"
    ---
    {{~/each}}

    {{~! place the real question at the end }}
    ### Input:
    "Premises": "{{premises}}"
    "Proposition": "{{proposition}}"

    ### Response:
    "Judgement": "Now we know that this deduction is {{select "validation" options=valid_validation logprobs='logprobs'}}"
    ''')

# Define the guidance program
useful_deduction = guidance(
    '''
    ### Instruction:
    Suppose you are one of the greatest AI scientists, logicians and mathematicians. Let us think step by step. Please determine whether the deduction of two given "Premises" to a "Proposition" is useful to determine whether the "Premises" entail or contradict the "Hypothesis", reply with Useful or Unuseful.

    {{~! display the few-shot examples ~}}
    {{~#each examples}}
    ### Input:
    "Premises": "{{this.premises}}"
    "Proposition": "{{this.proposition}}"
    "Hypothesis": "{{this.conclusion}}"

    ### Response:
    "Judgement": "Now we know that this deduction is {{this.usefulness}} to determine whether the Premises entail or contradict the Hypothesis."
    ---
    {{~/each}}

    {{~! place the real question at the end }}
    ### Input:
    "Premises": "{{premises}}"
    "Proposition": "{{proposition}}"
    "Hypothesis": "{{hypothesis}}"

    ### Response:
    "Judgement": "Now we know that this deduction is {{select "usefulness" options=valid_usefulness logprobs='logprobs'}} to determine whether the Premises entail or contradict the Hypothesis."
    ''')

# Define the guidance program
duplicated_deduction = guidance(
    '''
    ### Instruction:
    Suppose you are one of the greatest AI scientists, logicians and mathematicians. Let us think step by step. Please determine whether the "Proposition" is duplicated with the "Premises", reply with True or False.

    {{~! display the few-shot examples ~}}
    {{~#each examples}}
    ### Input:
    "Premises": "{{this.premises}}"
    "Proposition": "{{this.proposition}}"

    ### Response:
    "Judgement": "Now we know that this proposition is {{this.duplicated}} with the premises."
    ---
    {{~/each}}

    {{~! place the real question at the end }}
    ### Input:
    "Premises": "{{premises}}"
    "Proposition": "{{proposition}}"

    ### Response:
    "Judgement": "Now we know that this proposition is {{select "duplicated" options=valid_duplicated logprobs='logprobs'}} with the premises."
    ''')

# Define the guidance program
sourced_deduction = guidance(
    '''
    ### Instruction:
    Suppose you are one of the greatest AI scientists, logicians and mathematicians. Let us think step by step. Please determine whether the "Proposition" is directly deduced from the "Premises" other than introducing unsourced informations by common sense reasoning, reply with True or False.

    {{~! display the few-shot examples ~}}
    {{~#each examples}}
    ### Input:
    "Premises": "{{this.premises}}"
    "Proposition": "{{this.proposition}}"

    ### Response:
    "Judgement": "Is this proposition directly deduced from the premises? {{this.sourced}}"
    ---
    {{~/each}}

    {{~! place the real question at the end }}
    ### Input:
    "Premises": "{{premises}}"
    "Proposition": "{{proposition}}"

    ### Response:
    "Judgement": "Is this proposition directly deduced from the premises? {{select "sourced" options=valid_sourced logprobs='logprobs'}}"
    ''')

# Define the guidance program
structure_program = guidance(
    '''
    ### Instruction:
    Suppose you are one of the greatest AI scientists, logicians and mathematicians. Let us think step by step. Read and analyze the "Premises" first, then judge whether the "Premises" entail or contradict the "Hypothesis".
    ----

    {{~! display the few-shot examples ~}}
    {{~#each examples}}
    ### Input:
    "Premises": "{{this.premises}}"
    "Hypothesis": "{{this.conclusion}}"

    ### Response:
    "Thoughts": "Let us think step by step. From the premises, we know that {{this.propositions}}"
    "Recall the Hypothesis": "{{this.conclusion}}"
    "Judgement": "Now we know that the Premises {{this.judgement}} the Hypothesis."
    ---
    {{~/each}}

    {{~! place the real question at the end }}
    ### Input:
    "Premises": "{{premises}}."
    "Hypothesis": "{{hypothesis}}"

    ### Response:
    "Thoughts": "Let us think step by step. From the premises, we know that {{gen "proposition" temperature=temperature max_tokens=max_tokens stop='\"\\n'}}. "
    "Recall the Hypothesis": "{{hypothesis}}"
    "Judgement": "Now we know that the Premises {{select "judgement" options=valid_judgement logprobs='logprobs'}} the Hypothesis."
    ''')

#从用户通过命令行参数指定的数据集中加载训练集数据，并将这些数据存储在变量data中。
data = datasets.load_dataset(args.dataset, split='train')

t = time.localtime()
#记录日志，使用前缀和时间用来命名
logfilename = f'results-autotnli-{args.save_suffix}--' + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                                     t) + '.jsonl'
#记录日志，先写入第一行目录
with open(logfilename, 'w') as f:
    f.write(time.strftime("%Y-%m-%d %H:%M:%S", t) + '\n')  # write each result as a new line
    f.write("Model: " + args.model + "\n")
    f.write("Dataset: " + args.dataset + "\n")
    f.write(f"Temperature:{args.temperature}\n")
    f.write(f"Max Tokens:{args.max_tokens}\n")
    f.write("bf16: True\n")
    f.write("--------------------------------\n")

correct_predictions = 0
cnt = 0
total_cnt = len(data)


data_list = []
#从给定的数据集data中提取前1000个元素（如果数据集包含这么多元素的话），并将这些元素存储到一个新的列表data_list中。这个迭代器的原理就是如果很长就取1000个，如果很短就迭代取所有
for i in data:
    if cnt == 1000:
        break
    data_list.append(i)
    cnt += 1

cnt = 0

for example in tqdm(data_list, desc="Evaluating", unit="example"):
    #desc参数设置进度条的描述为"Evaluating"，unit参数设置进度条的单位为"example"。
    #更新label为entail或者contradict，为什么？是在新加一个字典项吗
    example.update({"label": 'entail' if example['label'] == 'entailment' else 'contradict'})
    cnt += 1
    conclusion = example['hypothesis']
    #这一步不知道为啥要用同样的符号分割再拼接
    premises = [s + '.' for s in example['premises'].split('.')]
    premises_cnt = len(example['premises'])
    propositions = ""
    failed_cnt = 0

    if args.verbose: print("[Premises]: \t", premises)
    if args.verbose: print("[Hypothesis]: \t", conclusion)

    ans_dict = {}

    #初始化一个列表用于存储数据，已知valid_judgement是一个选项集，只有两个元素
    #所以这一步可能是在计数，把选项集的每一项作为key，value初始化为0
    for i in valid_judgement:
        ans_dict[i] = 0

    for i in range(args.sc_cnt):
        #通过多次调用 structure_program 函数来获取不同的判断结果，并统计每种结果出现的次数。而structure_program是之前创建好的prompt
        #这里的examples是之前定义好的那些例子
        out = structure_program(
            examples=examples,
            premises=(' '.join(premises)),
            hypothesis=conclusion,
            valid_judgement=valid_judgement,
            temperature=0.7,
            max_tokens=args.max_tokens
        )
        ans_dict[out['judgement']] = ans_dict[out['judgement']] + 1

    ans, ans_cnt = '', 0
    for i in ans_dict.keys():
        #可能是在统计大多数
        if ans_dict[i] > ans_cnt:
            ans = i
            ans_cnt = ans_dict[i]
    #如果大多数的结果和标签一致，那么就认为这个被正确预测了
    if ans == example["label"]:
        correct_predictions += 1
    #展示结果
    print("[Prediction]: ", ans)
    print("[Actual]: ", example["label"])
    #准确率
    accuracy = correct_predictions / cnt

    print("[Running Average Accuracy]: ", accuracy)
    #example是选出来的1000条数据中的每个子条，为什么example有json_name这个属性？得打开数据看一看
    result = {
        "json_name": example["json_name"],
        "prediction": ans,
        "actual": example["label"],
        "accuracy": accuracy,
        "generated_propositions": propositions,
    }
    with open(logfilename, 'a') as f:
        f.write(json.dumps(result) + '\n') 
