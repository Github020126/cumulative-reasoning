import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool  #ProcessPool类通常用于创建进程池，可以并行执行多个任务，提高程序的执行效率。
from concurrent.futures import TimeoutError  #将concurrent.futures模块中的TimeoutError类导入到当前作用域中，以便在后续代码中使用。当异步操作超时时，会抛出TimeoutError异常。

from eval.grader import * #自行定义的grader
from utils.parser import *
from utils.utils import load_jsonl
from utils.python_executor import PythonExecutor


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, #因为是在评估，所以不使用为训练设计的prompt
             use_train_prompt_format=False, code_concat = False, max_func_call = 4, 
             code_exec_warning = False, max_code_fix_retries = 4, execute=False):
    assert samples or file_path, "samples or file_path must be provided"#字符串，作为断言失败时抛出的异常消息。
    if not samples:
        samples = list(load_jsonl(file_path))
    # dedup by idx
    #这段代码的目的是确保samples列表中的每个样本都有一个唯一的索引，并且列表按照这个索引排序。
                 #如果样本已经包含索引，则直接排序；如果没有，则添加索引并排序。
    if 'idx' in samples[0]:
        #只判断第一个元素是否有序号就可以了，如果第一个没有序号那么就可以认为后面的也没有序号
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
            #在else块中，这段代码使用列表推导式为samples列表中的每个元素添加一个'idx'键，
            #其值是元素的索引（由enumerate函数提供）。这样，每个元素都会包含一个唯一的索引。
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]
            #enumerate(samples)：这是一个函数，它返回一个枚举对象，该对象包含两个值：一个是索引（从0开始），另一个是列表samples中的元素。在列表推导式中，enumerate(samples)会逐个遍历samples列表中的元素。
            #for idx, sample in enumerate(samples)：这是列表推导式的循环部分，它将enumerate(samples)返回的每个元素（一个包含索引和样本的元组）解包为两个变量idx和sample。idx是当前元素的索引，sample是当前元素本身。
            #dict(idx=idx, **sample)：这是一个字典构造函数，它创建一个新的字典。idx=idx是将索引idx作为键，其值也是idx。
            #**sample是Python中的解包操作符，它将sample字典中的所有键值对作为参数传递给dict函数，从而将sample字典中的所有内容添加到新创建的字典中。

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
        #这意味着如果samples列表中的元素多于max_num_samples，那么只有前max_num_samples个元素会被保留用于后续处理。如果samples列表中的元素少于max_num_samples，则整个列表都会被保留。
    # parse gt
    for sample in samples:
        #这段代码的作用是为每个样本解析正确的答案和相关的上下文信息，并将这些信息存储在样本字典中，以便后续的评估和比较。
        #gt_cot是与正确答案相关的上下文
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)

    # execute
    if ('pred' not in samples[0]) or execute:
        if "pal" in prompt_type:
            executor = PythonExecutor(get_answer_expr="solution()")
        else:
            executor = PythonExecutor(get_answer_from_stdout=True)
    #如果prompt_type字符串中包含子字符串"pal"，则创建一个PythonExecutor对象，并且设置get_answer_expr参数为"solution()"。这意味着在执行代码时，PythonExecutor会尝试调用名为solution()的函数来获取答案。
    #如果prompt_type不包含"pal"，则创建另一个PythonExecutor对象，并且设置get_answer_from_stdout参数为True。这意味着PythonExecutor会从代码执行的标准输出中获取答案。
        
        for sample in tqdm(samples, desc="Execute"):
            sample['pred'] = []
            sample['report'] = []
            for code in sample['code']:
                #这部分代码的作用是对于每个样本中的每个代码片段，使用PythonExecutor执行代码，并收集执行结果和报告，然后将这些信息存储在样本字典中。
                pred, report = run_execute(executor, code, prompt_type, execute=True)
                sample['pred'].append(pred)
                sample['report'].append(report)
    #定义了参数来自于样本的哪些标签，这个列表迭代式倒是挺有意思，可以学习一下
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]
    
    scores = []
    timeout_cnt = 0 

    with ProcessPool() as pool:
            #使用ProcessPool创建一个进程池，并将其绑定到变量pool上。
            #在with语句块内部，可以安全地使用pool来执行多进程任务。当with语句块执行完毕后，会自动释放进程池资源，不需要手动关闭或清理。
        future = pool.map(math_equal_process, params, timeout=3)
            #创建一个Pool对象，然后使用map方法将math_equal_process函数应用于params列表中的每个元素，并且设置超时时间为3秒。处理的结果会被赋值给变量future
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)#从iterator中获取下一个结果，并将其赋值给变量result。
                    scores.append(result)#把result加到score列表中
                except StopIteration:
                    #如果所有数据都处理完了那么就结束迭代
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) #进度条更新

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        # - 这行代码将 scores 列表中的一部分赋值给当前样本的 'score' 键。
        #这部分从 scores 列表的 idx 索引开始，到 idx + len(sample['pred']) 结束。len(sample['pred']) 表示当前样本中预测结果的数量。
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad
    #这段代码的作用是确保 score_mat 中的所有子列表长度一致，如果某个子列表长度不足，则用其最后一个元素进行填充，以达到统一的长度。
    #这样做通常是为了在后续处理中能够对齐数据，例如在进行矩阵运算或者数据分析时。缺少多说就用最后一个元素填充多少
    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_str = f"Num samples: {len(samples)}\n" \
        f"Num scores: {len(scores)}\n" \
        f"Timeout samples: {timeout_cnt}\n" \
        f"Empty samples: {len([s for s in samples if not s['pred'][-1]])}\n" \
        f"Prompt type: {prompt_type}\n" \
        f"use_train_prompt_format: {use_train_prompt_format}\n" \
        f"code_concat: {code_concat}\n" \
        f"max_func_call: {max_func_call}\n" \
        f"code_exec_warning: {code_exec_warning}\n"\
        f"max_code_fix_retries: {max_code_fix_retries}\n"\
        f"Mean score: {mean_score}\n"

    # each type score
    if "type" in samples[0]:
        #统计每种类型的得分
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                #如果不在就添加key
                type_scores[sample['type']] = []
            #如果在就把分数的最后一个元素添加到列表中？
            type_scores[sample['type']].append(sample['score'][-1])
        #这一步是在确定数据类型将 type_scores 字典中的值 v 转换为 NumPy 数组。计算这个数组的平均值。将平均值乘以 100，可能用于转换为百分比形式。最后，将结果四舍五入到小数点后 1 位。
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_str += f"Type scores: {type_scores}\n"

    # each subject score
    #和type基本相同，subject和type的区别和联系是什么？
    if "subject" in samples[0]:
        subject_scores = {}
        for sample in samples:
            if sample['subject'] not in subject_scores:
                subject_scores[sample['subject']] = []
            subject_scores[sample['subject']].append(sample['score'][-1])
        subject_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in subject_scores.items()}
        subject_scores = {k: v for k, v in sorted(subject_scores.items(), key=lambda item: item[0])}
        result_str += f"Type scores: {subject_scores}\n"

    # each level score
    if "level" in samples[0]:
        level_scores = {}
        for sample in samples:
            if sample['level'] not in level_scores:
                level_scores[sample['level']] = []
            level_scores[sample['level']].append(sample['score'][-1])
        level_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in level_scores.items()}
        level_scores = {k: v for k, v in sorted(level_scores.items(), key=lambda item: item[0])}
        result_str += f"Level scores: {level_scores}\n"

    print(result_str)
    return result_str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tora")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
