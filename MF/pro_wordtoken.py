import re
import pandas as pd
import itertools
from collections import defaultdict, Counter


def generate_ngrams(text, n):
    """生成指定长度的n-gram"""
    ngrams = [''.join(chars) for chars in
              itertools.islice(itertools.zip_longest(*[text[i:] for i in range(n)]), len(text) - n + 1) if
              None not in chars]
    return ngrams


def collect_top_ngrams(label, ngram_counts, n, top_percent_threshold, top_tokens_dict):
    """收集前threshold的高频n-gram并添加到字典中"""
    if n in ngram_counts:
        total_tokens = sum(ngram_counts[n].values())  # 计算当前标签和n-gram长度的总token数
        threshold = total_tokens * top_percent_threshold
        top_tokens = sorted(ngram_counts[n].items(), key=lambda x: x[1], reverse=True)
        accumulated_count = 0
        for token, count in top_tokens:
            accumulated_count += count
            if accumulated_count > threshold:
                break
        top_tokens_list = [token for token, count in top_tokens[:top_tokens.index((token, count)) + 1]]
        top_tokens_dict[label].extend(top_tokens_list)


def match_ntld_to_wtl(ntld, wtl):
    """将NTLD转换为基于WTL的变长Token序列"""
    word_token_sequence = []
    i = 0
    while i < len(ntld):
        for token in wtl:
            if ntld[i:i + len(token)] == token:
                word_token_sequence.append(token)
                i += len(token)
                break
        else:
            word_token_sequence.append('-')
            i += 1
    return word_token_sequence


def process_csv(file_path):
    # 读取原始CSV文件到DataFrame
    df = pd.read_csv(file_path, encoding='utf-8')
    token_counts = defaultdict(lambda: defaultdict(Counter))
    # 初始化用于存储n-gram的列
    df['2gram'] = ''
    df['3gram'] = ''
    # 遍历DataFrame的每一行来生成n-gram并更新token_counts
    for index, row in df.iterrows():
        label = row['Label']
        tld = row['NTLD']

        # 生成2-gram, 3-gram和4-gram
        ngrams_2 = generate_ngrams(tld, 2)
        ngrams_3 = generate_ngrams(tld, 3)
        ngrams_4 = generate_ngrams(tld, 4)

        # 只保存2，3
        df.at[index, '2gram'] = ','.join(ngrams_2)
        df.at[index, '3gram'] = ','.join(ngrams_3)
        # 更新统计信息
        for n, ngrams in [(2, ngrams_2), (3, ngrams_3), (4, ngrams_4)]:
            for ngram in ngrams:
                token_counts[label][n][ngram] += 1

    # 选出并收集前1%的Token及其频率
    top_percent_threshold = 0.01
    top_tokens_dict = defaultdict(list)

    # 收集2-gram, 3-gram和4-gram的高频token
    for label, ngram_counts in token_counts.items():
        for n in [2, 3, 4]:
            collect_top_ngrams(label, ngram_counts, n, top_percent_threshold, top_tokens_dict)

    # 合并所有高频n-gram到一个列表HWTL中，总数5015
    HWTL = []
    for label, tokens in top_tokens_dict.items():
        HWTL.extend(tokens)
    '''
    additional_strings = [
        "co.", ".uk", "co.u", "o.u", "o.uk", ".co.", ".r", ".ru", ".u", ".s", ".se", ".cc",
        ".r", ".ru", ".na", ".nam", "name", "j.r", "5.", "5.c", "5.co", "j.ru", "x.r", "x.ru",
        "b.r", "b.ru", "e.r", "e.ru", "4.", "4.c", "4.co", "1.c", "t.ru", "f.ru", "d.r", "c.r",
        "c.ru", "1.co", "t.r", "f.r", "l.r", "l.ru", "y.r", "y.ru", "1.", "d.ru", "3.co", "3.",
        "m.ru", "m.r", "r.r", "3.c", "8.co", "r.ru", "h.r", "8.", "7.co", "7.c", "7.", "w.ru",
        "w.r", "i.r", "i.ru", "u.ru", "u.r", "6.co", "6.c", "6.", "8.c", "h.ru", "0.", "0.c",
        "n.ru", "p.r", "p.ru", "s.r", "s.ru", "v.r", "v.ru", "n.r", "2.c", "0.co", "2.", "d.r",
        "d.ru", "2.co", "t.se", "q.r", "lm.r", "g.cc", "f.se", "f.s", "q.ru", "ciax", "b.se",
        "b.s", "b.cc", "t.s", "m1", "xz.c", "cwxx", "9.co", "e1", "wbhi", "esj7", "wdwx", "qrh.",
        "icbk", "9.c", "ytck", "yfox", "yffn", "yaaq", "f5", "h0", "xy.r", "wf.r", "xqtx", "j6",
        "j7", "wtfi", "yu.i", "gvxw", "9.", "jwxb", "xrjw", "w.ru", "w.r", "u.cc", "s.cc", "o.se",
        "o.s", "o.cc", "ld.r", "l.cc", "k.cc", "j.se", "a.na", "j.s", "j.ru", "j.r", "i.cc", "h.se",
        "h.s", "g.se", "g.s", "e.na", "vijm", "c.cc", "n4", "s3", "vc.r", "ktso", "o.ru", "o.r",
        "nxfn", "nhkb", "cshm", "mrny", "mebn", "dccx", "lynb", "lyhv", "lgjx", "ex.r", "exkq",
        "kgfs", "oets", "kat.", "k.ru", "k.r", "jr.r", "josr", "fggi", "fxni", "imtf", "g.r",
        "g.ru", "huty", "hswa", "ge.r", "btvq", "os.r", "vakd", "aisy", "gnjt", "sj7", "uqjv",
        "t2", "xj6", "tl.i", "tidg", "a.r", "a.ru", "sq.r", "slbw", "shdy", "ab.r", "rhka",
        "ound", "reli", "apfx", "bhoh", "qwst", "ql.r", "pxsr", "pmta", "pkcv", "pfxi", "pdmm",
        "bphf", "bpsi", "owgf", "y.cc"
    ]
    HWTL.extend(additional_strings)
    '''
    HWTL = list(set(HWTL))

    # 生成EWTL
    with open('dataset/english_words.txt', 'r') as file:
        english_dictionary = set(map(str.strip, file.readlines()))
    EWTL = set(english_dictionary)

    # 合并HWTL和EWTL并去重保存到WTL里，总数6530（0.01时）
    WTL = set(HWTL).union(EWTL)
    WTL = sorted(WTL, key=len, reverse=True)
    with open('dataset/WTL.txt', 'w') as file:
        for word in WTL:
            file.write(f"{word}\n")

    # 对原csv每一行ntld做匹配操作，并保存回原csv文件新增一列‘Word Token’
    for index, row in df.iterrows():
        ntld = row['NTLD']
        word_token_sequence = match_ntld_to_wtl(ntld, WTL)
        df.at[index, 'Word Token'] = ' '.join(word_token_sequence)

    # 保存回原CSV文件
    df.to_csv(file_path, index=False, encoding='utf-8')


# CSV文件路径
csv_file_path = 'dataset/processed_data.csv'
process_csv(csv_file_path)