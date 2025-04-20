import json
import re
import pandas as pd
import urllib.request
import math
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

# Public Suffix List获取网址
PSL_URL = "https://publicsuffix.org/list/public_suffix_list.dat"


# 下载Public Suffix List
def fetch_public_suffix_list():
    with urllib.request.urlopen(PSL_URL) as response:
        content = response.read().decode('utf-8')
    cleaned_content = '\n'.join([line for line in content.splitlines() if not line.startswith("//")])
    return cleaned_content


# 解析Public Suffix List
def parse_public_suffix_list(content):
    suffixes = set()
    for line in content.splitlines():
        if not line or line.startswith("="):
            continue
        suffixes.add(line.strip())
    return suffixes


# 构建正则表达式模式
def build_regex_pattern(suffixes):
    escaped_suffixes = [re.escape(suffix) for suffix in suffixes]
    pattern = r'\.(' + '|'.join(escaped_suffixes) + ')$'
    return re.compile(pattern, re.IGNORECASE)


# 分割DGA域名的tld和ntld
def split_and_tokenize_domain(domain, tld_pattern):
    match = tld_pattern.search(domain)
    if match:
        tld = match.group(1)  # 提取 TLD（不包括点）
        n_tld = domain[:-len(tld) - 1]  # 提取 N-TLD（不包括 TLD 和最后的点）
        tld_tokens = list(tld)
        n_tld_tokens = list(n_tld)
        return tld, tld_tokens, n_tld_tokens
    else:
        return "", [], list(domain)


def main():
    # 获取并解析Public Suffix List作为tld名单，手动添加不符合域名系统规则的特殊“tld”
    psl_content = fetch_public_suffix_list()
    suffixes = parse_public_suffix_list(psl_content)
    suffixes.update(['co.c', 'cz.c'])
    tld_pattern = build_regex_pattern(suffixes)

    # 读取CSV数据集文件
    csv_file_path = 'dataset/processed_data.csv'
    df = pd.read_csv(csv_file_path)

    # 创建一个新的列表来存储结果
    results = []

    # 处理每个域名
    for domain in df['Domain']:  # 假设 'domain' 是包含域名的列名
        tld, _, n_tld_tokens = split_and_tokenize_domain(domain, tld_pattern)
        if tld:
            ntld_length = len(n_tld_tokens)  # N-TLD 长度
            len_tld_str_1 = str(ntld_length // 1)  # 除以1（实际就是原长度）
            len_tld_str_2 = str(math.ceil(ntld_length / 2))  # 除以2（向上取整）
            len_tld_str_3 = str(math.ceil(ntld_length / 3))  # 除以3（向上取整）

            # 组合成新的字符串
            tld_comb_1 = f"{len_tld_str_1}{tld}"
            tld_comb_2 = f"{len_tld_str_2}{tld}"
            tld_comb_3 = f"{len_tld_str_3}{tld}"

            # 将六个字符串放入一个列表中
            com_tld_len = [len_tld_str_1, tld_comb_1, len_tld_str_2, tld_comb_2, len_tld_str_3, tld_comb_3]
            results.append(com_tld_len)
        else:
            # 如果无法匹配 TLD，则使用默认值或跳过
            results.append([None] * 6)

    # 将结果转换为 DataFrame 并添加为新列
    new_df = pd.DataFrame(results,
                          columns=['len_tld_1', 'tld_comb_1', 'len_tld_2', 'tld_comb_2', 'len_tld_3', 'tld_comb_3'])
    new_combined_df = pd.concat([df, new_df], axis=1)
    new_combined_df['com_tld_len'] = new_combined_df[
        ['len_tld_1', 'tld_comb_1', 'len_tld_2', 'tld_comb_2', 'len_tld_3', 'tld_comb_3']].apply(
        lambda row: row.tolist(), axis=1)

    # 删除中间生成的列
    new_combined_df.drop(columns=['len_tld_1', 'tld_comb_1', 'len_tld_2', 'tld_comb_2', 'len_tld_3', 'tld_comb_3'],
                         inplace=True)

    # 将合并后的 DataFrame 保存回原始 CSV 文件
    new_combined_df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()