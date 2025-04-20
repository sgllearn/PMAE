import json
import re
import pandas as pd
import urllib.request

# Public Suffix List获取网址
PSL_URL = "https://publicsuffix.org/list/public_suffix_list.dat"  # 请确保这是有效的URL

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
        tld = match.group(0)
        n_tld = domain[:-len(tld)]
        tld_tokens = list(tld)
        n_tld_tokens = list(n_tld)
        return tld_tokens, n_tld_tokens
    else:
        return [], list(domain)

# 计算N-TLD的统计特征features
def calculate_n_tld_stats(n_tld_tokens):
    n_tld = ''.join(n_tld_tokens)
    length = len(n_tld)
    all_digits = 1 if n_tld.isdigit() else 0  #是否全为数字
    all_letters = 1 if n_tld.isalpha() else 0  #是否全为字母
    digit_count = sum(1 for char in n_tld if char.isdigit())
    letter_count = sum(1 for char in n_tld if char.isalpha())
    special_char_count = length - digit_count - letter_count
    digit_ratio = digit_count / length if length > 0 else 0   #数字占比
    letter_ratio = letter_count / length if length > 0 else 0 #字母占比
    special_char_ratio = special_char_count / length if length > 0 else 0 #特殊字符占比

    stats = [
        length,
        all_digits,
        all_letters,
        round(digit_ratio, 3),
        round(letter_ratio, 3),
        round(special_char_ratio, 3)
    ]

    return stats

def main():
    # 获取并解析Public Suffix List作为tld名单，手动添加不符合域名系统规则的特殊“tld”
    psl_content = fetch_public_suffix_list()
    suffixes = parse_public_suffix_list(psl_content)
    suffixes.update(['co.c', 'cz.c'])
    tld_pattern = build_regex_pattern(suffixes)

    # 读取CSV数据集文件
    csv_file_path = 'dataset/sampled_data.csv'
    df = pd.read_csv(csv_file_path)

    # 创建一个新的DataFrame来存储结果方便保存
    results = []

    # 处理数据集中的每个域名
    for _, row in df.iterrows():
        domain = row['Domain']
        label = row['Label']
        tld_tokens, n_tld_tokens = split_and_tokenize_domain(domain, tld_pattern)

        # 获取完整的N-TLD字符串
        n_tld = ''.join(n_tld_tokens)

        # 计算N-TLD的统计特征
        n_tld_stats = calculate_n_tld_stats(n_tld_tokens)

        # 将结果添加到列表中（每个结果是一个字典）
        results.append({
            'Domain': domain,
            'Label': label,
            'TLDToken': json.dumps(tld_tokens),  # TLD Tokens 单独保存为JSON字符串
            'NTLD': n_tld,  # 完整的N-TLD字符串
            'CharToken': json.dumps(n_tld_tokens),  # N-TLD Tokens 单独保存为JSON字符串
            'Feature': n_tld_stats  # N-TLD Stats 直接保存为数值列表
        })

    # 将结果列表转换为DataFrame
    result_df = pd.DataFrame(results)

    # 将结果DataFrame写入新的CSV文件
    output_csv_file_path = 'dataset/processed_data_sampled.csv'
    result_df.to_csv(output_csv_file_path, index=False)

    print(f"处理完成，结果已保存到 {output_csv_file_path}")

if __name__ == "__main__":
    main()