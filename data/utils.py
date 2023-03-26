import re


# 过滤文本最大长度和最小长度
min_length = 10
max_length = 100
# 定义正则表达式模式
pattern1 = re.compile(r"[^\w\s]{2,}")
pattern2 = re.compile(r"[\u4e00-\u9fa5]")
# 定义正则表达式模式，用于匹配非正常符号
pattern3 = re.compile('[^\u4e00-\u9fa5a-zA-Z0-9\s]')


# 判断字符串中是否有连续符号
def has_continuous_symbol(string):
    return bool(pattern1.search(string))


# 判断字符串中是否有中文
def has_chinese(string):
    return bool(pattern2.search(string))


# 判断字符串长度是否过长
def is_short_long(sent):
    return len(sent) <= min_length or len(sent) >= max_length


# 精细切分句子
def cut_sent(para):
    '''https://blog.csdn.net/blmoistawinde/article/details/82379256'''
    # 如果.:后面不是”‘数字大小写字母，则在.:和后面字符中间插入\n
    para = re.sub('([。！？；\.;\?])([^”’\da-zA-Z\s])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = re.sub('([；;])([^；;])', r'\1\n\2', para)  # 分号
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它

    return para.split("\n")


# 清洗文本函数
def clean_text(text):
    if has_continuous_symbol(text):
        return False
    if not has_chinese(text):
        return False
    if is_short_long(text):
        return False
    if "and" in text or "AND" in text or "or" in text or "OR" in text:
        return False

    return True