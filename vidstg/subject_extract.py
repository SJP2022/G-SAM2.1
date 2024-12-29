import spacy

# 加载英文语言模型
nlp = spacy.load("en_core_web_sm")

# 定义一个函数来提取句子中所有的主语
def extract_all_subjects(sentence):
    doc = nlp(sentence)
    subjects = []
    for token in doc:
        if token.dep_ == "nsubj":
            subjects.append(token.text)
    return subjects

# 示例句子
sentence = "A man and woman dance together in a spacious room with large windows, showcasing their graceful movements."

# 提取句子中所有的主语
subjects = extract_all_subjects(sentence)

# 打印句子中所有的主语
print("所有的主语:", subjects)