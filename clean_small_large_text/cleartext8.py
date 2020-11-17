import os
# 删除小文本或大文本
def file_path(path):
    for file in os.listdir(path):
        filepath = path + "/" + file
        del_small_file(filepath)
def del_small_file(file_name):
    size = os.path.getsize(file_name)
    min_file_size = 1 * 1024
    max_file_size = 10 * 1024
    if size < min_file_size or size > max_file_size:
        os.remove(file_name)


if __name__ == '__main__':
    path = r'../dataset/data/股票'
    file_path(path)
