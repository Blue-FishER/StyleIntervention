# 获取指定batch的文件的路径
import os


def get_file_path(dir_path, batch, file_kind="png"):
    paths = []
    # 文件必须以三位的batch开头
    begin_str = str(batch).zfill(3)

    for file in os.listdir(dir_path):
        if file.startswith(begin_str) and file.endswith(file_kind):
            paths.append(os.path.join(dir_path, file))

    return paths
