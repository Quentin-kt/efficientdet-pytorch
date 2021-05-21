import os
import torch
import numpy as np
import shutil

# 初始化
classes = ["ChJ", "BX", "ZhAW", "CK", "QF", "TJ", "JG", "ShG", "PL", "ZhGAJ", "CQBG", "YWChR", "ShL", "FSh"]


# 将文件夹内所有txt文件转换为一个tensor
def file_lines_to_tensor(dir_path, whether_confidence):
    total_file_count = sum([len(files) for root, dirs, files in os.walk(dir_path)])
    numpy = np.zeros((total_file_count, len(classes)))
    for filepath, dir_path, filenames in os.walk(dir_path):
        idx = -1
        for name in filenames:
            # 初始化下标和最大概率
            idx += 1
            max_confidence = np.zeros(len(classes))
            with open(os.path.join(filepath, name), "r", encoding="utf-8") as txt:
                for line in txt:
                    for idx_class in range(len(classes)):
                        if classes[idx_class] in line:
                            if whether_confidence:
                                confidence = float(line.split()[1])
                                if confidence > max_confidence[idx_class]:
                                    max_confidence[idx_class] = confidence
                            else:
                                max_confidence[idx_class] = 1
                            break
            numpy[idx][:] = max_confidence
    tensor = torch.from_numpy(numpy)
    return tensor


"""
 评估
    损失函数——多标签二分类交叉熵（Binary Cross Entropy）
"""
detection = file_lines_to_tensor(r'input/input_1/detection-results', True)
truth = file_lines_to_tensor(r'input/ground-truth', False)
loss = torch.nn.BCELoss()
total_loss = float(loss(detection, truth))
print(total_loss)

"""
 将评估结果存入results
"""
results_files_path = r'results/results_1'
if os.path.exists(results_files_path):  # if it exist already
    # reset the results directory
    shutil.rmtree(results_files_path)
os.makedirs(results_files_path)
with open(results_files_path + "/results.txt", 'a') as results_file:
    results_file.write("total = " + str(total_loss) + "\n")
