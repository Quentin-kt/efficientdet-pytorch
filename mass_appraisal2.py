import os
import shutil
import matplotlib.pyplot as plt
import numpy as np


# 重命名文件中的特定字符串
def alter(file, old_str, new_str):
    with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            if old_str in line:
                line = line.replace(old_str, new_str)
            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)


"""
变量初始化
"""
# 批量评估的起始点
start_idx = 1
end_idx = 100
appraisal_path = r"./appraisal.py"
new_input = 'input/input_' + str(start_idx)
new_result = 'results/results_' + str(start_idx)
alter(appraisal_path, 'results/results_1', new_result)
alter(appraisal_path, 'input/input_1', new_input)
if os.path.exists("results"):  # if it exist already
    shutil.rmtree("results")

"""
批量评估
"""
os.system("python ./get_gt_txt.py")
for idx in range(start_idx, end_idx + 1):
    old_input = 'input/input_' + str(idx)
    new_input = 'input/input_' + str(idx + 1)
    old_result = 'results/results_' + str(idx)
    new_result = 'results/results_' + str(idx + 1)
    print('###############################################')
    print('第' + str(idx) + '次评估')
    os.system("python ./appraisal.py")
    alter(appraisal_path, old_input, new_input)
    alter(appraisal_path, old_result, new_result)
alter(appraisal_path, 'input/input_' + str(end_idx + 1), 'input/input_1')
alter(appraisal_path, 'results/results_' + str(end_idx + 1), 'results/results_1')

"""
评估结果汇总与取优
"""
# 初始化
loss_summary_path = "results/loss_summary.txt"
loss_list = []
if os.path.exists(loss_summary_path):  # if it exist already
    os.remove(loss_summary_path)
# 遍历results.txt
for idx in range(1, end_idx - start_idx + 2):
    real_idx = start_idx + idx - 1
    result_path = "results/results_" + str(real_idx) + "/results.txt"
    key_str = "total = "
    with open(result_path, "r", encoding="utf-8") as result_txt:
        for line in result_txt:
            if key_str in line:
                # 将loss结果写入txt文件
                with open(loss_summary_path, "a", encoding="utf-8") as loss_summary_txt:
                    loss_summary_txt.write('Epoch' + str(real_idx) + '_' + line)
                loss_list.append(line)
                # 删去字符串首尾的无关字符
                loss_list[idx - 1] = loss_list[idx - 1].strip(key_str)
                loss_list[idx - 1] = loss_list[idx - 1].strip('\n')
                break
loss_list = [float(x) for x in loss_list]
loss_min = min(loss_list)
loss_min_idx = loss_list.index(loss_min) + start_idx
with open(loss_summary_path, "a", encoding="utf-8") as loss_summary_txt:
    loss_summary_txt.write('loss_min=' + '第' + str(loss_min_idx) + '次训练——' + str(loss_min))
print('###############################################')
print('loss_min=' + '第' + str(loss_min_idx) + '次训练——' + str(loss_min))

"""
评估结果绘图
"""
x = np.arange(end_idx - start_idx + 1)
y = [float(x) for x in loss_list]
plt.figure()
plt.grid(True)  # 网格线
plt.title('bce_loss_min = ' + str(loss_min))
plt.xlabel('index')
plt.ylabel('Average loss')
plt.plot(x, y, 'o-')
plt.savefig("results/loss_summary.jpg")
