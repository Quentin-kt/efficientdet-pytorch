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
get_map_path = r"./get_map.py"
new_input = 'input/input_' + str(start_idx)
new_result = 'results/results_' + str(start_idx)
alter(get_map_path, 'results/results_1', new_result)
alter(get_map_path, 'input/input_1', new_input)
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
    os.system("python ./get_map.py")
    alter(get_map_path, old_input, new_input)
    alter(get_map_path, old_result, new_result)
alter(get_map_path, 'input/input_' + str(end_idx + 1), 'input/input_1')
alter(get_map_path, 'results/results_' + str(end_idx + 1), 'results/results_1')

"""
评估结果汇总与取优
"""
# 初始化
map_summary_path = "results/map_summary.txt"
map_list = []
if os.path.exists(map_summary_path):  # if it exist already
    os.remove(map_summary_path)
# 遍历results.txt
for idx in range(1, end_idx - start_idx + 2):
    real_idx = start_idx + idx - 1
    result_path = "results/results_" + str(real_idx) + "/results.txt"
    key_str = "mAP = "
    with open(result_path, "r", encoding="utf-8") as result_txt:
        for line in result_txt:
            if key_str in line:
                # 将map结果写入txt文件
                with open(map_summary_path, "a", encoding="utf-8") as map_summary_txt:
                    map_summary_txt.write('Epoch' + str(real_idx) + '_' + line)
                map_list.append(line)
                map_list[idx - 1] = map_list[idx - 1].strip('mAP = ')
                map_list[idx - 1] = map_list[idx - 1].strip('%\n')
                break
map_list = [float(x) for x in map_list]
map_max = max(map_list)
map_max_idx = map_list.index(max(map_list)) + start_idx
with open(map_summary_path, "a", encoding="utf-8") as map_summary_txt:
    map_summary_txt.write('map_max=' + '第' + str(map_max_idx) + '次训练——' + str(map_max) + '%')
print('###############################################')
print('map_max=' + '第' + str(map_max_idx) + '次训练——' + str(map_max) + '%')

"""
评估结果绘图
"""
x = np.arange(end_idx - start_idx + 1)
y = [float(x) for x in map_list]
plt.figure()
plt.grid(True)  # 网格线
plt.title('mAP_max = ' + str(map_max))
plt.xlabel('index')
plt.ylabel('mAP')
plt.plot(x, y, 'o-')
plt.savefig("results/map_summary.jpg")
