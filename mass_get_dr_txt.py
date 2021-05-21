import os
import shutil



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
efficientdet_path = r"./efficientdet.py"
get_dr_txt_path = r"./get_dr_txt.py"
new_pth = "Epoch" + str(start_idx) + ".pth"
new_input = 'input/input_' + str(start_idx)
alter(efficientdet_path, 'Epoch1.pth', new_pth)
alter(get_dr_txt_path, 'input/input_1', new_input)
if os.path.exists("input"):  # if it exist already
    shutil.rmtree("input")
"""
批量预测
"""
for idx in range(start_idx, end_idx + 1):
    old_pth = "Epoch" + str(idx) + ".pth"
    new_pth = "Epoch" + str(idx + 1) + ".pth"
    old_input = 'input/input_' + str(idx)
    new_input = 'input/input_' + str(idx + 1)
    os.system("python ./get_dr_txt.py")
    alter(efficientdet_path, old_pth, new_pth)
    alter(get_dr_txt_path, old_input, new_input)
alter(efficientdet_path, 'Epoch' + str(end_idx + 1) + '.pth', 'Epoch1.pth')
alter(get_dr_txt_path, 'input/input_' + str(end_idx + 1), 'input/input_1')
print('Finish!')
