import numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图片格式
sns.set_theme(style="darkgrid", font='Times New Roman', font_scale=1.2)
fig = plt.figure(figsize=(7.0, 4.8))

def smooth(read_path, save_path, file_name, x='step', y='total_reward', z='label', weight=0.96):
    data = pd.read_csv(read_path + file_name)
    scalar = data[y].values
    label = data[z]
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last*weight + (1-weight)*point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save_file = save_path + 'smooth_' + file_name
    save = pd.DataFrame({x: data[x].values, y: smoothed})
    save['label'] = label
    save.to_csv(save_file)
    return save_file

def draw_main_results():
    # # 平滑预处理maskDQN原始reward数据
    # new_file1 = smooth(read_path='./data/asterix/', save_path='./data/asterix/', file_name='DQN_18_1701844949.csv')
    # new_file2 = smooth(read_path='./data/asterix/', save_path='./data/asterix/', file_name='DQN_18_1701854879.csv')
    # new_file3 = smooth(read_path='./data/asterix/', save_path='./data/asterix/', file_name='DQN_18_1701859621.csv')
    #
    # # 平滑预处理mask原始reward数据
    # mask_file1 = smooth(read_path='./data/asterix/', save_path='./data/asterix/', file_name='maskDQN_18_1701866078.csv')
    # mask_file2 = smooth(read_path='./data/asterix/', save_path='./data/asterix/', file_name='maskDQN_18_1701867786.csv')
    # mask_file3 = smooth(read_path='./data/asterix/', save_path='./data/asterix/', file_name='maskDQN_18_1701872038.csv')
    #
    # # 平滑预处理DQN原始reward数据
    # dqn_file1 = smooth(read_path='./data/asterix/', save_path='./data/asterix/', file_name='DQN_5_1701845123.csv')
    # dqn_file2 = smooth(read_path='./data/asterix/', save_path='./data/asterix/', file_name='DQN_5_1701854867.csv')
    # dqn_file3 = smooth(read_path='./data/asterix/', save_path='./data/asterix/', file_name='DQN_5_1701859637.csv')

    # # 读取maskDQN平滑后的数据
    # worstDQN1 = pd.read_csv(new_file1)
    # worstDQN2 = pd.read_csv(new_file2)
    # worstDQN3 = pd.read_csv(new_file3)
    #
    # # # 读取mask平滑后的数据
    # mask1 = pd.read_csv(mask_file1)
    # mask2 = pd.read_csv(mask_file2)
    # mask3 = pd.read_csv(mask_file3)
    #
    # # 读取DQN平滑后的数据
    # bestDQN1 = pd.read_csv(dqn_file1)
    # bestDQN2 = pd.read_csv(dqn_file2)
    # bestDQN3 = pd.read_csv(dqn_file3)

    # # # 拼接到一起
    # df_worstDQN = worstDQN1._append(worstDQN2._append(worstDQN3))
    # df_mask = mask1._append(mask2._append(mask3))
    # df_bestDQN = bestDQN1._append(bestDQN2._append(bestDQN3))

    # # 画图
    # # sns.lineplot(x='step', y="total_reward", data=mDQN1)
    # sns.lineplot(x='step', y="total_reward", data=df_worstDQN, label='full')
    # sns.lineplot(x='step', y="total_reward", data=df_mask, label='mask')
    # sns.lineplot(x='step', y="total_reward", data=df_bestDQN, label='minimal')
    pass

def draw_invalid_act_minatar():
    breakout_1_name = smooth(read_path='./data/breakout/', save_path="./data/breakout/", file_name='action_nums_1702091173.csv', y='act_num')
    breakout_2_name = smooth(read_path='./data/breakout/', save_path="./data/breakout/", file_name='action_nums_1702093529.csv', y='act_num')
    breakout_3_name = smooth(read_path='./data/breakout/', save_path="./data/breakout/", file_name='action_nums_1702094485.csv', y='act_num')

    asterix_1_name = smooth(read_path='./data/asterix/', save_path="./data/asterix/", file_name='action_nums_1702057838.csv', y='act_num')
    asterix_2_name = smooth(read_path='./data/asterix/', save_path="./data/asterix/", file_name='action_nums_1702057896.csv', y='act_num')
    asterix_3_name = smooth(read_path='./data/asterix/', save_path="./data/asterix/", file_name='action_nums_1702058084.csv', y='act_num')

    breakout_full_1_name = smooth(read_path='./data/breakout/', save_path="./data/breakout/", file_name='DQN_action_num_1702534604.csv', y='invalid_action_num')
    breakout_full_2_name = smooth(read_path='./data/breakout/', save_path="./data/breakout/", file_name='DQN_action_num_1702534802.csv', y='invalid_action_num')
    breakout_full_3_name = smooth(read_path='./data/breakout/', save_path="./data/breakout/", file_name='DQN_action_num_1702535781.csv', y='invalid_action_num')

    asterix_full_1_name = smooth(read_path='./data/asterix/', save_path="./data/asterix/", file_name='DQN_action_num_1702521987.csv', y='invalid_action_num')
    asterix_full_2_name = smooth(read_path='./data/asterix/', save_path="./data/asterix/", file_name='DQN_action_num_1702523126.csv', y='invalid_action_num')
    asterix_full_3_name = smooth(read_path='./data/asterix/', save_path="./data/asterix/", file_name='DQN_action_num_1702529091.csv', y='invalid_action_num')

    breakout_1 = pd.read_csv(breakout_1_name)
    breakout_2 = pd.read_csv(breakout_2_name)
    breakout_3 = pd.read_csv(breakout_3_name)

    asterix_1 = pd.read_csv(asterix_1_name)
    asterix_2 = pd.read_csv(asterix_2_name)
    asterix_3 = pd.read_csv(asterix_3_name)

    breakout_full_1 = pd.read_csv(breakout_full_1_name)
    breakout_full_2 = pd.read_csv(breakout_full_2_name)
    breakout_full_3 = pd.read_csv(breakout_full_3_name)

    asterix_full_1 = pd.read_csv(asterix_full_1_name)
    asterix_full_2 = pd.read_csv(asterix_full_2_name)
    asterix_full_3 = pd.read_csv(asterix_full_3_name)

    df_breakout = breakout_1._append(breakout_2._append(breakout_3))
    df_asterix = asterix_1._append(asterix_2._append(asterix_3))
    df_breakout_full = breakout_full_1._append(breakout_full_2._append(breakout_full_3))
    df_asterix_full = asterix_full_1._append(asterix_full_2._append(asterix_full_3))

    df_breakout['act_num'] /=15
    df_asterix['act_num'] /= 13

    ax1 = fig.add_subplot(111)
    plt.title('MinAtar')
    ax1.set_xlabel('episode')
    ax1.set_ylabel('invalid action num')

    sns.lineplot(x='step', y='act_num', data=df_breakout, label='Breakout ours', color='darkblue')
    sns.lineplot(x='step', y='act_num', data=df_asterix, label='Asterix ours', color='darkred')
    sns.lineplot(x='step', y='invalid_action_num', data=df_breakout_full, label='Breakout full', color='cornflowerblue')
    sns.lineplot(x='step', y='invalid_action_num', data=df_asterix_full, label='Asterix full', color='salmon')

    # 保存需要的图片
    plt.savefig('result_act_minatar', dpi=300)

def draw_invalid_act_RTS2():
    nomask_1_name = smooth(read_path='./data/RTS2/', save_path="./data/RTS2/", file_name='no_mask_num_invalid_action_1.csv', x='Step', y='Value', z='Wall time')
    nomask_2_name = smooth(read_path='./data/RTS2/', save_path="./data/RTS2/", file_name='no_mask_num_invalid_action_2.csv', x='Step', y='Value', z='Wall time')
    nomask_3_name = smooth(read_path='./data/RTS2/', save_path="./data/RTS2/", file_name='no_mask_num_invalid_action_3.csv', x='Step', y='Value', z='Wall time')

    automask_1_name = smooth(read_path='./data/RTS2/', save_path="./data/RTS2/", file_name='automask_num_invalid_action_1.csv', x='Step', y='Value', z='Wall time')
    automask_2_name = smooth(read_path='./data/RTS2/', save_path="./data/RTS2/", file_name='automask_num_invalid_action_2.csv', x='Step', y='Value', z='Wall time')
    automask_3_name = smooth(read_path='./data/RTS2/', save_path="./data/RTS2/", file_name='automask_num_invalid_action_3.csv', x='Step', y='Value', z='Wall time')

    nomask_1 = pd.read_csv(nomask_1_name)
    nomask_2 = pd.read_csv(nomask_2_name)
    nomask_3 = pd.read_csv(nomask_3_name)

    automask_1 = pd.read_csv(automask_1_name)
    automask_2 = pd.read_csv(automask_2_name)
    automask_3 = pd.read_csv(automask_3_name)

    df_nomask = nomask_1._append(nomask_2._append(nomask_3))
    df_automask = automask_1._append(automask_2._append(automask_3))

    df_nomask['Value'] /= 130
    df_automask['Value'] /=130

    ax1 = fig.add_subplot(111)
    plt.title('μRTS2')
    ax1.set_xlabel('episode')
    ax1.set_ylabel('invalid action num')

    sns.lineplot(x='Step', y='Value', data=df_nomask, label='no masks', color='cornflowerblue')
    sns.lineplot(x='Step', y='Value', data=df_automask, label='ours', color='darkblue')

    # 保存需要的图片
    plt.savefig('result_act_RTS2', dpi=300)

# draw_invalid_act_minatar()
draw_invalid_act_RTS2()
plt.show()
