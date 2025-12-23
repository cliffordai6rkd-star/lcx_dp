import os 
import shutil

def delete_episodes(episodes, dir_path):
    """
    删除指定目录下的episode目录
    Args:
        episodes: 要删除的episode编号列表
        dir_path: 目录路径
    """
    temp_trash_dir = dir_path + "_trash"
    os.makedirs(temp_trash_dir, exist_ok=True)
    for episode in episodes:
        # 构建目录名
        dirname = f"episode_{str(episode).zfill(4)}"  
        episode_path = os.path.join(dir_path, dirname)
        
        # 检查是否存在并删除
        if os.path.exists(episode_path):
            # @TODO: move to the temp trash dir
            pass
            # if os.path.isdir(episode_path):
            #     shutil.rmtree(episode_path)  # 删除整个目录
            #     print(f"已删除目录: {episode_path}")
            # else:
            #     os.remove(episode_path)  # 如果是文件则用remove
            #     print(f"已删除文件: {episode_path}")
        else:
            print(f"路径不存在: {episode_path}")

def cor_rate(invalid, directory):
    invalid_count = len(invalid)
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return 0
    count = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            count += 1
    if count == 0:
        print(f"目录中没有子目录: {directory}")
        return 0
    valid = count - invalid_count
    cor_rate = valid / count
    print(f"总数据量: {count}, 无效数据量: {invalid_count}, 正确率: {cor_rate:.2%}")
    return cor_rate

if __name__ == "__main__":
    
    episodes_to_delete = []
    dir = "/workspace/dataset/data/bread_picking"
    
    delete_episodes(episodes_to_delete, dir)
    result = cor_rate(episodes_to_delete, dir)  
    print(f"最终正确率: {result:.2%}")