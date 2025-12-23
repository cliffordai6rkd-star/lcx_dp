      
import numpy as np
import torch
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import mplcursors, os, random
from dataset.lerobot.reader import RerunEpisodeReader, ActionType, ObservationType

REPO_DIR = "/mnt/nas/zyx_checkpoints/dino/repo/dinov3"

def get_dinov3_model(model_type):
    name = None; 
    match model_type.lower():
        case "s+":
            name = "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
            model_type = "dinov3_vits16plus"
        case "b":
            name = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            model_type = "dinov3_vitb16"
        case "small":
            name = "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"
            model_type = "dinov3_convnext_small"
        case "base":
            name = "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
            model_type = "dinov3_convnext_base"
        case _:
            raise ValueError(f"Could not find the related checkpoint for the model {model_type}")
    model_path = os.path.join(REPO_DIR, '../..', name)
    # model_path = name
    print(f'Build model from {model_path} for {model_type}')
    # processor = AutoImageProcessor.from_pretrained(model_path)
    # model = AutoModel.from_pretrained(model_path, device_map="auto")
    model = torch.hub.load(REPO_DIR, model_type, source='local', weights=model_path)
    model = model.eval()    
    model = model.to("cuda")
    return model

def extract_dino_features(batch_imgs, model):
    """提取DINO特征"""
    # inputs = processor(images=video_frames, return_tensors="pt").to(model.device)
    # resize
    inputs = torch.tensor(batch_imgs, dtype=torch.float32).to("cuda")
    resize = Resize((128, 160), antialias=True)
    x = resize(inputs)
    print(f'resize shape: {x.shape}')
   
    with torch.inference_mode():
        # with torch.autocast('cuda', dtype=torch.bfloat16):
        feats = model.get_intermediate_layers(x, n=range(12), reshape=True, norm=True)
        # print(f'feats shape: {type(feats)} {len(feats)}')
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        x = x.view(dim, -1)
        
        # feats = model(**inputs)

        # feats = feats.last_hidden_state[:,0, :].cpu().numpy()
        # feats = feats.pooler_output.cpu().numpy()
        return x

def get_traj_pics(data_path, data_type, cam_key, skip_num, episode_dir, key_transform=None):
    start = 0; end = 0
    reader = RerunEpisodeReader(data_path, action_type=ActionType.END_EFFECTOR_POSE,
                observation_type=ObservationType.MASK, camera_keys=[cam_key], data_type=data_type,
                camera_keys_transformation=key_transform)
    
    if 'episode' in episode_dir:
        episode_number = int(episode_dir.lstrip("episode_"))
        episode_id = episode_number
        cur_data = reader.return_episode_data(episode_id, skip_num)
        if cur_data is None:
            return None, None
        imgs = []
        for step_data in cur_data:
            new_key = cam_key
            if key_transform is not None and cam_key in key_transform:
                new_key = key_transform[cam_key]
            cur_img = step_data["colors"][new_key]
            imgs.append(cur_img)
        end = len(imgs)
        imgs = np.stack(imgs, axis=0)
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        return imgs, range(start, end)

def tsne_compare_videos(features_list, indices_list, save_path="domain_traj_tsne.png", labels=["Video 1", "Video 2",]):

    # 合并特征并降维
    all_feats = np.vstack(features_list)
    tsne_2d = TSNE(n_components=2, random_state=42).fit_transform(all_feats)

    assert len(labels) == len(features_list)
    
    # 绘图配置
    plt.figure(figsize=(12, 10))
    markers = ['o', '^', '8']  # 视频标记区分
    cmaps = ['Reds', 'Blues','cool']  # 视频颜色区分
    scatters = []
    
    # 逐个绘制视频散点
    start = 0
    for i, (feats, indices) in enumerate(zip(features_list, indices_list)):
        end = start + len(feats)
        scatter = plt.scatter(tsne_2d[start:end, 0], tsne_2d[start:end, 1],
                             c=range(len(feats)), cmap=cmaps[i], marker=markers[i],
                             label=labels[i], s=30, alpha=0.8)
        scatters.append(scatter)
        start = end
        plt.colorbar(scatter, fraction=0.02, pad=0.04).set_label(f'{labels[i]} Frame Index')
    
    cursor = mplcursors.cursor(scatters, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        vid_idx = scatters.index(sel.artist)  # 视频索引
        frame_idx = indices_list[vid_idx][sel.index]  # 原始帧序号
        sel.annotation.set_text(f"{labels[vid_idx]}\nFrame: {frame_idx}")
    
    # 保存配置
    plt.title("t-SNE Comparison of different domain for same task traj", fontsize=14)
    plt.xlabel('t-SNE 1'), plt.ylabel('t-SNE 2'), plt.legend(), plt.grid(alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight', dpi=300), plt.close()
    print(f"对比图已保存至: {save_path}")

def get_pair_image_N_indices_list(data_paths, episode_dirs, data_type, human_key, robot_key, cam_key_human2robot_transform):
    imgs = []; indices_list = []
    for i, path in enumerate(data_paths):
        episode_dir = episode_dirs[i]
        cam_key = human_key if "human" in data_type[i] else robot_key
        key_transform = None if "robot" in data_type[i] else cam_key_human2robot_transform
        img, indices = get_traj_pics(path, data_type[i], cam_key, skip_num[i], episode_dir, key_transform)
        if img is None:
            break
        imgs.append(img); indices_list.append(indices)
    return imgs, indices_list

if __name__ == "__main__":
    # 配置参数（支持扩展到更多视频）
    data_paths = ["dataset/data/1212_duo_unitree_bread_n_picking——214ep", "dataset/data/1221_duo_unitree_bread_picking_human_114ep"]
    episode_pair = None # or [human_episode, robot_episode]
    episode_pair = ["episode_0012", "episode_0020"]
    skip_num = [1, 3]
    data_type = ["human", "robot"]
    # human_key = "head_color"; robot_key = "head_color"
    human_key = "right_hand_fisheye_color"; robot_key = "right_fisheye_color"
    common_key = robot_key
    cam_key_human2robot_transform = {
        "left_hand_fisheye_color": "left_fisheye_color", "right_hand_fisheye_color": "right_fisheye_color"
    }
    save_path = "dino_tsne_humanvsrobot_" + common_key + ".png"
    start_idx, end_idx = None, None  # 统一帧区间（可按需修改）
    
    model = get_dinov3_model("b")
    data_ok = False; imgs = []; indices_list = []; episode_dirs = []
    if episode_pair is None:
        while not data_ok:
            imgs = []; indices_list = []; episode_dirs = []
            # 随机各抽取一个episode
            for path in data_paths:
                dirs = os.listdir(path)
                id = random.randint(0, len(dirs)-1)
                episode_dirs.append(dirs[id])
            imgs, indices_list = get_pair_image_N_indices_list(data_paths, 
                episode_dirs, data_type, human_key, robot_key, cam_key_human2robot_transform)
            if not len(imgs) == len(data_paths): data_ok = False
            else: data_ok = True
    else:
        episode_dirs = episode_pair
        imgs, indices_list = get_pair_image_N_indices_list(data_paths, 
                episode_pair, data_type, human_key, robot_key, cam_key_human2robot_transform)
    print(f'load data from data path for {episode_dirs} {len(imgs)} {imgs[0].shape} {indices_list}')
        
    # 批量处理
    features_list = []
    for i, batch_img in enumerate(imgs, 0):
        feats = extract_dino_features(batch_img, model)
        print(feats.shape)
        features_list.append(feats)
        print(f"data {data_paths[i]}/{episode_dirs[i]}: {len(batch_img)}帧，特征形状{feats.shape}")
    
    # 对比可视化
    tsne_compare_videos(features_list, indices_list, save_path, labels=data_type)

    