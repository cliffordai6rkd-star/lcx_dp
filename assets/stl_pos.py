import trimesh
import numpy as np

# 1. 加载你的 STL 文件
file_name = 'assets/scene_objects/bread_picking/bread_picking_scene.STL'
mesh = trimesh.load(file_name)

# 2. 获取网格的包围盒几何中心 (原始单位，通常是毫米 mm)
# 你也可以用 mesh.centroid 获取物理质心
raw_center = mesh.bounding_box.centroid

# 3. 应用你的 MuJoCo 缩放比例 (scale="0.001 0.001 0.001")
scale = 0.001
scaled_center = raw_center * scale

# 4. 计算反向偏移值 (为了把中心移回 0,0,0，我们需要取反)
offset = -scaled_center

# 5. 输出结果
print(f"--- {file_name} 几何信息 ---")
print(f"原始网格中心 (mm) : {np.round(raw_center, 2)}")
print(f"缩放后中心 (m)    : {np.round(scaled_center, 4)}")
print("=" * 40)
print("✅ 请将以下属性直接复制到你的 MuJoCo XML 的 <asset> mesh 标签中:")
print(f'refpos="{offset[0]:.4f} {offset[1]:.4f} {offset[2]:.4f}"')
print("=" * 40)