# idol3D 兼容性分支

本分支基于上游 commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`，服务于 idol3D 的本地深度/相机推理。代码来源仍归原作者；本分支不是官方 release，也不改变模型权重许可证。

## 依赖和行为变化

- 支持声明调整为 Python >=3.12，直接依赖升级到 2026-11-12 核查的最新正式版本下限，允许 NumPy 2；最新解释器仍须独立验证。
- 基础安装仅包含深度/相机所需依赖；export、colmap、video、gs、cli、app、bench、xformers、dev 按 extras 选择。纯深度调用不加载 COLMAP、GS 或视频模块。
- 移除 MoviePy。最新版 MoviePy 2.2.1 仍要求 Pillow <12，与 Pillow 13 冲突；原视频功能只需帧编码，改用现有 ImageIO + FFmpeg，保持 H.264/帧率/CRF，并保证异常时关闭编码器。
- GS 依赖使用已发布 gsplat，而非浮动或固定 Git URL；未运行 CUDA 原生扩展验证前不宣称 GS 功能兼容。
- 复合导出格式在输出前完整验证；未安装可选依赖时提示相应 extra。批次视频避免覆盖同名输出。
- requirements.txt 引用本项目元数据，避免第二份不同步的包列表。补充实际使用但未声明的 addict、tqdm。
- 移除核心对 evo 的依赖：其配置模块硬编码用户主目录 `.evo`。相机中心的 Sim(3) 最小二乘配准改用 NumPy SVD，保留正旋转/尺度/平移和相机方向变换，退化输入明确报错，并用已知变换测试。修复 NumPy 2 已移除的 `np.math`。

## 安装与测试

从此 fork 根目录执行：

```powershell
python -m pip install -e .
python -m pip install -e ".[video,dev]"
python -m pytest tests
```

CLI 用户需要 `.[cli]`，Gradio 需要 `.[app]`，不再保证只安装基础包后就能运行所有 CLI/演示功能。纯深度使用 `from depth_anything_3.api import DepthAnything3`。

调用方应在首次 import 前将缓存、临时目录和配置路径指向应用便携目录。此分支不自动为用户下载模型；测试视频为合成小图。测试通过不能替代真实模型、CUDA、相机质量和全部可选组件验证。精确验证结果保存在父项目的开发测试记录。
