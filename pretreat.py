import cv2
import numpy as np
import os
import glob
import random
import sys
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


# 修复字体警告问题
def fix_matplotlib_fonts():
    """解决中文字符渲染问题"""
    try:
        # 尝试设置支持中文的字体
        font_list = fm.findSystemFonts()
        chinese_fonts = [
            'SimHei', 'Microsoft YaHei', 'STHeiti', 'STKaiti',
            'KaiTi', 'SimSun', 'FangSong', 'STSong'
        ]

        # 查找系统中可用的中文字体
        available_fonts = []
        for font in font_list:
            try:
                prop = fm.FontProperties(fname=font)
                name = prop.get_name()
                if any(cn_font in name for cn_font in chinese_fonts):
                    available_fonts.append(name)
            except:
                pass

        # 设置字体
        if available_fonts:
            plt.rcParams['font.sans-serif'] = available_fonts
            plt.rcParams['axes.unicode_minus'] = False
            logging.info(f"设置中文字体: {', '.join(available_fonts[:3])}")
        else:
            # 回退到默认字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            logging.warning("未找到中文字体，使用默认字体")
    except Exception as e:
        logging.error(f"设置字体失败: {str(e)}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']


# 修复字体问题
fix_matplotlib_fonts()


class DataAugmentor:
    def __init__(self, input_dir, output_dir, target_size=(224, 224)):
        """
        初始化数据增强器

        参数:
        input_dir: 原始图像目录
        output_dir: 增强后图像保存目录
        target_size: 输出图像尺寸 (宽, 高)
        """
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.target_size = target_size

        # 打印路径信息以便调试
        logging.info(f"输入目录: {self.input_dir}")
        logging.info(f"输出目录: {self.output_dir}")

        # 检查输入目录是否存在
        if not os.path.exists(self.input_dir):
            logging.error(f"错误: 输入目录不存在 - {self.input_dir}")
            sys.exit(1)

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"创建输出目录: {self.output_dir}")

        # 预处理配置
        self.gamma_value = 1.2
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = (8, 8)

        # 增强配置
        self.rotation_range = (-15, 15)
        self.crop_range = (0.85, 0.95)
        self.noise_sigma_range = (0.01, 0.05)
        self.hsv_saturation_range = (0.8, 1.2)
        self.hsv_value_range = (0.8, 1.2)
        self.blur_kernels = [3, 5]

    def load_images(self):
        """加载目录中的所有图像并返回详细统计信息"""
        logging.info("\n扫描图像文件...")
        logging.info(f"支持的格式: {', '.join(['*' + ext for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])}")

        # 获取所有可能的图像文件
        all_files = glob.glob(os.path.join(self.input_dir, '*'))
        if not all_files:
            logging.warning(f"警告: 输入目录 '{self.input_dir}' 中没有找到任何文件")

        logging.info(f"目录中找到 {len(all_files)} 个文件")

        images = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        skipped_files = []
        image_paths = []

        for file_path in all_files:
            # 检查是否是文件
            if not os.path.isfile(file_path):
                skipped_files.append((file_path, "目录/非文件"))
                continue

            # 检查文件扩展名
            filename = os.path.basename(file_path)
            ext = os.path.splitext(file_path)[1].lower()

            if ext in valid_extensions:
                # 尝试读取图像
                try:
                    img = cv2.imread(file_path)
                    if img is not None and img.size > 0:
                        images.append(img)
                        image_paths.append(file_path)
                        logging.debug(f"成功加载: {filename} (尺寸: {img.shape})")
                    else:
                        skipped_files.append((file_path, "OpenCV无法读取 - 可能损坏或不支持的格式"))
                except Exception as e:
                    skipped_files.append((file_path, f"读取错误: {str(e)}"))
            else:
                skipped_files.append((file_path, f"不支持的格式: {ext}"))

        # 打印加载统计
        logging.info(f"成功加载 {len(images)} 张图像")
        logging.info(f"跳过 {len(skipped_files)} 个文件:")

        if skipped_files:
            logging.info("\n跳过文件详情:")
            for file_path, reason in skipped_files:
                filename = os.path.basename(file_path)
                logging.info(f"- {filename}: {reason}")

        return images, image_paths

    def apply_clahe(self, img):
        """应用自适应直方图均衡化"""
        try:
            # 转换为LAB颜色空间
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # 创建CLAHE对象并应用于亮度通道
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size
            )
            cl = clahe.apply(l)

            # 合并通道并转换回BGR
            enhanced_lab = cv2.merge((cl, a, b))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logging.error(f"CLAHE处理失败: {str(e)}")
            return img

    def apply_gamma_correction(self, img):
        """应用伽马校正"""
        try:
            # 归一化图像
            img_normalized = img.astype(np.float32) / 255.0
            # 应用伽马校正
            gamma_corrected = np.power(img_normalized, 1.0 / self.gamma_value)
            # 转换回0-255范围
            return (gamma_corrected * 255).astype(np.uint8)
        except Exception as e:
            logging.error(f"伽马校正失败: {str(e)}")
            return img

    def augment_image(self, img):
        """
        对单张图像应用增强技术

        参数:
        img: 输入图像 (BGR格式)

        返回:
        增强后的图像
        """
        # 基础几何变换
        if random.random() > 0.5:
            img = cv2.flip(img, 1)  # 水平翻转

        # 随机旋转
        angle = random.uniform(*self.rotation_range)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 随机裁剪
        crop_factor = random.uniform(*self.crop_range)
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        y = random.randint(0, h - new_h)
        x = random.randint(0, w - new_w)
        img = img[y:y + new_h, x:x + new_w]

        # 调整到目标尺寸
        img = cv2.resize(img, self.target_size)

        # 像素级变换
        if random.random() > 0.3:
            sigma = random.uniform(*self.noise_sigma_range)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img + noise * 255, 0, 255).astype(np.uint8)

        # HSV空间增强
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= random.uniform(*self.hsv_saturation_range)  # 饱和度
            hsv[..., 2] *= random.uniform(*self.hsv_value_range)  # 明度
            hsv = np.clip(hsv, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        except Exception as e:
            logging.error(f"HSV转换失败: {str(e)}")

        # 高斯模糊
        if random.random() > 0.4:
            ksize = random.choice(self.blur_kernels)
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # 直方图均衡化
        if random.random() > 0.5:
            try:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_eq = cv2.equalizeHist(l)
                enhanced_lab = cv2.merge((l_eq, a, b))
                img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            except Exception as e:
                logging.error(f"直方图均衡化失败: {str(e)}")

        return img

    def process_and_augment(self, num_augmentations=10, preprocess_methods=None):
        """
        处理整个数据集并生成增强图像

        参数:
        num_augmentations: 每张原始图像生成的增强图像数量
        preprocess_methods: 应用的预处理方法列表 (None表示使用['clahe', 'gamma'])
        """
        if preprocess_methods is None:
            preprocess_methods = ['clahe', 'gamma']

        images, image_paths = self.load_images()

        if not images:
            logging.error("没有可处理的图像 - 退出")
            return

        total_images = len(images)
        augmented_count = 0

        logging.info(f"\n开始处理 {total_images} 张图像...")
        logging.info(f"每张图像生成 {num_augmentations} 个增强版本")
        logging.info(f"预处理方法: {', '.join(preprocess_methods)}")

        for idx, (img, img_path) in enumerate(zip(images, image_paths)):
            filename = os.path.basename(img_path)
            logging.info(f"处理图像 {idx + 1}/{total_images}: {filename}")

            # 应用预处理
            processed_img = img.copy()

            # 确保图像有内容
            if processed_img is None or processed_img.size == 0:
                logging.error(f"图像 {filename} 为空，跳过处理")
                continue

            # 应用预处理方法
            for method in preprocess_methods:
                try:
                    if method == 'clahe':
                        processed_img = self.apply_clahe(processed_img)
                    elif method == 'gamma':
                        processed_img = self.apply_gamma_correction(processed_img)
                except Exception as e:
                    logging.error(f"预处理 '{method}' 失败: {str(e)}")

            # 保存原始预处理后的图像
            orig_filename = f"orig_{idx:04d}_{filename}"
            orig_path = os.path.join(self.output_dir, orig_filename)

            # 检查图像数据是否有效
            if processed_img is None or processed_img.size == 0:
                logging.error(f"处理后的图像 {orig_filename} 为空，跳过保存")
            else:
                # 确保图像是有效的
                if processed_img.dtype != np.uint8:
                    processed_img = processed_img.astype(np.uint8)

                # 保存图像
                success = cv2.imwrite(orig_path, processed_img)
                if success:
                    logging.debug(f"保存预处理图像: {orig_filename}")
                else:
                    logging.error(f"保存失败: {orig_path}")

            # 生成增强图像
            for aug_idx in range(num_augmentations):
                # 从原始图像开始增强
                if img is None or img.size == 0:
                    logging.error(f"原始图像 {filename} 为空，跳过增强")
                    continue

                # 应用增强
                augmented_img = self.augment_image(processed_img.copy())

                # 检查增强后的图像是否有效
                if augmented_img is None or augmented_img.size == 0:
                    logging.error(f"增强后的图像为空，跳过保存")
                    continue

                # 确保图像是有效的
                if augmented_img.dtype != np.uint8:
                    augmented_img = augmented_img.astype(np.uint8)

                # 保存增强图像
                aug_filename = f"aug_{idx:04d}_{aug_idx:02d}_{filename}"
                aug_path = os.path.join(self.output_dir, aug_filename)
                success = cv2.imwrite(aug_path, augmented_img)

                if success:
                    augmented_count += 1
                    logging.debug(f"保存增强图像: {aug_filename}")
                else:
                    logging.error(f"保存失败: {aug_path}")

        logging.info(f"\n处理完成! 共生成 {augmented_count} 张增强图像")
        logging.info(f"总计图像: {total_images + augmented_count} (原始: {total_images}, 增强: {augmented_count})")
        logging.info(f"输出目录: {self.output_dir}")

        # 可视化结果
        self.visualize_results()

    def visualize_results(self, num_samples=3):
        """可视化处理前后的图像对比"""
        logging.info("\n尝试可视化结果...")
        image_paths = glob.glob(os.path.join(self.output_dir, '*.*'))
        if not image_paths:
            logging.warning("没有找到输出图像")
            return

        # 使用支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(15, 10))

        # 获取原始图像和对应的增强图像
        orig_images = sorted([p for p in image_paths if 'orig_' in os.path.basename(p)])
        if not orig_images:
            logging.warning("没有找到原始图像进行可视化")
            return

        num_samples = min(num_samples, len(orig_images))

        for i in range(num_samples):
            orig_path = orig_images[i]
            orig_img = cv2.imread(orig_path)
            if orig_img is None:
                logging.warning(f"无法读取图像: {orig_path}")
                continue

            # 确保图像是RGB格式
            if len(orig_img.shape) == 2:  # 灰度图像
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
            else:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            # 显示原始图像
            plt.subplot(num_samples, 4, i * 4 + 1)
            plt.imshow(orig_img)
            plt.title("原始图像" if i == 0 else "")
            plt.axis('off')

            # 显示3个增强版本
            base_name = os.path.basename(orig_path).split('_', 2)[-1]
            aug_paths = [p for p in image_paths if
                         f"_{base_name}" in os.path.basename(p) and 'aug_' in os.path.basename(p)]

            for j in range(3):
                ax = plt.subplot(num_samples, 4, i * 4 + j + 2)

                if j < len(aug_paths):
                    aug_img = cv2.imread(aug_paths[j])
                    if aug_img is None:
                        logging.warning(f"无法读取增强图像: {aug_paths[j]}")
                        plt.text(0.5, 0.5, "读取失败", ha='center', va='center')
                    else:
                        # 确保图像是RGB格式
                        if len(aug_img.shape) == 2:  # 灰度图像
                            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_GRAY2RGB)
                        else:
                            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)

                        plt.imshow(aug_img)
                        plt.title("增强图像" if i == 0 and j == 0 else "")
                else:
                    # 如果增强图像不足3个，留空
                    plt.text(0.5, 0.5, "无图像", ha='center', va='center')

                plt.axis('off')

        plt.tight_layout()
        vis_path = os.path.join(self.output_dir, 'visualization.png')
        plt.savefig(vis_path, dpi=100, bbox_inches='tight')
        logging.info(f"保存可视化结果到: {vis_path}")
        plt.close()  # 关闭图形以释放内存


def create_sample_images(input_dir, num_samples=3):
    """创建示例图像用于测试"""
    logging.info(f"\n创建示例输入目录: {input_dir}")
    os.makedirs(input_dir, exist_ok=True)

    # 创建示例图像
    for i in range(num_samples):
        # 创建纯色图像
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        color = [
            (255, 0, 0),  # 红色
            (0, 255, 0),  # 绿色
            (0, 0, 255)  # 蓝色
        ][i % 3]
        img[:, :] = color

        # 添加文本
        text = f"Sample {i + 1}"
        cv2.putText(img, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        # 保存图像
        img_path = os.path.join(input_dir, f"sample_{i + 1}.jpg")
        success = cv2.imwrite(img_path, img)
        if success:
            logging.info(f"创建示例图像: {img_path}")
        else:
            logging.error(f"创建示例图像失败: {img_path}")

    return num_samples


def main():
    # 获取当前脚本所在目录
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 配置输入输出目录
    INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "augmented_output")

    # 如果输入目录不存在，创建示例目录和图像
    if not os.path.exists(INPUT_DIR) or not os.listdir(INPUT_DIR):
        num_created = create_sample_images(INPUT_DIR)
        logging.info(f"创建了 {num_created} 个示例图像在 {INPUT_DIR}")
    else:
        logging.info(f"使用现有输入目录: {INPUT_DIR}")

    # 创建增强器
    augmentor = DataAugmentor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_size=(224, 224)
    )

    # 执行数据预处理和增强
    augmentor.process_and_augment(
        num_augmentations=5,  # 每张原始图像生成5个增强版本
        preprocess_methods=['clahe', 'gamma']  # 使用CLAHE和伽马校正
    )


if __name__ == "__main__":
    main()