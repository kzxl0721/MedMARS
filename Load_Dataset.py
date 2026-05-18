# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
import clip  # 替换 bert_embedding 为 clip
import re
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Set, Optional

def load_clip_model(rank, clip_path):
    device = torch.device(f"cuda:{rank}")
    clip_model, _ = clip.load(clip_path, device=device, jit=False)
    return clip_model, device


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text, keyword_positions = sample['image'], sample['label'], sample['text'], sample['keyword_positions']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        keyword_positions = torch.Tensor(keyword_positions)
        sample = {'image': image, 'label': label, 'text': text, 'keyword_positions': keyword_positions}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text, keyword_positions = sample['image'], sample['label'], sample['text'], sample['keyword_positions']
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        keyword_positions = torch.Tensor(keyword_positions)
        sample = {'image': image, 'label': label, 'text': text, 'keyword_positions': keyword_positions}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

import nltk
import re
from collections import defaultdict
from typing import List, Tuple, Set


class AutoKeywordExtractor:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("正在下载必要的NLTK数据...")
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)

        # 定义目标词性标签
        self.target_pos_tags = {
            'spatial': ['JJ', 'JJR', 'JJS'],  # 形容词（包括比较级和最高级）
            'quantity': ['CD', 'DT'],  # 基数词和限定词
            'directional': ['JJ', 'NN', 'NNS']  # 方向性名词和形容词
        }

        # 预定义一些常见的医学/解剖学关键词模式
        self.medical_patterns = {
            'directional': r'\b(left|right|upper|lower|middle|anterior|posterior|lateral|medial|proximal|distal|small|moderate|visible|center|occupying|medium-sized|center-right|center-left)\b',
            'quantity': r'\b(one|two|three|four|five|six|seven|eight|nine|ten|zero|no|single|double|multiple|bilateral|unilateral|well-defined|lobulated|round|oval|flat|top|bottom)\b',
            'distribution': r'\b(all|entire|partial|distributed|scattered|diffuse|focal|localized|widespread|irregularly|large|irregular|highlighted|prominent|well-defined|blurred)\b'
        }

    def extract_keywords_by_pos(self, text: str) -> Set[str]:
        """
        基于词性标注提取关键词
        """
        # 预处理文本
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        tokens = nltk.word_tokenize(text_clean)

        # 词性标注
        pos_tags = nltk.pos_tag(tokens)

        keywords = set()

        for word, pos in pos_tags:
            # 检查是否为目标词性
            if (pos in self.target_pos_tags['spatial'] or
                    pos in self.target_pos_tags['quantity'] or
                    pos in self.target_pos_tags['directional']):

                # 进一步筛选：检查是否符合医学相关模式
                if self._is_medical_keyword(word):
                    keywords.add(word)

        return keywords

    def extract_keywords_by_pattern(self, text: str) -> Set[str]:
        """
        基于正则表达式模式提取关键词
        """
        keywords = set()
        text_lower = text.lower()

        for category, pattern in self.medical_patterns.items():
            matches = re.findall(pattern, text_lower)
            keywords.update(matches)

        return keywords

    def _is_medical_keyword(self, word: str) -> bool:
        """
        判断单词是否为医学相关关键词
        """
        # 检查是否匹配任何医学模式
        for pattern in self.medical_patterns.values():
            if re.search(pattern, word, re.IGNORECASE):
                return True
        return False

    def extract_contextual_keywords(self, corpus: List[str], min_frequency: int = 2) -> Set[str]:
        """
        基于语料库上下文提取关键词
        """
        # 统计词频和上下文
        word_contexts = defaultdict(list)
        word_freq = defaultdict(int)

        for text in corpus:
            text_clean = re.sub(r'[^\w\s]', '', text.lower())
            tokens = nltk.word_tokenize(text_clean)
            pos_tags = nltk.pos_tag(tokens)

            for i, (word, pos) in enumerate(pos_tags):
                word_freq[word] += 1

                # 获取上下文（前后各2个词）
                context = []
                for j in range(max(0, i - 2), min(len(tokens), i + 3)):
                    if j != i:
                        context.append(tokens[j])
                word_contexts[word].extend(context)

        # 基于频率和上下文筛选关键词
        keywords = set()
        medical_context_words = {'lung', 'pulmonary', 'infection', 'area', 'areas', 'infected', 'lesion'}

        for word, freq in word_freq.items():
            if freq >= min_frequency:
                # 检查是否在医学上下文中出现
                context_words = set(word_contexts[word])
                if context_words.intersection(medical_context_words):
                    if self._is_medical_keyword(word):
                        keywords.add(word)

        return keywords

    def extract_keywords_mask(self, text: str, corpus: List[str] = None, max_tokens: int = 77) -> Tuple[
        List[str], List[int]]:
        """
        从文本中提取关键词位置掩码

        Args:
            text: 输入文本
            corpus: 语料库（用于上下文分析）
            max_tokens: 最大token数量限制

        Returns:
            tokens: 分词后的token列表
            keyword_positions: 关键词位置掩码 (1=关键词, 0=非关键词)
        """
        # 方法1：基于词性标注提取
        keywords_pos = self.extract_keywords_by_pos(text)

        # 方法2：基于模式匹配提取
        keywords_pattern = self.extract_keywords_by_pattern(text)

        # 方法3：如果提供了语料库，基于上下文提取
        keywords_context = set()
        if corpus:
            keywords_context = self.extract_contextual_keywords(corpus)

        # 合并所有方法提取的关键词
        all_keywords = keywords_pos.union(keywords_pattern).union(keywords_context)

        # 文本预处理和分词
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text_clean.split()

        # 限制token数量
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        # 生成关键词位置掩码
        keyword_positions = []
        for token in tokens:
            if token in all_keywords:
                keyword_positions.append(1)
            else:
                keyword_positions.append(0)
        return tokens, keyword_positions

    def analyze_keywords(self, text: str, corpus: List[str] = None) -> dict:
        """
        分析提取的关键词，返回详细信息
        """
        keywords_pos = self.extract_keywords_by_pos(text)
        keywords_pattern = self.extract_keywords_by_pattern(text)
        keywords_context = set()

        if corpus:
            keywords_context = self.extract_contextual_keywords(corpus)

        return {
            'pos_based': list(keywords_pos),
            'pattern_based': list(keywords_pattern),
            'context_based': list(keywords_context),
            'all_keywords': list(keywords_pos.union(keywords_pattern).union(keywords_context))
        }


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224,
                 clip_model_name: str = None) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.extractor = AutoKeywordExtractor()
        
        # 初始化CLIP模型和预处理器
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_path = "./nets/ViT-B-32.pt"
        self.clip_model, _ = clip.load(self.clip_path, device=self.device, jit=False)
        self.clip_model.eval()  # 设为评估模式

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
            
    def get_token_level_features(self, text: str):
        """
        获取token级别的CLIP文本特征 - 仅使用embedding层
        返回: (token_features, attention_mask)
        """
        # 对文本进行tokenization
        text_tokens = clip.tokenize([text]).to(self.device)  # [1, 77]
        
        with torch.no_grad():
            # 仅获取token embeddings + 位置编码
            x = self.clip_model.token_embedding(text_tokens).type(self.clip_model.dtype)  # [1, 77, 512]
            x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            
            # 创建attention mask
            attention_mask = (text_tokens != 0).float()  # [1, 77]

            
            return x.cpu().numpy().squeeze(), attention_mask.cpu().numpy().squeeze()

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]  
        mask_filename = image_filename[: -3] + "png"  
        
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        
        # 处理文本和关键词掩码
        text = self.rowtext[mask_filename]
        text_lines = text.split('\n')
        
        # 将文本行合并
        combined_text = ' '.join(text_lines)
        
        # 提取关键词位置掩码
        tokens, keyword_positions = self.extractor.extract_keywords_mask(combined_text, max_tokens=77)
        
        # 重新构建文本用于CLIP编码
        clip_text = ' '.join(tokens)
        if len(clip_text) > 200:
            clip_text = clip_text[:200]
            
        # 使用修改后的方法获取token级别的特征 
        token_features, attention_mask = self.get_token_level_features(clip_text) # token_features.shape, attention_mask.shape (77, 512) (77,)
        text = token_features  # 现在是token级别的特征
        
        # 确保keyword_positions长度与77一致
        if len(keyword_positions) < 77:
            keyword_positions.extend([0] * (77 - len(keyword_positions)))
        elif len(keyword_positions) > 77:
            keyword_positions = keyword_positions[:77]

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {
            'image': image, 
            'label': mask, 
            'text': text,  # 现在是 [77, 512] 的token级别特征
            'keyword_positions': keyword_positions,
            'attention_mask': attention_mask  # 添加attention mask
        }

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename
