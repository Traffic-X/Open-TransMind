# !/usr/bin/env python3
import json
import numpy as np
from collections import OrderedDict

from data.util.tokenizer import *
from modeling.backbones.vit_retrieval import CLIP
from fastreid.data.data_utils import read_image
from data.transforms.build import build_transforms_lazy


def build_model(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length,
                vocab_size, transformer_width, transformer_heads, transformer_layers, qkv_bias, pre_norm,
                proj, patch_bias):

    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        qkv_bias=qkv_bias,
        pre_norm=pre_norm,
        proj=proj,
        patch_bias=patch_bias
    )

    return model


def load_pretrained(model, pretrained):
    state_dict = paddle.load(pretrained)['model']
    new_state_dict = {}
    for k in state_dict:
        new_k = k[9:] # remove 'backbones.'
        new_state_dict[new_k] = state_dict[k]
    model.set_state_dict(new_state_dict)
    return model


def get_transforms():
    # Data Preprocessing
    from paddle.vision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    # from passl.datasets.preprocess.transforms import ToRGB

    preprocess = Compose([Resize([224,224]),
                        ToTensor(),
                        ])
    image_mean = paddle.to_tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
    image_std = paddle.to_tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])

    Transforms = build_transforms_lazy(
                        is_train=False,
                        size_test=[224, 224],
                        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                    )
    return Transforms


def get_images_texts(images_root, texts_root, transform_ops, test=False):
    images = []
    texts = []

    f = open(texts_root, 'r')
    for line in f.readlines():
        text = line.strip()
        texts.append(text)

    idx_dic = {}
    cnts = 0
    images_names = os.listdir(images_root)
    for image_name in images_names:
        img = read_image(os.path.join(images_root, image_name), "RGB")
        img = transform_ops(img)
        images.append(img)
        idx_dic[cnts] = image_name
        cnts += 1

    if test:
        images = images[:20]
        texts = texts[:20]

    return images, texts, idx_dic


def get_similarity(images, texts):

    tokenizer = SimpleTokenizer()
    n = len(images)
    stage = 1000
    image_features_list = []
    text_features_list = []
    for i in range(0, n, stage):
        print(f'{i}/{n}')
        start = i
        end = min(n, i + stage)
        image_input = paddle.to_tensor(images[start: end])

        text_tokens = [tokenizer.encode("" + desc) for desc in texts[start: end]]
        text_input = paddle.zeros((len(text_tokens), 77), dtype="int64")
        sot_token = tokenizer.encoder['<|startoftext|>']
        eot_token = tokenizer.encoder['<|endoftext|>']
        for i, tokens in enumerate(text_tokens):
            tokens = [sot_token] + tokens + [eot_token]
            text_input[i, :len(tokens)] = paddle.to_tensor(tokens)

        with paddle.no_grad():
            import numpy as np
            image_input = paddle.to_tensor(image_input)
            text_input = paddle.to_tensor(text_input)
            image_features = model.encode_image(image_input)
            image_features_list.append(image_features)
            text_features = model.encode_text(text_input)
            text_features_list.append(text_features)
 
    image_features = paddle.concat(image_features_list, axis=0)
    text_features = paddle.concat(text_features_list, axis=0)

    image_features /= image_features.norm(axis=-1, keepdim=True)
    text_features /= text_features.norm(axis=-1, keepdim=True)
    print('image_features.shape', image_features.shape)
    print('text_features.shape', text_features.shape)
    similarity = paddle.matmul(text_features, image_features.t()).cpu().numpy()
    return similarity


def infer(similarity, idx_dic, texts):
    similarity_argsort = np.argsort(-similarity, axis=1)
    print('similarity_argsort.shape', similarity_argsort.shape)

    topk = 10
    result_list = []
    for i in range (len(similarity_argsort)):
        dic = {'text': texts[i], 'image_names': []}
        for j in range(topk):
            dic['image_names'].append(idx_dic[similarity_argsort[i,j]])
        result_list.append(dic)
    with open('infer_json.json', 'w') as f:
        f.write(json.dumps({'results': result_list}, indent=4))
        


if __name__ == '__main__':
    # pretrained = './pretrained/vitbase_clip.pdparams'
    pretrained = "./outputs/vitbase_retrieval/model_final.pdmodel"
    images_root = './datasets/test/test_images/'
    texts_root = './datasets/test/test_text.txt' 


    model = build_model(embed_dim=512,
                        image_resolution=224,
                        vision_layers=12,
                        vision_width=768,
                        vision_patch_size=32,
                        context_length=77,
                        vocab_size=49408,
                        transformer_width=512,
                        transformer_heads=8,
                        transformer_layers=12,
                        qkv_bias=True,
                        pre_norm=True,
                        proj=True,
                        patch_bias=False)

    model = load_pretrained(model, pretrained)

    transform_ops = get_transforms()

    images, texts, idx_dic = get_images_texts(images_root, texts_root, transform_ops, test=False)

    similarity = get_similarity(images, texts)

    infer(similarity, idx_dic, texts)