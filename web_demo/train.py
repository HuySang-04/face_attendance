import os
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import DATA_PATH, LIST_EMBEDDING, LIST_NAME
from keras_facenet import FaceNet

model = FaceNet()

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    brightness_range=[0.5, 2.5],
    zoom_range=0.1,
    rotation_range=10
)

def augmentation_image(image, num=2):
    image = np.expand_dims(image, axis=0)
    aug_images = [image[0]]
    aug_image = data_gen.flow(image, batch_size=1)
    for _ in range(num):
        aug_images.append(next(aug_image)[0].astype(np.uint8))
    return aug_images


def compute_embeddings_for_user(user_name):
    user_path = os.path.join(DATA_PATH, user_name)
    embeds = []

    if not os.path.exists(user_path):
        print(f"Không tìm thấy thư mục {user_path}")
        return None

    for file in os.listdir(user_path):
        try:
            image = Image.open(f"{user_path}/{file}").convert("RGB")
        except:
            print(f"Không thể mở ảnh: {file}")
            continue

        image_np = np.array(image)
        for img_aug in augmentation_image(image_np):
            embed = model.embeddings([img_aug])
            embeds.append(embed[0])

    return np.array(embeds) if embeds else None


def load_existing_data():
    if os.path.exists(LIST_EMBEDDING) and os.path.exists(LIST_NAME):
        embed_lists = np.load(LIST_EMBEDDING, allow_pickle=True)
        names_lists = np.load(LIST_NAME, allow_pickle=True)
        return embed_lists, names_lists
    return np.empty((0, 512)), np.array([])


def save_data(embed_lists, names_lists):
    np.save(LIST_EMBEDDING, embed_lists)
    np.save(LIST_NAME, names_lists)
    print("Đã lưu dữ liệu embeddings thành công.")


def update_embeddings(user_name):
    print(f"Đang cập nhật embeddings cho: {user_name}")

    embed_lists, names_lists = load_existing_data()
    new_embeds = compute_embeddings_for_user(user_name)
    if new_embeds is None:
        print(f"Không tạo được embeddings cho {user_name}")
        return False

    if user_name in names_lists:
        idx_to_keep = np.where(names_lists != user_name)
        embed_lists = embed_lists[idx_to_keep]
        names_lists = names_lists[idx_to_keep]
        print(f"Đã xóa embedding cũ của {user_name}")

    embed_lists = np.vstack([embed_lists, new_embeds])
    names_lists = np.concatenate([names_lists, np.array([user_name] * len(new_embeds))])

    save_data(embed_lists, names_lists)
    print(f"Cập nhật embeddings thành công cho {user_name}")

    return True
