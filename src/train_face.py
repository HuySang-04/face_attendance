from utils import *

model = FaceNet()
embed_lists = []
names_lists = []

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

for usr in os.listdir(DATA_PATH):
    embeds = []
    usr_path = os.path.join(DATA_PATH, usr)

    for file in os.listdir(usr_path):
        try:
            image = Image.open(f'{usr_path}/{file}').convert('RGB')
        except:
            print(f'Cannot open image: {file}')
            continue

        image_np = np.array(image)
        for img_aug in augmentation_image(image_np):
            embed = model.embeddings([img_aug])
            embeds.append(embed[0])


    if embeds:
        embed_lists.extend(embeds)
        names_lists.extend([usr] * len(embeds))
        print(f"Processed for {usr}")

embed_lists = np.array(embed_lists)
np.save(LIST_EMBEDDING, embed_lists)
np.save(LIST_NAME, np.array(names_lists))