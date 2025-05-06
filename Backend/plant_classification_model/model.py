from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import  tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

model_path = r"C:\AplicatieLicenta-PlantScan\Backend\plant_classification_model\transferModel.keras"

@register_keras_serializable()
class HubFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, model_url, input_shape, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.model_url = model_url
        self.input_shape = input_shape
        self.trainable = trainable
        self.feature_extractor = hub.KerasLayer(
            model_url,
            input_shape=input_shape,
            trainable=trainable,
            name=kwargs.get('name', 'hub_layer')
        )

    def call(self, inputs):
        return self.feature_extractor(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_url": self.model_url,
            "input_shape": self.input_shape,
            "trainable": self.trainable
        })
        return config

# Load the model with both custom objects
model = load_model(
    model_path,
    custom_objects={
        'HubFeatureExtractor': HubFeatureExtractor,
        'KerasLayer': hub.KerasLayer
    },
    compile=False
)

classLabels = {
    "20": "fire lily", "2": "canterbury bells", "44": "bolero deep blue",
    "0": "pink primrose", "33": "mexican aster", "26": "prince of wales feathers",
    "6": "moon orchid", "15": "globe-flower", "24": "grape hyacinth",
    "25": "corn poppy", "78": "toad lily", "38": "siam tulip", "23": "red ginger",
    "66": "spring crocus", "34": "alpine sea holly", "31": "garden phlox",
    "9": "globe thistle", "5": "tiger lily", "92": "ball moss",
    "32": "love in the mist", "8": "monkshood", "101": "blackberry lily",
    "13": "spear thistle", "18": "balloon flower", "99": "blanket flower",
    "12": "king protea", "48": "oxeye daisy", "14": "yellow iris",
    "60": "cautleya spicata", "30": "carnation", "63": "silverbush",
    "67": "bearded iris", "62": "black-eyed susan", "68": "windflower",
    "61": "japanese anemone", "19": "giant white arum lily",
    "37": "great masterwort", "3": "sweet pea", "85": "tree mallow",
    "100": "trumpet creeper", "41": "daffodil", "21": "pincushion flower",
    "1": "hard-leaved pocket orchid", "53": "sunflower",
    "65": "osteospermum", "69": "tree poppy", "84": "desert-rose",
    "98": "bromelia", "86": "magnolia", "4": "english marigold",
    "91": "bee balm", "27": "stemless gentian", "96": "mallow",
    "56": "gaura", "39": "lenten rose", "46": "marigold",
    "58": "orange dahlia", "47": "buttercup", "54": "pelargonium",
    "35": "ruby-lipped cattleya", "90": "hippeastrum",
    "28": "artichoke", "70": "gazania", "89": "canna lily",
    "17": "peruvian lily", "97": "mexican petunia", "7": "bird of paradise",
    "29": "sweet william", "16": "purple coneflower", "51": "wild pansy",
    "83": "columbine", "11": "colt's foot", "10": "snapdragon",
    "95": "camellia", "22": "fritillary", "49": "common dandelion",
    "43": "poinsettia", "52": "primula", "71": "azalea",
    "64": "californian poppy", "79": "anthurium", "75": "morning glory",
    "36": "cape flower", "55": "bishop of llandaff", "59": "pink-yellow dahlia",
    "81": "clematis", "57": "geranium", "74": "thorn apple",
    "40": "barbeton daisy", "94": "bougainvillea", "42": "sword lily",
    "82": "hibiscus", "77": "lotus lotus", "87": "cyclamen", "93": "foxglove",
    "80": "frangipani", "73": "rose", "88": "watercress", "72": "water lily",
    "45": "wallflower", "76": "passion flower", "50": "petunia"
}


def predict_file(file):
    img = Image.open(file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]

    top_index = int(np.argmax(prediction))
    label = classLabels.get(str(top_index), f"Label necunoscut ({top_index})")
    prob = float(prediction[top_index])

    return {"label": label, "prob": prob}