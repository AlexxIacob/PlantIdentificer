    from flask import Flask, request, jsonify
    from tensorflow.keras.models import load_model
    import numpy as np
    from PIL import Image



    model = load_model("C:\AplicatieLicenta-PlantScan\Backend\transferModel.h5")

    def preprocess_image(image_path):
        img = Image.open(image_path).resize((224, 224))  # modifică dimensiunea după cum ai antrenat
        img_array = np.array(img) / 255.0  # normalizare, dacă ai făcut-o
        img_array = np.expand_dims(img_array, axis=0)  # (1, h, w, c)
        return img_array

    test_input = preprocess_image("$RSJC3QO.jpeg")  # înlocuiește cu o imagine de test validă
    prediction = model.predict(test_input)
    print("Predicție:", prediction)

