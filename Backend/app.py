from flask import Flask
from flask import Blueprint
from api_routes.chatbot_route import chatbot_bp
from api_routes.plantpredict_route import plantpredict_bp

app = Flask(__name__)

app.register_blueprint(plantpredict_bp)
app.register_blueprint(chatbot_bp)

if __name__ == '__main__':
    app.run(debug=True)
