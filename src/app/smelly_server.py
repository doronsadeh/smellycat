from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable WebSocket support


@app.route('/')
def index():
    return jsonify({"message": "Server is running"})


# Route to handle POST requests and emit data to clients
@app.route('/update', methods=['POST'])
def update_data():
    try:
        data = request.json.get('datapoint', [])
        if len(data) != 5:
            return jsonify({'error': 'Invalid data format. Expecting 4 values.'}), 400

        # Emit data to all connected clients via WebSocket
        socketio.emit('update_chart', {'data': data})
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    socketio.run(app, debug=True)
