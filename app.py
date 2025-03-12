import eventlet
eventlet.monkey_patch(dns=False)  # Disable Eventlet's DNS patching to avoid dnspython conflicts

import os
from flask import Flask, request, jsonify, session, render_template_string
from flask_socketio import SocketIO, emit, join_room, leave_room

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a strong secret in production
socketio = SocketIO(app, async_mode='eventlet')

# In-memory storage (use a database in production)
users = {}  # {phone_number: name}
online_users = set()
ADMIN_PASSWORD = 'jack'

# HTML template styled to look like a modern iPhone with an Apple wallpaper and responsive scaling
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>iPhone Replica</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <!-- Using system fonts (San Francisco) -->
  <link href="https://fonts.googleapis.com/css?family=San+Francisco:400,500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    /* Reset */
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
    }
    /* iPhone frame – fixed size for design but scales on smaller screens */
    .iphone {
      width: 375px;
      height: 812px;
      max-width: 100%;
      background: url('https://www.apple.com/v/iphone/home/af/images/overview/hero/hero_iphone__d6e06xbkqay6_large.jpg') no-repeat center center;
      background-size: cover;
      border: 16px solid #333;
      border-radius: 50px;
      position: relative;
      box-shadow: 0 20px 30px rgba(0,0,0,0.5);
      overflow: hidden;
    }
    /* Notch with reduced width (126px) */
    .notch {
      position: absolute;
      top: 0;
      left: 50%;
      transform: translateX(-50%);
      width: 126px;
      height: 20px;
      background: #000;
      border-bottom-left-radius: 15px;
      border-bottom-right-radius: 15px;
      z-index: 10;
      margin-top: 4px;
    }
    /* Header (appears at top of each app screen) */
    .header {
      position: absolute;
      top: 0;
      width: 100%;
      height: 50px;
      background: rgba(255,255,255,0.9);
      display: flex;
      align-items: center;
      padding: 0 15px;
      border-bottom: 1px solid #ddd;
      z-index: 5;
      font-size: 16px;
    }
    /* Screen container */
    .app-view {
      position: absolute;
      top: 50px;
      bottom: 0;
      width: 100%;
      background: #f9f9f9;
      overflow: hidden;
    }
    /* Each screen page */
    .screen {
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
      display: none;
      flex-direction: column;
      padding: 60px 15px 15px 15px; /* Top padding to allow for header */
    }
    /* Home Screen styling */
    #homeScreen {
      display: none;
      background: transparent;
    }
    .home-grid {
      display: grid;
      grid-template-columns: repeat(4, 70px);
      gap: 20px;
      justify-content: center;
      margin-top: 40px;
    }
    .app-icon {
      width: 70px;
      height: 70px;
      background: #fff;
      border: 1px solid #ccc;
      border-radius: 15px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: transform 0.2s;
    }
    .app-icon:hover {
      transform: translateY(-5px);
    }
    .app-icon i { font-size: 28px; color: #333; margin-bottom: 5px; }
    .app-icon span { font-size: 11px; color: #555; }
    /* Dock at bottom */
    .dock {
      position: absolute;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      gap: 20px;
      background: rgba(255,255,255,0.9);
      padding: 10px 20px;
      border-radius: 30px;
      box-shadow: 0 3px 8px rgba(0,0,0,0.2);
    }
    /* Form elements */
    input[type="text"], input[type="password"] {
      width: 100%;
      padding: 10px;
      margin: 10px 0;
      border: 1px solid #ccc;
      border-radius: 5px;
    }
    button {
      padding: 10px 15px;
      border: none;
      border-radius: 5px;
      background: #007aff;
      color: #fff;
      cursor: pointer;
      margin: 5px 0;
    }
    /* Messages App */
    #conversation {
      flex: 1;
      overflow-y: auto;
      padding: 10px;
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 8px;
      margin-bottom: 10px;
    }
    #messageForm {
      display: flex;
      gap: 5px;
    }
    #messageForm input[type="text"] {
      flex: 1;
      padding: 8px;
      border: 1px solid #ccc;
      border-radius: 5px;
    }
    #messageForm button {
      padding: 8px 12px;
      background: #007aff;
      border: none;
      border-radius: 5px;
      color: #fff;
    }
    /* Center login and admin screens */
    #loginScreen, #adminLoginScreen {
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }
  </style>
</head>
<body>
<div class="iphone">
  <div class="notch"></div>
  <div id="header" class="header"></div>
  <div class="app-view">
    <!-- Login Screen -->
    <div id="loginScreen" class="screen">
      <div style="width: 100%; text-align: center;">
        <h2>Enter Phone Number</h2>
        <input type="text" id="phoneNumberInput" placeholder="Phone Number">
        <div>
          <button onclick="login()">Unlock</button>
          <button onclick="register()">Register New Number</button>
        </div>
      </div>
    </div>
    <!-- Home Screen -->
    <div id="homeScreen" class="screen">
      <div class="home-grid">
        <div class="app-icon" onclick="showScreen('textApp')">
          <i class="fa-solid fa-comment"></i>
          <span>Messages</span>
        </div>
        <div class="app-icon" onclick="showScreen('facetimeApp')">
          <i class="fa-solid fa-video"></i>
          <span>FaceTime</span>
        </div>
        <div class="app-icon" onclick="showScreen('voiceApp')">
          <i class="fa-solid fa-phone"></i>
          <span>Voice</span>
        </div>
        <div class="app-icon" onclick="showScreen('onlineUsersApp')">
          <i class="fa-solid fa-users"></i>
          <span>Online</span>
        </div>
        <div class="app-icon" onclick="showScreen('adminLoginScreen')">
          <i class="fa-solid fa-user-shield"></i>
          <span>Admin</span>
        </div>
      </div>
      <div class="dock">
        <div class="app-icon" onclick="showScreen('textApp')">
          <i class="fa-solid fa-comment"></i>
        </div>
        <div class="app-icon" onclick="showScreen('voiceApp')">
          <i class="fa-solid fa-phone"></i>
        </div>
        <div class="app-icon" onclick="showScreen('facetimeApp')">
          <i class="fa-solid fa-video"></i>
        </div>
      </div>
    </div>
    <!-- Messages App -->
    <div id="textApp" class="screen">
      <div id="conversation"></div>
      <form id="messageForm" onsubmit="sendMessage(event)">
        <input type="text" id="recipientInput" placeholder="To: Number or group:">
        <input type="text" id="messageInput" placeholder="Message">
        <button type="submit">Send</button>
      </form>
      <button onclick="joinGroup()">Join Group</button>
    </div>
    <!-- FaceTime App -->
    <div id="facetimeApp" class="screen">
      <h2>FaceTime</h2>
      <input type="text" id="facetimeRecipient" placeholder="Recipient Phone Number">
      <button onclick="startCall('video')">Call</button>
      <div id="facetimeStatus"></div>
      <video id="localVideo" autoplay muted style="width: 100px;"></video>
      <video id="remoteVideo" autoplay style="width: 100%;"></video>
    </div>
    <!-- Voice App -->
    <div id="voiceApp" class="screen">
      <h2>Voice</h2>
      <input type="text" id="callRecipient" placeholder="Recipient Phone Number">
      <button onclick="startCall('audio')">Call</button>
      <div id="callStatus"></div>
      <audio id="remoteAudio" autoplay></audio>
    </div>
    <!-- Online Users App -->
    <div id="onlineUsersApp" class="screen">
      <h2>Online Users</h2>
      <ul id="onlineUsersList"></ul>
    </div>
    <!-- Admin Login Screen -->
    <div id="adminLoginScreen" class="screen">
      <div style="width: 100%; text-align: center;">
        <h2>Admin Login</h2>
        <input type="password" id="adminPasswordInput" placeholder="Password">
        <button onclick="adminLogin()">Login</button>
      </div>
    </div>
    <!-- Admin App -->
    <div id="adminApp" class="screen">
      <h2>Admin Panel</h2>
      <input type="text" id="newPhoneNumber" placeholder="Phone Number">
      <input type="text" id="newName" placeholder="Name">
      <button onclick="addUser()">Add User</button>
    </div>
  </div>
</div>
<script src="https://cdn.socket.io/4.6.1/socket.io.min.js"></script>
<script>
  var socket = io();
  showScreen('loginScreen');

  function showScreen(screenId) {
    var screens = document.getElementsByClassName('screen');
    for (var i = 0; i < screens.length; i++) {
      screens[i].style.display = 'none';
    }
    document.getElementById(screenId).style.display = 'flex';
    updateHeader(screenId);
  }

  function updateHeader(screenId) {
    var header = document.getElementById('header');
    if (screenId === 'homeScreen' || screenId === 'loginScreen') {
      header.innerHTML = '<span style="font-size: 16px;">9:41</span><span style="margin-left: auto; font-size: 16px;">🔋 100%</span><span style="margin-left: 10px; font-size: 16px;">📡</span>';
    } else {
      var title = {
        'textApp': 'Messages',
        'facetimeApp': 'FaceTime',
        'voiceApp': 'Voice',
        'onlineUsersApp': 'Online Users',
        'adminApp': 'Admin Panel',
        'adminLoginScreen': 'Admin Login'
      }[screenId] || '';
      header.innerHTML = "<i class='fa-solid fa-house' style='margin-right:10px; cursor:pointer;' onclick='showScreen(\"homeScreen\")'></i>" + title;
    }
  }

  function login() {
    var phoneNumber = document.getElementById('phoneNumberInput').value.trim();
    fetch('/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ phone_number: phoneNumber })
    })
    .then(res => res.json())
    .then(data => {
      if (data.status === 'success') {
        showScreen('homeScreen');
      } else {
        alert(data.message);
      }
    });
  }

  function register() {
    var phoneNumber = document.getElementById('phoneNumberInput').value.trim();
    if (!phoneNumber) {
      alert("Please enter a phone number to register.");
      return;
    }
    fetch('/register', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ phone_number: phoneNumber })
    })
    .then(res => res.json())
    .then(data => {
      if (data.status === 'success') {
        alert("Registration successful!");
        showScreen('homeScreen');
      } else {
        alert(data.message);
      }
    });
  }

  function adminLogin() {
    var password = document.getElementById('adminPasswordInput').value;
    fetch('/admin/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ password: password })
    })
    .then(res => res.json())
    .then(data => {
      if (data.status === 'success') {
        showScreen('adminApp');
      } else {
        alert(data.message);
      }
    });
  }

  function addUser() {
    var phoneNumber = document.getElementById('newPhoneNumber').value;
    var name = document.getElementById('newName').value;
    fetch('/admin/add_user', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ phone_number: phoneNumber, name: name })
    })
    .then(res => res.json())
    .then(data => {
      alert(data.status === 'success' ? 'User added' : data.message);
    });
  }

  socket.on('online_users', function(users) {
    var list = document.getElementById('onlineUsersList');
    list.innerHTML = '';
    users.forEach(user => {
      var li = document.createElement('li');
      li.textContent = user;
      list.appendChild(li);
    });
  });

  function sendMessage(e) {
    e.preventDefault();
    var recipient = document.getElementById('recipientInput').value.trim();
    var message = document.getElementById('messageInput').value.trim();
    if (message) {
      if (recipient.startsWith('group:')) {
        socket.emit('send_message', { group: recipient.substring(6), message: message });
      } else {
        socket.emit('send_message', { recipient: recipient, message: message });
      }
      document.getElementById('messageInput').value = '';
    }
  }

  function joinGroup() {
    var group = prompt('Enter group name:');
    if (group) socket.emit('join_group', { group: group });
  }

  socket.on('receive_message', function(data) {
    var conv = document.getElementById('conversation');
    var msg = document.createElement('div');
    msg.style.marginBottom = '10px';
    msg.textContent = data.group ? '[' + data.group + '] ' + data.sender + ': ' + data.message : data.sender + ': ' + data.message;
    conv.appendChild(msg);
    conv.scrollTop = conv.scrollHeight;
  });

  let localStream, peerConnection;
  async function startCall(type) {
    const recipient = type === 'audio' ? document.getElementById('callRecipient').value : document.getElementById('facetimeRecipient').value;
    if (!recipient) return;
    localStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: type === 'video' });
    if (type === 'video') document.getElementById('localVideo').srcObject = localStream;
    peerConnection = new RTCPeerConnection();
    localStream.getTracks().forEach(track => peerConnection.addTrack(track, localStream));
    peerConnection.onicecandidate = e => { if (e.candidate) socket.emit('ice_candidate', { recipient: recipient, candidate: e.candidate }); };
    peerConnection.ontrack = e => {
      if (type === 'audio') document.getElementById('remoteAudio').srcObject = e.streams[0];
      else document.getElementById('remoteVideo').srcObject = e.streams[0];
    };
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    socket.emit('call_user', { recipient: recipient, offer: offer });
  }

  socket.on('incoming_call', async data => {
    if (confirm('Call from ' + data.caller + '. Accept?')) {
      localStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: document.getElementById('facetimeApp').style.display === 'flex' });
      if (localStream.getVideoTracks().length) document.getElementById('localVideo').srcObject = localStream;
      peerConnection = new RTCPeerConnection();
      localStream.getTracks().forEach(track => peerConnection.addTrack(track, localStream));
      peerConnection.ontrack = e => {
        var remote = document.getElementById(localStream.getVideoTracks().length ? 'remoteVideo' : 'remoteAudio');
        remote.srcObject = e.streams[0];
      };
      peerConnection.onicecandidate = e => { if (e.candidate) socket.emit('ice_candidate', { recipient: data.caller, candidate: e.candidate }); };
      await peerConnection.setRemoteDescription(new RTCSessionDescription(data.offer));
      const answer = await peerConnection.createAnswer();
      await peerConnection.setLocalDescription(answer);
      socket.emit('answer_call', { caller: data.caller, answer: answer });
    }
  });

  socket.on('call_answered', async data => {
    await peerConnection.setRemoteDescription(new RTCSessionDescription(data.answer));
  });

  socket.on('ice_candidate', async data => {
    await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
  });
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/login', methods=['POST'])
def login():
    phone_number = request.json.get('phone_number')
    if phone_number in users:
        session['phone_number'] = phone_number
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Phone number not found'}), 404

@app.route('/register', methods=['POST'])
def register():
    phone_number = request.json.get('phone_number')
    if not phone_number:
        return jsonify({'status': 'error', 'message': 'Phone number is required'}), 400
    if phone_number in users:
        return jsonify({'status': 'error', 'message': 'Phone number already exists'}), 400
    users[phone_number] = phone_number
    session['phone_number'] = phone_number
    return jsonify({'status': 'success'})

@app.route('/admin/login', methods=['POST'])
def admin_login():
    password = request.json.get('password')
    if password == ADMIN_PASSWORD:
        session['admin'] = True
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Invalid password'}), 401

@app.route('/admin/add_user', methods=['POST'])
def add_user():
    if not session.get('admin'):
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    phone_number = request.json.get('phone_number')
    name = request.json.get('name')
    if phone_number in users:
        return jsonify({'status': 'error', 'message': 'Phone number exists'}), 400
    users[phone_number] = name
    return jsonify({'status': 'success'})

@socketio.on('connect')
def handle_connect():
    phone_number = session.get('phone_number')
    if phone_number:
        online_users.add(phone_number)
        emit('online_users', list(online_users), broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    phone_number = session.get('phone_number')
    if phone_number in online_users:
        online_users.remove(phone_number)
        emit('online_users', list(online_users), broadcast=True)

@socketio.on('send_message')
def handle_send_message(data):
    sender = session.get('phone_number')
    recipient = data.get('recipient')
    message = data.get('message')
    if recipient:
        emit('receive_message', {'sender': sender, 'message': message}, room=recipient)
    else:
        group = data.get('group')
        emit('receive_message', {'sender': sender, 'message': message, 'group': group}, room=group)

@socketio.on('join_group')
def handle_join_group(data):
    group = data.get('group')
    join_room(group)
    emit('joined_group', {'group': group}, room=group)

@socketio.on('call_user')
def handle_call_user(data):
    recipient = data.get('recipient')
    offer = data.get('offer')
    caller = session.get('phone_number')
    emit('incoming_call', {'caller': caller, 'offer': offer}, room=recipient)

@socketio.on('answer_call')
def handle_answer_call(data):
    caller = data.get('caller')
    answer = data.get('answer')
    emit('call_answered', {'answer': answer}, room=caller)

@socketio.on('ice_candidate')
def handle_ice_candidate(data):
    recipient = data.get('recipient')
    candidate = data.get('candidate')
    emit('ice_candidate', {'candidate': candidate}, room=recipient)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)
