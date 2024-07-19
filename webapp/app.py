import base64
import datetime
import cv2
import numpy as np
import torch
import os
import bcrypt
from sqlalchemy import create_engine, MetaData, Table, select, engine
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import aliased
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, url_for, redirect, abort, current_app
from flask_login import LoginManager, UserMixin, login_required, logout_user, login_user, current_user
from sqlalchemy import func

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = '123456'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost/monitor'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static')

# 初始化Flask-Login
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# 数据库模型 - 用户表
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(50), nullable=False)
    callnumber = db.Column(db.String(11), nullable=False)
    mail = db.Column(db.String(50), nullable=False)
    classroom = db.Column(db.Integer, db.ForeignKey('room.number'), nullable=False)

    def __init__(self, username, password, callnumber, mail, classroom):
        self.username = username
        self.password = password
        self.callnumber = callnumber
        self.mail = mail
        self.classroom = classroom


# 数据库模型 - 房间表
class Room(db.Model):
    number = db.Column(db.String(10), primary_key=True)  # 主键
    student = db.Column(db.Integer, nullable=False)  # 学生数量
    status = db.Column(db.String(1), nullable=False)  # 状态，Y表示可用，N表示不可用

    def __init__(self, number, student, status):
        self.number = number
        self.student = student
        self.status = status


# 数据库模型 - 预测表
class Forecast(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.String(10), nullable=False)  # 教室号d
    talk = db.Column(db.Integer, nullable=False)  # 聊天人数
    study = db.Column(db.Integer, nullable=False)  # 学习人数
    watch = db.Column(db.Integer, nullable=False)  # 观看视频人数
    sleep = db.Column(db.Integer, nullable=False)  # 睡觉人数
    phone = db.Column(db.Integer, nullable=False)  # 使用手机人数
    time = db.Column(db.DateTime, nullable=False)  # 记录时间
    video = db.Column(db.Integer, nullable=False)  # 使用手机人数

    # 构造函数
    def __init__(self, number, talk, study, watch, sleep, phone, time, video):
        self.number = number
        self.talk = talk
        self.study = study
        self.watch = watch
        self.sleep = sleep
        self.phone = phone
        self.time = time
        self.video = video


# 类别名称映射
class_names = {
    0: 'talk',
    1: 'study',
    2: 'watch',
    3: 'sleep',
    4: 'play_phone'
}
device = torch.device('cpu')
model_path = r'D:\Class\240701\YOLO1\ultralytics-main/best.pt'
model = YOLO(model_path).to(device)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode == "RGBA" else img),  # 如果是RGBA则转换为RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_counter = 0  # 全局计数器


# 加密密码
def hash_password(password):
    # Generate a salt
    salt = bcrypt.gensalt()
    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')


# 验证密码
def check_password(hashed_password, candidate_password):
    return bcrypt.checkpw(candidate_password.encode('utf-8'), hashed_password.encode('utf-8'))


# 在注册时加密密码


# 在保存图像的函数中
def save_image_with_counter(image, forecast_id):
    # 确定保存图片的目录
    save_dir = app.config['UPLOAD_FOLDER']

    # 构建新的文件名，使用预报记录的ID
    new_filename = f'{forecast_id}.jpg'

    # 完整路径
    full_path = os.path.join(save_dir, new_filename)

    # 保存图像
    cv2.imwrite(full_path, image)
    print(f"保存带有边界框和标签的图像到 {full_path}")

    return full_path


# 预测上传的图像
def predict_image(image_path):
    # 读取图像文件
    img = Image.open(image_path)
    # 使用YOLOv8进行预测
    results = model(img)[0]
    # 获取检测到的边界框信息
    boxes = results.boxes
    # 在原始图像上绘制检测到的边界框和标注类别
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # 转换PIL图像为OpenCV格式
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls)
        confidence = float(box.conf)
        # 绘制边界框
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 标注类别标签和置信度
        label = f"{class_names[class_id]} ({confidence:.2f})"
        cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 返回预测结果
    predictions = []
    for r in results.boxes.data.tolist():
        class_id = int(r[5])
        confidence = r[4]
        bbox = r[:4]
        predictions.append({
            'class_id': class_id,
            'confidence': confidence,
            'bbox': bbox,
        })

    # 计算各类别的人数
    counts = {class_name: 0 for class_name in class_names.values()}
    for prediction in predictions:
        class_id = int(prediction['class_id'])
        if class_id in class_names:
            counts[class_names[class_id]] += 1
    # 创建一个新的Forecast记录
    new_forecast = Forecast(
        number=current_user.classroom,
        talk=counts['talk'],
        study=counts['study'],
        watch=counts['watch'],
        sleep=counts['sleep'],
        phone=counts['play_phone'],
        time=datetime.datetime.now(),  # 使用当前时间
        video=0
    )
    # 将新记录添加到数据库会话
    db.session.add(new_forecast)
    # 提交更改以保存到数据库
    db.session.commit()
    # 保存带有边界框和标签的图像
    output_img_path = save_image_with_counter(img_cv, new_forecast.id)
    cv2.imwrite(output_img_path, img_cv)
    print(f"保存带有边界框和标签的图像到 {output_img_path}")
    return {'predictions': predictions, 'image_path': output_img_path}


def predict_video(video_path):
    max_video = db.session.query(func.max(Forecast.video)).scalar() or 0
    new_video = max_video + 1
    video_capture = cv2.VideoCapture(video_path)
    frame_results = []

    fps = video_capture.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    interval = 3  # 每三秒处理一次
    frame_interval = int(fps * interval)  # 每三秒对应的帧数

    frame_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = model(img_pil)[0]
            boxes = results.boxes

            # 在原始图像上绘制检测到的边界框和标注类别
            img_cv = frame.copy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls)
                confidence = float(box.conf)
                # 绘制边界框
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 标注类别标签和置信度
                label = f"{class_names[class_id]} ({confidence:.2f})"
                cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                frame_results.append({
                    'timestamp': video_capture.get(cv2.CAP_PROP_POS_MSEC),
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                })

            # 计算各类别的人数
            counts = {class_name: 0 for class_name in class_names.values()}
            for box in boxes:
                class_id = int(box.cls)
                if class_id in class_names:
                    counts[class_names[class_id]] += 1

            # 创建一个新的Forecast记录
            new_forecast = Forecast(
                number=current_user.classroom,
                talk=counts['talk'],
                study=counts['study'],
                watch=counts['watch'],
                sleep=counts['sleep'],
                phone=counts['play_phone'],
                time=datetime.datetime.now(),  # 使用当前时间
                video=new_video
            )

            # 将新记录添加到数据库会话
            db.session.add(new_forecast)
            db.session.commit()

            # 保存带有边界框和标签的图像
            save_image_with_counter(img_cv, new_forecast.id)

        frame_count += 1

    video_capture.release()
    return frame_results


@app.route('/predict_image', methods=['POST'])
@login_required
def handle_predict_image():
    file = request.files['file']
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        prediction_result = predict_image(file_path)
        return jsonify(prediction_result)
    return jsonify({'message': 'No file provided'})


@app.route('/predict_video', methods=['POST'])
@login_required
def handle_predict_video():
    file = request.files['file']
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        frame_results = predict_video(file_path)
        return jsonify({"frames": frame_results})
    return jsonify({'message': 'No file provided'})


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# 登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password(user.password, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))  # 修改这里
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        hashed_password = hash_password(password)
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('register.html', error="Username already exists.")
        new_user = User(username=username, password=hashed_password, callnumber='13383613201', mail='1695522966@qq.com',
                        classroom='0101')  # 密码已加密存储
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/api/video-tags')
def get_unique_video_tags():
    try:
        # 创建Forecast表的别名
        forecast_alias = aliased(Forecast)
        # 使用SQLAlchemy ORM方式创建查询，选择Forecast表中的video列，
        # 并去除重复值，同时排除None和0
        query = (
            db.session.query(forecast_alias.video)
            .filter(forecast_alias.video is not None)  # 排除None
            .filter(forecast_alias.video != 0)  # 排除0
            .distinct()
        )
        # 执行查询并获取结果
        video_tags = [row[0] for row in query.all()]
        return jsonify(video_tags)
    except Exception as e:
        current_app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500


@app.route('/api/detection-counts', methods=['GET'])
@login_required
def get_detection_counts():
    tag = request.args.get('tag')
    if not tag or tag == '0':
        return jsonify({'error': 'Invalid or missing tag parameter'}), 400

    # 查询与当前用户关联且与给定tag匹配的所有预测记录
    forecasts = Forecast.query.filter_by(number=current_user.classroom).filter(Forecast.video == tag).all()
    start_time = 0
    # 将查询结果转换为列表，其中每个元素都是一个字典，包含时间戳和各类别计数
    data = [
        {
            'time': start_time + i * 5,
            'talk': forecast.talk,
            'study': forecast.study,
            'watch': forecast.watch,
            'sleep': forecast.sleep,
            'phone': forecast.phone
        }
        for i, forecast in enumerate(forecasts)
    ]

    return jsonify(data)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/show')
@login_required
def show():
    return render_template('show.html')


@app.route('/get_usage_stats')
@login_required
def get_usage_stats():
    # 创建Room表的别名
    roomalias = aliased(Room)
    # 查询当前用户关联的班级的预测结果数量
    total_forecasts = db.session.query(Forecast).join(roomalias, Forecast.number == roomalias.number).filter(
        roomalias.number == current_user.classroom).count()
    return jsonify({'total_forecasts': total_forecasts})


@app.route('/history', methods=['GET'])
@login_required
def history():
    # 查询与当前用户关联的所有预测记录
    forecasts = Forecast.query.filter_by(number=current_user.classroom).all()
    histories = []
    for forecast in forecasts:
        # 将每个预测记录转换为字典格式
        history = {
            'timestamp': forecast.time.strftime("%Y-%m-%d %H:%M:%S"),
            'image_path': f'/static/{forecast.id}.jpg',  # 假设每条记录对应一个静态文件
            'prediction_result': {
                'talk': forecast.talk,
                'study': forecast.study,
                'watch': forecast.watch,
                'sleep': forecast.sleep,
                'play_phone': forecast.phone,
                'video': forecast.video
            }
        }
        histories.append(history)
    return jsonify(histories)


# 主页路由，现在需要登录才能访问
@app.route('/')
@login_required
def home():
    forecast_records = Forecast.query.filter_by(number=current_user.classroom).order_by(Forecast.time.desc()).first()
    if forecast_records:
        forecast_data = {
            'number': forecast_records.number,
            'talk': forecast_records.talk,
            'study': forecast_records.study,
            'watch': forecast_records.watch,
            'sleep': forecast_records.sleep,
            'phone': forecast_records.phone,
            'time': forecast_records.time.strftime("%Y-%m-%d %H:%M:%S"),
            'video': forecast_records.video
        }
    else:
        forecast_data = None
    return render_template('Rec_Plane.html', forecast=forecast_data)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # 保存文件到静态目录
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # 调用预测函数，传入文件路径
        result_dict = predict_image(file_path)
        # 从返回的字典中获取预测结果和图像路径
        predictions = result_dict['predictions']
        detected_image_path = result_dict['image_path']
        # 使用返回的路径读取图像并编码为Base64
        with open(detected_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # 清理临时文件
        os.remove(file_path)  # 清除原始上传的文件
        # 包含Base64编码的图像和预测结果
        response_data = {
            'predictions': predictions,
            'annotatedImage': encoded_string
        }

        return jsonify(response_data)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # 创建所有定义的数据库表
    app.run(host='0.0.0.0', port=5000, debug=True)
