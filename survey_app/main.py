from flask import Flask, render_template, redirect, url_for, request, jsonify, Response
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
import secrets
import numpy as np
import csv
import random
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from wtforms import StringField, PasswordField, BooleanField, IntegerField, validators
from wtforms.validators import InputRequired, Length, DataRequired, EqualTo, NumberRange
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user


app = Flask(__name__)

""" We activate this after the survey finished!
@app.route("/")
def index():
    return render_template("nomore.html", current_user=None)
"""


secret_key = secrets.token_hex(16)
db_user = os.environ.get("CLOUD_SQL_USERNAME")
db_password = os.environ.get("CLOUD_SQL_PASSWORD")
db_name = os.environ.get("CLOUD_SQL_DATABASE_NAME")
db_connection_name = os.environ.get("CLOUD_SQL_CONNECTION_NAME")
project_id = os.environ.get("PROJECT_ID")
instance_name = os.environ.get("CLOUD_SQL_INSTANCE_NAME")
flask_secret_key = os.environ.get("FLASK_SECRET_KEY")


app.config[
    "SQLALCHEMY_DATABASE_URI"
] = f"mysql+pymysql://{db_user}:{db_password}@{instance_name}/{db_name}?unix_socket=/cloudsql/{db_connection_name}"
app.config["SECRET_KEY"] = flask_secret_key
Bootstrap(app)
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Check which datasets exists
DATASETS = ["cplfw", "calfw", "xqlfw", "mlfw", "base"]

ADMIN_USERNAME = "admin"  # Name of the admin user

TIME_PER_QUESTION = 10  # Seconds

START_ID = 1  # ID of the first user in the system

NUM_USERS = 60  # Number of users in the system


# Names for the survey to appear on the dashboard
SURVEY_NAMES = {
    "base": "Survey 1",
    "calfw": "Survey 2",
    "cplfw": "Survey 3",
    "xqlfw": "Survey 4",
    "mlfw": "Survey 5",
}


# Descriptions of the survey, appear on the dashboard
SURVEY_DESCRIPTIONS = {
    "base": "Baseline survey.",
    "calfw": "Faces in different age.",
    "cplfw": "Faces in different pose.",
    "xqlfw": "Faces in different image qualities.",
    "mlfw": "Faces with or without masks.",
}


# Read the lists of scores for each dataset (this way we can select the least confident pairs for the machines)
DATA = {}
for dataset in DATASETS:
    DATA[dataset] = []
    with open(f"./static/lists/{dataset}.txt", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        all_data = [tuple(row) for row in reader]
    for uid in range(1, NUM_USERS + 1):
        DATA[dataset].append([lst for lst in all_data if int(lst[0]) in [uid]])


# This table is used to keep track of all the users in the system
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    password = db.Column(db.String(80))
    gender = db.Column(db.Integer)
    ethnicity = db.Column(db.Integer)
    age = db.Column(db.Integer)
    registration_date = db.Column(db.DateTime, default=datetime.utcnow)


# This table below is used to store the user's answers to the questions
class Survey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    dataset = db.Column(db.String(5))
    question_id = db.Column(db.Integer)
    pair_id = db.Column(db.Integer)
    certainty = db.Column(db.Integer)
    prediction = db.Column(db.String(3))
    starttime = db.Column(db.DateTime)
    endtime = db.Column(db.DateTime)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# This makes "current_user" available to templates
@app.context_processor
def inject_user():
    return {"current_user": current_user}


# This form is used to register a new user
class RegistrationForm(FlaskForm):
    username = StringField("username", validators=[InputRequired(), Length(min=5, max=15)])
    password = PasswordField("password", validators=[InputRequired(), Length(min=5, max=15)])
    password_repeat = PasswordField("password_repeat", validators=[DataRequired(), EqualTo("password")])
    age = IntegerField("age", [validators.optional()])
    ethnicity = IntegerField("ethnicity", [validators.optional()])
    gender = IntegerField("gender", [validators.optional()])


# This is needed to delete a user
class DeletionForm(FlaskForm):
    username = StringField("username")


# This form is used to login the user
class LoginForm(FlaskForm):
    username = StringField("username", validators=[InputRequired(), Length(min=5, max=15)])
    password = PasswordField("password", validators=[InputRequired(), Length(min=5, max=15)])
    remember = BooleanField("remember")


# This form is used to collect the user's answers to the questions
class DataForm(FlaskForm):
    dataset = StringField("dataset", validators=[InputRequired()])
    question_id = IntegerField("question_id", validators=[InputRequired()])
    pair_id = IntegerField("pair_id", validators=[InputRequired()])
    prediction = StringField("prediction", validators=[InputRequired()])
    certainty = IntegerField("certainty", validators=[InputRequired()])
    starttime = IntegerField("starttime", validators=[InputRequired()])


# This function is used to delete a user from the database and all the records of the user
@app.route("/delete_user", methods=["GET", "POST"])
@login_required
def delete_user():
    form = DeletionForm()
    if current_user.username == ADMIN_USERNAME:
        if form.username.data != ADMIN_USERNAME:
            user_id = User.query.filter_by(username=form.username.data).first().id
            if user_id:
                # Delete answers from survey database
                answers = Survey.query.filter_by(user_id=user_id).with_entities(Survey.id).all()
                for answer_id in answers:
                    db.session.delete(Survey.query.get(answer_id))
                # Delete data from user database
                db.session.delete(User.query.get(user_id))
                # Commit changes
                db.session.commit()
                return redirect(url_for("dashboard"))
            return render_template("error_template.html", message="User not found")
        return render_template("error_template.html", message="You cannot delete the admin user")
    else:
        return render_template("error_template.html", message="Only admin can do this")


# Default route when someone enters the site
@app.route("/", methods=["GET", "POST"])
def index():
    db.create_all()  # Use for instantiating the database ... only needed once
    form = RegistrationForm()
    if request.method == "GET":
        return render_template("index.html", form=form)
    elif form.validate_on_submit():
        available_ids = [5]
        # Find the first available id
        new_id = None
        for i in available_ids:
            if i not in [elem[0] for elem in User.query.with_entities(User.id).all()]:
                new_id = i
                break
        if new_id is None:
            return render_template(
                "error_template.html",
                message="Thank you for your interest. There are currently no more Surveys available, we might add additional Surveys in the future. Please check back later.",
            )
        # Add user to database
        hashed_password = generate_password_hash(form.password.data, method="sha1")
        new_user = User(
            id=new_id if form.username.data != ADMIN_USERNAME else 9999,
            username=form.username.data,
            password=hashed_password,
            age=form.age.data,
            ethnicity=form.ethnicity.data,
            gender=form.gender.data,
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("index.html", form=form, error="Username not available or passwords differ!")


# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if request.method == "GET":
        return render_template("login.html", form=form)
    elif form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for("dashboard"))
            else:
                return render_template("login.html", form=form, error="Invalid password!")
        else:
            return render_template("login.html", form=form, error="Invalid username!")
    return render_template("error_template.html", message="Error, please contact the survey owner! CODE: login")


# Check if username is available
@app.route("/check_username", methods=["POST"])
def check_username():
    username = request.form["username"]
    user = User.query.filter_by(username=username).first()
    if user is not None:
        return jsonify({"available": False})
    else:
        return jsonify({"available": True})


# Dashboard
@app.route("/dashboard")
@login_required
def dashboard():
    # login_user(User.query.get(1), remember=True)  # DEBUGGING
    if current_user.username == ADMIN_USERNAME:
        u_status = {}
        acc_user = {}
        acc_ai = {}
        usernames = [elem[0] for elem in User.query.with_entities(User.username).order_by(User.id).all() if elem[0] != ADMIN_USERNAME]
        register = [
            elem[0] for elem in User.query.with_entities(User.registration_date).order_by(User.id).all() if elem[0] != ADMIN_USERNAME
        ]
        for dataset in DATASETS:
            u_status[dataset] = []
            acc_user[dataset] = []
            acc_ai[dataset] = []
            for username in usernames:
                if username == ADMIN_USERNAME:
                    continue
                uid = User.query.filter_by(username=username).with_entities(User.id).first()[0]
                u_status[dataset].append(
                    len(Survey.query.filter_by(user_id=uid, dataset=dataset).with_entities(Survey.question_id).all())
                    / len(DATA[dataset][uid - 1])
                    if len(DATA[dataset][uid - 1]) != 0
                    else "-"
                )

                question_ids = [
                    elem[0] for elem in Survey.query.filter_by(user_id=uid, dataset=dataset).with_entities(Survey.question_id).all()
                ]
                preds_user = [
                    elem[0] == "1" for elem in Survey.query.filter_by(user_id=uid, dataset=dataset).with_entities(Survey.prediction).all()
                ]
                labels = [elem[4] == "True" for elem in np.asarray(DATA[dataset][uid - 1])[question_ids]]
                preds_ai = [elem[3] == "True" for elem in np.asarray(DATA[dataset][uid - 1])[question_ids]]

                acc_user[dataset].append(np.sum(np.equal(preds_user, labels)) / len(question_ids))
                acc_ai[dataset].append(np.sum(np.equal(preds_ai, labels)) / len(question_ids))
        completed_base_surveys = len([e for e in u_status["base"] if e == 1])
        completed_all_surveys = 0
        for idx in range(len(usernames)):
            cnt = 0
            for dataset in DATASETS:
                if u_status[dataset][idx] == 1:
                    cnt += 1
            if cnt == 5:
                completed_all_surveys += 1

        return render_template(
            "admin.html",
            u_status=u_status,
            completed_base_surveys=completed_base_surveys,
            completed_all_surveys=completed_all_surveys,
            acc_user=acc_user,
            acc_ai=acc_ai,
            register=register,
            usernames=usernames,
            status="",
        )
    else:
        info = []
        for dataset in DATASETS:
            # Calculate Acc for recorded user data and AI from DATA
            question_ids = [
                elem[0] for elem in Survey.query.filter_by(user_id=current_user.id, dataset=dataset).with_entities(Survey.question_id).all()
            ]

            preds_user = [
                elem[0] == "1"
                for elem in Survey.query.filter_by(user_id=current_user.id, dataset=dataset).with_entities(Survey.prediction).all()
            ]
            labels = [elem[4] == "True" for elem in np.asarray(DATA[dataset][current_user.id - 1])[question_ids]]
            preds_ai = [elem[3] == "True" for elem in np.asarray(DATA[dataset][current_user.id - 1])[question_ids]]

            acc_ai = np.sum(np.equal(preds_ai, labels)) / len(question_ids)
            acc_user = np.sum(np.equal(preds_user, labels)) / len(question_ids)

            info.append(
                {
                    "dataset": dataset,
                    "num_answers": len(question_ids),
                    "max_questions": len(DATA[dataset][current_user.id - 1]),
                    "name": SURVEY_NAMES[dataset],
                    "acc_ai": acc_ai,
                    "acc_user": acc_user,
                    "description": SURVEY_DESCRIPTIONS[dataset],
                    "duration": (TIME_PER_QUESTION * len(DATA[dataset][current_user.id - 1])) // 60,
                    "status": (
                        "start"
                        if Survey.query.filter_by(user_id=current_user.id, dataset=dataset).count() == 0
                        else "done"
                        if Survey.query.filter_by(user_id=current_user.id, dataset=dataset).count()
                        == len(DATA[dataset][current_user.id - 1])
                        else "continue"
                    ),
                }
            )
        return render_template("dashboard.html", info=sorted(info, key=lambda x: x["name"]), name=current_user.username)


# A Survey Question
@app.route("/question/<dataset>", methods=["GET"])
@login_required  # DEBUGGING
def question(dataset):
    # login_user(User.query.get(1), remember=True) # DEBUGGING

    form = DataForm()
    survey = Survey.query.filter_by(user_id=current_user.id, dataset=dataset).order_by(desc(Survey.question_id)).first()
    if survey:  # DEBUGGING
        question_id = int(survey.question_id) + 1
    else:
        question_id = 0

    if question_id < len(DATA[dataset][current_user.id - 1]):
        pair_id = int(DATA[dataset][current_user.id - 1][question_id][1])
        pred_ai = DATA[dataset][current_user.id - 1][question_id][2] == "True"
        cert_ai = float(DATA[dataset][current_user.id - 1][question_id][2])
        # DEBUGGING
        return render_template(
            "question.html",
            form=form,
            img1=os.path.join(
                f"https://storage.googleapis.com/face-verification-survey.appspot.com/datasets/{dataset}",
                f"{pair_id:04d}_0.png",
            ),
            img2=os.path.join(
                f"https://storage.googleapis.com/face-verification-survey.appspot.com/datasets/{dataset}",
                f"{pair_id:04d}_1.png",
            ),
            dataset=dataset,
            ai_prediction_on=False,  # question_id % 2, # Enable for AI predictions to be shown
            ai_prediction=pred_ai,
            ai_certainty=cert_ai,
            question_id=question_id,
            pair_id=pair_id,
            max_questions=len(DATA[dataset][current_user.id - 1]),
        )
    elif question_id == len(DATA[dataset][current_user.id - 1]):
        return redirect(url_for("finished"))
    return redirect(url_for("dashboard"))


@app.route("/finished")
@login_required
def finished():
    return render_template("finished.html")


# Collect a answer of a survey question
@app.route("/collect", methods=["POST"])
@login_required
def collect():
    # login_user(User.query.get(1), remember=True) # DEBUGGING
    form = DataForm()
    if form.validate_on_submit():
        answer = Survey.query.filter_by(user_id=current_user.id, dataset=form.dataset.data, pair_id=form.pair_id.data).first()
        if answer:
            answer.prediction = form.prediction.data == "yes"
            answer.certainty = form.certainty.data
            answer.starttime = datetime.fromtimestamp(form.starttime.data / 1000).strftime("%Y-%m-%d %H:%M:%S")
            answer.endtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            new_answer = Survey(
                user_id=current_user.id,
                dataset=form.dataset.data,
                question_id=form.question_id.data,
                pair_id=form.pair_id.data,
                certainty=form.certainty.data,
                prediction=form.prediction.data == "yes",
                starttime=datetime.fromtimestamp(form.starttime.data / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                endtime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            db.session.add(new_answer)
        db.session.commit()

        return redirect(url_for("question", dataset=form.dataset.data, question_id=int(form.question_id.data) + 1))
    return render_template("error_template.html", message="Error, please contact the survey owner! CODE: collect")


# Logout
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("goodbye"))


# Privacy Policy
@app.route("/privacypolicy")
def privacypolicy():
    return render_template("privacypolicy.html")


# Goodbye Page
@app.route("/goodbye")
def goodbye():
    return render_template("goodbye.html")


# Run the app
if __name__ == "__main__":
    app.run()
